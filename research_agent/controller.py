"""Deterministic lifecycle controller for safe research iterations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract
from .logger import ResearchLogger
from .metrics import MetricsValidationError, evaluate_predictions
from .runner import CandidateCallable, ExperimentRunner, RunnerResult
from .research_memory import ResearchMemory
from .safety import ExperimentProposal, SafetyReport, SafetyValidator
from .state import ResearchState


@dataclass(frozen=True)
class IterationResult:
    experiment_id: str
    decision: str
    state: ResearchState
    runner_result: RunnerResult | None = None
    metrics: Mapping[str, Any] | None = None
    error: str | None = None


class ExperimentController:
    """Coordinates one controlled experiment without changing the baseline."""

    def __init__(
        self,
        *,
        logger: ResearchLogger,
        runner: ExperimentRunner,
        validator: SafetyValidator,
        contract: BenchmarkContract = BENCHMARK_CONTRACT,
        state: ResearchState | None = None,
        max_iterations: int | None = None,
        research_memory: ResearchMemory | None = None,
    ) -> None:
        self.logger = logger
        self.runner = runner
        self.validator = validator
        self.contract = contract
        persisted_state = logger.store.read_root_json("state.json")
        self.state = state or (ResearchState.from_dict(persisted_state) if persisted_state else ResearchState())
        self.max_iterations = max_iterations
        self.research_memory = research_memory
        self._historical_configs = [
            record["config"] for record in logger.store.read_iterations() if "config" in record
        ]
        self._persist_state()

    def run_iteration(
        self,
        proposal: ExperimentProposal,
        candidate: CandidateCallable,
        *,
        code_diff: str = "",
    ) -> IterationResult:
        if self.state.stopped:
            message = f"controller stopped: {self.state.stop_reason}"
            self.logger.log_action(
                "iteration_skipped_stopped",
                experiment_id=proposal.experiment_id,
                details={"reason": self.state.stop_reason},
            )
            return IterationResult(proposal.experiment_id, "skipped", self.state, error=message)

        self.logger.log_action(
            "hypothesis_selected",
            experiment_id=proposal.experiment_id,
            details={"hypothesis": proposal.hypothesis, "rationale": proposal.rationale},
        )
        parent_experiment_id = proposal.parent_experiment_id or self.state.current_best_experiment_id
        patch_path = self.logger.record_code_diff(
            proposal.experiment_id,
            code_diff or self._configuration_diff(proposal),
        )
        report = self.validator.validate(proposal, historical_configs=self._historical_configs)
        if not report.passed:
            self.state.invalid_proposals += 1
            self.logger.log_action(
                "safety_rejected",
                experiment_id=proposal.experiment_id,
                details={"violations": list(report.violations)},
            )
            return self._finish_without_run(
                proposal,
                decision="rejected",
                error="; ".join(report.violations),
                recovery=self._restore_message(),
                safety_report=report,
                patch_path=str(patch_path),
                parent_experiment_id=parent_experiment_id,
                count_toward_budget=False,
            )

        self._historical_configs.append(proposal.config)
        self.logger.log_action("safety_passed", experiment_id=proposal.experiment_id)
        runner_result = self.runner.run(
            experiment_id=proposal.experiment_id,
            hypothesis=proposal.hypothesis,
            config=proposal.config,
            candidate=candidate,
            timeout_seconds=proposal.runtime_budget_seconds,
        )
        if runner_result.status != "completed" or runner_result.output is None:
            return self._finish_without_run(
                proposal,
                decision="failed",
                error=runner_result.error or f"runner status: {runner_result.status}",
                recovery=self._restore_message(),
                safety_report=report,
                patch_path=str(patch_path),
                runtime_seconds=runner_result.runtime_seconds,
                runner_result=runner_result,
                parent_experiment_id=parent_experiment_id,
            )

        try:
            self.logger.log_action("evaluation_started", experiment_id=proposal.experiment_id)
            metric_result = evaluate_predictions(
                runner_result.output.user_ids,
                runner_result.output.labels,
                runner_result.output.scores,
                split=proposal.selection_split,
            )
            metrics = metric_result.as_dict()
            self.logger.log_action(
                "metrics_received",
                experiment_id=proposal.experiment_id,
                details={name: metrics[name] for name in self.contract.metric_names},
            )
        except MetricsValidationError as exc:
            return self._finish_without_run(
                proposal,
                decision="failed",
                error=f"metrics validation failed: {exc}",
                recovery=self._restore_message(),
                safety_report=report,
                patch_path=str(patch_path),
                runtime_seconds=runner_result.runtime_seconds,
                runner_result=runner_result,
                parent_experiment_id=parent_experiment_id,
            )

        delta = float(metrics[self.contract.primary_metric]) - self.state.current_best_primary
        is_screening_run = proposal.config.get("fidelity") == "low"
        is_parity_run = proposal.config.get("run_type") == "parity"
        if is_parity_run:
            decision = "parity"
            recovery = None
            self.logger.log_action(
                "parity_completed",
                experiment_id=proposal.experiment_id,
                details={"delta_from_reference": delta},
            )
        elif is_screening_run:
            # Cheap runs rank candidates for promotion; they never compete with
            # the full-budget incumbent or consume the scientific stop budget.
            decision = "screened"
            recovery = self._restore_message()
            self.logger.log_action(
                "candidate_screened",
                experiment_id=proposal.experiment_id,
                details={"delta_from_full_incumbent": delta},
            )
            self.state.screening_runs_completed += 1
        elif delta > self.contract.acceptance_threshold:
            decision = "accepted"
            self.state.current_best_experiment_id = proposal.experiment_id
            self.state.current_best_primary = float(metrics[self.contract.primary_metric])
            self.state.consecutive_non_improvements = 0
            recovery = None
            self.logger.log_action(
                "candidate_accepted",
                experiment_id=proposal.experiment_id,
                details={"delta_primary": delta},
            )
        else:
            decision = "inconclusive" if delta > 0 else "rejected"
            self.state.consecutive_non_improvements += 1
            recovery = self._restore_message()
            self.logger.log_action(
                "candidate_not_promoted",
                experiment_id=proposal.experiment_id,
                details={"decision": decision, "delta_primary": delta},
            )
        return self._complete_iteration(
            proposal,
            decision=decision,
            metrics=metrics,
            delta_primary=delta,
            runtime_seconds=runner_result.runtime_seconds,
            error=None,
            recovery=recovery,
            safety_report=report,
            patch_path=str(patch_path),
            runner_result=runner_result,
            parent_experiment_id=parent_experiment_id,
            count_toward_budget=not is_screening_run,
        )

    def record_manual_intervention(
        self,
        *,
        description: str,
        reason: str,
        effect: str,
        experiment_id: str | None = None,
    ) -> None:
        self.logger.record_manual_intervention(
            description=description,
            reason=reason,
            effect=effect,
            experiment_id=experiment_id,
        )

    def record_implementation_failure(self) -> None:
        """Persist a generated-code failure without spending experiment budget."""
        self.state.implementation_failures += 1
        self._persist_state()

    def begin_plateau_restart(self) -> None:
        """Record a plateau and allow the LLM to investigate a new direction."""
        if self.state.consecutive_non_improvements < self.contract.non_improvement_limit:
            return
        self.state.plateau_restarts += 1
        self.state.consecutive_non_improvements = 0
        self.logger.log_action(
            "plateau_restart_requested",
            details={
                "restart_number": self.state.plateau_restarts,
                "trigger": f"{self.contract.non_improvement_limit} consecutive non-improvements",
                "next_step": "LLM must propose a new research direction",
            },
        )
        self._persist_state()

    def _finish_without_run(
        self,
        proposal: ExperimentProposal,
        *,
        decision: str,
        error: str,
        recovery: str,
        safety_report: SafetyReport,
        patch_path: str,
        runtime_seconds: float = 0.0,
        runner_result: RunnerResult | None = None,
        parent_experiment_id: str,
        count_toward_budget: bool = True,
    ) -> IterationResult:
        self.state.consecutive_non_improvements += 1
        self.logger.log_action(
            "accepted_candidate_restored",
            experiment_id=proposal.experiment_id,
            details={"current_best_experiment_id": self.state.current_best_experiment_id},
        )
        return self._complete_iteration(
            proposal,
            decision=decision,
            metrics={},
            delta_primary=None,
            runtime_seconds=runtime_seconds,
            error=error,
            recovery=recovery,
            safety_report=safety_report,
            patch_path=patch_path,
            runner_result=runner_result,
            parent_experiment_id=parent_experiment_id,
            count_toward_budget=count_toward_budget,
        )

    def _complete_iteration(
        self,
        proposal: ExperimentProposal,
        *,
        decision: str,
        metrics: Mapping[str, Any],
        delta_primary: float | None,
        runtime_seconds: float,
        error: str | None,
        recovery: str | None,
        safety_report: SafetyReport,
        patch_path: str,
        runner_result: RunnerResult | None,
        parent_experiment_id: str,
        count_toward_budget: bool = True,
    ) -> IterationResult:
        if count_toward_budget and proposal.config.get("run_type") != "parity":
            self.state.completed_iterations += 1
            self.state.full_evaluations_completed += 1
        self._apply_stop_rule()
        record = {
            "experiment_id": proposal.experiment_id,
            "parent_experiment_id": parent_experiment_id,
            "hypothesis": proposal.hypothesis,
            "rationale": proposal.rationale,
            "config": dict(proposal.config),
            "changed_factors": list(proposal.changed_factors),
            "direction_id": proposal.research_direction_id,
            "search_strategy": proposal.search_strategy,
            "search_region_id": proposal.search_region_id,
            "metrics": dict(metrics),
            "delta_primary": delta_primary,
            "runtime_seconds": runtime_seconds,
            "runner_metadata": dict(runner_result.output.metadata) if runner_result and runner_result.output else {},
            "decision": decision,
            "error": error,
            "recovery": recovery,
            "safety": {"passed": safety_report.passed, "violations": list(safety_report.violations)},
            "code_diff_path": patch_path,
            "state_after": self.state.as_dict(),
        }
        completed_record = self.logger.record_iteration(record)
        if self.research_memory is not None:
            self.research_memory.append_iteration(
                completed_record,
                source_run=self.logger.store.root.name,
            )
        self._persist_state()
        return IterationResult(
            proposal.experiment_id,
            decision,
            self.state,
            runner_result=runner_result,
            metrics=metrics,
            error=error,
        )

    def _apply_stop_rule(self) -> None:
        if self.state.current_best_primary >= self.contract.target_primary:
            self.state.stop_reason = (
                f"target validation primary reached: {self.state.current_best_primary:.4f} "
                f">= {self.contract.target_primary:.4f}"
            )
        else:
            budget = self.max_iterations or self.contract.max_experiments
            if self.state.completed_iterations >= budget:
                self.state.stop_reason = f"configured experiment budget reached: {budget}"

    def _persist_state(self) -> None:
        self.logger.store.write_root_json("state.json", self.state.as_dict())

    def _restore_message(self) -> str:
        return f"restored accepted candidate pointer: {self.state.current_best_experiment_id}"

    @staticmethod
    def _configuration_diff(proposal: ExperimentProposal) -> str:
        return (
            "# No source-code change was applied.\n"
            "# Controlled configuration change for " + proposal.experiment_id + "\n"
            + repr(dict(proposal.config))
            + "\n"
        )
