"""Deterministic lifecycle controller for safe research iterations."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract, ComparisonValidity
from .logger import ResearchLogger
from .metrics import MetricsValidationError, evaluate_predictions
from .runner import CandidateCallable, ExperimentRunner, RunnerResult
from .safety import ExperimentProposal, SafetyReport, SafetyValidator
from .state import ResearchState


@dataclass(frozen=True)
class IterationResult:
    experiment_id: str
    decision: str
    state: ResearchState
    runner_result: RunnerResult | None = None
    metrics: Mapping[str, Any] | None = None
    comparison_validity: ComparisonValidity | None = None
    error: str | None = None


Clock = Callable[[], float]


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
        require_seed_confirmation: bool = False,
        clock: Clock = time.time,
    ) -> None:
        self.logger = logger
        self.runner = runner
        self.validator = validator
        self.contract = contract
        persisted_state = logger.store.read_root_json("state.json")
        iteration_records = logger.store.read_iterations()
        history_state = (
            ResearchState.from_dict(iteration_records[-1]["state_after"])
            if iteration_records and isinstance(iteration_records[-1].get("state_after"), dict)
            else None
        )
        recovered = ResearchState.from_dict(persisted_state) if persisted_state else None
        if history_state and (
            recovered is None
            or history_state.completed_iterations > recovered.completed_iterations
        ):
            recovered = history_state
            logger.log_action(
                "state_reconciled_from_iteration_history",
                details={"completed_iterations": history_state.completed_iterations},
            )
        resolutions = logger.store.read_root_json("promotion_resolutions.json") or {}
        resolution_states = [
            ResearchState.from_dict(value["state_after"])
            for value in resolutions.values()
            if isinstance(value, dict) and isinstance(value.get("state_after"), dict)
        ]
        if resolution_states:
            latest_resolution = max(
                resolution_states,
                key=lambda item: (item.completed_iterations, item.valid_comparisons),
            )
            if recovered is None or (
                latest_resolution.completed_iterations,
                latest_resolution.valid_comparisons,
            ) > (recovered.completed_iterations, recovered.valid_comparisons):
                recovered = latest_resolution
                logger.log_action(
                    "state_reconciled_from_promotion_resolution",
                    details={
                        "completed_iterations": recovered.completed_iterations,
                        "valid_comparisons": recovered.valid_comparisons,
                    },
                )
        self.state = state or recovered or ResearchState()
        self.max_iterations = max_iterations
        self.require_seed_confirmation = require_seed_confirmation
        self.clock = clock
        if self.state.started_at_epoch_seconds is None:
            self.state.started_at_epoch_seconds = self.clock()
        # Anchor for THIS invocation; paused time between invocations is never
        # added to the research budget.
        self._invocation_started_at = self.clock()
        self._historical_configs = [
            record["config"]
            for record in iteration_records
            if record.get("metrics")
            and record.get("decision")
            in {"screened", "accepted", "rejected", "inconclusive", "invalid"}
        ]
        self._apply_stop_rule()
        self._persist_state()

    def run_iteration(
        self,
        proposal: ExperimentProposal,
        candidate: CandidateCallable,
        *,
        code_diff: str = "",
    ) -> IterationResult:
        self._apply_stop_rule()
        usable_deadline = (
            self.contract.max_wall_clock_seconds
            - self.contract.finalization_reserve_seconds
        )
        if (
            not self.state.stopped
            and self.state.elapsed_seconds + proposal.runtime_budget_seconds > usable_deadline
        ):
            self._stop(
                "wall_clock_budget",
                "insufficient wall-clock budget for the next trial while preserving "
                f"the {self.contract.finalization_reserve_seconds:g}-second finalization reserve",
            )
        if self.state.stopped:
            self._persist_state()
            message = f"controller stopped: {self.state.stop_reason}"
            self.logger.log_action(
                "iteration_skipped_stopped",
                experiment_id=proposal.experiment_id,
                details={"reason": self.state.stop_reason},
            )
            return IterationResult(
                proposal.experiment_id,
                "skipped",
                self._state_snapshot(),
                error=message,
            )

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
            )

        self.logger.log_action("safety_passed", experiment_id=proposal.experiment_id)
        runner_result = self._run_with_transient_retry(proposal, candidate)
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

        self._historical_configs.append(proposal.config)
        incumbent_id = proposal.comparison_incumbent_id or parent_experiment_id
        incumbent_metadata, incumbent_config = self._incumbent_evidence(
            incumbent_id,
            proposal.config,
            runner_result.output.metadata,
        )
        comparison = ComparisonValidity.assess(
            candidate_experiment_id=proposal.experiment_id,
            config=proposal.config,
            selection_split=proposal.selection_split,
            metrics=metrics,
            runner_metadata=runner_result.output.metadata,
            incumbent_primary=self.state.current_best_primary,
            incumbent_experiment_id=incumbent_id,
            incumbent_metadata=incumbent_metadata,
            incumbent_config=incumbent_config,
            contract=self.contract,
        )
        delta = comparison.delta_primary
        if not comparison.valid:
            decision = "screened" if comparison.fidelity != "full" else "invalid"
            recovery = None
            self.logger.log_action(
                "comparison_invalid",
                experiment_id=proposal.experiment_id,
                details=comparison.as_dict(),
            )
        elif proposal.config.get("calibration_only") is True:
            decision = "calibrated"
            recovery = None
            self.logger.log_action(
                "screen_full_calibration_completed",
                experiment_id=proposal.experiment_id,
                details={
                    "primary": comparison.primary_score,
                    "source_screen_experiment_id": parent_experiment_id,
                },
            )
        elif (
            delta is not None
            and self._is_meaningful_improvement(delta)
            and self.require_seed_confirmation
        ):
            decision = "pending_confirmation"
            recovery = None
            self.logger.log_action(
                "candidate_pending_seed_confirmation",
                experiment_id=proposal.experiment_id,
                details={"single_seed_delta_primary": delta},
            )
        elif delta is not None and self._is_meaningful_improvement(delta):
            decision = "accepted"
            self.state.current_best_experiment_id = proposal.experiment_id
            assert comparison.primary_score is not None
            self.state.current_best_primary = comparison.primary_score
            self.state.valid_comparisons += 1
            self.state.consecutive_non_improvements = 0
            recovery = None
            self.logger.log_action(
                "candidate_accepted",
                experiment_id=proposal.experiment_id,
                details={"delta_primary": delta},
            )
        else:
            decision = "inconclusive" if delta is not None and delta > 0 else "rejected"
            self.state.valid_comparisons += 1
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
            comparison_validity=comparison,
        )

    def resolve_seed_confirmation(
        self,
        experiment_id: str,
        certificate: Mapping[str, Any],
    ) -> IterationResult:
        """Apply an independently persisted three-seed promotion certificate."""
        record = next(
            (
                item
                for item in reversed(self.logger.store.read_iterations())
                if item.get("experiment_id") == experiment_id
            ),
            None,
        )
        if record is None or record.get("decision") != "pending_confirmation":
            raise ValueError("seed confirmation must resolve a pending full-fidelity candidate")
        normalized = self._validate_seed_certificate(record, certificate)
        confirmations = self.logger.store.read_root_json("promotion_confirmations.json") or {}
        persisted_certificate = confirmations.get(experiment_id)
        if not isinstance(persisted_certificate, dict):
            raise ValueError("seed confirmation must be backed by persisted runner evidence")
        persisted_normalized = self._validate_seed_certificate(record, persisted_certificate)
        if normalized != persisted_normalized:
            raise ValueError("seed confirmation differs from the persisted runner evidence")
        resolutions = self.logger.store.read_root_json("promotion_resolutions.json") or {}
        existing = resolutions.get(experiment_id)
        if existing:
            if existing.get("certificate") != normalized:
                raise ValueError("resolved seed certificate cannot be changed or replayed")
            stored_state = ResearchState.from_dict(existing.get("state_after") or {})
            self.state = stored_state
            return IterationResult(
                experiment_id,
                str(existing["decision"]),
                self._state_snapshot(),
                metrics=record.get("metrics") or {},
            )
        candidate_scores = normalized["candidate_scores"]
        self.state.valid_comparisons += 1
        self.state.runner_attempts += int(normalized["confirmation_attempts"])
        if normalized["confirmed"] is True:
            decision = "accepted"
            self.state.current_best_experiment_id = experiment_id
            self.state.current_best_primary = sum(float(score) for score in candidate_scores) / 3
            self.state.consecutive_non_improvements = 0
            error = None
        else:
            decision = "rejected"
            self.state.consecutive_non_improvements += 1
            error = "three-seed margin did not satisfy the promotion threshold"
        self._apply_stop_rule()
        resolution = {
            "selected_experiment_id": experiment_id,
            "decision": decision,
            "certificate": normalized,
            "state_after": self.state.as_dict(),
        }
        resolutions[experiment_id] = resolution
        self.logger.store.write_root_json("promotion_resolutions.json", resolutions)
        self._persist_state()
        self.logger.log_action(
            "seed_confirmation_resolved",
            experiment_id=experiment_id,
            details={"decision": decision, "certificate": normalized},
        )
        return IterationResult(
            experiment_id,
            decision,
            self._state_snapshot(),
            metrics=record.get("metrics") or {},
            error=error,
        )

    def _validate_seed_certificate(
        self,
        record: Mapping[str, Any],
        certificate: Mapping[str, Any],
    ) -> dict[str, Any]:
        experiment_id = str(record["experiment_id"])
        comparator_id = str(record.get("comparison_incumbent_id") or "baseline")
        if certificate.get("selected_experiment_id") != experiment_id:
            raise ValueError("seed certificate belongs to a different candidate")
        if certificate.get("comparator_experiment_id") != comparator_id:
            raise ValueError("seed certificate uses the wrong comparison incumbent")
        if certificate.get("seeds") != [0, 1, 2]:
            raise ValueError("seed certificate must use the fixed seeds 0, 1, and 2")
        candidate_scores = certificate.get("candidate_scores")
        comparator_scores = certificate.get("comparator_scores")
        if not isinstance(candidate_scores, list) or len(candidate_scores) != 3:
            raise ValueError("seed certificate must contain three candidate scores")
        if not isinstance(comparator_scores, list) or len(comparator_scores) != 3:
            raise ValueError("seed certificate must contain three comparator scores")
        try:
            candidate_values = [float(value) for value in candidate_scores]
            comparator_values = [float(value) for value in comparator_scores]
        except (TypeError, ValueError) as exc:
            raise ValueError("seed certificate scores must be numeric") from exc
        if not all(math.isfinite(value) for value in candidate_values + comparator_values):
            raise ValueError("seed certificate scores must be finite")
        candidate_groups = certificate.get("candidate_comparison_groups")
        comparator_groups = certificate.get("comparator_comparison_groups")
        if (
            not isinstance(candidate_groups, list)
            or not isinstance(comparator_groups, list)
            or len(candidate_groups) != 3
            or candidate_groups != comparator_groups
            or not all(
                isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                for value in candidate_groups
            )
        ):
            raise ValueError("seed certificate comparison lineage is not matched")
        deltas = [
            candidate - comparator
            for candidate, comparator in zip(candidate_values, comparator_values)
        ]
        mean_delta = sum(deltas) / 3
        wins = sum(delta > self.contract.improvement_threshold for delta in deltas)
        confirmed = (
            self._is_meaningful_improvement(mean_delta)
            and wins >= 2
        )
        raw_mean_delta = certificate.get("mean_delta")
        if isinstance(raw_mean_delta, bool) or not isinstance(
            raw_mean_delta,
            (int, float),
        ):
            raise ValueError("seed certificate mean delta must be numeric")
        recorded_mean_delta = float(raw_mean_delta)
        if not math.isfinite(recorded_mean_delta) or not math.isclose(
            recorded_mean_delta,
            mean_delta,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("seed certificate mean delta does not match its scores")
        if (
            not isinstance(certificate.get("wins"), int)
            or isinstance(certificate.get("wins"), bool)
            or certificate.get("wins") != wins
            or not isinstance(certificate.get("confirmed"), bool)
            or certificate.get("confirmed") is not confirmed
        ):
            raise ValueError("seed certificate decision does not match recomputed evidence")
        attempts = certificate.get("confirmation_attempts")
        expected_attempts = 6 if comparator_id == "baseline" else 3
        if (
            not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or attempts != expected_attempts
        ):
            raise ValueError("seed certificate has invalid attempt accounting")
        checkpoint = certificate.get("submission_checkpoint_path")
        if confirmed:
            if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
                raise ValueError("confirmed seed certificate lacks its seed-0 checkpoint")
            try:
                Path(checkpoint).resolve().relative_to(
                    self.logger.store.runs_dir.resolve()
                )
            except ValueError as exc:
                raise ValueError(
                    "confirmed seed checkpoint is outside the append-only run artifacts"
                ) from exc
        return {
            **dict(certificate),
            "candidate_scores": candidate_values,
            "comparator_scores": comparator_values,
            "mean_delta": mean_delta,
            "wins": wins,
            "confirmed": confirmed,
        }
    def record_critic_rejection(
        self,
        proposal: ExperimentProposal,
        *,
        reasons: Sequence[str],
    ) -> IterationResult:
        """Record a planning rejection without creating or running an experiment."""
        message = "; ".join(reasons) or "proposal critic rejected the candidate"
        self.logger.log_action(
            "proposal_rejected_before_execution",
            experiment_id=proposal.experiment_id,
            details={"reasons": list(reasons)},
        )
        self.state.planning_rejections += 1
        self._persist_state()
        return IterationResult(
            experiment_id=proposal.experiment_id,
            decision="critic_rejected",
            state=self._state_snapshot(),
            error=message,
        )

    def ensure_cycle_capacity(self, required_trials: int) -> bool:
        """Stop before a cycle when its screen/promotion envelope cannot fit."""
        if required_trials <= 0:
            raise ValueError("required_trials must be positive")
        self._apply_stop_rule()
        budget = self.max_iterations or self.contract.max_iterations
        if (
            not self.state.stopped
            and self.state.completed_iterations + required_trials > budget
        ):
            self._stop(
                "iteration_budget",
                "insufficient trial budget for an atomic research cycle: "
                f"remaining={budget - self.state.completed_iterations}, "
                f"required={required_trials}",
            )
            self._persist_state()
        return not self.state.stopped

    def complete_cycle(self) -> None:
        self.state.completed_cycles += 1
        self.state.last_pause_reason = None
        self._persist_state()

    def set_screen_promotion_width(
        self,
        width: int,
        *,
        rank_correlation: float | None,
    ) -> None:
        if width not in {1, 2}:
            raise ValueError("screen promotion width must be one or two")
        previous = self.state.screen_promotion_width
        self.state.screen_promotion_width = width
        self._persist_state()
        self.logger.log_action(
            "screen_promotion_policy_updated",
            details={
                "previous_width": previous,
                "new_width": width,
                "rank_correlation": rank_correlation,
                "threshold": 0.5,
            },
        )

    def ensure_wall_clock_capacity(self, required_seconds: float) -> bool:
        if required_seconds <= 0:
            raise ValueError("required_seconds must be positive")
        self._apply_stop_rule()
        usable_deadline = (
            self.contract.max_wall_clock_seconds
            - self.contract.finalization_reserve_seconds
        )
        if (
            not self.state.stopped
            and self.state.elapsed_seconds + required_seconds > usable_deadline
        ):
            self._stop(
                "wall_clock_budget",
                "insufficient wall-clock budget for full promotion and matched-seed confirmation",
            )
            self._persist_state()
        return not self.state.stopped

    def stop_search_exhausted(self, reason: str) -> None:
        """Pause for new search-space evidence; this is not a competition stop rule."""
        if self.state.stopped:
            return
        self.state.last_pause_reason = reason
        self._persist_state()
        self.logger.log_action(
            "search_paused_exhausted",
            details={"reason": reason, "terminal": False},
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

    def begin_plateau_restart(self) -> None:
        """Compatibility hook: a competition plateau is terminal, not resettable."""
        if self.state.consecutive_non_improvements < self.contract.non_improvement_limit:
            return
        self.logger.log_action(
            "plateau_restart_refused",
            details={
                "trigger": f"{self.contract.non_improvement_limit} consecutive non-improvements",
                "reason": "the benchmark stopping contract makes the plateau terminal",
            },
        )
        self._apply_stop_rule()
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
    ) -> IterationResult:
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
        comparison_validity: ComparisonValidity | None = None,
    ) -> IterationResult:
        self.state.completed_iterations += 1
        self._apply_stop_rule()
        record = {
            "experiment_id": proposal.experiment_id,
            "parent_experiment_id": parent_experiment_id,
            "comparison_incumbent_id": (
                proposal.comparison_incumbent_id or parent_experiment_id
            ),
            "hypothesis": proposal.hypothesis,
            "rationale": proposal.rationale,
            "config": dict(proposal.config),
            "changed_factors": list(proposal.changed_factors),
            "direction_id": proposal.research_direction_id,
            "search_strategy": proposal.search_strategy,
            "search_region_id": proposal.search_region_id,
            "metrics": dict(metrics),
            "delta_primary": delta_primary,
            "comparison_validity": comparison_validity.as_dict() if comparison_validity else None,
            "runtime_seconds": runtime_seconds,
            "runner_metadata": dict(runner_result.output.metadata) if runner_result and runner_result.output else {},
            "decision": decision,
            "error": error,
            "recovery": recovery,
            "safety": {"passed": safety_report.passed, "violations": list(safety_report.violations)},
            "code_diff_path": patch_path,
            "state_after": self.state.as_dict(),
        }
        self.logger.record_iteration(record)
        self._persist_state()
        return IterationResult(
            proposal.experiment_id,
            decision,
            self._state_snapshot(),
            runner_result=runner_result,
            metrics=metrics,
            comparison_validity=comparison_validity,
            error=error,
        )

    def _apply_stop_rule(self) -> None:
        anchor = getattr(self, "_invocation_started_at", None)
        if anchor is not None:
            self.state.elapsed_seconds = self.state.active_runtime_seconds + max(
                0.0, self.clock() - anchor
            )
        if self.state.stop_reason is not None:
            return
        if self.state.consecutive_non_improvements >= self.contract.non_improvement_limit:
            self._stop(
                "plateau",
                f"{self.contract.non_improvement_limit} consecutive valid non-improvements",
            )
            return
        budget = self.max_iterations or self.contract.max_iterations
        if self.state.completed_iterations >= budget:
            self._stop("iteration_budget", f"configured iteration budget reached: {budget}")
            return
        if self.state.elapsed_seconds >= self.contract.max_wall_clock_seconds:
            self._stop(
                "wall_clock_budget",
                f"wall-clock budget reached: {self.contract.max_wall_clock_seconds:g} seconds",
            )

    def _stop(self, code: str, reason: str) -> None:
        self.state.stop_reason_code = code
        self.state.stop_reason = reason

    def _is_meaningful_improvement(self, delta: float) -> bool:
        threshold = self.contract.improvement_threshold
        return delta > threshold and not math.isclose(
            delta,
            threshold,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    def checkpoint_active_runtime(self) -> None:
        """Fold this invocation's elapsed time into the carried-forward total."""
        anchor = getattr(self, "_invocation_started_at", None)
        if anchor is None:
            return
        now = self.clock()
        self.state.active_runtime_seconds += max(0.0, now - anchor)
        self._invocation_started_at = now
        self.state.elapsed_seconds = self.state.active_runtime_seconds
        self._persist_state()

    def _persist_state(self) -> None:
        self.logger.store.write_root_json("state.json", self.state.as_dict())

    def _state_snapshot(self) -> ResearchState:
        return ResearchState.from_dict(self.state.as_dict())

    def _run_with_transient_retry(
        self,
        proposal: ExperimentProposal,
        candidate: CandidateCallable,
    ) -> RunnerResult:
        result: RunnerResult | None = None
        for attempt in range(1, 3):
            self.state.runner_attempts += 1
            result = self.runner.run(
                experiment_id=proposal.experiment_id,
                hypothesis=proposal.hypothesis,
                config=proposal.config,
                candidate=candidate,
                timeout_seconds=proposal.runtime_budget_seconds,
            )
            if (
                result.status == "completed"
                or result.failure_kind != "transient"
                or attempt == 2
            ):
                return result
            self.logger.log_action(
                "transient_trial_retry_scheduled",
                experiment_id=proposal.experiment_id,
                details={"completed_attempt": attempt, "maximum_attempts": 2},
            )
        assert result is not None
        return result

    def _restore_message(self) -> str:
        return f"restored accepted candidate pointer: {self.state.current_best_experiment_id}"

    def _incumbent_evidence(
        self,
        incumbent_id: str,
        candidate_config: Mapping[str, Any],
        candidate_metadata: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if incumbent_id == "baseline":
            shared = {
                key: candidate_metadata.get(key)
                for key in (
                    "data_sha256",
                    "evaluator_sha256",
                    "preprocessing_sha256",
                    "staging_code_sha256",
                    "feature_schema_sha256",
                    "comparison_group_id",
                    "seed",
                )
            }
            return (
                {
                    **shared,
                    "model_code_sha256": "organizer-baseline-reference",
                    "score_provenance": "organizer_published_five_seed_mean",
                },
                {
                    "epochs": self.contract.full_max_epochs,
                    "patience": self.contract.full_patience,
                    "seed": candidate_config.get("seed"),
                },
            )
        for record in reversed(self.logger.store.read_iterations()):
            if record.get("experiment_id") == incumbent_id:
                return (
                    dict(record.get("runner_metadata") or {}),
                    dict(record.get("config") or {}),
                )
        return {}, {}

    @staticmethod
    def _configuration_diff(proposal: ExperimentProposal) -> str:
        return (
            "# No source-code change was applied.\n"
            "# Controlled configuration change for " + proposal.experiment_id + "\n"
            + repr(dict(proposal.config))
            + "\n"
        )
