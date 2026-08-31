"""Self-extending LLM research loop for in-scope PyTorch FM capabilities."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Any, Mapping

from .agent_team import GeneratedCandidate, LLMResearchTeam, build_isolated_candidate
from .controller import ExperimentController
from .fidelity import FidelityManager
from .logger import ResearchLogger
from .runner import CandidateCallable, PreparedData
from .safety import ExperimentProposal
from .models.torch_fm import run_torch_fm_candidate
from .planner import ResearchDirection
from .research_memory import ResearchMemory


class BroadAutonomousLoop:
    """Lets the LLM execute complete, checked candidates in isolated workspaces."""

    max_repairs_per_candidate = 1
    novelty_cooldown = 3

    def __init__(
        self,
        controller: ExperimentController,
        logger: ResearchLogger,
        team: LLMResearchTeam,
        research_memory: ResearchMemory | None = None,
    ) -> None:
        self.controller, self.logger, self.team = controller, logger, team
        self.fidelity = FidelityManager()
        self.research_memory = research_memory
        self._screened_candidates: dict[str, tuple[Any, ExperimentProposal, CandidateCallable, str]] = {}

    def run(self, max_cycles: int) -> int:
        completed = 0
        self._run_trusted_parity_once()
        for _ in range(max_cycles):
            if self.controller.state.stopped:
                break
            history = self.logger.store.read_iterations()
            plan, meta = self._propose_novel(history)
            if plan is None:
                continue
            hypothesis_id = self._next_hypothesis_id()
            experiment_id = self._next_experiment_id()
            fingerprint = _proposal_fingerprint(plan)
            self.logger.log_action(
                "llm_broad_hypothesis_proposed",
                experiment_id=experiment_id,
                details={**plan.__dict__, "llm": meta, "hypothesis_id": hypothesis_id, "fingerprint": fingerprint},
            )
            critique, critique_meta = self.team.critique(plan)
            automatic = (
                critique.decision == "approved" and not plan.requires_human_review
                and plan.implementation_surface != "human_review" and _normalize_model_family(plan.model_family) == "fm"
            )
            self.logger.log_action("llm_critic_completed", details={"decision": critique.decision, "rationale": critique.rationale, "llm": critique_meta, "automatic_execution": automatic})
            if not automatic:
                self.logger.log_action("human_review_required", details={"proposal": plan.__dict__, "reason": critique.rationale})
                continue

            built = self._build_isolated_candidate(plan, experiment_id)
            if built is None:
                self.controller.record_implementation_failure()
                continue
            candidate, source, config, verification = built
            workspace = self.logger.store.root / "candidate_workspaces" / experiment_id
            proposal = ExperimentProposal(
                experiment_id=experiment_id,
                parent_experiment_id=self.controller.state.current_best_experiment_id,
                hypothesis=plan.hypothesis,
                rationale=plan.rationale,
                config=config,
                changed_factors=(plan.controlled_change,),
                model_family="fm",
                research_direction_id=plan.area,
                search_strategy="llm_isolated_candidate",
                search_region_id=f"region_{plan.area}",
            )
            self.logger.log_action("isolated_candidate_preflight_passed", experiment_id=experiment_id, details={"workspace": str(workspace), "verification": verification})
            iteration = self.controller.run_iteration(proposal, candidate, code_diff=source)
            if iteration.decision == "screened":
                self._screened_candidates[proposal.experiment_id] = (plan, proposal, candidate, source)
            promoted = self._maybe_promote(iteration)
            completed += 1 + int(promoted is not None)
            review, review_meta = self.team.review(self.logger.store.read_iterations())
            self.logger.log_action("llm_evidence_review_completed", details={**review, "llm": review_meta})
        return completed

    def _maybe_promote(self, iteration: Any) -> Any | None:
        """Promote the global screening leader rather than waiting per area."""
        if not iteration.metrics:
            return None
        history = self.logger.store.read_iterations()
        screens = [
            item for item in history
            if item.get("search_strategy") == "llm_isolated_candidate"
            and item.get("config", {}).get("fidelity") == "low"
            and isinstance(item.get("metrics", {}).get("primary"), (int, float))
        ]
        if len(screens) < self.fidelity.min_low_fidelity_trials:
            return None
        promoted_parents = {
            item.get("parent_experiment_id") for item in history
            if item.get("config", {}).get("fidelity") == "full"
        }
        leader = max(
            (item for item in screens if item.get("experiment_id") not in promoted_parents),
            key=lambda item: float(item["metrics"]["primary"]),
            default=None,
        )
        if leader is None or leader["experiment_id"] not in self._screened_candidates:
            return None
        plan, proposal, candidate, source = self._screened_candidates[leader["experiment_id"]]
        direction = ResearchDirection(
            direction_id="global_autonomous_screening", hypothesis=plan.hypothesis, rationale=plan.rationale,
            search_space={}, success_evidence="full-budget validation improvement over the incumbent",
            evaluation_budget={"low_epochs": 4, "full_epochs": 40, "patience": 4}, strategy="llm_isolated_candidate",
        )
        if not self.fidelity.should_promote(leader["metrics"], self.controller.state.current_best_primary, proposal=proposal, history=history, global_pool=True):
            return None
        promoted = self.fidelity.promote(proposal, direction, experiment_id=self._next_experiment_id())
        self.logger.log_action("autonomous_candidate_promoted", experiment_id=promoted.experiment_id, details={"parent_experiment_id": proposal.experiment_id, "area": plan.area})
        return self.controller.run_iteration(promoted, candidate, code_diff=source)

    def _run_trusted_parity_once(self) -> None:
        """Establish the reviewed PyTorch FM path before LLM extensions."""
        if self.controller.state.completed_iterations or any(
            event["action"] == "trusted_pytorch_parity_started" for event in self.logger.store.read_events()
        ):
            return
        experiment_id = self._next_experiment_id()
        config = {
            **_candidate_config({}),
            "loss": "pointwise",
            "epochs": 40,
            "patience": 4,
            "fidelity": "full",
            "run_type": "parity",
            "extension_name": "trusted_pytorch_fm_parity",
        }
        proposal = ExperimentProposal(
            experiment_id=experiment_id,
            parent_experiment_id=self.controller.state.current_best_experiment_id,
            hypothesis="The reviewed PyTorch FM path can reproduce the canonical data.py-to-official-evaluator benchmark flow.",
            rationale="Framework parity is required before autonomous hooks are compared; it changes no data, feature, sampling, or metric behavior.",
            config=config,
            changed_factors=("trusted_pytorch_fm_parity",),
            model_family="fm",
            research_direction_id="framework_validation",
            search_strategy="trusted_parity",
            search_region_id="region_framework_validation",
        )
        self.logger.log_action("trusted_pytorch_parity_started", experiment_id=experiment_id)
        self.controller.run_iteration(proposal, run_torch_fm_candidate, code_diff="trusted host runtime: research_agent/models/torch_fm.py")

    def _propose_novel(self, history: list[Mapping[str, Any]]) -> tuple[Any | None, dict[str, Any]]:
        blocked = self._recent_proposal_memory()
        state = {
            **self.controller.state.as_dict(),
            "proposal_memory": blocked,
            "novelty_rule": "Do not repeat a blocked fingerprint, controlled change, or area unless new completed metric evidence directly justifies it.",
        }
        for attempt in range(3):
            plan, meta = _propose_with_memory(
                self.team,
                history,
                state,
                self.research_memory.planner_summary() if self.research_memory is not None else None,
            )
            fingerprint = _proposal_fingerprint(plan)
            if fingerprint not in {item["fingerprint"] for item in blocked}:
                return plan, meta
            self.logger.log_action(
                "hypothesis_suppressed_as_duplicate",
                details={"fingerprint": fingerprint, "planning_attempt": attempt + 1, "reason": "recently blocked or attempted without new evidence"},
            )
            blocked.append({"fingerprint": fingerprint, "reason": "duplicate proposal rejected"})
        return None, {}

    def _recent_proposal_memory(self) -> list[dict[str, str]]:
        events = self.logger.store.read_events()
        proposals: dict[str, dict[str, str]] = {}
        blocked: set[str] = set()
        for event in events:
            details = event.get("details", {})
            if event["action"] == "llm_broad_hypothesis_proposed":
                proposals[str(event.get("experiment_id"))] = {
                    "fingerprint": str(details.get("fingerprint", "")),
                    "area": str(details.get("area", "")),
                    "controlled_change": str(details.get("controlled_change", "")),
                    "reason": "recent proposal",
                }
            if event["action"] in {"isolated_candidate_rejected", "candidate_abandoned", "candidate_failure_recorded"}:
                blocked.add(str(event.get("experiment_id")))
        recent = list(proposals.items())[-self.novelty_cooldown :]
        return [{**details, "reason": "blocked" if key in blocked else details["reason"]} for key, details in recent if details["fingerprint"]]

    def _next_experiment_id(self) -> str:
        numbers = [
            int(match.group(1))
            for event in self.logger.store.read_events()
            if (match := re.fullmatch(r"exp_(\d+)", str(event.get("experiment_id") or "")))
        ]
        return f"exp_{max(numbers, default=0) + 1:03d}"

    def _next_hypothesis_id(self) -> str:
        count = sum(event["action"] == "llm_broad_hypothesis_proposed" for event in self.logger.store.read_events())
        return f"hyp_{count + 1:05d}"

    def _build_isolated_candidate(
        self, plan: Any, experiment_id: str
    ) -> tuple[CandidateCallable, str, dict[str, Any], Mapping[str, Any]] | None:
        """Preflight a complete candidate; hook registration is never a training gate."""
        failure = ""
        attempted_sources: set[str] = set()
        for attempt in range(self.max_repairs_per_candidate + 1):
            if attempt:
                decision, decision_meta = self.team.decide_candidate_recovery(plan, failure)
                self.logger.log_action("candidate_recovery_decided", experiment_id=experiment_id, details={**decision, "llm": decision_meta, "attempt": attempt})
                if decision["decision"] != "repair":
                    self.logger.log_action("candidate_abandoned", experiment_id=experiment_id, details={"reason": decision["rationale"]})
                    return None
            generated, code_meta = _code_with_failure_context(self.team, plan, failure)
            if isinstance(generated, str):  # compatibility with focused test doubles
                generated = GeneratedCandidate(generated, {})
            source, config = generated.source, _candidate_config(generated.config_patch)
            config["epochs"] = 4
            config["fidelity"] = "low"
            source_hash = hashlib.sha256(source.encode()).hexdigest()
            if source_hash in attempted_sources:
                failure = "unchanged generated source after repair request"
                self.logger.log_action("candidate_abandoned", experiment_id=experiment_id, details={"reason": failure})
                self._record_candidate_failure(experiment_id, failure, attempt, source)
                return None
            attempted_sources.add(source_hash)
            workspace = self.logger.store.root / "candidate_workspaces" / experiment_id
            try:
                candidate = build_isolated_candidate(source, workspace)
                self.logger.log_action(
                    "isolated_candidate_static_check_passed",
                    experiment_id=experiment_id,
                    details={"attempt": attempt, "source_sha256": source_hash},
                )
                self._preflight(candidate, config, experiment_id)
                verification = {"static_check": "passed", "preflight": "passed", "attempt": attempt, "source_sha256": source_hash}
                self.logger.log_action("isolated_candidate_preflight_completed", experiment_id=experiment_id, details={"attempt": attempt, "config": config})
                return candidate, source, config, verification
            except Exception as exc:  # Generated candidate failures must become repair evidence.
                failure = f"{type(exc).__name__}: {exc}"
                self.logger.log_action("isolated_candidate_preflight_failed", experiment_id=experiment_id, details={"error": failure, "llm": code_meta, "attempt": attempt})
                self._record_candidate_failure(experiment_id, failure, attempt, source)
        self.logger.log_action("isolated_candidate_rejected", experiment_id=experiment_id, details={"reason": f"repair budget exhausted: {failure}"})
        return None

    def _record_candidate_failure(self, experiment_id: str, failure: str, attempt: int, source: str) -> None:
        event = self.logger.log_action(
            "candidate_failure_recorded",
            experiment_id=experiment_id,
            details={
                "attempt_id": f"{experiment_id}_attempt_{attempt + 1:02d}",
                "failure_class": _failure_class(failure),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "reason": failure,
            },
        )
        if self.research_memory is not None:
            self.research_memory.append_failure_event(event, source_run=self.logger.store.root.name)

    def _preflight(self, candidate: CandidateCallable, config: Mapping[str, Any], experiment_id: str) -> None:
        """Run a small real data.py-backed execution before screening training."""
        prepared = self.controller.runner._load_prepared_data()
        sample = replace(
            prepared,
            train_rows=prepared.train_rows[:4096], validation_rows=prepared.validation_rows[:4096],
            train_features=prepared.train_features[:4096], validation_features=prepared.validation_features[:4096],
            train_labels=prepared.train_labels[:4096], validation_labels=prepared.validation_labels[:4096],
            train_user_ids=prepared.train_user_ids[:4096], validation_user_ids=prepared.validation_user_ids[:4096],
        )
        preflight_config = {**config, "epochs": 1, "patience": 1, "batch_size": min(int(config.get("batch_size", 1024)), 1024)}
        preflight_dir = self.logger.store.root / "candidate_workspaces" / experiment_id / "preflight"
        preflight_dir.mkdir(parents=True, exist_ok=True)
        output = candidate(sample, preflight_config, preflight_dir)
        if not (len(output.user_ids) == len(output.labels) == len(output.scores) == len(sample.validation_rows)):
            raise RuntimeError("preflight output must align exactly to the canonical validation sample")


def _normalize_model_family(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"fm", "factorization machine", "unchanged baseline model family", "baseline fm", "unchanged fm", "pytorch fm", "existing pytorch fm"}:
        return "fm"
    if "baseline model family" in normalized or "factorization machine" in normalized or "pytorch fm" in normalized:
        return "fm"
    return normalized


def _candidate_config(patch: Mapping[str, Any]) -> dict[str, Any]:
    """Start every screen from parity, recording only supported candidate settings."""
    config = {
        "loss": "pointwise", "learning_rate": 0.001, "l2": 1e-6,
        "embedding_dim": 16, "batch_size": 8192, "seed": 0,
        "epochs": 4, "patience": 3, "fidelity": "low",
        "extension_name": str(patch.get("extension_name", "autonomous_extension")),
    }
    allowed = {"loss", "learning_rate", "l2", "embedding_dim", "batch_size", "seed", "epochs", "patience", "extension_name"}
    config.update({key: value for key, value in patch.items() if key in allowed})
    return config


def _code_with_failure_context(team: Any, plan: Any, failure: str) -> tuple[Any, dict[str, Any]]:
    try:
        return team.code(plan, failure=failure)
    except TypeError:
        # Compatibility for focused test doubles and legacy adapters.
        return team.code(plan)


def _propose_with_memory(
    team: Any,
    history: list[Mapping[str, Any]],
    state: Mapping[str, Any],
    memory: Mapping[str, Any] | None,
) -> tuple[Any, dict[str, Any]]:
    """Keep focused test doubles compatible with the planner input."""
    try:
        return team.propose(history, state, research_memory=memory)
    except TypeError:
        return team.propose(history, state)


def _proposal_fingerprint(plan: Any) -> str:
    material = "|".join(" ".join(value.lower().split()) for value in (plan.hypothesis, plan.controlled_change))
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _failure_class(failure: str) -> str:
    lowered = failure.lower()
    if "does not implement" in lowered or "inconsistent with the claimed" in lowered:
        return "hypothesis_not_implemented"
    if "prohibited isolated-runtime" in lowered or "prohibited isolated" in lowered:
        return "unsafe_source"
    return "integration_or_preflight_failed"
