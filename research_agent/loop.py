"""Closed-loop orchestration across planner, search, safety, runner, and review."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .controller import ExperimentController, IterationResult
from .critic import ProposalCritic
from .fidelity import FidelityManager
from .logger import ResearchLogger
from .planner import ResearchDirection, ResearchPlanner
from .review import EvidenceReviewer
from .regions import SearchRegionManager
from .runner import CandidateCallable
from .search import SearchController, SearchSpaceExhausted, SearchState


@dataclass(frozen=True)
class CycleResult:
    direction: ResearchDirection
    iteration: IterationResult
    promoted_iteration: IterationResult | None


class AutonomousResearchLoop:
    """Runs evidence-based cycles; no hypothesis sequence is hard-coded."""

    def __init__(
        self,
        *,
        controller: ExperimentController,
        logger: ResearchLogger,
        planner: ResearchPlanner,
        search: SearchController,
        critic: ProposalCritic,
        reviewer: EvidenceReviewer,
        fidelity: FidelityManager,
        regions: SearchRegionManager | None = None,
        candidate: CandidateCallable,
    ) -> None:
        self.controller = controller
        self.logger = logger
        self.planner = planner
        self.search = search
        self.critic = critic
        self.reviewer = reviewer
        self.fidelity = fidelity
        self.regions = regions or SearchRegionManager()
        self.candidate = candidate

    def run(self, max_cycles: int) -> list[CycleResult]:
        results: list[CycleResult] = []
        for _ in range(max_cycles):
            if self.controller.state.stopped:
                break
            history = self.logger.store.read_iterations()
            review = self.reviewer.review(history, self.controller.state)
            self.logger.log_action(
                "evidence_reviewed",
                details={
                    "action": review.action,
                    "rationale": review.rationale,
                    "regions": [snapshot.__dict__ for snapshot in self.regions.snapshots(history)],
                },
            )
            if review.action == "restart":
                self.controller.begin_plateau_restart()
            try:
                direction = self.planner.propose(history, self.controller.state)
            except SearchSpaceExhausted as exc:
                self.logger.log_action("research_search_exhausted", details={"reason": str(exc)})
                break
            llm_metadata = getattr(self.planner, "last_metadata", None)
            if llm_metadata:
                self.logger.log_action("llm_hypothesis_generated", details=dict(llm_metadata))
            self.logger.log_action("research_direction_proposed", details=direction.as_dict())
            for specialist in getattr(self.planner, "last_specialist_metadata", []):
                self.logger.log_action("llm_specialist_consulted", details=specialist)
            search_state = self.regions.choose_search_state(direction, history, self.controller.state, review)
            try:
                proposal = self.search.propose_trial(direction, self.controller.state, history, search_state=search_state)
            except SearchSpaceExhausted as exc:
                self.logger.log_action(
                    "search_direction_exhausted",
                    details={"direction_id": direction.direction_id, "reason": str(exc)},
                )
                if hasattr(self.planner, "mark_exhausted"):
                    self.planner.mark_exhausted(direction.direction_id)
                continue
            critic_result = self.critic.review(proposal, history)
            self.logger.log_action(
                "proposal_critic_reviewed",
                experiment_id=proposal.experiment_id,
                details={"approved": critic_result.approved, "reasons": list(critic_result.reasons)},
            )
            if not critic_result.approved:
                iteration = self.controller.run_iteration(proposal, self.candidate)
                results.append(CycleResult(direction, iteration, None))
                continue

            iteration = self.controller.run_iteration(proposal, self.candidate)
            promoted = self._maybe_promote(direction, proposal, iteration)
            results.append(CycleResult(direction, iteration, promoted))
        return results

    def _maybe_promote(
        self,
        direction: ResearchDirection,
        proposal,
        iteration: IterationResult,
    ) -> IterationResult | None:
        # A low-fidelity near miss may be rejected as a final result while
        # still being exactly the evidence ASHA should re-check at full budget.
        if not iteration.metrics:
            return None
        if not self.fidelity.should_promote(
            iteration.metrics,
            self.controller.state.current_best_primary,
            proposal=proposal,
            history=self.logger.store.read_iterations(),
        ):
            return None
        history = self.logger.store.read_iterations()
        experiment_id = self.search._next_experiment_id(history)
        promoted = self.fidelity.promote(proposal, direction, experiment_id=experiment_id)
        critic_result = self.critic.review(promoted, history)
        self.logger.log_action(
            "promotion_critic_reviewed",
            experiment_id=promoted.experiment_id,
            details={"approved": critic_result.approved, "reasons": list(critic_result.reasons)},
        )
        if not critic_result.approved:
            return None
        return self.controller.run_iteration(promoted, self.candidate)
