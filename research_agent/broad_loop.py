"""End-to-end broad-proposal loop using the LLM specialist roles."""
from __future__ import annotations

from typing import Any

from .agent_team import LLMResearchTeam, build_isolated_candidate
from .controller import ExperimentController
from .llm_planner import LLMPlanningError
from .logger import ResearchLogger
from .safety import ExperimentProposal


class BroadAutonomousLoop:
    def __init__(self, controller: ExperimentController, logger: ResearchLogger, team: LLMResearchTeam) -> None:
        self.controller, self.logger, self.team = controller, logger, team

    def run(self, max_cycles: int) -> int:
        completed = 0
        for _ in range(max_cycles):
            if self.controller.state.stopped:
                break
            history = self.logger.store.read_iterations()
            plan, meta = self.team.propose(history, self.controller.state.as_dict())
            self.logger.log_action("llm_broad_hypothesis_proposed", details={**plan.__dict__, "llm": meta})
            critique, critique_meta = self.team.critique(plan)
            normalized_family = _normalize_model_family(plan.model_family)
            automatic = critique.decision == "approved" and not plan.requires_human_review and normalized_family == "fm"
            self.logger.log_action("llm_critic_completed", details={"decision": critique.decision, "rationale": critique.rationale, "llm": critique_meta, "automatic_execution": automatic})
            if not automatic:
                self.logger.log_action("human_review_required", details={"proposal": plan.__dict__, "reason": critique.rationale})
                break
            experiment_id = f"exp_{self.controller.state.completed_iterations + 1:03d}"
            source, code_meta = self.team.code(plan)
            workspace = self.logger.store.root / "candidate_workspaces" / experiment_id
            try:
                candidate = build_isolated_candidate(source, workspace)
            except LLMPlanningError as exc:
                self.logger.log_action("generated_candidate_rejected", experiment_id=experiment_id, details={"error": str(exc), "llm": code_meta})
                continue
            proposal = ExperimentProposal(
                experiment_id=experiment_id,
                parent_experiment_id=self.controller.state.current_best_experiment_id,
                hypothesis=plan.hypothesis,
                rationale=plan.rationale,
                config={"loss": "pointwise", "learning_rate": 0.001, "l2": 1e-6, "embedding_dim": 16, "batch_size": 8192, "seed": 0, "epochs": 4, "agent_change": plan.controlled_change},
                changed_factors=(plan.controlled_change,),
                model_family=normalized_family,
                research_direction_id=plan.area,
                search_strategy="llm_broad_proposal",
                search_region_id=f"region_{plan.area}",
            )
            self.logger.log_action("coding_subagent_completed", experiment_id=experiment_id, details={"workspace": str(workspace), "llm": code_meta})
            self.controller.run_iteration(proposal, candidate, code_diff=source)
            completed += 1
            review, review_meta = self.team.review(self.logger.store.read_iterations())
            self.logger.log_action("llm_evidence_review_completed", details={**review, "llm": review_meta})
        return completed


def _normalize_model_family(value: str) -> str:
    """LLM prose such as 'unchanged baseline model family' still means FM."""
    normalized = value.strip().lower()
    if normalized in {"fm", "factorization machine", "unchanged baseline model family", "baseline fm", "unchanged fm"}:
        return "fm"
    if "baseline model family" in normalized or "factorization machine" in normalized:
        return "fm"
    return normalized
