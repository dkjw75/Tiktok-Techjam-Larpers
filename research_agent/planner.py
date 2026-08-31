"""Evidence-driven research-orchestrator interface.

The default implementation is local and deterministic because this project has
no configured LLM credential. It has the same narrow output boundary intended
for a future LLM adapter: a research direction, never an executable trial.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .state import ResearchState


@dataclass(frozen=True)
class ResearchDirection:
    direction_id: str
    hypothesis: str
    rationale: str
    search_space: Mapping[str, Any]
    success_evidence: str
    evaluation_budget: Mapping[str, Any]
    strategy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction_id": self.direction_id,
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "search_space": dict(self.search_space),
            "success_evidence": self.success_evidence,
            "evaluation_budget": dict(self.evaluation_budget),
            "strategy": self.strategy,
        }


class ResearchPlanner(Protocol):
    """Boundary implemented by the OpenAI planner and test doubles."""

    def propose(self, history: Sequence[Mapping[str, Any]], state: ResearchState) -> ResearchDirection: ...


class EvidencePlanner:
    """Proposes a research direction from logged outcomes, not fixed trials."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed

    def propose(
        self,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
    ) -> ResearchDirection:
        evidence_review = next(
            (
                item.get("evidence_review")
                for item in reversed(history)
                if item.get("record_type") == "evidence_review"
            ),
            None,
        )
        experiment_history = [
            item for item in history if item.get("record_type") != "evidence_review"
        ]
        directions = self._available_directions(experiment_history)
        attempted = {item.get("direction_id") for item in experiment_history}
        unexplored = [direction for direction in directions if direction.direction_id not in attempted]
        allocation = (
            str(evidence_review.get("allocation_status"))
            if isinstance(evidence_review, Mapping)
            else "EXPLORE"
        )
        latest_direction = (
            str(experiment_history[-1].get("direction_id"))
            if experiment_history and experiment_history[-1].get("direction_id")
            else None
        )
        if allocation == "EXPLOIT" and latest_direction is not None:
            preferred = tuple(
                item for item in directions if item.direction_id == latest_direction
            )
            pool = preferred or unexplored or directions
        elif allocation == "ABANDON" and latest_direction is not None:
            alternatives = tuple(
                item for item in directions if item.direction_id != latest_direction
            )
            pool = alternatives or unexplored or directions
        else:
            pool = unexplored or directions
        rng = random.Random(self.seed + state.completed_iterations)
        chosen = rng.choice(pool)

        if not experiment_history:
            rationale = "No prior agent-run evidence exists, so begin with broad bootstrap exploration."
        else:
            recent = experiment_history[-3:]
            recent_decisions = [item.get("decision") for item in recent]
            rationale = (
                f"Recent decisions were {recent_decisions}; choose a {'new' if unexplored else 'diverse'} "
                "research direction rather than repeat an identical configuration."
            )
        if isinstance(evidence_review, Mapping):
            rationale = (
                f"{rationale} Structured evidence allocation is "
                f"{evidence_review.get('allocation_status', 'unknown')}: "
                f"{evidence_review.get('rationale', 'no rationale recorded')}"
            )
        return ResearchDirection(
            direction_id=chosen.direction_id,
            hypothesis=chosen.hypothesis,
            rationale=rationale,
            search_space=chosen.search_space,
            success_evidence=chosen.success_evidence,
            evaluation_budget=chosen.evaluation_budget,
            strategy="exploration" if unexplored else "diverse_restart",
        )

    @staticmethod
    def _available_directions(history: Sequence[Mapping[str, Any]]) -> tuple[ResearchDirection, ...]:
        # These are approved research domains, not a scheduled hypothesis list.
        # Exact values are selected later by SearchController.
        return (
            ResearchDirection(
                direction_id="rank_ensemble",
                hypothesis="Blending within-user ranks across FM members that differ in feature set and training objective may beat any single member, because their errors are decorrelated even when some members are individually weaker than the baseline.",
                rationale="Single-model directions are exhausted: every loss change lost to the baseline and every wide feature set lost to the 5-field baseline. Ensembling is the only untested direction whose gain does not require a better individual model.",
                search_space={
                    "loss": ["ensemble"],
                    "member_set": ["core4", "core5", "core6"],
                },
                success_evidence="Full-fidelity three-seed mean validation primary improves by more than 0.002, targeting at least 0.003, with at least two seed wins.",
                evaluation_budget={
                    "low_epochs": 4,
                    "low_patience": 2,
                    "full_epochs": 40,
                    "full_patience": 4,
                },
                strategy="exploitation",
            ),
            ResearchDirection(
                direction_id="listwise_fm_ranking",
                hypothesis="A per-user full-slate softmax objective may align FM training with GAUC and nDCG@5 better than pointwise or sampled-pair training.",
                rationale="The first pairwise direction underperformed, so test the organizer-identified untried listwise objective before any more FM tuning.",
                search_space={
                    "loss": ["listwise"],
                    "objective_variant": ["t1", "t05", "t1_bce25"],
                },
                success_evidence="Full-fidelity three-seed mean validation primary improves by more than 0.002, targeting at least 0.003.",
                evaluation_budget={
                    "low_epochs": 4,
                    "low_patience": 2,
                    "full_epochs": 40,
                    "full_patience": 4,
                },
                strategy="bootstrap",
            ),
        )
