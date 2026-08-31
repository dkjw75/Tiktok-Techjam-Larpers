"""Multi-fidelity promotion policy for evidence-based compute allocation."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping

from .planner import ResearchDirection
from .safety import ExperimentProposal


class FidelityManager:
    """Promotes only candidates with recorded validation evidence."""

    reduction_factor = 3
    min_low_fidelity_trials = 3

    def should_promote(
        self,
        metrics: Mapping[str, Any],
        incumbent_primary: float,
        *,
        proposal: ExperimentProposal | None = None,
        history: list[Mapping[str, Any]] | None = None,
        global_pool: bool = False,
    ) -> bool:
        primary = metrics.get("primary")
        if not isinstance(primary, (int, float)):
            return False
        if proposal is None or history is None:
            return float(primary) >= incumbent_primary - 0.002
        rung = [
            float(item["metrics"]["primary"])
            for item in history
            if (global_pool or item.get("direction_id") == proposal.research_direction_id)
            and item.get("config", {}).get("fidelity") == "low"
            and (not global_pool or item.get("search_strategy") == "llm_isolated_candidate")
            and isinstance(item.get("metrics", {}).get("primary"), (int, float))
        ]
        if not any(item.get("experiment_id") == proposal.experiment_id for item in history):
            rung.append(float(primary))
        if len(rung) < self.min_low_fidelity_trials:
            return False
        survivors = max(1, math.ceil(len(rung) / self.reduction_factor))
        return sorted(rung, reverse=True).index(float(primary)) < survivors

    def promote(
        self,
        proposal: ExperimentProposal,
        direction: ResearchDirection,
        *,
        experiment_id: str,
    ) -> ExperimentProposal:
        config = dict(proposal.config)
        config["epochs"] = int(direction.evaluation_budget["full_epochs"])
        config["patience"] = int(direction.evaluation_budget.get("patience", 4))
        config["fidelity"] = "full"
        # Full validation is a fixed three-seed confirmation ensemble; screening
        # remains single-seed to protect compute.
        config["confirmation_seeds"] = (0, 1, 2)
        if isinstance(config.get("_locked_settings"), Mapping):
            config["_locked_settings"] = {
                **config["_locked_settings"],
                "epochs": config["epochs"],
                "patience": config["patience"],
            }
        return replace(
            proposal,
            experiment_id=experiment_id,
            parent_experiment_id=proposal.experiment_id,
            config=config,
            search_strategy="asha_promotion",
        )
