"""Multi-fidelity promotion policy for evidence-based compute allocation."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping

from .planner import ResearchDirection
from .safety import ExperimentProposal


class FidelityManager:
    """Promotes only candidates with recorded validation evidence."""

    def should_promote(self, metrics: Mapping[str, Any], incumbent_primary: float) -> bool:
        """Return whether a screen is rankable, never whether it beats the full champion."""
        primary = metrics.get("primary")
        return (
            not isinstance(primary, bool)
            and isinstance(primary, (int, float))
            and math.isfinite(float(primary))
        )

    def promote(
        self,
        proposal: ExperimentProposal,
        direction: ResearchDirection,
        *,
        experiment_id: str,
    ) -> ExperimentProposal:
        config = dict(proposal.config)
        config["epochs"] = int(direction.evaluation_budget["full_epochs"])
        config["patience"] = int(direction.evaluation_budget.get("full_patience", 4))
        config["fidelity"] = "full"
        return replace(
            proposal,
            experiment_id=experiment_id,
            parent_experiment_id=proposal.experiment_id,
            config=config,
            search_strategy="promotion",
        )
