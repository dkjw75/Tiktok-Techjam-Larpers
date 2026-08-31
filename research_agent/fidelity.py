"""Multi-fidelity promotion policy for evidence-based compute allocation."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping

from .planner import ResearchDirection
from .safety import ExperimentProposal


class FidelityManager:
    """Promotes candidates only when short-budget evidence is trustworthy."""

    reduction_factor = 3
    min_low_fidelity_trials = 3
    calibration_pair_count = 3
    calibration_min_correlation = 0.25
    calibration_screen_tolerance = 0.002

    def promotion_status(self, history: list[Mapping[str, Any]]) -> dict[str, Any]:
        """Report whether short screens have earned authority to select full runs."""
        screens = {
            str(item.get("experiment_id")): float(item["metrics"]["primary"])
            for item in history
            if item.get("config", {}).get("fidelity") == "low"
            and isinstance(item.get("metrics", {}).get("primary"), (int, float))
        }
        pairs = [
            (screens[str(item.get("parent_experiment_id"))], float(item["metrics"]["primary"]))
            for item in history
            if item.get("config", {}).get("fidelity") == "full"
            and str(item.get("parent_experiment_id")) in screens
            and isinstance(item.get("metrics", {}).get("primary"), (int, float))
        ]
        correlation = _pearson_correlation(pairs)
        enough_pairs = len(pairs) >= self.calibration_pair_count
        reliable = enough_pairs and correlation is not None and correlation >= self.calibration_min_correlation
        return {
            "calibration_pairs": len(pairs),
            "screen_full_correlation": correlation,
            "screening_reliable": reliable,
            "policy": (
                "calibrating" if not enough_pairs
                else "screening_reliable" if reliable
                else "screening_not_predictive"
            ),
        }

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
        status = self.promotion_status(history)
        # While learning whether short and full scores agree, allow only close
        # contenders through. Screens remain a safety/cheap-failure check, not
        # a licence to spend a full budget on clearly weak candidates.
        if float(primary) < incumbent_primary - self.calibration_screen_tolerance:
            return False
        if status["policy"] == "screening_not_predictive":
            return False
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
        survives_rung = sorted(rung, reverse=True).index(float(primary)) < survivors
        if not survives_rung:
            return False
        # Once calibrated, a screen must actually beat the full incumbent.
        return not status["screening_reliable"] or float(primary) > incumbent_primary

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


def _pearson_correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in pairs)
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)
