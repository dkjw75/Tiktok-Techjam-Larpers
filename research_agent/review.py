"""Evidence review for deciding whether the next cycle explores or refines."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from .state import ResearchState


@dataclass(frozen=True)
class ReviewDecision:
    action: str
    rationale: str
    allocation_status: str = "EXPLORE"
    metric_trends: Mapping[str, Any] = field(default_factory=dict)
    seed_evidence: Mapping[str, Any] = field(default_factory=dict)
    complementarity_evidence: Mapping[str, Any] = field(default_factory=dict)
    failure_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "rationale": self.rationale,
            "allocation_status": self.allocation_status,
            "metric_trends": dict(self.metric_trends),
            "seed_evidence": dict(self.seed_evidence),
            "complementarity_evidence": dict(self.complementarity_evidence),
            "failure_count": self.failure_count,
        }


class EvidenceReviewer:
    """Interprets recorded results without selecting low-level configurations."""

    def review(
        self,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
        resolutions: Mapping[str, Any] | None = None,
    ) -> ReviewDecision:
        resolutions = resolutions or {}
        if state.stopped:
            return ReviewDecision(
                "finished",
                state.stop_reason or "controller stopped",
                allocation_status="ABANDON",
            )
        if not history:
            return ReviewDecision("explore", "No completed experiments exist yet.")
        scored = [item for item in history if (item.get("metrics") or {}).get("primary") is not None]
        recent = scored[-3:]
        trends = {
            metric: [float((item.get("metrics") or {})[metric]) for item in recent]
            for metric in ("GAUC", "nDCG@5", "primary")
            if all((item.get("metrics") or {}).get(metric) is not None for item in recent)
        }
        failures = sum(
            item.get("decision") in {"failed", "invalid", "critic_rejected"}
            for item in history[-5:]
        )
        latest = history[-1]
        latest_resolution = resolutions.get(str(latest.get("experiment_id")))
        effective_decision = (
            latest_resolution.get("decision")
            if isinstance(latest_resolution, Mapping)
            else latest.get("decision")
        )
        certificate = (
            latest_resolution.get("certificate")
            if isinstance(latest_resolution, Mapping)
            else None
        )
        seed_evidence: dict[str, Any] = {}
        if isinstance(certificate, Mapping):
            scores = [float(value) for value in certificate.get("candidate_scores", [])]
            seed_evidence = {
                "scores": scores,
                "mean": mean(scores) if scores else None,
                "std": pstdev(scores) if len(scores) > 1 else None,
                "wins": certificate.get("wins"),
                "confirmed": certificate.get("confirmed"),
            }
        metadata = dict(latest.get("runner_metadata") or {})
        complementarity = {
            "held_half_primary": metadata.get("blend_weight_held_half_primary"),
            "fit_half_primary": metadata.get("blend_weight_fit_half_primary"),
            "ensemble_delta_if_added": metadata.get("ensemble_delta_if_added"),
            "measured": metadata.get("ensemble_delta_if_added") is not None,
        }
        if effective_decision == "accepted":
            return ReviewDecision(
                "refine",
                "The latest result cleared comparable and seed gates; refine only within its evidenced region.",
                allocation_status="EXPLOIT",
                metric_trends=trends,
                seed_evidence=seed_evidence,
                complementarity_evidence=complementarity,
                failure_count=failures,
            )
        latest_region = latest.get("search_region_id") or latest.get("direction_id")
        region_history = [
            item
            for item in history
            if (item.get("search_region_id") or item.get("direction_id"))
            == latest_region
        ]
        recent_region = region_history[-3:]
        negative_decisions = {"rejected", "invalid", "failed", "critic_rejected"}

        def is_negative(item: Mapping[str, Any]) -> bool:
            resolution = resolutions.get(str(item.get("experiment_id")))
            decision = (
                resolution.get("decision")
                if isinstance(resolution, Mapping)
                else item.get("decision")
            )
            if decision in negative_decisions:
                return True
            delta = item.get("delta_primary")
            return (
                decision == "screened"
                and isinstance(delta, (int, float))
                and not isinstance(delta, bool)
                and float(delta) <= 0.0
            )

        recent_failures = [
            item
            for item in recent_region
            if is_negative(item)
        ]
        allocation = (
            "ABANDON"
            if len(recent_region) == 3 and len(recent_failures) == 3
            else "EXPLORE"
        )
        rationale = (
            "The recent region has three negative results; abandon that exact mechanism and choose a materially different direction."
            if allocation == "ABANDON"
            else "Evidence is mixed; explore one controlled, diverse direction."
        )
        return ReviewDecision(
            "explore",
            rationale,
            allocation_status=allocation,
            metric_trends=trends,
            seed_evidence=seed_evidence,
            complementarity_evidence=complementarity,
            failure_count=failures,
        )
