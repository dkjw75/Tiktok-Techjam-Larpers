"""Evidence review for deciding whether the next cycle explores or refines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .state import ResearchState


@dataclass(frozen=True)
class ReviewDecision:
    action: str
    rationale: str


class EvidenceReviewer:
    """Interprets recorded results without selecting low-level configurations."""

    def review(self, history: Sequence[Mapping[str, Any]], state: ResearchState) -> ReviewDecision:
        if state.stopped:
            return ReviewDecision("finished", state.stop_reason or "controller stopped")
        if not history:
            return ReviewDecision("explore", "No completed experiments exist yet.")
        if history[-1].get("decision") == "accepted":
            return ReviewDecision("refine", "The latest result was accepted; inspect nearby evidence before changing direction.")
        return ReviewDecision("explore", "Evidence is mixed; reserve the next cycle for a diverse direction.")
