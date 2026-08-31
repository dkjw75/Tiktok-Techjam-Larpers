"""Persistent state for the deterministic research-controller loop."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ResearchState:
    current_best_experiment_id: str = "baseline"
    current_best_primary: float = 0.6015
    completed_iterations: int = 0
    screening_runs_completed: int = 0
    full_evaluations_completed: int = 0
    invalid_proposals: int = 0
    implementation_failures: int = 0
    consecutive_non_improvements: int = 0
    plateau_restarts: int = 0
    stop_reason: str | None = None

    @property
    def stopped(self) -> bool:
        return self.stop_reason is not None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ResearchState":
        return cls(**value)
