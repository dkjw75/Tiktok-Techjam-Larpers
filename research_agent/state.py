"""Persistent state for the deterministic research-controller loop."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, cast


@dataclass
class ResearchState:
    current_best_experiment_id: str = "baseline"
    current_best_primary: float = 0.6016
    completed_iterations: int = 0
    completed_cycles: int = 0
    runner_attempts: int = 0
    planning_rejections: int = 0
    screen_promotion_width: int = 1
    valid_comparisons: int = 0
    consecutive_non_improvements: int = 0
    plateau_restarts: int = 0
    started_at_epoch_seconds: float | None = None
    # Research runtime ACTUALLY spent computing, carried across invocations.
    # Wall-clock since first start would also count hours the agent was paused,
    # which must not be spendable as if it were research effort.
    active_runtime_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    stop_reason_code: str | None = None
    stop_reason: str | None = None
    last_pause_reason: str | None = None

    @property
    def stopped(self) -> bool:
        return self.stop_reason is not None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ResearchState":
        known = {item.name for item in fields(cls)}
        payload = {key: item for key, item in value.items() if key in known}
        return cls(**cast(dict[str, Any], payload))
