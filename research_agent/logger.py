"""Structured event and iteration logging for research-agent runs."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .store import ArtifactStore


Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchLogger:
    """Writes all material research actions to append-only storage."""

    def __init__(self, store: ArtifactStore, clock: Clock = utc_now) -> None:
        self.store = store
        self.clock = clock

    def log_action(
        self,
        action: str,
        *,
        experiment_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not action.strip():
            raise ValueError("action must not be empty")
        event = {
            "timestamp": self._timestamp(),
            "action": action,
            "experiment_id": experiment_id,
            "details": details or {},
        }
        self.store.append_event(event)
        self._refresh_readable_summary()
        return event

    def record_iteration(self, record: dict[str, Any]) -> dict[str, Any]:
        required = {"experiment_id", "hypothesis", "rationale", "config", "decision"}
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"iteration record missing required fields: {', '.join(missing)}")
        completed = {"timestamp": self._timestamp(), **record}
        self.store.append_iteration(completed)
        metrics = completed.get("metrics", {})
        self.store.append_metric_summary(
            {
                "experiment_id": completed["experiment_id"],
                "parent_experiment_id": completed.get("parent_experiment_id"),
                "decision": completed["decision"],
                "GAUC": metrics.get("GAUC"),
                "nDCG@5": metrics.get("nDCG@5"),
                "primary": metrics.get("primary"),
                "delta_primary": completed.get("delta_primary"),
                "runtime_seconds": completed.get("runtime_seconds"),
            }
        )
        self.log_action(
            "iteration_recorded",
            experiment_id=completed["experiment_id"],
            details={"decision": completed["decision"]},
        )
        self._refresh_readable_summary()
        return completed

    def record_manual_intervention(
        self,
        *,
        description: str,
        reason: str,
        effect: str,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        intervention = {
            "timestamp": self._timestamp(),
            "experiment_id": experiment_id,
            "description": description,
            "reason": reason,
            "effect": effect,
        }
        self.store.append_intervention(intervention)
        self.log_action(
            "manual_intervention_recorded",
            experiment_id=experiment_id,
            details={"reason": reason},
        )
        self._refresh_readable_summary()
        return intervention

    def record_code_diff(self, experiment_id: str, diff_text: str) -> Path:
        path = self.store.write_patch(experiment_id, diff_text)
        self.log_action(
            "code_diff_recorded",
            experiment_id=experiment_id,
            details={"patch_path": str(path)},
        )
        return path

    def _timestamp(self) -> str:
        return self.clock().astimezone(timezone.utc).isoformat()

    def _refresh_readable_summary(self) -> None:
        """Keep a human-readable run log current even if a run stops abruptly."""
        from .reporter import MarkdownReporter

        MarkdownReporter(self.store).write()
