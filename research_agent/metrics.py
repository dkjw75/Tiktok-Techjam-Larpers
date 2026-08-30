"""Validation-only adapter around the immutable official evaluator."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from evaluate import evaluate

from .contracts import BENCHMARK_CONTRACT


class MetricsValidationError(ValueError):
    """Raised when predictions cannot safely be passed to the evaluator."""


@dataclass(frozen=True)
class MetricResult:
    """Official metrics plus evaluator identity for an experiment record."""

    gauc: float
    ndcg_at_5: float
    primary: float
    users: int
    rows: int
    evaluator_sha256: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "GAUC": self.gauc,
            "nDCG@5": self.ndcg_at_5,
            "primary": self.primary,
            "users": self.users,
            "rows": self.rows,
            "evaluator_sha256": self.evaluator_sha256,
        }


def evaluate_predictions(
    user_ids: Sequence[Any],
    labels: Sequence[Any],
    scores: Sequence[Any],
    *,
    split: str = "valid",
    allow_test: bool = False,
) -> MetricResult:
    """Validate candidate predictions and call the official evaluator.

    Candidate selection is restricted to validation. Passing ``split='test'``
    requires an explicit final-confirmation opt-in.
    """
    if split == BENCHMARK_CONTRACT.test_split and not allow_test:
        raise MetricsValidationError("test evaluation requires allow_test=True")
    if split != BENCHMARK_CONTRACT.selection_split and split != BENCHMARK_CONTRACT.test_split:
        raise MetricsValidationError(f"unsupported evaluation split: {split!r}")

    users = list(user_ids)
    label_values = _one_dimensional("labels", labels, dtype=float)
    score_values = _one_dimensional("scores", scores, dtype=float)
    if not (len(users) == len(label_values) == len(score_values)):
        raise MetricsValidationError("user_ids, labels, and scores must have equal lengths")
    if not len(users):
        raise MetricsValidationError("cannot evaluate empty predictions")
    if not np.isfinite(score_values).all():
        raise MetricsValidationError("scores must be finite; NaN and Inf are not allowed")
    if not np.isfinite(label_values).all() or not np.isin(label_values, (0.0, 1.0)).all():
        raise MetricsValidationError("labels must be finite binary long_view values")

    official = evaluate(users, label_values.astype(int).tolist(), score_values.tolist())
    return MetricResult(
        gauc=float(official["GAUC"]),
        ndcg_at_5=float(official["nDCG@5"]),
        primary=float(official["primary"]),
        users=int(official["users"]),
        rows=int(official["rows"]),
        evaluator_sha256=_evaluator_sha256(),
    )


def _one_dimensional(name: str, values: Sequence[Any], *, dtype: type) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array: NDArray[Any] = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise MetricsValidationError(f"{name} must be one-dimensional")
    return array


def _evaluator_sha256() -> str:
    evaluator_path = Path(__file__).resolve().parent.parent / "evaluate.py"
    content = evaluator_path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()
