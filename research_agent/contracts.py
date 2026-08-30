"""Immutable benchmark facts shared by every research-agent component.

This module intentionally contains no data loading, model training, or metric
implementation. Those responsibilities remain in data.py and evaluate.py.
"""
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import re
from typing import Any, Mapping


class Fidelity(str, Enum):
    """Evaluation budgets with distinct decision authority."""

    SCREEN = "low"
    FULL = "full"


ORGANIZER_EVALUATOR_SHA256 = "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ComparisonValidity:
    """Typed proof that a candidate may be compared with the champion."""

    candidate_experiment_id: str
    fidelity: str
    selection_split: str
    primary_score: float | None
    incumbent_primary: float
    incumbent_experiment_id: str
    delta_primary: float | None
    valid: bool
    reasons: tuple[str, ...] = ()

    @classmethod
    def assess(
        cls,
        *,
        candidate_experiment_id: str,
        config: Mapping[str, Any],
        selection_split: str,
        metrics: Mapping[str, Any],
        runner_metadata: Mapping[str, Any],
        incumbent_primary: float,
        incumbent_experiment_id: str,
        incumbent_metadata: Mapping[str, Any],
        incumbent_config: Mapping[str, Any],
        contract: "BenchmarkContract",
    ) -> "ComparisonValidity":
        fidelity = str(config.get("fidelity", ""))
        reasons: list[str] = []
        if fidelity != Fidelity.FULL.value:
            reasons.append("candidate is not full fidelity")
        if fidelity == Fidelity.FULL.value:
            if config.get("epochs") != contract.full_max_epochs:
                reasons.append(
                    f"full fidelity requires max epochs={contract.full_max_epochs}"
                )
            if config.get("patience") != contract.full_patience:
                reasons.append(
                    f"full fidelity requires patience={contract.full_patience}"
                )
            epochs_run = runner_metadata.get("epochs_run")
            if (
                isinstance(epochs_run, bool)
                or not isinstance(epochs_run, int)
                or not 1 <= epochs_run <= contract.full_max_epochs
            ):
                reasons.append("full fidelity requires valid epochs_run termination evidence")
            if runner_metadata.get("configured_epochs") != contract.full_max_epochs:
                reasons.append("full fidelity requires configured epoch-budget evidence")
            if runner_metadata.get("effective_patience") != contract.full_patience:
                reasons.append("full fidelity requires effective patience evidence")
            stopped_by = runner_metadata.get("stopped_by")
            if stopped_by == "max_epochs_truncated":
                reasons.append("max-epoch termination is truncated and not comparable")
            elif stopped_by != "early_stopping":
                reasons.append("full fidelity requires a recognized stopped_by reason")
            elif (
                stopped_by == "early_stopping"
                and isinstance(epochs_run, int)
                and epochs_run <= contract.full_patience
            ):
                reasons.append("early stopping occurred before the patience window was observable")
            if metrics.get("evaluator_sha256") != contract.evaluator_sha256:
                reasons.append("metric evaluator hash does not match the organizer evaluator")
            if runner_metadata.get("evaluator_sha256") != contract.evaluator_sha256:
                reasons.append("runner evaluator hash does not match the organizer evaluator")
            for field in (
                "data_sha256",
                "preprocessing_sha256",
                "staging_code_sha256",
                "feature_schema_sha256",
                "model_code_sha256",
                "comparison_group_id",
            ):
                value = runner_metadata.get(field)
                if not isinstance(value, str) or not _SHA256.fullmatch(value):
                    reasons.append(f"full fidelity requires valid {field} lineage")
            if runner_metadata.get("seed") != config.get("seed"):
                reasons.append("runner seed evidence does not match the configuration")
            for field in (
                "data_sha256",
                "evaluator_sha256",
                "preprocessing_sha256",
                "staging_code_sha256",
                "feature_schema_sha256",
                "comparison_group_id",
            ):
                if incumbent_metadata.get(field) != runner_metadata.get(field):
                    reasons.append(f"candidate and incumbent {field} lineage differ")
            if incumbent_metadata.get("seed") != runner_metadata.get("seed"):
                reasons.append("candidate and incumbent seeds are not matched")
            if incumbent_config.get("epochs") != contract.full_max_epochs:
                reasons.append("incumbent lacks the full epoch budget")
            if incumbent_config.get("patience") != contract.full_patience:
                reasons.append("incumbent lacks the full patience setting")
        if selection_split != contract.selection_split:
            reasons.append("candidate was not selected on the validation split")
        raw_primary = metrics.get(contract.primary_metric)
        primary_score: float | None = None
        if isinstance(raw_primary, bool) or not isinstance(raw_primary, (int, float)):
            reasons.append("candidate has no numeric primary metric")
        else:
            primary_score = float(raw_primary)
            if not math.isfinite(primary_score):
                reasons.append("candidate primary metric must be finite")
        delta = None if primary_score is None else primary_score - incumbent_primary
        return cls(
            candidate_experiment_id=candidate_experiment_id,
            fidelity=fidelity,
            selection_split=selection_split,
            primary_score=primary_score,
            incumbent_primary=incumbent_primary,
            incumbent_experiment_id=incumbent_experiment_id,
            delta_primary=delta,
            valid=not reasons,
            reasons=tuple(reasons),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_experiment_id": self.candidate_experiment_id,
            "fidelity": self.fidelity,
            "selection_split": self.selection_split,
            "primary_score": self.primary_score,
            "incumbent_primary": self.incumbent_primary,
            "incumbent_experiment_id": self.incumbent_experiment_id,
            "delta_primary": self.delta_primary,
            "valid": self.valid,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class BenchmarkContract:
    """The fixed KuaiRand-Pure benchmark rules the agent must obey."""

    data_dir: Path = Path("KuaiRand-Pure/data")
    label: str = "long_view"
    train_split: str = "train"
    validation_split: str = "valid"
    test_split: str = "test"
    gauc_metric: str = "GAUC"
    ndcg_metric: str = "nDCG@5"
    primary_metric: str = "primary"
    improvement_threshold: float = 0.002
    non_improvement_limit: int = 3
    max_iterations: int = 50
    max_wall_clock_seconds: float = 6 * 60 * 60
    finalization_reserve_seconds: float = 30 * 60
    full_max_epochs: int = 40
    full_patience: int = 4
    evaluator_sha256: str = ORGANIZER_EVALUATOR_SHA256

    @property
    def selection_split(self) -> str:
        """Candidates are selected only with validation evidence."""
        return self.validation_split

    @property
    def protected_modules(self) -> tuple[str, ...]:
        """Modules that agent experiments must not modify."""
        return ("evaluate.py", "baseline.py", "data.py", "submit.py")

    @property
    def metric_names(self) -> tuple[str, str, str]:
        return (self.gauc_metric, self.ndcg_metric, self.primary_metric)

    @property
    def max_experiments(self) -> int:
        """Compatibility alias for callers not yet renamed to iterations."""
        return self.max_iterations


BENCHMARK_CONTRACT = BenchmarkContract()
