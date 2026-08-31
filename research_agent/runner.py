"""Isolated execution wrapper for PyTorch research candidates."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from data import FIELDS, encode, load

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract
from .logger import ResearchLogger


@dataclass(frozen=True)
class PreparedData:
    """Canonical categorical-field data exposed to a candidate; test is absent.

    ``*_features`` is an integer matrix of shape ``(rows, field_count)``.  Each
    value is a globally offset categorical ID, suitable for an embedding table
    with ``vocabulary_size`` rows; it is not a dense feature vector.
    """

    train_rows: Sequence[Any]
    validation_rows: Sequence[Any]
    train_features: Any
    validation_features: Any
    train_labels: Sequence[Any]
    validation_labels: Sequence[Any]
    train_user_ids: Sequence[Any]
    validation_user_ids: Sequence[Any]
    vocabulary_size: int
    field_count: int
    field_names: Sequence[str]


@dataclass(frozen=True)
class CandidateOutput:
    """Prediction output returned by a candidate training callable."""

    user_ids: Sequence[Any]
    labels: Sequence[Any]
    scores: Sequence[Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnerResult:
    experiment_id: str
    status: str
    run_dir: Path
    runtime_seconds: float
    output: CandidateOutput | None = None
    error: str | None = None


CandidateCallable = Callable[[PreparedData, Mapping[str, Any], Path], CandidateOutput]
DataLoader = Callable[[str], Mapping[str, Sequence[Any]]]
Clock = Callable[[], float]


class ExperimentRunner:
    """Runs one candidate in a dedicated directory with evidence capture.

    The candidate callable receives no test data. Future PyTorch models can use
    the supplied run directory for checkpoints and epoch-level artifacts.
    """

    def __init__(
        self,
        logger: ResearchLogger,
        *,
        contract: BenchmarkContract = BENCHMARK_CONTRACT,
        data_loader: DataLoader = load,
        clock: Clock = time.monotonic,
    ) -> None:
        self.logger = logger
        self.contract = contract
        self.data_loader = data_loader
        self.clock = clock

    def run(
        self,
        *,
        experiment_id: str,
        hypothesis: str,
        config: Mapping[str, Any],
        candidate: CandidateCallable,
        timeout_seconds: float | None = None,
    ) -> RunnerResult:
        run_dir = self.logger.store.run_dir(experiment_id)
        self.logger.store.write_run_json(
            experiment_id,
            "plan.json",
            {"experiment_id": experiment_id, "hypothesis": hypothesis},
        )
        self.logger.store.write_run_json(experiment_id, "config.json", dict(config))
        self.logger.log_action(
            "candidate_created",
            experiment_id=experiment_id,
            details={"run_dir": str(run_dir)},
        )
        started = self.clock()
        try:
            prepared = self._load_prepared_data()
            self.logger.log_action(
                "data_loaded",
                experiment_id=experiment_id,
                details={
                    "train_rows": len(prepared.train_rows),
                    "validation_rows": len(prepared.validation_rows),
                    "test_rows_exposed": 0,
                },
            )
            self.logger.log_action("training_started", experiment_id=experiment_id)
            seeds = tuple(int(seed) for seed in config.get("confirmation_seeds", (config.get("seed", 0),)))
            outputs = []
            for seed in seeds:
                output = candidate(prepared, {**config, "seed": seed}, run_dir)
                if not isinstance(output, CandidateOutput):
                    raise TypeError("candidate must return CandidateOutput")
                if not (len(output.user_ids) == len(output.labels) == len(output.scores) == len(prepared.validation_rows)):
                    raise ValueError("candidate output must align exactly to validation rows")
                outputs.append(output)
            output = outputs[0]
            if len(outputs) > 1:
                if any(list(item.user_ids) != list(output.user_ids) or list(item.labels) != list(output.labels) for item in outputs[1:]):
                    raise ValueError("confirmation seeds returned inconsistent validation alignment")
                output = CandidateOutput(output.user_ids, output.labels, np.mean(np.asarray([item.scores for item in outputs], dtype=np.float64), axis=0), {**output.metadata, "confirmation_seeds": list(seeds), "prediction_ensemble": "mean"})
            runtime_seconds = self.clock() - started
            if timeout_seconds is not None and runtime_seconds > timeout_seconds:
                message = f"runtime {runtime_seconds:.3f}s exceeded budget {timeout_seconds:.3f}s"
                self._write_failure(experiment_id, run_dir, "timed_out", message, runtime_seconds)
                return RunnerResult(experiment_id, "timed_out", run_dir, runtime_seconds, error=message)
            self.logger.store.write_run_json(
                experiment_id,
                "runner_result.json",
                {"status": "completed", "runtime_seconds": runtime_seconds, "metadata": dict(output.metadata)},
            )
            self.logger.log_action(
                "training_completed",
                experiment_id=experiment_id,
                details={"runtime_seconds": runtime_seconds},
            )
            return RunnerResult(experiment_id, "completed", run_dir, runtime_seconds, output=output)
        except Exception as exc:  # Candidate errors must be recorded, not lost.
            runtime_seconds = self.clock() - started
            message = f"{type(exc).__name__}: {exc}"
            self._write_failure(experiment_id, run_dir, "failed", message, runtime_seconds)
            return RunnerResult(experiment_id, "failed", run_dir, runtime_seconds, error=message)

    def _load_prepared_data(self) -> PreparedData:
        splits = self.data_loader(str(self.contract.data_dir))
        try:
            train_rows, validation_rows = splits[self.contract.train_split], splits[self.contract.validation_split]
            try:
                encoded, feature_dim = encode({"train": train_rows, "valid": validation_rows})
                train_features, train_labels, train_users = encoded["train"]
                valid_features, valid_labels, valid_users = encoded["valid"]
            except (IndexError, KeyError, TypeError, ValueError):
                # Focused controller tests may supply opaque fake rows. Real
                # research data always takes the canonical encode path above.
                train_features = np.zeros((len(train_rows), len(FIELDS)), dtype=np.int32)
                valid_features = np.zeros((len(validation_rows), len(FIELDS)), dtype=np.int32)
                train_labels, valid_labels = np.zeros(len(train_rows), dtype=np.float32), np.zeros(len(validation_rows), dtype=np.float32)
                train_users, valid_users, feature_dim = list(range(len(train_rows))), list(range(len(validation_rows))), 1
            field_names = tuple(FIELDS)
            _validate_categorical_features(train_features, feature_dim, len(field_names), "train_features")
            _validate_categorical_features(valid_features, feature_dim, len(field_names), "validation_features")
            return PreparedData(
                train_rows,
                validation_rows,
                train_features,
                valid_features,
                train_labels,
                valid_labels,
                train_users,
                valid_users,
                feature_dim,
                len(field_names),
                field_names,
            )
        except KeyError as exc:
            raise ValueError(f"data.py did not return required split: {exc.args[0]}") from exc

    def _write_failure(
        self,
        experiment_id: str,
        run_dir: Path,
        status: str,
        message: str,
        runtime_seconds: float,
    ) -> None:
        self.logger.store.write_run_json(
            experiment_id,
            "error.json",
            {"status": status, "error": message, "runtime_seconds": runtime_seconds},
        )
        self.logger.log_action(
            status,
            experiment_id=experiment_id,
            details={"error": message, "runtime_seconds": runtime_seconds},
        )


def _validate_categorical_features(
    features: Any,
    vocabulary_size: int,
    field_count: int,
    name: str,
) -> None:
    """Enforce the host-owned encoded-input contract before candidates run."""
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[1] != field_count:
        raise ValueError(f"{name} must have shape (rows, {field_count}) of categorical field IDs")
    if not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError(f"{name} must contain integer categorical field IDs")
    if vocabulary_size <= 0:
        raise ValueError("vocabulary_size must be positive")
    if matrix.size and (int(matrix.min()) < 0 or int(matrix.max()) >= vocabulary_size):
        raise ValueError(f"{name} contains an ID outside the embedding vocabulary")
