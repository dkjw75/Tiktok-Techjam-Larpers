"""Isolated execution wrapper for PyTorch research candidates."""
from __future__ import annotations

import errno
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import data as benchmark_data
from data import load

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract
from .data_boundary import load_staged_splits, stage_research_splits
from .logger import ResearchLogger
from .lineage import candidate_code_sha256
from .store import ArtifactStore, RunReservation, RunReservationError


@dataclass(frozen=True)
class PreparedData:
    """Prepared data exposed to a candidate; test rows are intentionally absent.

    `prediction_rows` exists only for the authorised finalization transaction. It
    is scored but NEVER used to fit anything: model selection, early stopping and
    blend weights all come from `validation_rows`. Keeping it a distinct field is
    what makes leaking the prediction split structurally impossible rather than a
    matter of remembering which argument to pass.
    """

    train_rows: Sequence[Any]
    validation_rows: Sequence[Any]
    prediction_rows: Sequence[Any] | None = None
    # Execution lineage (dataset/evaluator/preprocessing/model-code hashes),
    # deliberately separate from the experiment configuration: configuration is
    # a scientific parameter, lineage is provenance. Candidates stamp it into
    # their checkpoint so a bundle can be bound to the inputs that produced it.
    lineage: Mapping[str, Any] | None = None


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
    failure_kind: str | None = None
    attempt_id: str | None = None
    reused: bool = False


CandidateCallable = Callable[[PreparedData, Mapping[str, Any], Path], CandidateOutput]
DataLoader = Callable[[str], Mapping[str, Sequence[Any]]]
Clock = Callable[[], float]
ExecutionMode = Literal["auto", "in_process", "subprocess"]

APPROVED_CANDIDATES: Mapping[str, str] = {
    "torch_fm": "research_agent.models.torch_fm:run_torch_fm_candidate",
    "organizer_fm": "research_agent.models.organizer_fm:run_organizer_fm_candidate",
    "ensemble_fm": "research_agent.models.ensemble_fm:run_ensemble_fm_candidate",
    "dispatch": "research_agent.models.dispatch:run_candidate",
}


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    timed_out: bool


class ExperimentRunner:
    """Runs one candidate in a dedicated attempt with durable evidence.

    Production candidates run in a fresh Python interpreter, so a timeout can
    terminate the complete process tree. Dependency-injected test callables
    retain an in-process path; that path cannot promise pre-emptive timeout and
    is deliberately never selected for the production ``data.load`` path.
    """

    def __init__(
        self,
        logger: ResearchLogger,
        *,
        contract: BenchmarkContract = BENCHMARK_CONTRACT,
        data_loader: DataLoader = load,
        clock: Clock = time.monotonic,
        execution_mode: ExecutionMode = "auto",
    ) -> None:
        if execution_mode not in {"auto", "in_process", "subprocess"}:
            raise ValueError(f"unsupported execution mode: {execution_mode}")
        self.logger = logger
        self.contract = contract
        self.data_loader = data_loader
        self.clock = clock
        self.execution_mode = execution_mode

    def run(
        self,
        *,
        experiment_id: str,
        hypothesis: str,
        config: Mapping[str, Any],
        candidate: CandidateCallable,
        timeout_seconds: float | None = None,
    ) -> RunnerResult:
        candidate_path = self._candidate_path(candidate)
        candidate_key = self._candidate_key(candidate_path)
        lineage = self._lineage_metadata(config, candidate)
        fingerprint = self._fingerprint(hypothesis, config, candidate_path, lineage)
        try:
            reservation = self.logger.store.reserve_run(experiment_id, fingerprint)
        except RunReservationError as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.logger.log_action(
                "run_reservation_failed",
                experiment_id=experiment_id,
                details={"error": message, "failure_kind": "deterministic"},
            )
            return RunnerResult(
                experiment_id,
                "failed",
                self.logger.store.runs_dir / experiment_id,
                0.0,
                error=message,
                failure_kind="deterministic",
            )

        if reservation.reused:
            return self._load_completed_result(reservation, expected_lineage=lineage)

        started = self.clock()
        try:
            self.logger.store.write_attempt_json(
                reservation,
                "plan.json",
                {
                    "experiment_id": experiment_id,
                    "attempt_id": reservation.attempt_id,
                    "hypothesis": hypothesis,
                    "fingerprint": fingerprint,
                },
            )
            self.logger.store.write_attempt_json(reservation, "config.json", dict(config))
            self.logger.log_action(
                "candidate_created",
                experiment_id=experiment_id,
                details={
                    "run_dir": str(reservation.run_dir),
                    "attempt_id": reservation.attempt_id,
                },
            )
            self.logger.store.transition_run(reservation, "running")
            if self._should_use_subprocess(candidate_path):
                return self._run_subprocess(
                    reservation,
                    candidate_key=candidate_key,
                    timeout_seconds=timeout_seconds,
                    started=started,
                    config=config,
                    lineage=lineage,
                )
            return self._run_in_process(
                reservation,
                candidate=candidate,
                timeout_seconds=timeout_seconds,
                started=started,
                config=config,
                lineage=lineage,
            )
        except BaseException as exc:
            runtime_seconds = self.clock() - started
            failure_kind = classify_exception(exc)
            status = _failure_status(failure_kind)
            message = f"{type(exc).__name__}: {exc}"
            try:
                self._write_failure(
                    reservation,
                    status,
                    failure_kind,
                    message,
                    runtime_seconds,
                )
            except Exception:
                self.logger.store.release_run(reservation)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return RunnerResult(
                experiment_id,
                "failed",
                reservation.run_dir,
                runtime_seconds,
                error=message,
                failure_kind=failure_kind,
                attempt_id=reservation.attempt_id,
            )

    def _run_in_process(
        self,
        reservation: RunReservation,
        *,
        candidate: CandidateCallable,
        timeout_seconds: float | None,
        started: float,
        config: Mapping[str, Any],
        lineage: Mapping[str, Any],
    ) -> RunnerResult:
        prepared = self._load_prepared_data()
        self._log_data_loaded(reservation, prepared)
        self.logger.log_action(
            "training_started",
            experiment_id=reservation.experiment_id,
            details={"attempt_id": reservation.attempt_id, "execution_mode": "in_process"},
        )
        output = candidate(prepared, config, reservation.run_dir)
        validate_candidate_alignment(output, prepared)
        runtime_seconds = self.clock() - started
        if timeout_seconds is not None and runtime_seconds > timeout_seconds:
            message = f"runtime {runtime_seconds:.3f}s exceeded budget {timeout_seconds:.3f}s"
            self._write_failure(reservation, "timed_out", "resource", message, runtime_seconds)
            return RunnerResult(
                reservation.experiment_id,
                "timed_out",
                reservation.run_dir,
                runtime_seconds,
                error=message,
                failure_kind="resource",
                attempt_id=reservation.attempt_id,
            )
        return self._complete(reservation, output, runtime_seconds, lineage=lineage)

    def _run_subprocess(
        self,
        reservation: RunReservation,
        *,
        candidate_key: str,
        timeout_seconds: float | None,
        started: float,
        config: Mapping[str, Any],
        lineage: Mapping[str, Any],
    ) -> RunnerResult:
        stage_name = (
            "research-input-"
            f"{lineage['data_sha256']}-{lineage['staging_code_sha256']}.json.gz"
        )
        staged_input = stage_research_splits(
            self.contract.data_dir,
            self.logger.store.root / "inputs" / stage_name,
            train_split=self.contract.train_split,
            validation_split=self.contract.validation_split,
            source_data_sha256=str(lineage["data_sha256"]),
            staging_code_sha256=str(lineage["staging_code_sha256"]),
        )
        job = {
            "candidate_key": candidate_key,
            "config": dict(config),
            "run_dir": str(reservation.run_dir.resolve()),
            "prepared_data_path": str(staged_input.resolve()),
            "train_split": self.contract.train_split,
            "validation_split": self.contract.validation_split,
            "source_data_sha256": lineage["data_sha256"],
            "staging_code_sha256": lineage["staging_code_sha256"],
            "lineage": dict(lineage),
        }
        job_path = self.logger.store.write_attempt_json(reservation, "worker_job.json", job)
        self.logger.log_action(
            "training_started",
            experiment_id=reservation.experiment_id,
            details={"attempt_id": reservation.attempt_id, "execution_mode": "subprocess"},
        )
        process_result = run_process(
            [sys.executable, "-m", "research_agent.worker", "--job", str(job_path.resolve())],
            timeout_seconds=timeout_seconds,
            stdout_path=reservation.run_dir / "stdout.log",
            stderr_path=reservation.run_dir / "stderr.log",
            cwd=Path(__file__).resolve().parents[1],
        )
        runtime_seconds = self.clock() - started
        if process_result.timed_out:
            message = f"worker exceeded hard timeout {timeout_seconds:.3f}s"
            self._write_failure(reservation, "timed_out", "resource", message, runtime_seconds)
            return RunnerResult(
                reservation.experiment_id,
                "timed_out",
                reservation.run_dir,
                runtime_seconds,
                error=message,
                failure_kind="resource",
                attempt_id=reservation.attempt_id,
            )

        worker_result = self.logger.store.read_attempt_json(reservation, "worker_result.json")
        if process_result.returncode != 0 or not worker_result:
            stderr = _read_tail(reservation.run_dir / "stderr.log")
            message = stderr or f"worker exited with code {process_result.returncode} without a result"
            failure_kind = str((worker_result or {}).get("failure_kind") or "transient")
            message = str((worker_result or {}).get("error") or message)
            self._write_failure(
                reservation,
                _failure_status(failure_kind),
                failure_kind,
                message,
                runtime_seconds,
            )
            return RunnerResult(
                reservation.experiment_id,
                "failed",
                reservation.run_dir,
                runtime_seconds,
                error=message,
                failure_kind=failure_kind,
                attempt_id=reservation.attempt_id,
            )
        output_payload = self.logger.store.read_attempt_json(reservation, "candidate_output.json")
        if not output_payload:
            message = "worker reported completion without candidate_output.json"
            self._write_failure(
                reservation,
                "transient_failed",
                "transient",
                message,
                runtime_seconds,
            )
            return RunnerResult(
                reservation.experiment_id,
                "failed",
                reservation.run_dir,
                runtime_seconds,
                error=message,
                failure_kind="transient",
                attempt_id=reservation.attempt_id,
            )
        return self._complete(
            reservation,
            _candidate_output_from_json(output_payload),
            runtime_seconds,
            lineage=lineage,
        )

    def _complete(
        self,
        reservation: RunReservation,
        output: CandidateOutput,
        runtime_seconds: float,
        *,
        lineage: Mapping[str, Any],
    ) -> RunnerResult:
        output = CandidateOutput(
            output.user_ids,
            output.labels,
            output.scores,
            {**dict(output.metadata), **dict(lineage)},
        )
        output_path = self.logger.store.write_attempt_json(
            reservation,
            "candidate_output.json",
            candidate_output_to_json(output),
        )
        checkpoint_path = Path(str(output.metadata.get("checkpoint_path", "")))
        checkpoint_sha256 = (
            _file_sha256(checkpoint_path) if checkpoint_path.is_file() else None
        )
        self.logger.store.write_attempt_json(
            reservation,
            "runner_result.json",
            {
                "status": "completed",
                "runtime_seconds": runtime_seconds,
                "metadata": dict(output.metadata),
                "attempt_id": reservation.attempt_id,
                "candidate_output_sha256": _file_sha256(output_path),
                "candidate_output_size_bytes": output_path.stat().st_size,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_size_bytes": (
                    checkpoint_path.stat().st_size if checkpoint_path.is_file() else None
                ),
            },
        )
        self.logger.store.transition_run(reservation, "completed")
        self.logger.log_action(
            "training_completed",
            experiment_id=reservation.experiment_id,
            details={"runtime_seconds": runtime_seconds, "attempt_id": reservation.attempt_id},
        )
        return RunnerResult(
            reservation.experiment_id,
            "completed",
            reservation.run_dir,
            runtime_seconds,
            output=output,
            attempt_id=reservation.attempt_id,
        )

    def _load_completed_result(
        self,
        reservation: RunReservation,
        *,
        expected_lineage: Mapping[str, Any],
    ) -> RunnerResult:
        result_payload = self.logger.store.read_attempt_json(reservation, "runner_result.json") or {}
        output_payload = self.logger.store.read_attempt_json(reservation, "candidate_output.json")
        if not output_payload:
            message = "completed attempt is missing candidate_output.json"
            self.logger.log_action(
                "cached_result_invalid",
                experiment_id=reservation.experiment_id,
                details={"attempt_id": reservation.attempt_id, "error": message},
            )
            return RunnerResult(
                reservation.experiment_id,
                "failed",
                reservation.run_dir,
                float(result_payload.get("runtime_seconds", 0.0)),
                error=message,
                failure_kind="deterministic",
                attempt_id=reservation.attempt_id,
                reused=True,
            )
        output_path = reservation.run_dir / "candidate_output.json"
        if (
            result_payload.get("candidate_output_sha256") != _file_sha256(output_path)
            or result_payload.get("candidate_output_size_bytes") != output_path.stat().st_size
        ):
            return self._invalid_cached_result(
                reservation,
                result_payload,
                "completed candidate output hash or size changed",
            )
        output = _candidate_output_from_json(output_payload)
        lineage_mismatches = [
            field
            for field, expected in expected_lineage.items()
            if output.metadata.get(field) != expected
        ]
        if lineage_mismatches:
            return self._invalid_cached_result(
                reservation,
                result_payload,
                "completed candidate lineage changed: "
                + ", ".join(sorted(lineage_mismatches)),
            )
        checkpoint_path = Path(str(output.metadata.get("checkpoint_path", "")))
        expected_checkpoint_hash = result_payload.get("checkpoint_sha256")
        if expected_checkpoint_hash is not None and (
            not checkpoint_path.is_file()
            or expected_checkpoint_hash != _file_sha256(checkpoint_path)
            or result_payload.get("checkpoint_size_bytes") != checkpoint_path.stat().st_size
        ):
            return self._invalid_cached_result(
                reservation,
                result_payload,
                "completed checkpoint hash or size changed",
            )
        try:
            validate_candidate_alignment(
                output,
                self._prepared_data_for_reuse(reservation, expected_lineage),
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return self._invalid_cached_result(
                reservation,
                result_payload,
                f"completed candidate alignment could not be revalidated: {exc}",
            )
        self.logger.log_action(
            "completed_attempt_reused",
            experiment_id=reservation.experiment_id,
            details={"attempt_id": reservation.attempt_id},
        )
        return RunnerResult(
            reservation.experiment_id,
            "completed",
            reservation.run_dir,
            float(result_payload.get("runtime_seconds", 0.0)),
            output=output,
            attempt_id=reservation.attempt_id,
            reused=True,
        )

    def _prepared_data_for_reuse(
        self,
        reservation: RunReservation,
        expected_lineage: Mapping[str, Any],
    ) -> PreparedData:
        job = self.logger.store.read_attempt_json(reservation, "worker_job.json")
        if job:
            splits = load_staged_splits(
                Path(str(job["prepared_data_path"])),
                source_data_sha256=str(expected_lineage["data_sha256"]),
                staging_code_sha256=str(expected_lineage["staging_code_sha256"]),
            )
            return PreparedData(
                train_rows=splits[self.contract.train_split],
                validation_rows=splits[self.contract.validation_split],
                lineage=dict(expected_lineage),
            )
        return self._load_prepared_data(expected_lineage)

    def _invalid_cached_result(
        self,
        reservation: RunReservation,
        result_payload: Mapping[str, Any],
        message: str,
    ) -> RunnerResult:
        self.logger.log_action(
            "cached_result_invalid",
            experiment_id=reservation.experiment_id,
            details={"attempt_id": reservation.attempt_id, "error": message},
        )
        return RunnerResult(
            reservation.experiment_id,
            "failed",
            reservation.run_dir,
            float(result_payload.get("runtime_seconds", 0.0)),
            error=message,
            failure_kind="deterministic",
            attempt_id=reservation.attempt_id,
            reused=True,
        )

    def _load_prepared_data(
        self,
        lineage: Mapping[str, Any] | None = None,
    ) -> PreparedData:
        splits = self.data_loader(str(self.contract.data_dir))
        try:
            return PreparedData(
                train_rows=splits[self.contract.train_split],
                validation_rows=splits[self.contract.validation_split],
                lineage=None if lineage is None else dict(lineage),
            )
        except KeyError as exc:
            raise ValueError(f"data.py did not return required split: {exc.args[0]}") from exc

    def _log_data_loaded(self, reservation: RunReservation, prepared: PreparedData) -> None:
        self.logger.log_action(
            "data_loaded",
            experiment_id=reservation.experiment_id,
            details={
                "attempt_id": reservation.attempt_id,
                "train_rows": len(prepared.train_rows),
                "validation_rows": len(prepared.validation_rows),
                "test_rows_exposed": 0,
            },
        )

    def _write_failure(
        self,
        reservation: RunReservation,
        status: str,
        failure_kind: str,
        message: str,
        runtime_seconds: float,
    ) -> None:
        self.logger.store.write_attempt_json(
            reservation,
            "error.json",
            {
                "status": status,
                "failure_kind": failure_kind,
                "error": message,
                "runtime_seconds": runtime_seconds,
                "attempt_id": reservation.attempt_id,
            },
        )
        self.logger.store.transition_run(
            reservation,
            status,
            failure_kind=failure_kind,
            error=message,
        )
        public_status = "timed_out" if status == "timed_out" else "failed"
        self.logger.log_action(
            public_status,
            experiment_id=reservation.experiment_id,
            details={
                "error": message,
                "runtime_seconds": runtime_seconds,
                "failure_kind": failure_kind,
                "attempt_id": reservation.attempt_id,
            },
        )

    def _should_use_subprocess(self, candidate_path: str) -> bool:
        if self.execution_mode == "in_process":
            return False
        if self.execution_mode == "subprocess":
            if not self._candidate_key(candidate_path):
                raise ValueError("subprocess candidate is not in the approved registry")
            return True
        if self.data_loader is load:
            if not self._candidate_key(candidate_path):
                raise ValueError("production candidate is not in the approved registry")
            return True
        return False

    @staticmethod
    def _candidate_key(candidate_path: str) -> str:
        for key, approved_path in APPROVED_CANDIDATES.items():
            if candidate_path == approved_path:
                return key
        return ""

    @staticmethod
    def _candidate_path(candidate: CandidateCallable) -> str:
        module = getattr(candidate, "__module__", "")
        qualname = getattr(candidate, "__qualname__", "")
        if not module or not qualname or "<locals>" in qualname:
            return ""
        return f"{module}:{qualname}"

    def _fingerprint(
        self,
        hypothesis: str,
        config: Mapping[str, Any],
        candidate_path: str,
        lineage: Mapping[str, Any],
    ) -> str:
        payload = {
            "hypothesis": hypothesis,
            "config": dict(config),
            "candidate": candidate_path,
            "data_dir": str(self.contract.data_dir),
            "train_split": self.contract.train_split,
            "validation_split": self.contract.validation_split,
            "lineage": dict(lineage),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _lineage_metadata(
        self,
        config: Mapping[str, Any],
        candidate: CandidateCallable,
    ) -> dict[str, Any]:
        root = Path(__file__).resolve().parents[1]
        evaluator_sha256 = _normalized_file_sha256(root / "evaluate.py")
        canonical_preprocessing_sha256 = _normalized_file_sha256(root / "data.py")
        staging_code_sha256 = _normalized_file_sha256(
            root / "research_agent" / "data_boundary.py"
        )
        preprocessing_sha256 = hashlib.sha256(
            f"{canonical_preprocessing_sha256}:{staging_code_sha256}".encode("ascii")
        ).hexdigest()
        model_code_sha256 = candidate_code_sha256(candidate)
        data_sha256 = _dataset_sha256(self.contract.data_dir, self.data_loader)
        feature_schema_sha256 = hashlib.sha256(
            json.dumps(list(benchmark_data.FIELDS), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        comparison_payload = {
            "data_sha256": data_sha256,
            "evaluator_sha256": evaluator_sha256,
            "preprocessing_sha256": preprocessing_sha256,
            "staging_code_sha256": staging_code_sha256,
            "feature_schema_sha256": feature_schema_sha256,
        }
        comparison_group_id = hashlib.sha256(
            json.dumps(comparison_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            **comparison_payload,
            "model_code_sha256": model_code_sha256,
            "comparison_group_id": comparison_group_id,
            "seed": config.get("seed"),
        }


def run_process(
    command: Sequence[str],
    *,
    timeout_seconds: float | None,
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path | None = None,
    termination_grace_seconds: float = 2.0,
) -> ProcessResult:
    """Run a command in its own process group and kill the tree on timeout."""
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    popen_kwargs: dict[str, Any] = {"cwd": cwd}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
            **popen_kwargs,
        )
        try:
            process.wait(timeout=timeout_seconds)
            return ProcessResult(process.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            terminate_process_tree(process, grace_seconds=termination_grace_seconds)
            return ProcessResult(process.returncode, timed_out=True)


def terminate_process_tree(process: subprocess.Popen[Any], *, grace_seconds: float = 2.0) -> None:
    """Terminate a process and descendants without relying on their cooperation."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0 and process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return

    kill_process_group = getattr(os, "killpg")
    try:
        kill_process_group(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            kill_process_group(process.pid, getattr(signal, "SIGKILL"))
        except ProcessLookupError:
            pass
        process.wait()


def import_callable(path: str) -> CandidateCallable:
    """Resolve a trusted ``module:qualname`` candidate reference."""
    module_name, separator, qualname = path.partition(":")
    if not separator or not module_name or not qualname:
        raise ValueError(f"invalid candidate path: {path!r}")
    target: Any = importlib.import_module(module_name)
    for segment in qualname.split("."):
        if segment == "<locals>" or segment.startswith("_"):
            raise ValueError(f"unsafe candidate path segment: {segment!r}")
        target = getattr(target, segment)
    if not callable(target):
        raise TypeError(f"candidate is not callable: {path}")
    return target


def candidate_output_to_json(output: CandidateOutput) -> dict[str, Any]:
    return {
        "user_ids": _jsonable_sequence(output.user_ids),
        "labels": _jsonable_sequence(output.labels),
        "scores": _jsonable_sequence(output.scores),
        "metadata": _jsonable(dict(output.metadata)),
    }


def load_completed_candidate_output(
    store: ArtifactStore,
    experiment_id: str,
) -> CandidateOutput:
    """Load one completed prediction vector after verifying its persisted bytes.

    This intentionally exposes only validation candidate output.  It does not
    load raw data and it refuses partial, failed, or hash-drifted attempts.
    """
    experiment_dir = (store.runs_dir / experiment_id).resolve()
    try:
        experiment_dir.relative_to(store.runs_dir.resolve())
    except ValueError as exc:
        raise ValueError("incumbent experiment identifier is unsafe") from exc
    status_path = experiment_dir / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"incumbent run status is unavailable: {experiment_id}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise ValueError(f"incumbent run is not completed: {experiment_id}")
    attempt_id = str(status.get("attempt_id") or "")
    if Path(attempt_id).name != attempt_id or not attempt_id.startswith("attempt-"):
        raise ValueError("incumbent attempt identifier is unsafe")
    attempt_dir = experiment_dir / "attempts" / attempt_id
    output_path = attempt_dir / "candidate_output.json"
    result_path = attempt_dir / "runner_result.json"
    if not output_path.is_file() or not result_path.is_file():
        raise FileNotFoundError("completed incumbent output evidence is incomplete")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("candidate_output_sha256") != _file_sha256(output_path)
        or result.get("candidate_output_size_bytes") != output_path.stat().st_size
    ):
        raise ValueError("completed incumbent output hash or size changed")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    return _candidate_output_from_json(payload)


def validate_candidate_alignment(output: CandidateOutput, prepared: PreparedData) -> None:
    """Require predictions for every validation row in canonical order."""
    if any(len(row) < 7 for row in prepared.validation_rows):
        raise ValueError("validation rows do not match the canonical seven-field schema")
    expected_users = [row[1] for row in prepared.validation_rows]
    expected_labels = [int(row[6]) for row in prepared.validation_rows]
    actual_users = _jsonable_sequence(output.user_ids)
    actual_labels = [int(value) for value in _jsonable_sequence(output.labels)]
    actual_scores = _jsonable_sequence(output.scores)
    if len(actual_scores) != len(expected_users):
        raise ValueError(
            "candidate scores must cover the complete validation split: "
            f"expected {len(expected_users)}, got {len(actual_scores)}"
        )
    if actual_users != expected_users:
        raise ValueError("candidate user_ids do not preserve canonical validation row order")
    if actual_labels != expected_labels:
        raise ValueError("candidate labels do not match canonical validation labels")


def _candidate_output_from_json(payload: Mapping[str, Any]) -> CandidateOutput:
    return CandidateOutput(
        user_ids=list(payload["user_ids"]),
        labels=list(payload["labels"]),
        scores=list(payload["scores"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _jsonable_sequence(values: Sequence[Any]) -> list[Any]:
    if hasattr(values, "tolist"):
        converted = values.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [_jsonable(value) for value in values]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def classify_exception(exc: BaseException) -> str:
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return "interrupted"
    if isinstance(exc, MemoryError):
        return "resource"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transient"
    if isinstance(exc, OSError) and exc.errno in {
        errno.EAGAIN,
        errno.EBUSY,
        errno.ECONNABORTED,
        errno.ECONNRESET,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.ETIMEDOUT,
    }:
        return "transient"
    return "deterministic"


def _failure_status(failure_kind: str) -> str:
    return {
        "transient": "transient_failed",
        "resource": "resource_failed",
        "interrupted": "interrupted",
    }.get(failure_kind, "deterministic_failed")


def _read_tail(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace")
    return content[-limit:].strip()


def _file_sha256(path: Path) -> str:
    """Raw-byte hash for generated binary artifacts (checkpoints, JSON output)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_file_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def _dataset_sha256(data_dir: Path, data_loader: DataLoader) -> str:
    paths = [
        data_dir / "video_features_basic_pure.csv",
        data_dir / "log_standard_4_08_to_4_21_pure.csv",
        data_dir / "log_standard_4_22_to_5_08_pure.csv",
    ]
    digest = hashlib.sha256()
    if all(path.is_file() for path in paths):
        for path in paths:
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    identity = (
        f"in-memory-test-fixture:{data_loader.__module__}:"
        f"{getattr(data_loader, '__qualname__', '')}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
