"""Append-only artifact storage for autonomous research runs."""
from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RunReservationError(RuntimeError):
    """Base error for a run directory that cannot be reserved safely."""


class RunBusyError(RunReservationError):
    """Raised when another live process owns an experiment reservation."""


class RunCollisionError(RunReservationError):
    """Raised when an experiment identifier is reused for different inputs."""


class RunNotRetryableError(RunReservationError):
    """Raised when an unchanged deterministic failure is attempted again."""


@dataclass(frozen=True)
class RunReservation:
    """One durable attempt within a logical experiment."""

    experiment_id: str
    attempt_id: str
    run_dir: Path
    fingerprint: str
    reused: bool = False


class ArtifactStore:
    """Stores run evidence without overwriting prior experiment history."""

    _SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    METRIC_COLUMNS = (
        "experiment_id",
        "parent_experiment_id",
        "decision",
        "GAUC",
        "nDCG@5",
        "primary",
        "delta_primary",
        "runtime_seconds",
    )

    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)
        self.events_path = self.root / "experiments.jsonl"
        self.iterations_path = self.root / "iterations.jsonl"
        self.metrics_path = self.root / "metrics.csv"
        self.interventions_path = self.root / "manual_interventions.jsonl"
        self.patches_dir = self.root / "patches"
        self.runs_dir = self.root / "runs"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.patches_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)

    def run_dir(self, experiment_id: str) -> Path:
        """Create a legacy one-directory run.

        New runner code should use :meth:`reserve_run` so interrupted attempts
        can be resumed without overwriting evidence. This method remains for
        compatibility with callers that deliberately require create-once
        behavior.
        """
        self._validate_identifier(experiment_id)
        self.initialize()
        path = self.runs_dir / experiment_id
        path.mkdir(exist_ok=False)
        return path

    def run_experiment_ids(self) -> tuple[str, ...]:
        """Return every durably allocated logical experiment identifier."""
        if not self.runs_dir.is_dir():
            return ()
        return tuple(sorted(path.name for path in self.runs_dir.iterdir() if path.is_dir()))

    def reserve_run(self, experiment_id: str, fingerprint: str) -> RunReservation:
        """Atomically reserve an attempt or reuse an identical completed one.

        A logical experiment owns an append-only ``attempts`` directory. A
        stale ``reserved`` or ``running`` attempt is marked ``interrupted`` on
        resume. Only transient/resource/interrupted outcomes may be retried
        unchanged; completed results are reused and deterministic failures
        require a new configuration or experiment identifier.
        """
        self._validate_identifier(experiment_id)
        if not fingerprint.strip():
            raise ValueError("fingerprint must not be empty")
        self.initialize()
        experiment_dir = self.runs_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        lock_path = experiment_dir / ".reservation.lock"
        self._acquire_lock(lock_path)
        try:
            state_path = experiment_dir / "status.json"
            state = self._read_json(state_path, strict=True)
            if state:
                existing_fingerprint = str(state.get("fingerprint", ""))
                if existing_fingerprint != fingerprint:
                    raise RunCollisionError(
                        f"experiment id {experiment_id!r} already belongs to different inputs"
                    )
                status = str(state.get("status", ""))
                attempt_id = str(state.get("attempt_id", ""))
                attempt_dir = experiment_dir / "attempts" / attempt_id
                if status == "completed":
                    self._release_lock(lock_path)
                    return RunReservation(
                        experiment_id,
                        attempt_id,
                        attempt_dir,
                        fingerprint,
                        reused=True,
                    )
                if status == "deterministic_failed":
                    raise RunNotRetryableError(
                        f"experiment {experiment_id!r} failed deterministically; change its inputs before retrying"
                    )
                if status in {"reserved", "running"}:
                    self._write_attempt_status(
                        attempt_dir,
                        {
                            **state,
                            "status": "interrupted",
                            "failure_kind": "interrupted",
                            "error": "previous process ended before recording a terminal status",
                        },
                    )

            attempts_dir = experiment_dir / "attempts"
            attempts_dir.mkdir(exist_ok=True)
            attempt_number = self._next_attempt_number(attempts_dir)
            attempt_id = f"attempt-{attempt_number:04d}"
            attempt_dir = attempts_dir / attempt_id
            attempt_dir.mkdir(exist_ok=False)
            reservation = RunReservation(experiment_id, attempt_id, attempt_dir, fingerprint)
            self._write_reservation_status(reservation, "reserved")
            return reservation
        except Exception:
            self._release_lock(lock_path)
            raise

    def transition_run(
        self,
        reservation: RunReservation,
        status: str,
        *,
        failure_kind: str | None = None,
        error: str | None = None,
    ) -> None:
        """Persist a run transition atomically and release terminal locks."""
        legal_transitions = {
            "reserved": {"running", "interrupted", "deterministic_failed"},
            "running": {
                "completed",
                "transient_failed",
                "deterministic_failed",
                "resource_failed",
                "timed_out",
                "interrupted",
            },
        }
        if status not in set().union(*legal_transitions.values()):
            raise ValueError(f"unsupported run status: {status}")
        current = self._read_json(reservation.run_dir / "status.json", strict=True)
        if not current:
            raise RunReservationError("run status is missing")
        if (
            current.get("experiment_id") != reservation.experiment_id
            or current.get("attempt_id") != reservation.attempt_id
            or current.get("fingerprint") != reservation.fingerprint
        ):
            raise RunCollisionError("reservation identity does not match persisted run status")
        current_status = str(current.get("status", ""))
        if status not in legal_transitions.get(current_status, set()):
            raise RunReservationError(
                f"illegal run transition: {current_status or 'missing'} -> {status}"
            )
        payload = {
            "experiment_id": reservation.experiment_id,
            "attempt_id": reservation.attempt_id,
            "fingerprint": reservation.fingerprint,
            "status": status,
            "failure_kind": failure_kind,
            "error": error,
        }
        self._write_attempt_status(reservation.run_dir, payload)
        if status not in {"reserved", "running"}:
            self._release_lock(self.runs_dir / reservation.experiment_id / ".reservation.lock")

    def release_run(self, reservation: RunReservation) -> None:
        """Release a reservation lock after an unexpected orchestration error."""
        self._release_lock(self.runs_dir / reservation.experiment_id / ".reservation.lock")

    def write_attempt_json(
        self,
        reservation: RunReservation,
        filename: str,
        payload: dict[str, Any],
    ) -> Path:
        """Atomically write evidence inside one immutable attempt directory."""
        if Path(filename).name != filename:
            raise ValueError("filename must not contain a directory")
        if not reservation.run_dir.exists():
            raise FileNotFoundError(f"run attempt does not exist: {reservation.attempt_id}")
        destination = reservation.run_dir / filename
        self._atomic_write_text(destination, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        return destination

    def read_attempt_json(self, reservation: RunReservation, filename: str) -> dict[str, Any] | None:
        if Path(filename).name != filename:
            raise ValueError("filename must not contain a directory")
        return self._read_json(reservation.run_dir / filename, strict=True)

    def completed_fingerprints(self) -> set[str]:
        """Return only successfully completed inputs for duplicate filtering."""
        self.initialize()
        fingerprints: set[str] = set()
        for status_path in self.runs_dir.glob("*/status.json"):
            state = self._read_json(status_path, strict=True)
            if state and state.get("status") == "completed" and state.get("fingerprint"):
                fingerprints.add(str(state["fingerprint"]))
        return fingerprints

    def append_event(self, event: dict[str, Any]) -> None:
        self._append_jsonl(self.events_path, event)

    def append_iteration(self, iteration: dict[str, Any]) -> None:
        self._append_jsonl(self.iterations_path, iteration)

    def append_intervention(self, intervention: dict[str, Any]) -> None:
        self._append_jsonl(self.interventions_path, intervention)

    def append_metric_summary(self, summary: dict[str, Any]) -> None:
        self.initialize()
        has_file = self.metrics_path.exists()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.METRIC_COLUMNS)
            if not has_file:
                writer.writeheader()
            writer.writerow({column: summary.get(column, "") for column in self.METRIC_COLUMNS})

    def write_patch(self, experiment_id: str, diff_text: str) -> Path:
        self._validate_identifier(experiment_id)
        self.initialize()
        destination = self.patches_dir / f"{experiment_id}.patch"
        destination.write_text(diff_text, encoding="utf-8")
        return destination

    def write_run_json(self, experiment_id: str, filename: str, payload: dict[str, Any]) -> Path:
        self._validate_identifier(experiment_id)
        if Path(filename).name != filename:
            raise ValueError("filename must not contain a directory")
        run_dir = self.runs_dir / experiment_id
        if not run_dir.exists():
            raise FileNotFoundError(f"run directory does not exist: {experiment_id}")
        destination = run_dir / filename
        self._atomic_write_text(destination, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        return destination

    def write_root_json(self, filename: str, payload: dict[str, Any]) -> Path:
        """Write a small current-state document at the artifact root."""
        if Path(filename).name != filename:
            raise ValueError("filename must not contain a directory")
        self.initialize()
        destination = self.root / filename
        self._atomic_write_text(destination, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        return destination

    def read_events(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.events_path)

    def read_iterations(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.iterations_path)

    def read_interventions(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.interventions_path)

    def read_root_json(self, filename: str) -> dict[str, Any] | None:
        if Path(filename).name != filename:
            raise ValueError("filename must not contain a directory")
        source = self.root / filename
        if not source.exists():
            return None
        return json.loads(source.read_text(encoding="utf-8"))

    def _append_jsonl(self, destination: Path, payload: dict[str, Any]) -> None:
        self.initialize()
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    @staticmethod
    def _read_jsonl(source: Path) -> list[dict[str, Any]]:
        if not source.exists():
            return []
        records: list[dict[str, Any]] = []
        lines = source.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                # A process may die between writing bytes and the terminating
                # newline. Preserve all prior evidence and ignore that tail.
        return records

    def _write_reservation_status(self, reservation: RunReservation, status: str) -> None:
        payload = {
            "experiment_id": reservation.experiment_id,
            "attempt_id": reservation.attempt_id,
            "fingerprint": reservation.fingerprint,
            "status": status,
            "failure_kind": None,
            "error": None,
        }
        self._write_attempt_status(reservation.run_dir, payload)

    def _write_attempt_status(self, attempt_dir: Path, payload: dict[str, Any]) -> None:
        self._atomic_write_text(
            attempt_dir / "status.json",
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        )
        experiment_status = attempt_dir.parent.parent / "status.json"
        self._atomic_write_text(
            experiment_status,
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        )

    @staticmethod
    def _next_attempt_number(attempts_dir: Path) -> int:
        numbers = []
        for path in attempts_dir.glob("attempt-*"):
            try:
                numbers.append(int(path.name.removeprefix("attempt-")))
            except ValueError:
                continue
        return max(numbers, default=0) + 1

    def _acquire_lock(self, lock_path: Path) -> None:
        now = time.time()
        payload = json.dumps({"pid": os.getpid(), "created_at_epoch_seconds": now})
        for _ in range(2):
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                owner = self._read_json(lock_path) or {}
                owner_pid = owner.get("pid")
                created_at = owner.get("created_at_epoch_seconds")
                age = now - float(created_at) if isinstance(created_at, (int, float)) else None
                # An empty/partial lock may be observed during the tiny window
                # between O_EXCL creation and its fsync. Treat it as busy until
                # it is old enough to be unambiguously stale. The seven-hour
                # ceiling also recovers a lock whose PID was later reused.
                if age is None and time.time() - lock_path.stat().st_mtime < 30.0:
                    raise RunBusyError("experiment reservation is being initialized")
                if (
                    isinstance(owner_pid, int)
                    and self._pid_is_running(owner_pid)
                    and (age is None or age < 7 * 60 * 60)
                ):
                    raise RunBusyError(f"experiment reservation is owned by live process {owner_pid}")
                lock_path.unlink(missing_ok=True)
                continue
            try:
                os.write(descriptor, payload.encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return
        raise RunBusyError(f"could not acquire experiment reservation: {lock_path.parent.name}")

    @staticmethod
    def _release_lock(lock_path: Path) -> None:
        lock_path.unlink(missing_ok=True)

    @staticmethod
    def _pid_is_running(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    @staticmethod
    def _read_json(source: Path, *, strict: bool = False) -> dict[str, Any] | None:
        if not source.exists():
            return None
        try:
            return json.loads(source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            if strict:
                raise RunReservationError(f"corrupted run evidence: {source}") from exc
            return None

    @staticmethod
    def _atomic_write_text(destination: Path, content: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _validate_identifier(self, identifier: str) -> None:
        if not self._SAFE_IDENTIFIER.fullmatch(identifier):
            raise ValueError(f"unsafe experiment identifier: {identifier!r}")
