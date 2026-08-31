"""Explicit, evidence-preserving recovery for an interrupted finalization."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .store import ArtifactStore


RECOVERABLE_SOURCE_STATUSES = {
    "test_access_started",
    "failed_after_test_boundary",
}
RECOVERY_AUTHORIZED_STATUS = "recovery_authorized"


@dataclass(frozen=True)
class FinalizationRecoveryResult:
    """Durable evidence for one manually authorized recovery transition."""

    previous_status: str
    recovery_record_path: Path
    recovery_record_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "status": RECOVERY_AUTHORIZED_STATUS,
            "previous_status": self.previous_status,
            "recovery_record_path": str(self.recovery_record_path),
            "recovery_record_sha256": self.recovery_record_sha256,
        }


def recover_interrupted_finalization(
    store: ArtifactStore,
    *,
    confirm_recovery: bool,
    reason: str,
    submission_path: str | Path | None = None,
) -> FinalizationRecoveryResult:
    """Authorize one retry only when the interrupted boundary left no outputs.

    The previous certificate is copied verbatim into a create-once recovery
    record before ``finalization.json`` changes. No prior evidence is deleted.
    The finalizer must explicitly recognize ``recovery_authorized`` before the
    transition can be used to cross the boundary again.
    """
    if not confirm_recovery:
        raise PermissionError(
            "finalization recovery requires explicit confirm_recovery=True"
        )
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("finalization recovery requires a non-empty reason")

    recovery_lock = store.root / ".finalization-recovery.lock"
    _acquire_recovery_lock(recovery_lock)
    try:
        return _recover_locked(
            store,
            reason=normalized_reason,
            submission_path=submission_path,
        )
    finally:
        recovery_lock.unlink(missing_ok=True)


def _recover_locked(
    store: ArtifactStore,
    *,
    reason: str,
    submission_path: str | Path | None,
) -> FinalizationRecoveryResult:
    """Perform recovery while this process exclusively owns recovery state."""

    certificate_path = store.root / "finalization.json"
    if not certificate_path.is_file():
        raise FileNotFoundError("finalization.json is missing")
    certificate_bytes = certificate_path.read_bytes()
    try:
        previous = json.loads(certificate_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError("finalization.json is not valid JSON") from exc
    if not isinstance(previous, dict):
        raise RuntimeError("finalization.json must contain a JSON object")
    previous_status = str(previous.get("status", ""))
    if previous_status not in RECOVERABLE_SOURCE_STATUSES:
        raise RuntimeError(
            "finalization recovery is allowed only from "
            "test_access_started or failed_after_test_boundary"
        )
    fingerprint = previous.get("fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise RuntimeError("interrupted finalization has no valid transaction fingerprint")

    lock_evidence = _inactive_lock_evidence(store.root / ".finalization.lock")
    output_evidence = _output_absence_evidence(
        store,
        previous,
        submission_path=submission_path,
    )
    now = datetime.now(timezone.utc)
    prior_sha256 = hashlib.sha256(certificate_bytes).hexdigest()
    record: dict[str, Any] = {
        "schema_version": 1,
        "action": "recovery_authorization_intent",
        "created_at_utc": now.isoformat(),
        "created_at_epoch_seconds": time.time(),
        "reason": reason,
        "previous_status": previous_status,
        "previous_finalization": previous,
        "previous_finalization_file": {
            "path": str(certificate_path.resolve()),
            "sha256": prior_sha256,
            "size_bytes": len(certificate_bytes),
        },
        "lock_evidence": lock_evidence,
        "output_absence_evidence": output_evidence,
    }
    record_bytes = (
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    records_dir = store.root / "finalization_recoveries"
    records_dir.mkdir(parents=True, exist_ok=True)
    record_path = records_dir / (
        f"recovery-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{record_sha256[:12]}.json"
    )
    _write_create_once(record_path, record_bytes)

    # Recheck the certificate immediately before the transition so a concurrent
    # actor cannot replace the evidence that was just archived.
    if not certificate_path.is_file() or certificate_path.read_bytes() != certificate_bytes:
        raise RuntimeError(
            "finalization.json changed during recovery; immutable recovery record retained"
        )
    store.write_root_json(
        "finalization.json",
        {
            "status": RECOVERY_AUTHORIZED_STATUS,
            "fingerprint": fingerprint,
            "recovered_from_status": previous_status,
            "recovery_authorized_at_utc": now.isoformat(),
            "recovery_reason": reason,
            "recovery_record_path": str(record_path.resolve()),
            "recovery_record_sha256": record_sha256,
            "previous_finalization_sha256": prior_sha256,
        },
    )
    return FinalizationRecoveryResult(
        previous_status=previous_status,
        recovery_record_path=record_path,
        recovery_record_sha256=record_sha256,
    )


def _acquire_recovery_lock(lock_path: Path) -> None:
    """Serialize recovery attempts; a pre-existing lock fails closed."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        # A crash between creating the lock and finishing recovery would
        # otherwise wedge recovery permanently. Reclaim ONLY a lock whose owner
        # is demonstrably dead; a live or unreadable owner still fails closed.
        if not _reclaim_dead_recovery_lock(lock_path):
            raise RuntimeError("another finalization recovery attempt is active") from exc
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as retry_exc:
            raise RuntimeError(
                "another finalization recovery attempt is active"
            ) from retry_exc
    try:
        payload = json.dumps(
            {"pid": os.getpid(), "created_at_epoch_seconds": time.time()},
            sort_keys=True,
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _reclaim_dead_recovery_lock(lock_path: Path) -> bool:
    """Archive and remove a recovery lock owned by a dead process.

    Returns True only when the lock was provably abandoned. A malformed lock is
    treated as live: we cannot prove its owner is gone, so we refuse rather than
    guess. Reclaiming leaves durable evidence -- it is never silent.
    """
    try:
        raw = lock_path.read_bytes()
    except OSError:
        return False
    try:
        payload = json.loads(raw.decode("utf-8"))
        owner = int(payload["pid"])
    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
        return False
    if ArtifactStore._pid_is_running(owner):
        return False
    archive = lock_path.with_name(f"{lock_path.name}.abandoned")
    archive.write_text(
        json.dumps(
            {
                "reclaimed_lock_sha256": hashlib.sha256(raw).hexdigest(),
                "reclaimed_lock_bytes": raw.decode("utf-8", errors="replace"),
                "dead_owner_pid": owner,
                "reclaimed_by_pid": os.getpid(),
                "reclaimed_at_epoch_seconds": time.time(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lock_path.unlink(missing_ok=True)
    return True


def _inactive_lock_evidence(lock_path: Path) -> dict[str, Any]:
    if not lock_path.exists():
        return {
            "path": str(lock_path.resolve()),
            "present": False,
            "owner_pid": None,
            "owner_pid_running": False,
        }
    raw = lock_path.read_bytes()
    try:
        value = json.loads(raw)
        owner_pid = int(value["pid"] if isinstance(value, dict) else value)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "finalization lock exists but its owner cannot be verified"
        ) from exc
    owner_running = ArtifactStore._pid_is_running(owner_pid)
    if owner_running:
        raise RuntimeError(
            f"finalization recovery refused: owner PID {owner_pid} is still active"
        )
    return {
        "path": str(lock_path.resolve()),
        "present": True,
        "owner_pid": owner_pid,
        "owner_pid_running": False,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _output_absence_evidence(
    store: ArtifactStore,
    previous: dict[str, Any],
    *,
    submission_path: str | Path | None,
) -> list[dict[str, Any]]:
    forbidden_fields = {
        "test_metrics",
        "test_primary",
        "test_GAUC",
        "test_nDCG@5",
        "submission_path",
        "submission_sha256",
        "submission_checked",
        "report_path",
    }
    present_fields = sorted(field for field in forbidden_fields if field in previous)
    if present_fields:
        raise RuntimeError(
            "finalization certificate already contains output/test evidence: "
            + ", ".join(present_fields)
        )

    original_target = previous.get("submission_target")
    if not isinstance(original_target, str) or not original_target:
        raise RuntimeError(
            "interrupted finalization did not persist its original submission target; "
            "recovery cannot prove output absence for this legacy transaction"
        )
    expected_submission = Path(original_target)
    if submission_path is not None and Path(submission_path).resolve() != expected_submission.resolve():
        raise RuntimeError(
            "recovery submission path differs from the interrupted transaction target"
        )
    exact_paths = {
        expected_submission,
        store.root / "final_submission.csv",
        store.root / "final_summary.json",
        store.root / "final_test_metrics.json",
        store.root / "test_metrics.json",
        store.root / "final_test_predictions.csv",
        store.root / "test_predictions.csv",
        # The ensemble finalizer writes prediction-split member ranks here
        # before it writes the submission. Their presence proves test-derived
        # output escaped the interrupted process even if the CSV did not.
        store.root / "ensemble_members.npz",
    }
    discovered = set(exact_paths)
    if store.root.is_dir():
        patterns = (
            "*final*submission*",
            "*final*summary*",
            "*test*metric*",
            "*test*prediction*",
        )
        for pattern in patterns:
            discovered.update(path for path in store.root.rglob(pattern) if path.is_file())

    existing = sorted(path for path in discovered if path.exists())
    if existing:
        raise RuntimeError(
            "finalization recovery refused because output/test artifacts exist: "
            + ", ".join(str(path) for path in existing)
        )
    return [
        {
            "path": str(path.resolve()),
            "present": False,
        }
        for path in sorted(discovered, key=lambda value: str(value))
    ]


def _write_create_once(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
