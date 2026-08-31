from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from research_agent.finalization_recovery import (
    RECOVERY_AUTHORIZED_STATUS,
    recover_interrupted_finalization,
)
from research_agent.finalize import _validate_recovery_authorization
from research_agent.store import ArtifactStore


class FinalizationRecoveryTests(unittest.TestCase):
    def _store_with_status(self, directory: str, status: str) -> ArtifactStore:
        store = ArtifactStore(directory)
        store.write_root_json(
            "finalization.json",
            {
                "status": status,
                "fingerprint": "f" * 64,
                "submission_target": str((store.root / "final_submission.csv").resolve()),
                "error": "interrupted",
            },
        )
        return store

    def test_recovery_archives_prior_certificate_and_authorizes_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_status(directory, "test_access_started")

            result = recover_interrupted_finalization(
                store,
                confirm_recovery=True,
                reason="operator killed leaking implementation before any output",
            )

            current = store.read_root_json("finalization.json")
            assert current is not None
            self.assertEqual(current["status"], RECOVERY_AUTHORIZED_STATUS)
            self.assertEqual(current["recovered_from_status"], "test_access_started")
            record = json.loads(result.recovery_record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["previous_finalization"]["status"], "test_access_started")
            self.assertEqual(record["previous_finalization_file"]["sha256"], current["previous_finalization_sha256"])
            self.assertFalse(record["lock_evidence"]["present"])
            self.assertTrue(
                all(not item["present"] for item in record["output_absence_evidence"])
            )
            evidence = _validate_recovery_authorization(
                store, current, "f" * 64
            )
            self.assertEqual(
                evidence["recovery_record_sha256"], result.recovery_record_sha256
            )

    def test_finalizer_rejects_recovery_for_a_different_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_status(directory, "test_access_started")
            recover_interrupted_finalization(
                store,
                confirm_recovery=True,
                reason="archive interrupted boundary",
            )
            current = store.read_root_json("finalization.json")
            assert current is not None
            with self.assertRaisesRegex(RuntimeError, "different transaction"):
                _validate_recovery_authorization(store, current, "e" * 64)

    def test_finalizer_rejects_tampered_recovery_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_status(directory, "test_access_started")
            result = recover_interrupted_finalization(
                store,
                confirm_recovery=True,
                reason="archive interrupted boundary",
            )
            result.recovery_record_path.write_text("{}\n", encoding="utf-8")
            current = store.read_root_json("finalization.json")
            assert current is not None
            with self.assertRaisesRegex(RuntimeError, "hash changed"):
                _validate_recovery_authorization(store, current, "f" * 64)

    def test_failed_after_boundary_is_recoverable_when_no_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_status(directory, "failed_after_test_boundary")

            result = recover_interrupted_finalization(
                store,
                confirm_recovery=True,
                reason="verified failure occurred before output persistence",
            )

            self.assertEqual(result.previous_status, "failed_after_test_boundary")

    def test_recovery_requires_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_status(directory, "test_access_started")
            with self.assertRaisesRegex(PermissionError, "explicit"):
                recover_interrupted_finalization(
                    store,
                    confirm_recovery=False,
                    reason="not authorized",
                )

    def test_recovery_refuses_non_boundary_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_status(directory, "failed_before_test_boundary")
            with self.assertRaisesRegex(RuntimeError, "allowed only"):
                recover_interrupted_finalization(
                    store,
                    confirm_recovery=True,
                    reason="wrong state",
                )

    def test_recovery_refuses_missing_transaction_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            store.write_root_json(
                "finalization.json",
                {"status": "test_access_started"},
            )
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                recover_interrupted_finalization(
                    store,
                    confirm_recovery=True,
                    reason="cannot bind retry to original transaction",
                )

    def test_recovery_attempts_are_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_status(directory, "test_access_started")
            (store.root / ".finalization-recovery.lock").write_text(
                json.dumps({"pid": os.getpid()}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "recovery attempt is active"):
                recover_interrupted_finalization(
                    store,
                    confirm_recovery=True,
                    reason="must serialize manual recovery",
                )

    def test_recovery_refuses_a_live_finalization_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_status(directory, "test_access_started")
            (store.root / ".finalization.lock").write_text(
                json.dumps({"pid": os.getpid()}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "still active"):
                recover_interrupted_finalization(
                    store,
                    confirm_recovery=True,
                    reason="must not race the active owner",
                )

    def test_recovery_records_an_inactive_lock_without_deleting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_status(directory, "test_access_started")
            lock = store.root / ".finalization.lock"
            lock.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")

            result = recover_interrupted_finalization(
                store,
                confirm_recovery=True,
                reason="prior finalization owner is no longer running",
            )

            record = json.loads(result.recovery_record_path.read_text(encoding="utf-8"))
            self.assertTrue(record["lock_evidence"]["present"])
            self.assertFalse(record["lock_evidence"]["owner_pid_running"])
            self.assertTrue(lock.exists())

    def test_recovery_refuses_existing_outputs_or_test_metrics(self) -> None:
        cases = (
            "final_submission.csv",
            "final_summary.json",
            "test_metrics.json",
            "ensemble_members.npz",
        )
        for filename in cases:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                store = self._store_with_status(directory, "test_access_started")
                (Path(directory) / filename).write_text("evidence\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "artifacts exist"):
                    recover_interrupted_finalization(
                        store,
                        confirm_recovery=True,
                        reason="outputs forbid a retry",
                    )

            with self.subTest(payload_field="test_metrics"), tempfile.TemporaryDirectory() as directory:
                store = ArtifactStore(directory)
                store.write_root_json(
                    "finalization.json",
                    {
                        "status": "failed_after_test_boundary",
                        "fingerprint": "f" * 64,
                        "test_metrics": {},
                    },
                )
                with self.assertRaisesRegex(RuntimeError, "contains output/test evidence"):
                    recover_interrupted_finalization(
                        store,
                        confirm_recovery=True,
                        reason="metrics forbid a retry",
                    )


class RecoveryStaleLockTests(unittest.TestCase):
    """A crash must not wedge recovery forever, but guessing is worse than waiting."""

    def _lock(self, directory, payload):
        path = Path(directory) / ".finalization-recovery.lock"
        path.write_bytes(payload)
        return path

    def test_lock_owned_by_a_live_process_is_refused(self):
        import os

        from research_agent.finalization_recovery import _reclaim_dead_recovery_lock

        with tempfile.TemporaryDirectory() as directory:
            lock = self._lock(
                directory,
                json.dumps({"pid": os.getpid(), "created_at_epoch_seconds": 1.0}).encode(),
            )
            self.assertFalse(_reclaim_dead_recovery_lock(lock))
            self.assertTrue(lock.exists(), "a live lock must never be removed")

    def test_lock_owned_by_a_dead_process_is_archived_and_reclaimed(self):
        from research_agent.finalization_recovery import _reclaim_dead_recovery_lock

        with tempfile.TemporaryDirectory() as directory:
            raw = json.dumps({"pid": 999_999_999, "created_at_epoch_seconds": 1.0}).encode()
            lock = self._lock(directory, raw)
            self.assertTrue(_reclaim_dead_recovery_lock(lock))
            self.assertFalse(lock.exists())
            archive = lock.with_name(f"{lock.name}.abandoned")
            self.assertTrue(archive.exists(), "reclaiming must leave evidence")
            evidence = json.loads(archive.read_text(encoding="utf-8"))
            self.assertEqual(evidence["dead_owner_pid"], 999_999_999)
            self.assertEqual(
                evidence["reclaimed_lock_sha256"], hashlib.sha256(raw).hexdigest()
            )

    def test_malformed_lock_fails_closed(self):
        from research_agent.finalization_recovery import _reclaim_dead_recovery_lock

        with tempfile.TemporaryDirectory() as directory:
            lock = self._lock(directory, b"not json at all")
            self.assertFalse(
                _reclaim_dead_recovery_lock(lock),
                "an unreadable owner cannot be proven dead",
            )
            self.assertTrue(lock.exists())

    def test_lock_without_a_pid_field_fails_closed(self):
        from research_agent.finalization_recovery import _reclaim_dead_recovery_lock

        with tempfile.TemporaryDirectory() as directory:
            lock = self._lock(directory, json.dumps({"created_at": 1.0}).encode())
            self.assertFalse(_reclaim_dead_recovery_lock(lock))
            self.assertTrue(lock.exists())
