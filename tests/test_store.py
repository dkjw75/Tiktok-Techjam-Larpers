import json
import tempfile
import unittest

from research_agent.store import (
    ArtifactStore,
    RunCollisionError,
    RunNotRetryableError,
)


class ArtifactStoreReservationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ArtifactStore(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_transient_failure_gets_new_attempt_and_is_not_completed_duplicate(self):
        first = self.store.reserve_run("exp_001", "fingerprint-a")
        self.store.transition_run(first, "running")
        self.store.transition_run(
            first,
            "transient_failed",
            failure_kind="transient",
            error="temporary outage",
        )

        second = self.store.reserve_run("exp_001", "fingerprint-a")

        self.assertEqual(first.attempt_id, "attempt-0001")
        self.assertEqual(second.attempt_id, "attempt-0002")
        self.assertNotIn("fingerprint-a", self.store.completed_fingerprints())
        self.store.transition_run(second, "running")
        self.store.transition_run(second, "completed")
        self.assertIn("fingerprint-a", self.store.completed_fingerprints())

    def test_completed_attempt_is_reused(self):
        first = self.store.reserve_run("exp_002", "fingerprint-b")
        self.store.transition_run(first, "running")
        self.store.transition_run(first, "completed")

        reused = self.store.reserve_run("exp_002", "fingerprint-b")

        self.assertTrue(reused.reused)
        self.assertEqual(reused.attempt_id, first.attempt_id)
        self.assertEqual(reused.run_dir, first.run_dir)

    def test_identifier_collision_with_different_inputs_is_rejected(self):
        first = self.store.reserve_run("exp_003", "fingerprint-c")
        self.store.transition_run(first, "running")
        self.store.transition_run(first, "completed")

        with self.assertRaises(RunCollisionError):
            self.store.reserve_run("exp_003", "fingerprint-other")

    def test_deterministic_failure_is_not_retried_unchanged(self):
        first = self.store.reserve_run("exp_004", "fingerprint-d")
        self.store.transition_run(first, "running")
        self.store.transition_run(first, "deterministic_failed", failure_kind="deterministic")

        with self.assertRaises(RunNotRetryableError):
            self.store.reserve_run("exp_004", "fingerprint-d")

    def test_stale_running_attempt_is_marked_interrupted_before_resume(self):
        first = self.store.reserve_run("exp_005", "fingerprint-e")
        self.store.transition_run(first, "running")
        self.store.release_run(first)  # Simulate process death after the lock disappears.

        resumed = self.store.reserve_run("exp_005", "fingerprint-e")
        first_status = json.loads((first.run_dir / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(first_status["status"], "interrupted")
        self.assertEqual(resumed.attempt_id, "attempt-0002")

    def test_jsonl_reader_preserves_records_before_partial_tail(self):
        self.store.append_event({"action": "completed"})
        with self.store.events_path.open("a", encoding="utf-8") as handle:
            handle.write('{"action": "partial"')

        self.assertEqual(self.store.read_events(), [{"action": "completed"}])

    def test_corrupted_status_fails_closed(self):
        first = self.store.reserve_run("exp_006", "fingerprint-f")
        self.store.release_run(first)
        (first.run_dir.parent.parent / "status.json").write_text("{broken", encoding="utf-8")

        with self.assertRaisesRegex(Exception, "corrupted run evidence"):
            self.store.reserve_run("exp_006", "fingerprint-f")

    def test_illegal_transition_is_rejected(self):
        first = self.store.reserve_run("exp_007", "fingerprint-g")

        with self.assertRaisesRegex(Exception, "illegal run transition"):
            self.store.transition_run(first, "completed")


if __name__ == "__main__":
    unittest.main()
