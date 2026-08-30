import csv
import tempfile
import unittest
from datetime import datetime, timezone

from research_agent.logger import ResearchLogger
from research_agent.reporter import MarkdownReporter
from research_agent.store import ArtifactStore


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ArtifactStore(self.tempdir.name)
        self.logger = ResearchLogger(
            self.store,
            clock=lambda: datetime(2026, 8, 29, tzinfo=timezone.utc),
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_actions_are_append_only_jsonl_records(self):
        self.logger.log_action("training_started", experiment_id="exp_001")
        self.logger.log_action("evaluation_started", experiment_id="exp_001")

        events = self.store.read_events()

        self.assertEqual([event["action"] for event in events], ["training_started", "evaluation_started"])
        self.assertEqual(events[0]["experiment_id"], "exp_001")

    def test_iteration_writes_jsonl_and_metric_summary(self):
        self.logger.record_iteration(
            {
                "experiment_id": "exp_001",
                "parent_experiment_id": "baseline",
                "hypothesis": "A controlled change may improve ranking.",
                "rationale": "Tested with validation only.",
                "config": {"seed": 0},
                "metrics": {"GAUC": 0.67, "nDCG@5": 0.54, "primary": 0.605},
                "delta_primary": 0.0035,
                "runtime_seconds": 12.5,
                "decision": "accepted",
            }
        )

        iterations = self.store.read_iterations()
        self.assertEqual(iterations[0]["decision"], "accepted")
        with self.store.metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["primary"], "0.605")

    def test_patch_and_manual_summary_are_preserved(self):
        patch = self.logger.record_code_diff("exp_001", "diff --git a/a.py b/a.py\n")
        self.logger.record_manual_intervention(
            experiment_id="exp_001",
            description="Reduced the allowed runtime.",
            reason="Laptop battery limit.",
            effect="Future runs use the smaller budget.",
        )

        report = MarkdownReporter(self.store).write()

        self.assertTrue(patch.exists())
        self.assertIn("Manual interventions: 1", report.read_text(encoding="utf-8"))
        self.assertEqual(len(self.store.read_interventions()), 1)

    def test_report_explicitly_records_zero_interventions(self):
        report = MarkdownReporter(self.store).write()
        self.assertIn("Manual interventions: 0", report.read_text(encoding="utf-8"))
