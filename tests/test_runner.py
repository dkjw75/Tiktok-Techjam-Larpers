import tempfile
import unittest
from pathlib import Path

from research_agent.logger import ResearchLogger
from research_agent.runner import CandidateOutput, ExperimentRunner
from research_agent.store import ArtifactStore


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ArtifactStore(self.tempdir.name)
        self.logger = ResearchLogger(self.store)

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _loader(_data_dir):
        return {
            "train": [("train",)],
            "valid": [("valid",)],
            "test": [("test",)],
        }

    def test_runner_isolates_artifacts_and_hides_test_rows(self):
        seen = {}

        def candidate(data, config, run_dir):
            seen["train"] = data.train_rows
            seen["valid"] = data.validation_rows
            seen["has_test"] = hasattr(data, "test_rows")
            seen["vocabulary_size"] = data.vocabulary_size
            seen["field_count"] = data.field_count
            (run_dir / "checkpoint.note").write_text("candidate artifact", encoding="utf-8")
            return CandidateOutput(["u"], [1], [0.9], {"framework": "pytorch"})

        result = ExperimentRunner(self.logger, data_loader=self._loader).run(
            experiment_id="exp_001",
            hypothesis="A safe candidate should run in isolation.",
            config={"seed": 0},
            candidate=candidate,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(seen["train"], [("train",)])
        self.assertEqual(seen["valid"], [("valid",)])
        self.assertFalse(seen["has_test"])
        self.assertEqual(seen["vocabulary_size"], 1)
        self.assertEqual(seen["field_count"], 5)
        self.assertTrue((result.run_dir / "config.json").exists())
        self.assertTrue((result.run_dir / "checkpoint.note").exists())

    def test_runner_records_candidate_failure(self):
        def broken_candidate(_data, _config, _run_dir):
            raise RuntimeError("training exploded")

        result = ExperimentRunner(self.logger, data_loader=self._loader).run(
            experiment_id="exp_002",
            hypothesis="Failures must preserve evidence.",
            config={},
            candidate=broken_candidate,
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("training exploded", result.error)
        self.assertTrue((result.run_dir / "error.json").exists())
        self.assertEqual(self.store.read_events()[-1]["action"], "failed")

    def test_runner_marks_budget_overrun(self):
        times = iter((0.0, 3.0))

        def candidate(_data, _config, _run_dir):
            return CandidateOutput(["u"], [1], [0.9])

        result = ExperimentRunner(
            self.logger,
            data_loader=self._loader,
            clock=lambda: next(times),
        ).run(
            experiment_id="exp_003",
            hypothesis="Budget overruns are visible.",
            config={},
            candidate=candidate,
            timeout_seconds=2.0,
        )

        self.assertEqual(result.status, "timed_out")
        self.assertIn("exceeded budget", result.error)
        self.assertTrue((result.run_dir / "error.json").exists())
