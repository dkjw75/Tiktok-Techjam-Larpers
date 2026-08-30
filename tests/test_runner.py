import tempfile
import unittest
from pathlib import Path
import sys
import time

from research_agent.contracts import BenchmarkContract
from research_agent.logger import ResearchLogger
from research_agent.runner import CandidateOutput, ExperimentRunner, PreparedData, run_process
from research_agent.store import ArtifactStore


def subprocess_candidate(data: PreparedData, _config, _run_dir):
    """Importable test candidate used to prove the worker boundary."""
    row = data.validation_rows[0]
    return CandidateOutput([row[1]], [row[6]], [0.75], {"worker_pid": __import__("os").getpid()})


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
            "valid": [
                (20220421, "u", "v1", "a1", "1", 1000.0, 1),
                (20220421, "u", "v2", "a2", "1", 1000.0, 0),
            ],
            "test": [("test",)],
        }

    def test_runner_isolates_artifacts_and_hides_test_rows(self):
        seen = {}

        def candidate(data, config, run_dir):
            seen["train"] = data.train_rows
            seen["valid"] = data.validation_rows
            seen["has_test"] = hasattr(data, "test_rows")
            (run_dir / "checkpoint.note").write_text("candidate artifact", encoding="utf-8")
            return CandidateOutput(
                ["u", "u"], [1, 0], [0.9, 0.1], {"framework": "pytorch"}
            )

        result = ExperimentRunner(self.logger, data_loader=self._loader).run(
            experiment_id="exp_001",
            hypothesis="A safe candidate should run in isolation.",
            config={"seed": 0},
            candidate=candidate,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(seen["train"], [("train",)])
        self.assertEqual(seen["valid"], self._loader("")["valid"])
        self.assertFalse(seen["has_test"])
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
        self.assertEqual(result.failure_kind, "deterministic")
        self.assertEqual(result.attempt_id, "attempt-0001")
        self.assertIn("training exploded", result.error)
        self.assertTrue((result.run_dir / "error.json").exists())
        self.assertEqual(self.store.read_events()[-1]["action"], "failed")

    def test_candidate_cannot_score_only_an_easy_validation_subset(self):
        def subset_candidate(_data, _config, _run_dir):
            return CandidateOutput(["valid"], [1], [1.0])

        runner = ExperimentRunner(self.logger, data_loader=self._loader)
        result = runner.run(
            experiment_id="exp_subset",
            hypothesis="A partial validation prediction must be rejected.",
            config={"seed": 0},
            candidate=subset_candidate,
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_kind, "deterministic")
        self.assertIn("complete validation split", result.error)

    def test_runner_marks_budget_overrun(self):
        times = iter((0.0, 3.0))

        def candidate(_data, _config, _run_dir):
            return CandidateOutput(["u", "u"], [1, 0], [0.9, 0.1])

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
        self.assertEqual(result.failure_kind, "resource")
        self.assertIn("exceeded budget", result.error)
        self.assertTrue((result.run_dir / "error.json").exists())

    def test_transient_failure_can_retry_same_experiment(self):
        calls = 0

        def flaky_candidate(_data, _config, _run_dir):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("temporary worker outage")
            return CandidateOutput(["u", "u"], [1, 0], [0.9, 0.1])

        runner = ExperimentRunner(self.logger, data_loader=self._loader)
        first = runner.run(
            experiment_id="exp_retry",
            hypothesis="Retry only transient infrastructure failures.",
            config={"seed": 0},
            candidate=flaky_candidate,
        )
        second = runner.run(
            experiment_id="exp_retry",
            hypothesis="Retry only transient infrastructure failures.",
            config={"seed": 0},
            candidate=flaky_candidate,
        )

        self.assertEqual(first.failure_kind, "transient")
        self.assertEqual(first.attempt_id, "attempt-0001")
        self.assertEqual(second.status, "completed")
        self.assertEqual(second.attempt_id, "attempt-0002")

    def test_completed_attempt_is_reused_without_rerunning_candidate(self):
        calls = 0

        def candidate(_data, _config, _run_dir):
            nonlocal calls
            calls += 1
            return CandidateOutput(["u", "u"], [1, 0], [0.9, 0.1])

        runner = ExperimentRunner(self.logger, data_loader=self._loader)
        first = runner.run(
            experiment_id="exp_cached",
            hypothesis="Identical completed work should be reusable.",
            config={"seed": 0},
            candidate=candidate,
        )
        second = runner.run(
            experiment_id="exp_cached",
            hypothesis="Identical completed work should be reusable.",
            config={"seed": 0},
            candidate=candidate,
        )

        self.assertEqual(first.status, "completed")
        self.assertTrue(second.reused)
        self.assertEqual(second.output.scores, [0.9, 0.1])
        self.assertEqual(calls, 1)

    def test_real_process_is_killed_at_hard_timeout(self):
        started = time.monotonic()
        result = run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=0.2,
            stdout_path=Path(self.tempdir.name) / "stdout.log",
            stderr_path=Path(self.tempdir.name) / "stderr.log",
        )

        self.assertTrue(result.timed_out)
        self.assertLess(time.monotonic() - started, 5.0)

    def test_default_production_path_runs_candidate_in_worker_process(self):
        try:
            from research_agent.models.torch_fm import run_torch_fm_candidate
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest("declared Torch integration dependency is unavailable")
            raise
        data_dir = Path(self.tempdir.name) / "data"
        data_dir.mkdir()
        (data_dir / "video_features_basic_pure.csv").write_text(
            "video_id,author_id\nv1,a1\nv2,a2\n",
            encoding="utf-8",
        )
        header = "date,user_id,video_id,tab,duration_ms,long_view\n"
        (data_dir / "log_standard_4_08_to_4_21_pure.csv").write_text(
            header
            + "20220408,u1,v1,1,1000,1\n"
            + "20220408,u1,v2,1,1000,0\n"
            + "20220408,u2,v1,1,1000,0\n"
            + "20220408,u2,v2,1,1000,1\n",
            encoding="utf-8",
        )
        (data_dir / "log_standard_4_22_to_5_08_pure.csv").write_text(
            header
            + "20220422,u1,v1,1,1000,1\n"
            + "20220422,u1,v2,1,1000,0\n"
            + "20220422,u2,v1,1,1000,0\n"
            + "20220422,u2,v2,1,1000,1\n",
            encoding="utf-8",
        )
        runner = ExperimentRunner(
            self.logger,
            contract=BenchmarkContract(data_dir=data_dir),
        )

        result = runner.run(
            experiment_id="exp_worker",
            hypothesis="Production execution must cross a process boundary.",
            config={
                "loss": "pointwise",
                "learning_rate": 0.01,
                "l2": 0.0,
                "epochs": 1,
                "patience": 1,
                "batch_size": 2,
                "seed": 0,
            },
            candidate=run_torch_fm_candidate,
            timeout_seconds=20.0,
        )

        self.assertEqual(result.status, "completed", result.error)
        self.assertNotEqual(result.output.metadata["worker_pid"], __import__("os").getpid())
        self.assertTrue((result.run_dir / "worker_job.json").exists())
        self.assertTrue((result.run_dir / "worker_result.json").exists())
        job = __import__("json").loads(
            (result.run_dir / "worker_job.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("data_dir", job)
        self.assertIn("prepared_data_path", job)
        started_events = [
            event for event in self.store.read_events() if event["action"] == "training_started"
        ]
        self.assertEqual(started_events[-1]["details"]["execution_mode"], "subprocess")
