import tempfile
import unittest
from pathlib import Path

from research_agent.runner import CandidateOutput, RunnerResult
from research_agent.seed_validation import confirm_promotion_candidate, confirm_selected_candidate
from research_agent.state import ResearchState
from research_agent.store import ArtifactStore


class FakeRunner:
    def __init__(self, root: Path, *, winning: bool = True):
        self.root = root
        self.winning = winning
        self.configs = []

    def run(self, *, experiment_id, hypothesis, config, candidate, timeout_seconds):
        self.configs.append(dict(config))
        checkpoint = self.root / f"{experiment_id}.pt"
        checkpoint.write_bytes(b"checkpoint")
        scores = [0.9, 0.1, 0.1, 0.9] if self.winning else [0.1, 0.9, 0.9, 0.1]
        if "comparator" in experiment_id:
            scores = [0.1, 0.9, 0.9, 0.1]
        return RunnerResult(
            experiment_id=experiment_id,
            status="completed",
            run_dir=self.root,
            runtime_seconds=0.1,
            output=CandidateOutput(
                ["u1", "u1", "u2", "u2"],
                [1, 0, 0, 1],
                scores,
                {
                    "checkpoint_path": str(checkpoint),
                    "stopped_by": "early_stopping",
                    "configured_epochs": 40,
                    "effective_patience": 4,
                    "comparison_group_id": "a" * 64,
                },
            ),
        )


class SeedValidationTests(unittest.TestCase):
    def _store_with_selected_candidate(self, directory: str) -> ArtifactStore:
        store = ArtifactStore(directory)
        store.write_root_json(
            "state.json",
            ResearchState(
                current_best_experiment_id="exp_001",
                current_best_primary=0.7,
                stop_reason_code="plateau",
                stop_reason="three valid non-improvements",
            ).as_dict(),
        )
        store.append_iteration(
            {
                "experiment_id": "exp_001",
                "parent_experiment_id": "baseline",
                "comparison_incumbent_id": "baseline",
                "hypothesis": "candidate",
                "rationale": "selected on validation",
                "config": {
                    "loss": "pairwise",
                    "learning_rate": 0.001,
                    "l2": 1e-6,
                    "fidelity": "full",
                    "epochs": 40,
                    "patience": 4,
                    "seed": 0,
                },
                "decision": "accepted",
                "comparison_validity": {"valid": True},
            }
        )
        return store

    def test_three_seed_confirmation_uses_seed_zero_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_selected_candidate(directory)
            runner = FakeRunner(Path(directory), winning=True)

            result = confirm_selected_candidate(store, runner)

            self.assertTrue(result.confirmed)
            self.assertEqual(result.seeds, (0, 1, 2))
            self.assertEqual(result.wins, 3)
            self.assertTrue(result.submission_checkpoint_path.endswith("seed-0.pt"))
            self.assertEqual([config["seed"] for config in runner.configs], [0, 1, 2])
            self.assertEqual(store.read_root_json("seed_confirmation.json")["submission_seed"], 0)

    def test_noisy_candidate_is_not_confirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_selected_candidate(directory)
            runner = FakeRunner(Path(directory), winning=False)

            result = confirm_selected_candidate(store, runner)

            self.assertFalse(result.confirmed)
            self.assertIsNone(result.submission_checkpoint_path)

    def test_promotion_uses_matched_organizer_baseline_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_selected_candidate(directory)
            runner = FakeRunner(Path(directory), winning=True)
            record = store.read_iterations()[-1]

            result = confirm_promotion_candidate(
                store,
                runner,
                record,
                candidate=lambda *_args: None,
            )

            self.assertTrue(result.confirmed)
            self.assertEqual(result.comparison_mode, "matched_seed_organizer_baseline")
            self.assertEqual(result.wins, 3)
            self.assertEqual(len(runner.configs), 6)
            cached = confirm_promotion_candidate(
                store,
                runner,
                record,
                candidate=lambda *_args: None,
            )
            self.assertEqual(cached.submission_bundle, result.submission_bundle)
            self.assertEqual(len(runner.configs), 6, "cached evidence must not retrain")


if __name__ == "__main__":
    unittest.main()
