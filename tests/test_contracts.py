import unittest

from research_agent.contracts import BENCHMARK_CONTRACT, BenchmarkContract, ComparisonValidity


def valid_metrics(primary=0.61):
    return {"primary": primary, "evaluator_sha256": BENCHMARK_CONTRACT.evaluator_sha256}


def valid_runner_metadata(**overrides):
    value = {
        "epochs_run": 7,
        "configured_epochs": 40,
        "effective_patience": 4,
        "stopped_by": "early_stopping",
        "evaluator_sha256": BENCHMARK_CONTRACT.evaluator_sha256,
        "data_sha256": "1" * 64,
        "preprocessing_sha256": "2" * 64,
        "staging_code_sha256": "6" * 64,
        "feature_schema_sha256": "3" * 64,
        "model_code_sha256": "4" * 64,
        "comparison_group_id": "5" * 64,
        "seed": 0,
    }
    value.update(overrides)
    return value


class BenchmarkContractTests(unittest.TestCase):
    def test_candidate_selection_is_validation_only(self):
        self.assertEqual(BENCHMARK_CONTRACT.selection_split, "valid")
        self.assertNotEqual(BENCHMARK_CONTRACT.selection_split, "test")

    def test_required_metrics_are_fixed(self):
        self.assertEqual(
            BENCHMARK_CONTRACT.metric_names,
            ("GAUC", "nDCG@5", "primary"),
        )

    def test_baseline_and_evaluator_are_protected(self):
        self.assertEqual(
            BENCHMARK_CONTRACT.protected_modules,
            ("evaluate.py", "baseline.py", "data.py", "submit.py"),
        )

    def test_stopping_parameters_match_project_contract(self):
        self.assertEqual(BENCHMARK_CONTRACT.improvement_threshold, 0.002)
        self.assertEqual(BENCHMARK_CONTRACT.non_improvement_limit, 3)
        self.assertEqual(BENCHMARK_CONTRACT.max_iterations, 50)
        self.assertEqual(BENCHMARK_CONTRACT.max_wall_clock_seconds, 21600)
        self.assertEqual(BENCHMARK_CONTRACT.finalization_reserve_seconds, 1800)

    def test_only_full_fidelity_validation_evidence_is_comparable(self):
        contract = BenchmarkContract()
        invalid = ComparisonValidity.assess(
            candidate_experiment_id="screen_001",
            config={"fidelity": "low"},
            selection_split="valid",
            metrics={"primary": 0.9},
            runner_metadata={},
            incumbent_primary=0.6,
            incumbent_experiment_id="baseline",
            incumbent_metadata={},
            incumbent_config={},
            contract=contract,
        )
        valid = ComparisonValidity.assess(
            candidate_experiment_id="full_001",
            config={"fidelity": "full", "epochs": 40, "patience": 4, "seed": 0},
            selection_split="valid",
            metrics=valid_metrics(),
            runner_metadata=valid_runner_metadata(),
            incumbent_primary=0.6,
            incumbent_experiment_id="baseline",
            incumbent_metadata=valid_runner_metadata(),
            incumbent_config={"epochs": 40, "patience": 4, "seed": 0},
            contract=contract,
        )

        self.assertFalse(invalid.valid)
        self.assertIn("not full fidelity", invalid.reasons[0])
        self.assertTrue(valid.valid)
        self.assertAlmostEqual(valid.delta_primary, 0.01)

    def test_full_label_without_organizer_budget_is_invalid(self):
        invalid = ComparisonValidity.assess(
            candidate_experiment_id="fake_full",
            config={"fidelity": "full", "epochs": 12},
            selection_split="valid",
            metrics={"primary": 0.7},
            runner_metadata={},
            incumbent_primary=0.6,
            incumbent_experiment_id="baseline",
            incumbent_metadata={},
            incumbent_config={},
            contract=BenchmarkContract(),
        )

        self.assertFalse(invalid.valid)
        self.assertIn("full fidelity requires max epochs=40", invalid.reasons)
        self.assertIn("full fidelity requires patience=4", invalid.reasons)
        self.assertIn("full fidelity requires valid epochs_run termination evidence", invalid.reasons)

    def test_unrecognized_full_termination_is_invalid(self):
        invalid = ComparisonValidity.assess(
            candidate_experiment_id="bad_termination",
            config={"fidelity": "full", "epochs": 40, "patience": 4, "seed": 0},
            selection_split="valid",
            metrics=valid_metrics(0.7),
            runner_metadata=valid_runner_metadata(epochs_run=40, stopped_by="timeout"),
            incumbent_primary=0.6,
            incumbent_experiment_id="baseline",
            incumbent_metadata=valid_runner_metadata(epochs_run=40, stopped_by="timeout"),
            incumbent_config={"epochs": 40, "patience": 4, "seed": 0},
            contract=BenchmarkContract(),
        )

        self.assertFalse(invalid.valid)
        self.assertTrue(any("recognized stopped_by" in reason for reason in invalid.reasons))

    def test_max_epochs_termination_is_truncated_and_invalid(self):
        invalid = ComparisonValidity.assess(
            candidate_experiment_id="short_max",
            config={"fidelity": "full", "epochs": 40, "patience": 4, "seed": 0},
            selection_split="valid",
            metrics=valid_metrics(0.7),
            runner_metadata=valid_runner_metadata(
                epochs_run=40,
                stopped_by="max_epochs_truncated",
            ),
            incumbent_primary=0.6,
            incumbent_experiment_id="baseline",
            incumbent_metadata=valid_runner_metadata(
                epochs_run=40,
                stopped_by="max_epochs_truncated",
            ),
            incumbent_config={"epochs": 40, "patience": 4, "seed": 0},
            contract=BenchmarkContract(),
        )

        self.assertFalse(invalid.valid)
        self.assertIn("max-epoch termination is truncated and not comparable", invalid.reasons)

    def test_candidate_and_incumbent_must_share_comparison_lineage(self):
        invalid = ComparisonValidity.assess(
            candidate_experiment_id="lineage_mismatch",
            config={"fidelity": "full", "epochs": 40, "patience": 4, "seed": 0},
            selection_split="valid",
            metrics=valid_metrics(0.7),
            runner_metadata=valid_runner_metadata(),
            incumbent_primary=0.6,
            incumbent_experiment_id="champion",
            incumbent_metadata=valid_runner_metadata(data_sha256="f" * 64),
            incumbent_config={"epochs": 40, "patience": 4, "seed": 0},
            contract=BenchmarkContract(),
        )

        self.assertFalse(invalid.valid)
        self.assertIn("candidate and incumbent data_sha256 lineage differ", invalid.reasons)


if __name__ == "__main__":
    unittest.main()
