import unittest

from research_agent.contracts import BENCHMARK_CONTRACT


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
            ("evaluate.py", "baseline.py"),
        )

    def test_stopping_parameters_match_project_contract(self):
        self.assertEqual(BENCHMARK_CONTRACT.improvement_threshold, 0.002)
        self.assertEqual(BENCHMARK_CONTRACT.acceptance_threshold, 0.001)
        self.assertEqual(BENCHMARK_CONTRACT.non_improvement_limit, 3)
        self.assertEqual(BENCHMARK_CONTRACT.target_primary, 0.65)
        self.assertEqual(BENCHMARK_CONTRACT.max_experiments, 20)


if __name__ == "__main__":
    unittest.main()
