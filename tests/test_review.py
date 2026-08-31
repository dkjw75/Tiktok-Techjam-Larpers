from __future__ import annotations

import unittest

from research_agent.review import EvidenceReviewer
from research_agent.state import ResearchState


class EvidenceReviewerTests(unittest.TestCase):
    def test_separates_metric_seed_failure_and_complementarity_evidence(self) -> None:
        history = [
            {
                "experiment_id": "exp_001",
                "decision": "pending_confirmation",
                "metrics": {"GAUC": 0.67, "nDCG@5": 0.538, "primary": 0.604},
                "runner_metadata": {
                    "blend_weight_fit_half_primary": 0.603,
                    "blend_weight_held_half_primary": 0.605,
                },
            }
        ]
        resolutions = {
            "exp_001": {
                "decision": "accepted",
                "certificate": {
                    "candidate_scores": [0.604, 0.603, 0.604],
                    "wins": 2,
                    "confirmed": True,
                },
            }
        }
        review = EvidenceReviewer().review(history, ResearchState(), resolutions)
        self.assertEqual(review.action, "refine")
        self.assertEqual(review.allocation_status, "EXPLOIT")
        self.assertEqual(review.metric_trends["GAUC"], [0.67])
        self.assertGreater(review.seed_evidence["std"], 0.0)
        self.assertEqual(review.complementarity_evidence["held_half_primary"], 0.605)
        self.assertFalse(review.complementarity_evidence["measured"])

    def test_three_negative_results_abandon_region_without_resetting_stop_state(self) -> None:
        history = [
            {
                "experiment_id": f"exp_{index:03d}",
                "decision": "rejected",
                "metrics": {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55},
                "runner_metadata": {},
            }
            for index in range(1, 4)
        ]
        state = ResearchState(consecutive_non_improvements=3)
        review = EvidenceReviewer().review(history, state)
        self.assertEqual(review.allocation_status, "ABANDON")
        self.assertEqual(state.consecutive_non_improvements, 3)


if __name__ == "__main__":
    unittest.main()
