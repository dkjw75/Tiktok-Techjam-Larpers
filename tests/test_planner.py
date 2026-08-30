import unittest

from research_agent.planner import EvidencePlanner
from research_agent.state import ResearchState


class PlannerTests(unittest.TestCase):
    def test_bootstrap_direction_contains_no_exact_trial_configuration(self):
        direction = EvidencePlanner(seed=7).propose([], ResearchState())

        self.assertIn(direction.direction_id, {"rank_ensemble", "listwise_fm_ranking"})
        self.assertTrue(direction.search_space)
        self.assertNotIn("seed", direction.search_space)
        self.assertNotIn("learning_rate", direction.search_space)
        self.assertIn("bootstrap", direction.rationale)

    def test_history_changes_planner_rationale_and_avoids_attempted_direction_when_possible(self):
        history = [
            {"direction_id": "pairwise_fm_ranking", "decision": "rejected"},
            {"direction_id": "rank_ensemble", "decision": "rejected"},
        ]
        direction = EvidencePlanner(seed=0).propose(history, ResearchState(completed_iterations=1))

        self.assertEqual(direction.direction_id, "listwise_fm_ranking")
        self.assertIn("Recent decisions", direction.rationale)
        self.assertEqual(direction.strategy, "exploration")
