import unittest

from research_agent.fidelity import FidelityManager
from research_agent.planner import EvidencePlanner
from research_agent.search import SearchController
from research_agent.state import ResearchState


class FidelityTests(unittest.TestCase):
    def test_promotion_requires_validation_evidence_and_uses_full_budget(self):
        direction = EvidencePlanner(seed=0).propose([], ResearchState())
        proposal = SearchController(seed=0).propose_trial(direction, ResearchState(), [])
        manager = FidelityManager()

        self.assertTrue(manager.should_promote({"primary": 0.6}, 0.6015))
        self.assertTrue(manager.should_promote({"primary": 0.1}, 0.6015))
        self.assertFalse(manager.should_promote({"primary": float("nan")}, 0.6015))
        promoted = manager.promote(proposal, direction, experiment_id="exp_002")
        self.assertEqual(promoted.config["fidelity"], "full")
        self.assertEqual(promoted.config["epochs"], direction.evaluation_budget["full_epochs"])
        self.assertEqual(promoted.config["epochs"], 40)
        self.assertEqual(promoted.config["patience"], 4)
        self.assertEqual(promoted.changed_factors, proposal.changed_factors)
