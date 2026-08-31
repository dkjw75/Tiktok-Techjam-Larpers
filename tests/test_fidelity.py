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
        self.assertFalse(manager.should_promote({"primary": 0.5}, 0.6015))
        promoted = manager.promote(proposal, direction, experiment_id="exp_002")
        self.assertEqual(promoted.config["fidelity"], "full")
        self.assertEqual(promoted.config["epochs"], direction.evaluation_budget["full_epochs"])
        self.assertEqual(promoted.changed_factors, proposal.changed_factors)

    def test_asha_promotes_only_the_top_low_fidelity_survivor(self):
        direction = EvidencePlanner(seed=0).propose([], ResearchState())
        proposal = SearchController(seed=0).propose_trial(direction, ResearchState(), [])
        records = [
            {"experiment_id": f"exp_{number:03d}", "direction_id": proposal.research_direction_id, "search_region_id": f"restart_{number}", "config": {"fidelity": "low"}, "metrics": {"primary": score}}
            for number, score in ((1, 0.60), (2, 0.61), (3, 0.62))
        ]
        proposal = proposal.__class__(**{**proposal.__dict__, "experiment_id": "exp_003"})
        manager = FidelityManager()
        self.assertTrue(manager.should_promote({"primary": 0.62}, 0.62, proposal=proposal, history=records))
        self.assertFalse(manager.should_promote({"primary": 0.60}, 0.62, proposal=proposal, history=records))

    def test_global_pool_promotes_across_distinct_research_areas(self):
        direction = EvidencePlanner(seed=0).propose([], ResearchState())
        proposal = SearchController(seed=0).propose_trial(direction, ResearchState(), [])
        proposal = proposal.__class__(**{**proposal.__dict__, "experiment_id": "exp_003"})
        records = [
            {"experiment_id": "exp_001", "direction_id": "training", "search_strategy": "llm_self_extending_capability", "config": {"fidelity": "low"}, "metrics": {"primary": 0.60}},
            {"experiment_id": "exp_002", "direction_id": "sampling", "search_strategy": "llm_self_extending_capability", "config": {"fidelity": "low"}, "metrics": {"primary": 0.61}},
            {"experiment_id": "exp_003", "direction_id": "feature", "search_strategy": "llm_self_extending_capability", "config": {"fidelity": "low"}, "metrics": {"primary": 0.62}},
        ]

        self.assertTrue(FidelityManager().should_promote(
            {"primary": 0.62}, 0.70, proposal=proposal, history=records, global_pool=True
        ))
