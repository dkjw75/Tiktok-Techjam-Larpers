import unittest

from research_agent.fidelity import FidelityManager
from research_agent.safety import ExperimentProposal


def low(experiment_id, primary):
    return {
        "experiment_id": experiment_id,
        "config": {"fidelity": "low"},
        "metrics": {"primary": primary},
        "direction_id": "model",
        "search_strategy": "llm_isolated_candidate",
    }


def full(experiment_id, parent_experiment_id, primary):
    return {
        "experiment_id": experiment_id,
        "parent_experiment_id": parent_experiment_id,
        "config": {"fidelity": "full"},
        "metrics": {"primary": primary},
    }


class FidelityManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = FidelityManager()
        self.proposal = ExperimentProposal(
            experiment_id="low_4",
            hypothesis="Test a single change.",
            rationale="Validation-only evidence.",
            config={"fidelity": "low"},
            changed_factors=("one",),
            research_direction_id="model",
            search_strategy="llm_isolated_candidate",
        )

    def test_clearly_weak_screen_is_not_promoted_while_calibrating(self):
        history = [low("low_1", 0.600), low("low_2", 0.599), low("low_3", 0.598)]
        self.assertFalse(
            self.manager.should_promote(
                {"primary": 0.590}, 0.6015, proposal=self.proposal, history=history, global_pool=True
            )
        )

    def test_unreliable_screens_stop_promoting_full_runs(self):
        history = [
            low("low_1", 0.600), full("full_1", "low_1", 0.604),
            low("low_2", 0.601), full("full_2", "low_2", 0.602),
            low("low_3", 0.602), full("full_3", "low_3", 0.600),
            low("low_4", 0.603),
        ]
        status = self.manager.promotion_status(history)
        self.assertEqual(status["policy"], "screening_not_predictive")
        self.assertFalse(
            self.manager.should_promote(
                {"primary": 0.603}, 0.6015, proposal=self.proposal, history=history, global_pool=True
            )
        )
