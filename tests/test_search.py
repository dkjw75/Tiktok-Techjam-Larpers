import unittest

from research_agent.planner import EvidencePlanner
from research_agent.research_tools import ResearchToolCatalog
from research_agent.search import SearchController, SearchSpaceExhausted
from research_agent.state import ResearchState


class SearchControllerTests(unittest.TestCase):
    def setUp(self):
        self.planner = EvidencePlanner(seed=0)
        self.search = SearchController(seed=0)

    def test_search_controller_selects_one_factor_not_a_fixed_template(self):
        direction = self.planner.propose([], ResearchState())
        proposal = self.search.propose_trial(direction, ResearchState(), [])

        self.assertEqual(len(proposal.changed_factors), 1)
        self.assertEqual(proposal.research_direction_id, direction.direction_id)
        self.assertEqual(proposal.config["fidelity"], "low")
        self.assertNotEqual(proposal.config[proposal.changed_factors[0]], self.search.BASELINE_CONFIG[proposal.changed_factors[0]])

    def test_pairwise_direction_changes_only_loss_conceptually(self):
        history = [{"direction_id": "pointwise_fm_optimization", "decision": "rejected"}]
        direction = self.planner.propose(history, ResearchState(completed_iterations=1))
        proposal = self.search.propose_trial(direction, ResearchState(completed_iterations=1), history)

        self.assertEqual(direction.direction_id, "pairwise_fm_ranking")
        self.assertEqual(proposal.changed_factors, ("loss",))
        self.assertEqual(proposal.config["loss"], "pairwise")

    def test_lab_pairwise_tool_sets_its_loss_without_a_legacy_direction_id(self):
        direction = ResearchToolCatalog().direction(
            "fm_pairwise_search", hypothesis="Pairwise ranking may improve validation ranking.", rationale="Test the implemented objective.",
        )
        proposal = self.search.propose_trial(direction, ResearchState(), [])
        self.assertEqual(proposal.changed_factors, ("loss",))
        self.assertEqual(proposal.config["loss"], "pairwise")
        self.assertEqual(proposal.search_strategy, "direction_bootstrap")

    def test_controller_never_falls_back_to_an_already_run_low_fidelity_configuration(self):
        direction = ResearchToolCatalog().direction("fm_pairwise_search", hypothesis="Pairwise ranking may improve validation ranking.", rationale="Test it.")
        history = []
        seen = set()
        for number in range(9):
            proposal = self.search.propose_trial(direction, ResearchState(completed_iterations=number), history)
            key = tuple(proposal.config[name] for name in ("loss", "learning_rate", "l2"))
            self.assertNotIn(key, seen)
            seen.add(key)
            history.append({"experiment_id": proposal.experiment_id, "direction_id": direction.direction_id, "config": proposal.config, "metrics": {"primary": 0.59}})
        with self.assertRaises(SearchSpaceExhausted):
            self.search.propose_trial(direction, ResearchState(completed_iterations=9), history)

    def test_joint_refinement_combines_the_best_distinct_training_signals(self):
        direction = ResearchToolCatalog().direction("fm_training_search", hypothesis="Training settings may help.", rationale="Refine evidence.")
        base = dict(self.search.BASELINE_CONFIG)
        history = [
            {"direction_id": direction.direction_id, "config": {**base, "learning_rate": 0.002, "fidelity": "low"}, "metrics": {"primary": 0.600}},
            {"direction_id": direction.direction_id, "config": {**base, "l2": 1e-5, "fidelity": "low"}, "metrics": {"primary": 0.601}},
            {"direction_id": direction.direction_id, "config": {**base, "learning_rate": 0.0005, "fidelity": "low"}, "metrics": {"primary": 0.599}},
        ]
        joint = self.search._joint_refinement(direction, base, history, ResearchState(current_best_primary=0.6015))
        self.assertEqual(joint["learning_rate"], 0.002)
        self.assertEqual(joint["l2"], 1e-5)

    def test_new_trial_ids_and_values_avoid_history_when_possible(self):
        direction = self.planner.propose([], ResearchState())
        history = [
            {
                "experiment_id": "exp_001",
                "direction_id": direction.direction_id,
                "changed_factors": ["learning_rate"],
                "config": {"learning_rate": 0.0005},
            }
        ]
        proposal = self.search.propose_trial(direction, ResearchState(completed_iterations=1), history)

        self.assertEqual(proposal.experiment_id, "exp_002")

    def test_bayesian_optimization_activates_after_comparable_numeric_trials(self):
        direction = self.planner.propose([], ResearchState())
        history = [
            {"direction_id": direction.direction_id, "config": {"learning_rate": value}, "metrics": {"primary": score}}
            for value, score in ((0.0005, 0.60), (0.001, 0.61), (0.002, 0.59), (0.001, 0.612), (0.0005, 0.601), (0.001, 0.613))
        ]
        _, algorithm = self.search._choose_value(direction, "learning_rate", history, ResearchState(completed_iterations=6))
        self.assertEqual(algorithm, "bayesian_optimization")

    def test_tpe_activates_after_comparable_categorical_trials(self):
        direction = type("Direction", (), {"direction_id": "categorical", "search_space": {"optimizer": ("adam", "adagrad", "sgd")}})()
        history = [
            {"direction_id": "categorical", "config": {"optimizer": value}, "metrics": {"primary": score}}
            for value, score in (("adam", .61), ("sgd", .59), ("adagrad", .60), ("adam", .612), ("sgd", .58), ("adam", .613))
        ]
        self.search.BASELINE_CONFIG = {**self.search.BASELINE_CONFIG, "optimizer": "adam"}
        _, algorithm = self.search._choose_value(direction, "optimizer", history, ResearchState(completed_iterations=6))
        self.assertEqual(algorithm, "tpe")
