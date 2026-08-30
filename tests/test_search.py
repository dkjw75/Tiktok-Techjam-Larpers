import unittest

from research_agent.planner import EvidencePlanner
from research_agent.search import SearchController
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

    def test_listwise_direction_changes_only_loss_conceptually(self):
        history = [
            {"direction_id": "pairwise_fm_ranking", "decision": "rejected"},
            {"direction_id": "rank_ensemble", "decision": "rejected"},
        ]
        direction = self.planner.propose(history, ResearchState(completed_iterations=1))
        proposal = self.search.propose_trial(direction, ResearchState(completed_iterations=1), history)

        self.assertEqual(direction.direction_id, "listwise_fm_ranking")
        self.assertEqual(proposal.changed_factors, ("loss",))
        self.assertEqual(proposal.config["loss"], "listwise")

    def test_listwise_direction_searches_objective_factors_after_bootstrap(self):
        history = [
            {
                "experiment_id": "exp_001",
                "direction_id": "listwise_fm_ranking",
                "decision": "screened",
                "changed_factors": ["loss"],
                "metrics": {"primary": 0.5},
                "config": {**self.search.BASELINE_CONFIG, "loss": "listwise"},
            }
        ]
        direction = self.planner.propose(
            [
                {"direction_id": "pairwise_fm_ranking"},
                {"direction_id": "rank_ensemble"},
            ],
            ResearchState(completed_iterations=1),
        )
        self.assertEqual(direction.direction_id, "listwise_fm_ranking")

        proposal = self.search.propose_trial(
            direction,
            ResearchState(
                current_best_experiment_id="exp_001",
                completed_iterations=1,
            ),
            history,
        )

        self.assertEqual(proposal.changed_factors, ("objective_variant",))
        self.assertEqual(proposal.config["loss"], "listwise")

    def test_screen_proposals_require_fresh_evidence_between_candidates(self):
        history = [
            {
                "experiment_id": "exp_001",
                "direction_id": "pairwise_fm_ranking",
                "changed_factors": ["loss"],
                "config": {**self.search.BASELINE_CONFIG, "loss": "pairwise"},
            }
        ]
        state = ResearchState(completed_iterations=1)
        direction = self.planner.propose(history, state)
        with self.assertRaisesRegex(ValueError, "sequentially"):
            self.search.propose_batch(direction, state, history, count=3)

        proposals = self.search.propose_batch(direction, state, history, count=1)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].config["fidelity"], "low")

    def test_direction_change_emits_bounded_coupled_parameter_envelope(self):
        direction = self.planner.propose([], ResearchState())
        self.assertEqual(direction.direction_id, "listwise_fm_ranking")

        proposals = self.search.propose_batch(direction, ResearchState(), [], count=1)

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].changed_factors, ("loss",))
        self.assertEqual(proposals[0].config["loss"], "listwise")
        self.assertEqual(proposals[0].parent_experiment_id, "baseline")

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

    def test_durable_run_ids_are_not_reallocated_after_orchestrator_crash(self):
        direction = self.planner.propose([], ResearchState())
        proposal = self.search.propose_trial(
            direction,
            ResearchState(),
            [],
            reserved_ids=("exp_001",),
        )

        self.assertEqual(proposal.experiment_id, "exp_002")
