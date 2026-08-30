import tempfile
import unittest

from research_agent.controller import ExperimentController
from research_agent.critic import CriticResult, ProposalCritic
from research_agent.fidelity import FidelityManager
from research_agent.logger import ResearchLogger
from research_agent.loop import AutonomousResearchLoop, _spearman_rank_correlation
from research_agent.planner import EvidencePlanner
from research_agent.review import EvidenceReviewer
from research_agent.runner import CandidateOutput, ExperimentRunner
from research_agent.safety import ExperimentProposal, SafetyValidator
from research_agent.search import SearchController
from research_agent.state import ResearchState
from research_agent.store import ArtifactStore


def loader(_data_dir):
    return {
        "train": [("train",)],
        "valid": [
            (20220421, "u", "v1", "a1", "1", 1000.0, 1),
            (20220421, "u", "v2", "a2", "1", 1000.0, 0),
        ],
        "test": [("test",)],
    }


def candidate(_data, _config, _run_dir):
    return CandidateOutput(
        ["u", "u"],
        [1, 0],
        [0.9, 0.1],
        metadata={"epochs_run": 8, "configured_epochs": 40, "effective_patience": 4, "stopped_by": "early_stopping"},
    )


class AutonomousLoopTests(unittest.TestCase):
    def test_periodic_cycle_calibrates_all_screen_ranks_at_full_fidelity(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=900)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(
                    current_best_primary=0.5,
                    completed_cycles=4,
                ),
            )
            loop = AutonomousResearchLoop(
                controller=controller,
                logger=logger,
                planner=EvidencePlanner(seed=0),
                search=SearchController(seed=0),
                critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(),
                fidelity=FidelityManager(),
                candidate=candidate,
            )

            result = loop.run(max_cycles=1)[0]

            self.assertEqual(len(result.screen_iterations), 3)
            self.assertEqual(len(result.calibration_iterations), 2)
            self.assertTrue(
                all(item.decision == "calibrated" for item in result.calibration_iterations)
            )
            evidence = store.read_root_json("screen_full_calibrations.json")["5"]
            self.assertEqual(len(evidence["pairs"]), 3)
            self.assertEqual(controller.state.completed_cycles, 5)

    def test_spearman_calibration_detects_preserved_and_reversed_rank(self):
        self.assertEqual(
            _spearman_rank_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]),
            1.0,
        )
        self.assertEqual(
            _spearman_rank_correlation([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]),
            -1.0,
        )

    def test_resume_resolves_pending_confirmation_before_new_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=900)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.5),
                max_iterations=1,
                require_seed_confirmation=True,
            )
            pending = controller.run_iteration(
                ExperimentProposal(
                    experiment_id="exp_pending",
                    hypothesis="A candidate requires confirmation.",
                    rationale="Exercise crash-resume ordering.",
                    config={
                        "candidate": "pending",
                        "fidelity": "full",
                        "epochs": 40,
                        "patience": 4,
                    },
                    changed_factors=("candidate",),
                    runtime_budget_seconds=60,
                ),
                candidate,
            )
            self.assertEqual(pending.decision, "pending_confirmation")
            self.assertTrue(controller.state.stopped)
            checkpoint = store.runs_dir / "exp_pending-promotion-candidate-seed-0" / "checkpoint.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            certificate = {
                "selected_experiment_id": "exp_pending",
                "comparator_experiment_id": "baseline",
                "seeds": [0, 1, 2],
                "candidate_scores": [0.9, 0.9, 0.9],
                "comparator_scores": [0.5, 0.5, 0.5],
                "mean_delta": 0.4,
                "wins": 3,
                "confirmed": True,
                "submission_checkpoint_path": str(checkpoint),
                "comparison_mode": "matched_seed_organizer_baseline",
                "candidate_comparison_groups": ["c" * 64] * 3,
                "comparator_comparison_groups": ["c" * 64] * 3,
                "confirmation_attempts": 6,
            }

            def confirmer(_record):
                store.write_root_json(
                    "promotion_confirmations.json",
                    {"exp_pending": certificate},
                )
                return certificate

            loop = AutonomousResearchLoop(
                controller=controller,
                logger=logger,
                planner=EvidencePlanner(seed=0),
                search=SearchController(seed=0),
                critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(),
                fidelity=FidelityManager(),
                candidate=candidate,
                promotion_confirmer=confirmer,
            )

            self.assertEqual(loop.run(max_cycles=1), [])
            self.assertEqual(controller.state.current_best_experiment_id, "exp_pending")
            self.assertEqual(controller.state.valid_comparisons, 1)
            actions = [event["action"] for event in store.read_events()]
            self.assertIn("pending_seed_confirmation_resumed", actions)

    def test_best_screen_is_promoted_without_comparison_to_full_champion(self):
        def late_winner(_data, config, _run_dir):
            is_full = config.get("fidelity") == "full"
            return CandidateOutput(
                ["u", "u"],
                [1, 0],
                [0.9, 0.1] if is_full else [0.1, 0.9],
                metadata={
                    "epochs_run": 8,
                    "configured_epochs": 40,
                    "effective_patience": 4,
                    "stopped_by": "early_stopping",
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=900)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.6),
            )
            loop = AutonomousResearchLoop(
                controller=controller,
                logger=logger,
                planner=EvidencePlanner(seed=0),
                search=SearchController(seed=0),
                critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(),
                fidelity=FidelityManager(),
                candidate=late_winner,
            )

            result = loop.run(max_cycles=1)[0]

            self.assertLess(result.iteration.metrics["primary"], 0.6)
            self.assertIsNotNone(result.promoted_iteration)
            self.assertEqual(result.promoted_iteration.decision, "accepted")

    def test_loop_logs_direction_critic_and_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=900)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.5),
            )
            loop = AutonomousResearchLoop(
                controller=controller,
                logger=logger,
                planner=EvidencePlanner(seed=0),
                search=SearchController(seed=0),
                critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(),
                fidelity=FidelityManager(),
                candidate=candidate,
            )

            results = loop.run(max_cycles=1)

            self.assertEqual(len(results), 1)
            actions = [event["action"] for event in store.read_events()]
            self.assertIn("research_direction_proposed", actions)
            self.assertIn("proposal_critic_reviewed", actions)
            self.assertEqual(store.read_iterations()[0]["direction_id"], results[0].direction.direction_id)
            self.assertEqual(results[0].iteration.decision, "screened")
            self.assertEqual(results[0].promoted_iteration.decision, "accepted")

    def test_critic_rejection_runs_zero_experiments(self):
        class RejectingCritic:
            def review(self, _proposal, _history):
                return CriticResult(False, ("deliberate rejection",))

        candidate_calls = []

        def counting_candidate(*args):
            candidate_calls.append(args)
            return candidate(*args)

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=900)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.5),
            )
            loop = AutonomousResearchLoop(
                controller=controller,
                logger=logger,
                planner=EvidencePlanner(seed=0),
                search=SearchController(seed=0),
                critic=RejectingCritic(),
                reviewer=EvidenceReviewer(),
                fidelity=FidelityManager(),
                candidate=counting_candidate,
            )

            results = loop.run(max_cycles=1)

            self.assertEqual(candidate_calls, [])
            self.assertEqual(store.read_iterations(), [])
            self.assertEqual(results[0].iteration.decision, "critic_rejected")
            self.assertEqual(controller.state.completed_iterations, 0)

    def test_cycle_does_not_start_without_screen_and_promotion_capacity(self):
        calls = []

        def counting_candidate(*args):
            calls.append(args)
            return candidate(*args)

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=900)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.5),
                max_iterations=3,
            )
            loop = AutonomousResearchLoop(
                controller=controller,
                logger=logger,
                planner=EvidencePlanner(seed=0),
                search=SearchController(seed=0),
                critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(),
                fidelity=FidelityManager(),
                candidate=counting_candidate,
            )

            self.assertEqual(loop.run(max_cycles=1), [])
            self.assertEqual(calls, [])
            self.assertEqual(controller.state.stop_reason_code, "iteration_budget")
