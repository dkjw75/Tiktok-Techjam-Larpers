import tempfile
import unittest

from research_agent.controller import ExperimentController
from research_agent.contracts import BenchmarkContract
from research_agent.logger import ResearchLogger
from research_agent.runner import CandidateOutput, ExperimentRunner
from research_agent.safety import ExperimentProposal, SafetyValidator
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


def strong_candidate(_data, _config, _run_dir):
    return CandidateOutput(
        ["u", "u"],
        [1, 0],
        [0.9, 0.1],
        metadata={"epochs_run": 8, "configured_epochs": 40, "effective_patience": 4, "stopped_by": "early_stopping"},
    )


def broken_candidate(_data, _config, _run_dir):
    raise RuntimeError("intentional runner failure")


def weak_candidate(_data, _config, _run_dir):
    return CandidateOutput(
        ["u", "u"],
        [1, 0],
        [0.1, 0.9],
        metadata={"epochs_run": 8, "configured_epochs": 40, "effective_patience": 4, "stopped_by": "early_stopping"},
    )


def proposal(experiment_id, **changes):
    values = {
        "experiment_id": experiment_id,
        "hypothesis": "A single controlled change may improve ranking.",
        "rationale": "Use validation evidence only.",
        "config": {"candidate": experiment_id, "fidelity": "full", "epochs": 40, "patience": 4},
        "changed_factors": ("candidate",),
        "runtime_budget_seconds": 60,
    }
    values.update(changes)
    return ExperimentProposal(**values)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ArtifactStore(self.tempdir.name)
        self.logger = ResearchLogger(self.store)
        self.runner = ExperimentRunner(self.logger, data_loader=loader)
        self.validator = SafetyValidator(max_runtime_seconds=60)

    def tearDown(self):
        self.tempdir.cleanup()

    def persist_confirmation(self, experiment_id, certificate):
        confirmations = self.store.read_root_json("promotion_confirmations.json") or {}
        confirmations[experiment_id] = certificate
        self.store.write_root_json("promotion_confirmations.json", confirmations)

    def test_improving_candidate_is_accepted_and_updates_state(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
        )

        result = controller.run_iteration(proposal("exp_001"), strong_candidate)

        self.assertEqual(result.decision, "accepted")
        self.assertEqual(controller.state.current_best_experiment_id, "exp_001")
        self.assertEqual(controller.state.current_best_primary, 1.0)
        self.assertTrue((self.store.root / "state.json").exists())
        record = self.store.read_iterations()[0]
        self.assertEqual(record["decision"], "accepted")
        self.assertEqual(record["parent_experiment_id"], "baseline")

    def test_production_candidate_cannot_become_champion_before_seed_confirmation(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
            require_seed_confirmation=True,
        )

        pending = controller.run_iteration(proposal("exp_seed_gate"), strong_candidate)

        self.assertEqual(pending.decision, "pending_confirmation")
        self.assertEqual(controller.state.current_best_experiment_id, "baseline")
        checkpoint = self.store.runs_dir / "exp_seed_gate-promotion-candidate-seed-0" / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        certificate = {
            "selected_experiment_id": "exp_seed_gate",
            "seeds": [0, 1, 2],
            "candidate_scores": [0.9, 0.91, 0.89],
            "comparator_scores": [0.5, 0.5, 0.5],
            "comparator_experiment_id": "baseline",
            "mean_delta": 0.4,
            "wins": 3,
            "confirmed": True,
            "candidate_comparison_groups": ["a" * 64] * 3,
            "comparator_comparison_groups": ["a" * 64] * 3,
            "confirmation_attempts": 6,
            "submission_checkpoint_path": str(checkpoint),
        }
        self.persist_confirmation("exp_seed_gate", certificate)
        resolved = controller.resolve_seed_confirmation("exp_seed_gate", certificate)

        self.assertEqual(resolved.decision, "accepted")
        self.assertEqual(controller.state.current_best_experiment_id, "exp_seed_gate")
        self.assertAlmostEqual(controller.state.current_best_primary, 0.9)

    def test_failed_seed_confirmation_counts_as_non_improvement(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
            require_seed_confirmation=True,
        )
        controller.run_iteration(proposal("exp_noisy"), strong_candidate)

        certificate = {
            "selected_experiment_id": "exp_noisy",
            "seeds": [0, 1, 2],
            "candidate_scores": [0.9, 0.4, 0.4],
            "comparator_scores": [0.5, 0.5, 0.5],
            "comparator_experiment_id": "baseline",
            "mean_delta": (0.4 - 0.1 - 0.1) / 3,
            "wins": 1,
            "confirmed": False,
            "candidate_comparison_groups": ["a" * 64] * 3,
            "comparator_comparison_groups": ["a" * 64] * 3,
            "confirmation_attempts": 6,
            "submission_checkpoint_path": None,
        }
        self.persist_confirmation("exp_noisy", certificate)
        resolved = controller.resolve_seed_confirmation("exp_noisy", certificate)

        self.assertEqual(resolved.decision, "rejected")
        self.assertEqual(controller.state.current_best_experiment_id, "baseline")
        self.assertEqual(controller.state.consecutive_non_improvements, 1)

        replayed = controller.resolve_seed_confirmation("exp_noisy", certificate)
        self.assertEqual(replayed.decision, "rejected")
        self.assertEqual(controller.state.consecutive_non_improvements, 1)
        self.assertEqual(controller.state.valid_comparisons, 1)
        self.assertEqual(controller.state.runner_attempts, 7)

    def test_seed_confirmation_rejects_tampered_or_non_finite_evidence(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
            require_seed_confirmation=True,
        )
        controller.run_iteration(proposal("exp_tampered"), strong_candidate)
        certificate = {
            "selected_experiment_id": "exp_tampered",
            "seeds": [0, 1, 2],
            "candidate_scores": [0.9, 0.9, 0.9],
            "comparator_scores": [0.5, 0.5, 0.5],
            "comparator_experiment_id": "baseline",
            "mean_delta": 0.4,
            "wins": 3,
            "confirmed": True,
            "candidate_comparison_groups": ["b" * 64] * 3,
            "comparator_comparison_groups": ["b" * 64] * 3,
            "confirmation_attempts": 6,
            "submission_checkpoint_path": None,
        }
        self.persist_confirmation("exp_tampered", certificate)

        with self.assertRaisesRegex(ValueError, "checkpoint"):
            controller.resolve_seed_confirmation("exp_tampered", certificate)

        non_finite = {**certificate, "candidate_scores": [float("nan"), 0.9, 0.9]}
        with self.assertRaisesRegex(ValueError, "finite"):
            controller.resolve_seed_confirmation("exp_tampered", non_finite)
        self.assertEqual(controller.state.current_best_experiment_id, "baseline")

    def test_controller_recovers_state_and_history_after_restart(self):
        test_contract = BenchmarkContract()
        first_controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
            contract=test_contract,
        )
        first_controller.run_iteration(proposal("exp_001"), strong_candidate)

        resumed = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            contract=test_contract,
        )
        duplicate = resumed.run_iteration(
            proposal("exp_002", config={"candidate": "exp_001", "fidelity": "full", "epochs": 40, "patience": 4}),
            strong_candidate,
        )

        self.assertEqual(resumed.state.current_best_experiment_id, "exp_001")
        self.assertEqual(duplicate.decision, "rejected")
        self.assertIn("duplicates", duplicate.error)

    def test_non_improvement_is_rejected_and_preserves_best_pointer(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_experiment_id="baseline", current_best_primary=1.0),
        )

        result = controller.run_iteration(proposal("exp_002"), strong_candidate)

        self.assertEqual(result.decision, "rejected")
        self.assertEqual(controller.state.current_best_experiment_id, "baseline")
        record = self.store.read_iterations()[0]
        self.assertIn("restored accepted candidate pointer", record["recovery"])

    def test_safety_rejection_never_invokes_runner(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
        )
        unsafe = proposal("exp_003", changed_factors=("a", "b"))

        result = controller.run_iteration(unsafe, strong_candidate)

        self.assertEqual(result.decision, "rejected")
        self.assertFalse((self.store.runs_dir / "exp_003").exists())
        self.assertIn("exactly one", result.error)

    def test_plateau_stops_after_three_valid_non_improvements(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.64),
        )

        failed = controller.run_iteration(proposal("exp_004"), broken_candidate)
        controller.run_iteration(proposal("exp_005"), weak_candidate)
        controller.run_iteration(proposal("exp_006"), weak_candidate)
        third = controller.run_iteration(proposal("exp_007"), weak_candidate)

        self.assertEqual(failed.decision, "failed")
        self.assertEqual(third.decision, "rejected")
        self.assertTrue(controller.state.stopped)
        self.assertEqual(controller.state.stop_reason_code, "plateau")
        self.assertEqual(controller.state.consecutive_non_improvements, 3)

    def test_plateau_restart_cannot_reset_terminal_counter(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.64),
        )
        for number in range(3):
            controller.run_iteration(proposal(f"exp_{number + 10:03d}"), weak_candidate)
        self.assertTrue(controller.state.stopped)
        self.assertEqual(controller.state.consecutive_non_improvements, 3)
        controller.begin_plateau_restart()
        self.assertEqual(controller.state.consecutive_non_improvements, 3)
        self.assertEqual(controller.state.plateau_restarts, 0)

    def test_low_fidelity_result_cannot_update_champion(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
        )

        result = controller.run_iteration(
            proposal("screen_001", config={"candidate": "screen_001", "fidelity": "low", "epochs": 4}),
            strong_candidate,
        )

        self.assertEqual(result.decision, "screened")
        self.assertFalse(result.comparison_validity.valid)
        self.assertEqual(controller.state.current_best_experiment_id, "baseline")
        self.assertEqual(controller.state.current_best_primary, 0.5)
        self.assertEqual(controller.state.consecutive_non_improvements, 0)

    def test_exact_epsilon_improvement_is_not_accepted(self):
        contract = BenchmarkContract(improvement_threshold=0.5)
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
            contract=contract,
        )

        result = controller.run_iteration(proposal("exp_epsilon"), strong_candidate)

        self.assertNotEqual(result.decision, "accepted")

    def test_improvement_above_epsilon_is_accepted(self):
        contract = BenchmarkContract(improvement_threshold=0.5)
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.499999),
            contract=contract,
        )

        result = controller.run_iteration(proposal("exp_above_epsilon"), strong_candidate)

        self.assertEqual(result.decision, "accepted")

    def test_iteration_budget_stops_controller_without_score_target(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=1.0),
            max_iterations=2,
        )
        controller.run_iteration(proposal("exp_020"), weak_candidate)
        controller.run_iteration(proposal("exp_021"), weak_candidate)

        self.assertTrue(controller.state.stopped)
        self.assertEqual(controller.state.stop_reason_code, "iteration_budget")

    def test_wall_clock_budget_is_checked_before_execution(self):
        now = [0.0]
        contract = BenchmarkContract(max_wall_clock_seconds=10)
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
            contract=contract,
            clock=lambda: now[0],
        )
        now[0] = 11.0

        result = controller.run_iteration(proposal("exp_late"), strong_candidate)

        self.assertEqual(result.decision, "skipped")
        self.assertEqual(controller.state.stop_reason_code, "wall_clock_budget")
        self.assertEqual(self.store.read_root_json("state.json")["stop_reason_code"], "wall_clock_budget")
        self.assertFalse((self.store.runs_dir / "exp_late").exists())

    def test_transient_retry_preserves_one_logical_iteration(self):
        calls = 0

        def transient_then_strong(data, config, run_dir):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("temporary worker outage")
            return strong_candidate(data, config, run_dir)

        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
        )
        result = controller.run_iteration(
            proposal("exp_retry_controller"),
            transient_then_strong,
        )

        self.assertEqual(result.decision, "accepted")
        self.assertEqual(controller.state.completed_iterations, 1)
        self.assertEqual(controller.state.runner_attempts, 2)

    def test_state_reconciles_newer_append_only_iteration_after_crash(self):
        stale = ResearchState(completed_iterations=1, current_best_primary=0.5)
        newer = ResearchState(
            completed_iterations=2,
            current_best_experiment_id="exp_002",
            current_best_primary=0.8,
        )
        self.store.write_root_json("state.json", stale.as_dict())
        self.store.append_iteration(
            {
                "experiment_id": "exp_002",
                "hypothesis": "recovery evidence",
                "rationale": "iteration committed before state write",
                "config": {"fidelity": "full"},
                "decision": "accepted",
                "state_after": newer.as_dict(),
            }
        )

        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
        )

        self.assertEqual(controller.state.completed_iterations, 2)
        self.assertEqual(controller.state.current_best_experiment_id, "exp_002")


class ActiveRuntimeAccountingTests(unittest.TestCase):
    """Paused time must not be spendable as if it were research effort."""

    def _controller(self, store, clock, state=None):
        logger = ResearchLogger(store)
        return ExperimentController(
            logger=logger,
            runner=ExperimentRunner(logger, data_loader=loader),
            validator=SafetyValidator(max_runtime_seconds=900),
            state=state,
            clock=clock,
        )

    def test_paused_time_between_invocations_is_not_charged(self):
        now = [1000.0]
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            first = self._controller(store, lambda: now[0])
            now[0] += 60.0                      # 60s of actual research
            first.checkpoint_active_runtime()
            carried = first.state.active_runtime_seconds
            self.assertAlmostEqual(carried, 60.0, places=6)

            now[0] += 10_000.0                  # ten thousand seconds paused
            second = self._controller(store, lambda: now[0], state=first.state)
            second._apply_stop_rule()
            # Only the 60 seconds of real work count.
            self.assertAlmostEqual(second.state.elapsed_seconds, 60.0, places=6)

    def test_active_runtime_accumulates_across_invocations(self):
        now = [500.0]
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            first = self._controller(store, lambda: now[0])
            now[0] += 30.0
            first.checkpoint_active_runtime()
            now[0] += 5_000.0                   # paused
            second = self._controller(store, lambda: now[0], state=first.state)
            now[0] += 45.0                      # more real research
            second.checkpoint_active_runtime()
            self.assertAlmostEqual(
                second.state.active_runtime_seconds, 75.0, places=6
            )
