from __future__ import annotations

import unittest
import tempfile
from unittest.mock import Mock, patch

from research_agent.contracts import BENCHMARK_CONTRACT
from research_agent.recertify import _select_best_eligible_screen
from research_agent.recertify import _validate_source_run
from research_agent.recertify import recertify_screen_candidate
from research_agent.controller import IterationResult
from research_agent.seed_validation import SeedConfirmationResult
from research_agent.state import ResearchState
from research_agent.store import ArtifactStore


def screen(experiment_id: str, primary: float, *, model: str = "fm_rank_ensemble"):
    return {
        "experiment_id": experiment_id,
        "hypothesis": "rank ensembling improves validation ranking",
        "comparison_incumbent_id": "baseline",
        "decision": "screened",
        "config": {"fidelity": "low", "loss": "ensemble", "member_set": "core6"},
        "metrics": {"primary": primary},
        "comparison_validity": {
            "selection_split": "valid",
            "reasons": ["candidate is not full fidelity"],
        },
        "runner_metadata": {"model": model},
    }


class RecertificationSelectionTests(unittest.TestCase):
    def test_recertification_holds_destination_finalization_mutex(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as destination_dir:
            source = ArtifactStore(source_dir)
            destination = ArtifactStore(destination_dir)
            acquired: list[str] = []

            def acquire(path):
                acquired.append(str(path))
                path.write_text('{"pid": 1}', encoding="utf-8")

            def locked(*_args, **_kwargs):
                self.assertTrue((destination.root / ".finalization.lock").exists())
                self.assertTrue((destination.root / ".recertification.lock").exists())
                return {"status": "certified"}

            with (
                patch(
                    "research_agent.recertify._acquire_finalization_lock",
                    side_effect=acquire,
                ),
                patch(
                    "research_agent.recertify._recertify_screen_candidate_locked",
                    side_effect=locked,
                ),
            ):
                result = recertify_screen_candidate(
                    source,
                    destination,
                    screen_experiment_id="exp_005",
                    candidate=lambda *_args: None,
                )

            self.assertEqual(result["status"], "certified")
            self.assertIn(str(source.root / ".finalization.lock"), acquired)
            self.assertIn(str(destination.root / ".finalization.lock"), acquired)
            self.assertIn(str(destination.root / ".recertification.lock"), acquired)
            self.assertFalse((destination.root / ".finalization.lock").exists())

    def test_recertification_rejects_same_source_and_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with self.assertRaisesRegex(ValueError, "different stores"):
                recertify_screen_candidate(
                    store,
                    store,
                    screen_experiment_id="exp_005",
                    candidate=lambda *_args: None,
                )

    def test_failed_seed_confirmation_reports_rejected_not_baseline_certified(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as destination_dir:
            source = ArtifactStore(source_dir)
            source.append_iteration(screen("exp_005", 0.6039))
            destination = ArtifactStore(destination_dir)
            run_calls: list[str] = []

            class FakeController:
                def __init__(self, *, logger, **_kwargs):
                    self.logger = logger

                def run_iteration(self, proposal, _candidate):
                    run_calls.append(proposal.experiment_id)
                    record = {
                        **screen("exp_001", 0.6039),
                        "config": dict(proposal.config),
                        "decision": "pending_confirmation",
                        "state_after": ResearchState(
                            completed_iterations=1,
                            stop_reason_code="iteration_budget",
                            stop_reason="focused scope completed",
                        ).as_dict(),
                    }
                    self.logger.store.append_iteration(record)
                    return IterationResult(
                        "exp_001",
                        "pending_confirmation",
                        ResearchState(),
                        metrics={"primary": 0.6039},
                    )

                def resolve_seed_confirmation(self, _experiment_id, _certificate):
                    return IterationResult(
                        "exp_001",
                        "rejected",
                        ResearchState(current_best_experiment_id="baseline"),
                    )

                def checkpoint_active_runtime(self):
                    return None

            rejected = SeedConfirmationResult(
                selected_experiment_id="exp_001",
                comparator_experiment_id="baseline",
                seeds=(0, 1, 2),
                candidate_scores=(0.603, 0.600, 0.601),
                comparator_scores=(0.601, 0.601, 0.601),
                mean_delta=0.0,
                wins=1,
                confirmed=False,
                submission_checkpoint_path=None,
                comparison_mode="matched_seed_organizer_baseline",
            )
            selected_confirmation = Mock()

            def fake_manifest(store, _contract, *, create, **_kwargs):
                manifest = {"immutable_fingerprint": "a" * 64}
                if create:
                    store.write_root_json("run_manifest.json", manifest)
                return manifest

            with (
                patch(
                    "research_agent.recertify.ensure_run_manifest",
                    side_effect=fake_manifest,
                ),
                patch(
                    "research_agent.recertify._validate_source_run",
                    return_value=ResearchState(
                        stop_reason_code="wall_clock_budget",
                        stop_reason="source budget exhausted",
                    ),
                ),
                patch("research_agent.recertify._validate_screen_lineage"),
                patch(
                    "research_agent.recertify._preflight_selected_checkpoint",
                    return_value={
                        "checkpoint_sha256": "b" * 64,
                        "validation_primary": 0.6039,
                        "test_materialized": False,
                    },
                ),
                patch("research_agent.recertify.ExperimentController", FakeController),
                patch(
                    "research_agent.recertify.confirm_promotion_candidate",
                    return_value=rejected,
                ),
                patch(
                    "research_agent.recertify.confirm_selected_candidate",
                    selected_confirmation,
                ),
                patch("research_agent.recertify.MarkdownReporter.write"),
            ):
                result = recertify_screen_candidate(
                    source,
                    destination,
                    screen_experiment_id="exp_005",
                    candidate=lambda *_args: None,
                )
                resumed = recertify_screen_candidate(
                    source,
                    destination,
                    screen_experiment_id="exp_005",
                    candidate=lambda *_args: None,
                )
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(resumed["status"], "rejected")
            self.assertEqual(result["promotion_decision"], "rejected")
            self.assertEqual(result["submission_bundle"], {})
            self.assertEqual(run_calls, ["exp_001"])
            selected_confirmation.assert_not_called()

    def test_source_with_any_finalization_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as destination_dir:
            source = ArtifactStore(source_dir)
            source.write_root_json(
                "run_manifest.json",
                {"placeholder": True},
            )
            source.write_root_json(
                "finalization.json",
                {"status": "test_access_started"},
            )
            with self.assertRaisesRegex(RuntimeError, "finalization/boundary"):
                # Manifest validation is deliberately bypassed here so the test
                # isolates the contamination guard.
                from unittest.mock import patch

                with patch(
                    "research_agent.recertify.ensure_run_manifest",
                    return_value={"immutable_fingerprint": "a" * 64},
                ):
                    recertify_screen_candidate(
                        source,
                        ArtifactStore(destination_dir),
                        screen_experiment_id="exp_005",
                        candidate=lambda *_args: None,
                    )

    def test_requested_screen_must_be_best_eligible_validation_screen(self):
        records = [screen("exp_005", 0.6039), screen("exp_007", 0.6037)]
        selected = _select_best_eligible_screen(
            records,
            "exp_005",
            contract=BENCHMARK_CONTRACT,
        )
        self.assertEqual(selected["experiment_id"], "exp_005")
        with self.assertRaisesRegex(RuntimeError, "not the best eligible"):
            _select_best_eligible_screen(
                records,
                "exp_007",
                contract=BENCHMARK_CONTRACT,
            )

    def test_full_single_model_and_nonvalidation_records_are_ineligible(self):
        full = screen("exp_001", 0.9)
        full["config"]["fidelity"] = "full"
        wrong_model = screen("exp_002", 0.8, model="fm")
        wrong_split = screen("exp_003", 0.7)
        wrong_split["comparison_validity"]["selection_split"] = "test"
        with self.assertRaisesRegex(RuntimeError, "no eligible"):
            _select_best_eligible_screen(
                [full, wrong_model, wrong_split],
                "exp_001",
                contract=BENCHMARK_CONTRACT,
            )

    def test_source_run_must_be_terminal_and_history_consistent(self):
        with tempfile.TemporaryDirectory() as source_dir:
            source = ArtifactStore(source_dir)
            record = screen("exp_005", 0.6039)
            record["state_after"] = {
                "completed_iterations": 1,
                "current_best_experiment_id": "baseline",
                "current_best_primary": 0.6016,
                "stop_reason_code": "wall_clock_budget",
            }
            source.append_iteration(record)
            source.write_root_json(
                "state.json",
                ResearchState(
                    completed_iterations=1,
                    current_best_experiment_id="baseline",
                    current_best_primary=0.6016,
                    stop_reason_code="wall_clock_budget",
                    stop_reason="source budget exhausted",
                ).as_dict(),
            )
            state = _validate_source_run(
                source,
                source.read_iterations(),
                contract=BENCHMARK_CONTRACT,
            )
            self.assertTrue(state.stopped)

            source.write_root_json(
                "state.json",
                ResearchState(
                    completed_iterations=1,
                    current_best_experiment_id="baseline",
                    current_best_primary=0.6016,
                ).as_dict(),
            )
            with self.assertRaisesRegex(RuntimeError, "terminal stop"):
                _validate_source_run(
                    source,
                    source.read_iterations(),
                    contract=BENCHMARK_CONTRACT,
                )

    def test_source_run_with_unresolved_reservation_is_rejected(self):
        with tempfile.TemporaryDirectory() as source_dir:
            source = ArtifactStore(source_dir)
            record = screen("exp_005", 0.6039)
            terminal = ResearchState(
                completed_iterations=1,
                current_best_experiment_id="baseline",
                current_best_primary=0.6016,
                stop_reason_code="wall_clock_budget",
                stop_reason="source budget exhausted",
            )
            record["state_after"] = terminal.as_dict()
            source.append_iteration(record)
            source.write_root_json("state.json", terminal.as_dict())
            run_dir = source.runs_dir / "exp_999"
            run_dir.mkdir(parents=True)
            (run_dir / ".reservation.lock").write_text("unresolved", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unresolved runs"):
                _validate_source_run(
                    source,
                    source.read_iterations(),
                    contract=BENCHMARK_CONTRACT,
                )


if __name__ == "__main__":
    unittest.main()
