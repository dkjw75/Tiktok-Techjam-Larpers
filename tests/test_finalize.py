from __future__ import annotations

import hashlib
import tempfile
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.contracts import BENCHMARK_CONTRACT
from research_agent.finalize import (
    FinalizationResult,
    _completed_result,
    _acquire_finalization_lock,
    _finalization_fingerprint,
    _require_finalizable,
    _selected_iteration,
    finalize_run,
)
from research_agent.manifest import ensure_run_manifest
from research_agent.state import ResearchState
from research_agent.store import ArtifactStore


class FinalizationTests(unittest.TestCase):
    def test_validation_preflight_runs_before_boundary_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            state = ResearchState(
                stop_reason_code="iteration_budget",
                stop_reason="focused scope completed",
            )
            store.write_root_json("state.json", state.as_dict())
            ensure_run_manifest(store, BENCHMARK_CONTRACT, create=True)
            order: list[str] = []
            result = FinalizationResult(
                selected_experiment_id="baseline",
                selection_primary=0.6016,
                submission_path=store.root / "final_submission.csv",
                submission_checked=True,
                test_metrics=None,
                report_path=store.root / "research_log.md",
            )

            def preflight(_selected, _contract):
                order.append("preflight")
                return None

            def execute(*_args, **_kwargs):
                order.append("boundary")
                return result

            with (
                patch("research_agent.finalize._preflight_selected_checkpoint", side_effect=preflight),
                patch("research_agent.finalize._execute_finalization", side_effect=execute),
            ):
                actual = finalize_run(store, confirm_final_evaluation=True)

            self.assertIs(actual, result)
            self.assertEqual(order, ["preflight", "boundary"])

    def test_preflight_failure_never_executes_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            store.write_root_json(
                "state.json",
                ResearchState(
                    stop_reason_code="iteration_budget",
                    stop_reason="focused scope completed",
                ).as_dict(),
            )
            ensure_run_manifest(store, BENCHMARK_CONTRACT, create=True)
            with (
                patch(
                    "research_agent.finalize._preflight_selected_checkpoint",
                    side_effect=RuntimeError("replay mismatch"),
                ),
                patch("research_agent.finalize._execute_finalization") as execute,
            ):
                with self.assertRaisesRegex(RuntimeError, "replay mismatch"):
                    finalize_run(store, confirm_final_evaluation=True)
            execute.assert_not_called()
            certificate = store.read_root_json("finalization.json")
            assert certificate is not None
            self.assertEqual(certificate["status"], "failed_before_test_boundary")

    def test_stale_partial_finalization_lock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / ".finalization.lock"
            lock.write_bytes(b"")
            stale = time.time() - 60
            os.utime(lock, (stale, stale))
            with self.assertRaisesRegex(RuntimeError, "owner cannot be verified"):
                _acquire_finalization_lock(lock)
            lock.unlink()

    def test_fresh_partial_finalization_lock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / ".finalization.lock"
            lock.write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "owner cannot be verified"):
                _acquire_finalization_lock(lock)

    def test_old_lock_owned_by_live_process_is_never_stolen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / ".finalization.lock"
            lock.write_text(
                __import__("json").dumps({"pid": os.getpid()}),
                encoding="utf-8",
            )
            old = time.time() - 24 * 60 * 60
            os.utime(lock, (old, old))
            with self.assertRaisesRegex(RuntimeError, "another finalization"):
                _acquire_finalization_lock(lock)

    def test_completed_finalization_rejects_tampered_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            submission = Path(directory) / "submission.csv"
            report = Path(directory) / "report.md"
            submission.write_text("original submission\n", encoding="utf-8")
            report.write_text("original report\n", encoding="utf-8")
            hashlib = __import__("hashlib")
            payload = {
                "selected_experiment_id": "baseline",
                "selection_primary": 0.6016,
                "submission_path": str(submission),
                "submission_checked": True,
                "test_metrics": None,
                "report_path": str(report),
                "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
                "submission_size_bytes": submission.stat().st_size,
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "report_size_bytes": report.stat().st_size,
            }
            submission.write_text("tampered submission\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "artifact hash changed"):
                _completed_result(payload)

    def test_finalization_rejects_model_code_drift_before_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            manifest = ensure_run_manifest(store, BENCHMARK_CONTRACT, create=True)
            selected = {
                "experiment_id": "exp_001",
                "runner_metadata": {
                    **{
                        field: manifest[field]
                        for field in (
                            "data_sha256",
                            "evaluator_sha256",
                            "preprocessing_sha256",
                            "staging_code_sha256",
                            "feature_schema_sha256",
                            "comparison_group_id",
                        )
                    },
                    "model_code_sha256": "0" * 64,
                },
            }

            with self.assertRaisesRegex(RuntimeError, "model code changed"):
                _finalization_fingerprint(
                    store,
                    ResearchState(
                        current_best_experiment_id="exp_001",
                        stop_reason_code="plateau",
                        stop_reason="converged",
                    ),
                    selected,
                    Path(directory) / "submission.csv",
                    BENCHMARK_CONTRACT,
                    manifest=manifest,
                )

    def test_baseline_is_selected_when_no_agent_candidate_was_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            self.assertIsNone(_selected_iteration(store, "baseline"))

    def test_missing_selected_experiment_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with self.assertRaisesRegex(ValueError, "missing"):
                _selected_iteration(store, "exp_999")

    def test_finalization_requires_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            state = ResearchState(stop_reason="configured experiment budget reached")
            with self.assertRaisesRegex(PermissionError, "explicit"):
                _require_finalizable(store, state, confirm_final_evaluation=False)

    def test_finalization_requires_research_to_have_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with self.assertRaisesRegex(RuntimeError, "converge or exhaust"):
                _require_finalizable(
                    store,
                    ResearchState(),
                    confirm_final_evaluation=True,
                )

    def test_low_fidelity_selected_candidate_cannot_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            store.append_iteration(
                {
                    "experiment_id": "exp_001",
                    "hypothesis": "screen candidate",
                    "rationale": "must not finalize",
                    "config": {"fidelity": "low"},
                    "decision": "accepted",
                    "comparison_validity": {"valid": True},
                }
            )
            state = ResearchState(
                current_best_experiment_id="exp_001",
                stop_reason_code="iteration_budget",
                stop_reason="configured experiment budget reached",
            )
            with self.assertRaisesRegex(RuntimeError, "full fidelity"):
                _require_finalizable(store, state, confirm_final_evaluation=True)

    def test_full_valid_candidate_can_finalize(self) -> None:
        from research_agent.lineage import candidate_code_sha256
        from research_agent.models.dispatch import run_candidate

        model_code = candidate_code_sha256(run_candidate)
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            store.append_iteration(
                {
                    "experiment_id": "exp_001",
                    "hypothesis": "full candidate",
                    "rationale": "valid comparison",
                    "config": {"fidelity": "full"},
                    "decision": "accepted",
                    "comparison_validity": {"valid": True},
                    "runner_metadata": {
                        "model": "fm_rank_ensemble",
                        "checkpoint_schema_version": 2,
                        "validation_score_sha256": "c" * 64,
                        "model_code_sha256": model_code,
                    },
                }
            )
            state = ResearchState(
                current_best_experiment_id="exp_001",
                stop_reason_code="plateau",
                stop_reason="converged after three non-improvements",
            )
            checkpoint = store.runs_dir / "exp_001-promotion-candidate-seed-0" / "checkpoint.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            certificate = {
                "selected_experiment_id": "exp_001",
                "comparator_experiment_id": "baseline",
                "seeds": [0, 1, 2],
                "candidate_scores": [0.61, 0.61, 0.61],
                "comparator_scores": [0.60, 0.60, 0.60],
                "mean_delta": 0.01,
                "wins": 3,
                "confirmed": True,
                "submission_seed": 0,
                "submission_checkpoint_path": str(checkpoint),
                "submission_bundle": {
                    "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                    "size_bytes": len(b"checkpoint"),
                    "schema_version": 2,
                    "model_kind": "fm_rank_ensemble",
                    "seed": 0,
                    "validation_score_sha256": "c" * 64,
                    "model_code_sha256": model_code,
                },
                "comparison_mode": "matched_seed_organizer_baseline",
                "candidate_comparison_groups": ["a" * 64] * 3,
                "comparator_comparison_groups": ["a" * 64] * 3,
                "confirmation_attempts": 6,
            }
            store.write_root_json("seed_confirmation.json", certificate)
            store.write_root_json(
                "promotion_confirmations.json",
                {"exp_001": certificate},
            )
            store.write_root_json(
                "promotion_resolutions.json",
                {
                    "exp_001": {
                        "decision": "accepted",
                        "certificate": certificate,
                        "state_after": state.as_dict(),
                    }
                },
            )
            selected = _require_finalizable(
                store,
                state,
                confirm_final_evaluation=True,
            )
            self.assertEqual(selected["experiment_id"], "exp_001")
            self.assertEqual(
                selected["runner_metadata"]["checkpoint_path"],
                str(checkpoint),
            )

    def test_unconfirmed_candidate_falls_back_to_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            store.append_iteration(
                {
                    "experiment_id": "exp_001",
                    "hypothesis": "full candidate",
                    "rationale": "valid but noisy",
                    "config": {"fidelity": "full"},
                    "decision": "accepted",
                    "comparison_validity": {"valid": True},
                }
            )
            store.write_root_json(
                "seed_confirmation.json",
                {
                    "selected_experiment_id": "exp_001",
                    "seeds": [0, 1, 2],
                    "submission_seed": 0,
                    "confirmed": False,
                    "submission_checkpoint_path": None,
                },
            )
            state = ResearchState(
                current_best_experiment_id="exp_001",
                stop_reason_code="plateau",
                stop_reason="converged",
            )

            self.assertIsNone(
                _require_finalizable(store, state, confirm_final_evaluation=True)
            )

    def test_infrastructure_stop_cannot_unlock_final_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            state = ResearchState(
                stop_reason_code="infrastructure_failure",
                stop_reason="worker could not be recovered",
            )
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                _require_finalizable(
                    store,
                    state,
                    confirm_final_evaluation=True,
                )

    def test_completed_finalization_is_reused_without_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            state = ResearchState(
                stop_reason_code="iteration_budget",
                stop_reason="budget reached",
            )
            store.write_root_json("state.json", state.as_dict())
            submission = Path(directory) / "final_submission.csv"
            report = Path(directory) / "research_log.md"
            submission.write_text("row_id,user_id,video_id,score\n", encoding="utf-8")
            report.write_text("# completed\n", encoding="utf-8")
            ensure_run_manifest(store, BENCHMARK_CONTRACT, create=True)
            fingerprint = _finalization_fingerprint(
                store,
                state,
                None,
                submission,
                BENCHMARK_CONTRACT,
            )
            store.write_root_json(
                "finalization.json",
                {
                    "status": "completed",
                    "fingerprint": fingerprint,
                    "selected_experiment_id": "baseline",
                    "selection_primary": 0.6016,
                    "submission_path": str(submission),
                    "submission_checked": True,
                    "test_metrics": None,
                    "report_path": str(report),
                    "submission_sha256": __import__("hashlib").sha256(
                        submission.read_bytes()
                    ).hexdigest(),
                    "submission_size_bytes": submission.stat().st_size,
                    "report_sha256": __import__("hashlib").sha256(
                        report.read_bytes()
                    ).hexdigest(),
                    "report_size_bytes": report.stat().st_size,
                },
            )

            with patch("research_agent.finalize._execute_finalization") as execute:
                result = finalize_run(
                    store,
                    submission_path=submission,
                    confirm_final_evaluation=True,
                )

            execute.assert_not_called()
            self.assertEqual(result.selected_experiment_id, "baseline")

    def test_concurrent_finalization_is_rejected_before_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            store.write_root_json(
                "state.json",
                ResearchState(
                    stop_reason_code="iteration_budget",
                    stop_reason="budget reached",
                ).as_dict(),
            )
            store.initialize()
            ensure_run_manifest(store, BENCHMARK_CONTRACT, create=True)
            (store.root / ".finalization.lock").write_text(
                str(__import__("os").getpid()),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "another finalization"):
                finalize_run(store, confirm_final_evaluation=True)
