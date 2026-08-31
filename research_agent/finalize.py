"""Final, explicitly authorised test confirmation and submission creation."""
from __future__ import annotations

import os
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract
from .reporter import MarkdownReporter
from .state import ResearchState
from .store import ArtifactStore
from .lineage import (
    candidate_code_sha256,
    normalized_file_sha256,
    verify_bundle_descriptor,
)
from .manifest import IMMUTABLE_LINEAGE_FIELDS, ensure_run_manifest


@dataclass(frozen=True)
class FinalizationResult:
    selected_experiment_id: str
    selection_primary: float
    submission_path: Path
    submission_checked: bool
    test_metrics: dict[str, Any] | None
    report_path: Path


def finalize_run(
    store: ArtifactStore,
    *,
    contract: BenchmarkContract = BENCHMARK_CONTRACT,
    submission_path: str | Path | None = None,
    confirm_final_evaluation: bool = False,
) -> FinalizationResult:
    """Write one valid submission and perform the permitted final test check.

    Test rows enter only here, after validation-based selection is complete.
    """
    store.initialize()
    lock_path = store.root / ".finalization.lock"
    _acquire_finalization_lock(lock_path)
    target = Path(submission_path) if submission_path else store.root / "final_submission.csv"
    fingerprint: str | None = None
    recovery_evidence: dict[str, str] | None = None
    transaction_started = False
    try:
        # Every mutable read that authorizes the one-way test boundary is
        # serialized by the same process lock.  In particular, the bundle
        # cannot be swapped by another cooperating finalizer between
        # certificate verification and inference.
        manifest = ensure_run_manifest(store, contract, create=False)
        persisted = store.read_root_json("state.json") or {}
        state = ResearchState.from_dict(persisted)
        selected = _require_finalizable(
            store,
            state,
            confirm_final_evaluation=confirm_final_evaluation,
        )
        fingerprint = _finalization_fingerprint(
            store,
            state,
            selected,
            target,
            contract,
            manifest=manifest,
        )
        completed = store.read_root_json("finalization.json")
        if completed and completed.get("status") == "completed":
            if completed.get("fingerprint") != fingerprint:
                raise RuntimeError("run was already finalized with different immutable inputs")
            return _completed_result(completed)
        if completed and completed.get("status") == "recovery_authorized":
            recovery_evidence = _validate_recovery_authorization(
                store, completed, fingerprint
            )
        elif completed and completed.get("status") in {
            "test_access_started",
            "failed_after_test_boundary",
        }:
            raise RuntimeError(
                "previous finalization did not commit a completed certificate; "
                "test access is fail-closed and will not be repeated automatically"
            )
        elif completed and completed.get("status") not in {
            "failed_before_test_boundary",
        }:
            raise RuntimeError(
                "finalization has an unresolved or unknown transaction status: "
                f"{completed.get('status')!r}"
            )
        running_payload: dict[str, Any] = {
            "status": "running",
            "fingerprint": fingerprint,
            "pid": os.getpid(),
        }
        if recovery_evidence is not None:
            running_payload["recovery_evidence"] = recovery_evidence
        store.write_root_json(
            "finalization.json",
            running_payload,
        )
        transaction_started = True
        loaded_checkpoint, expected_checkpoint_sha256 = _load_certified_selected_checkpoint(
            store,
            selected,
        )
        preflight_evidence = _preflight_selected_checkpoint(
            selected,
            contract,
            loaded_checkpoint=loaded_checkpoint,
            checkpoint_sha256=expected_checkpoint_sha256,
        )
        return _execute_finalization(
            store,
            state,
            selected,
            target,
            contract,
            fingerprint,
            preflight_evidence=preflight_evidence,
            recovery_evidence=recovery_evidence,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            loaded_checkpoint=loaded_checkpoint,
        )
    except BaseException as exc:
        if transaction_started:
            current = store.read_root_json("finalization.json") or {}
            crossed_test_boundary = current.get("status") == "test_access_started"
            failure_payload: dict[str, Any] = {
                "status": (
                    "failed_after_test_boundary"
                    if crossed_test_boundary
                    else "failed_before_test_boundary"
                ),
                "fingerprint": fingerprint,
                "submission_target": str(target.resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            }
            if recovery_evidence is not None:
                failure_payload["recovery_evidence"] = recovery_evidence
            store.write_root_json(
                "finalization.json",
                failure_payload,
            )
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def _execute_finalization(
    store: ArtifactStore,
    state: ResearchState,
    selected: dict[str, Any] | None,
    target: Path,
    contract: BenchmarkContract,
    fingerprint: str,
    *,
    preflight_evidence: dict[str, Any] | None,
    recovery_evidence: dict[str, str] | None,
    expected_checkpoint_sha256: str | None,
    loaded_checkpoint: Any | None,
) -> FinalizationResult:
    target.parent.mkdir(parents=True, exist_ok=True)
    boundary_payload: dict[str, Any] = {
        "status": "test_access_started",
        "fingerprint": fingerprint,
        "submission_target": str(target.resolve()),
        "preflight_evidence": preflight_evidence,
    }
    if recovery_evidence is not None:
        boundary_payload["recovery_evidence"] = recovery_evidence
    store.write_root_json(
        "finalization.json",
        boundary_payload,
    )
    if selected is None:
        _make_official_baseline_submission(target, contract)
        test_metrics = None
    else:
        model = (selected.get("runner_metadata") or {}).get("model")
        if model == "fm_rank_ensemble":
            test_metrics = _write_selected_ensemble_submission(
                target,
                selected,
                contract,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                loaded_checkpoint=loaded_checkpoint,
            )
        else:
            test_metrics = _write_selected_torch_submission(target, selected, contract)
    from data import load
    from submit import read_submission

    splits = load(str(contract.data_dir))
    read_submission(target, splits[contract.test_split])
    final_selected_id = selected["experiment_id"] if selected is not None else "baseline"
    selection_primary = state.current_best_primary if selected is not None else 0.6016
    summary: dict[str, Any] = {
        "selected_experiment_id": final_selected_id,
        "research_champion_experiment_id": state.current_best_experiment_id,
        "selection_primary": selection_primary,
        "submission_path": str(target),
        "submission_checked": True,
        "test_GAUC": test_metrics.get("GAUC") if test_metrics else "unavailable (official baseline submission; no agent-selected candidate)",
        "test_nDCG@5": test_metrics.get("nDCG@5") if test_metrics else "unavailable (official baseline submission; no agent-selected candidate)",
        "test_primary": test_metrics.get("primary") if test_metrics else "unavailable (official baseline submission; no agent-selected candidate)",
    }
    store.write_root_json("final_summary.json", summary)
    report_path = MarkdownReporter(store).write()
    submission_sha256 = _file_sha256(target)
    report_sha256 = _file_sha256(report_path)
    result = FinalizationResult(
        selected_experiment_id=final_selected_id,
        selection_primary=selection_primary,
        submission_path=target,
        submission_checked=True,
        test_metrics=test_metrics,
        report_path=report_path,
    )
    completed_payload: dict[str, Any] = {
        "status": "completed",
        "fingerprint": fingerprint,
        "submission_target": str(target.resolve()),
        "preflight_evidence": preflight_evidence,
        "selected_experiment_id": result.selected_experiment_id,
        "selection_primary": result.selection_primary,
        "submission_path": str(result.submission_path),
        "submission_checked": result.submission_checked,
        "test_metrics": result.test_metrics,
        "report_path": str(result.report_path),
        "submission_sha256": submission_sha256,
        "submission_size_bytes": target.stat().st_size,
        "report_sha256": report_sha256,
        "report_size_bytes": report_path.stat().st_size,
    }
    if recovery_evidence is not None:
        completed_payload["recovery_evidence"] = recovery_evidence
    store.write_root_json(
        "finalization.json",
        completed_payload,
    )
    return result


def _completed_result(payload: dict[str, Any]) -> FinalizationResult:
    submission_path = Path(str(payload["submission_path"]))
    report_path = Path(str(payload["report_path"]))
    if not submission_path.is_file() or not report_path.is_file():
        raise RuntimeError("completed finalization certificate references missing artifacts")
    expected_artifacts = (
        (submission_path, "submission_sha256", "submission_size_bytes"),
        (report_path, "report_sha256", "report_size_bytes"),
    )
    for path, hash_field, size_field in expected_artifacts:
        if payload.get(hash_field) != _file_sha256(path):
            raise RuntimeError(f"completed finalization artifact hash changed: {path}")
        if payload.get(size_field) != path.stat().st_size:
            raise RuntimeError(f"completed finalization artifact size changed: {path}")
    return FinalizationResult(
        selected_experiment_id=str(payload["selected_experiment_id"]),
        selection_primary=float(payload["selection_primary"]),
        submission_path=submission_path,
        submission_checked=bool(payload["submission_checked"]),
        test_metrics=(
            dict(payload["test_metrics"])
            if isinstance(payload.get("test_metrics"), dict)
            else None
        ),
        report_path=report_path,
    )


def _validate_recovery_authorization(
    store: ArtifactStore,
    payload: dict[str, Any],
    fingerprint: str,
) -> dict[str, str]:
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError("recovery authorization belongs to a different transaction")
    record_path = Path(str(payload.get("recovery_record_path", "")))
    if not record_path.is_file():
        raise RuntimeError("recovery authorization record is missing")
    try:
        record_path.resolve().relative_to(
            (store.root / "finalization_recoveries").resolve()
        )
    except ValueError as exc:
        raise RuntimeError("recovery authorization record is outside the run store") from exc
    record_bytes = record_path.read_bytes()
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    if payload.get("recovery_record_sha256") != record_sha256:
        raise RuntimeError("recovery authorization record hash changed")
    try:
        record = json.loads(record_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError("recovery authorization record is invalid") from exc
    required_record_fields = {
        "schema_version",
        "action",
        "created_at_utc",
        "created_at_epoch_seconds",
        "reason",
        "previous_status",
        "previous_finalization",
        "previous_finalization_file",
        "lock_evidence",
        "output_absence_evidence",
    }
    if set(record) != required_record_fields:
        raise RuntimeError("recovery authorization record schema is incomplete")
    if record.get("schema_version") != 1 or record.get("action") != "recovery_authorization_intent":
        raise RuntimeError("recovery authorization record has an unsupported schema")
    previous = record.get("previous_finalization")
    if not isinstance(previous, dict) or previous.get("fingerprint") != fingerprint:
        raise RuntimeError("recovery record does not bind the original transaction")
    if record.get("previous_status") not in {
        "test_access_started",
        "failed_after_test_boundary",
    } or previous.get("status") != record.get("previous_status"):
        raise RuntimeError("recovery record has an invalid prior boundary status")
    archived_bytes = (
        json.dumps(previous, indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")
    archived_sha256 = hashlib.sha256(archived_bytes).hexdigest()
    archived_file = record.get("previous_finalization_file")
    if (
        not isinstance(archived_file, dict)
        or archived_file.get("sha256") != archived_sha256
        or archived_file.get("size_bytes") != len(archived_bytes)
        or archived_sha256 != payload.get("previous_finalization_sha256")
    ):
        raise RuntimeError("recovery record does not match the archived certificate")
    evidence = record.get("output_absence_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise RuntimeError("recovery record has no complete output-absence evidence")
    evidence_paths: list[Path] = []
    for item in evidence:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "present"}
            or not isinstance(item.get("path"), str)
            or not item.get("path")
            or item.get("present") is not False
        ):
            raise RuntimeError("recovery record has malformed output-absence evidence")
        evidence_paths.append(Path(item["path"]))
    target = previous.get("submission_target")
    if not isinstance(target, str) or not target:
        raise RuntimeError(
            "interrupted certificate does not bind its original submission target; "
            "this legacy transaction cannot be retried safely"
        )
    if Path(target).resolve() not in {path.resolve() for path in evidence_paths}:
        raise RuntimeError("recovery evidence omits the original submission target")
    existing = [path for path in evidence_paths if path.exists()]
    if existing:
        raise RuntimeError("recovery record contains output evidence")
    return {
        "recovery_record_path": str(record_path),
        "recovery_record_sha256": record_sha256,
        "recovered_from_status": str(payload.get("recovered_from_status", "")),
    }


def _finalization_fingerprint(
    store: ArtifactStore,
    state: ResearchState,
    selected: dict[str, Any] | None,
    target: Path,
    contract: BenchmarkContract,
    *,
    manifest: dict[str, Any] | None = None,
) -> str:
    manifest = manifest or ensure_run_manifest(store, contract, create=False)
    checkpoint_sha256 = None
    if selected is not None:
        metadata = selected.get("runner_metadata") or {}
        mismatches = [
            field
            for field in IMMUTABLE_LINEAGE_FIELDS
            if metadata.get(field) != manifest.get(field)
        ]
        if mismatches:
            raise RuntimeError(
                "selected candidate lineage differs from final benchmark inputs: "
                + ", ".join(sorted(mismatches))
            )
        from .models.dispatch import run_candidate

        expected_model_code = metadata.get("model_code_sha256")
        if not isinstance(expected_model_code, str) or not expected_model_code:
            raise RuntimeError("selected candidate has no model-code lineage")
        current_model_code = candidate_code_sha256(run_candidate)
        if current_model_code != expected_model_code:
            raise RuntimeError(
                "selected candidate model code changed after certification: "
                f"expected {expected_model_code}, got {current_model_code}"
            )
        confirmation = store.read_root_json("seed_confirmation.json") or {}
        checkpoint = Path(
            str(
                confirmation.get("submission_checkpoint_path")
                or selected.get("runner_metadata", {}).get("checkpoint_path", "")
            )
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"selected checkpoint is unavailable: {checkpoint}")
        checkpoint_sha256 = _file_sha256(checkpoint)
    payload = {
        "terminal_state": {
            "current_best_experiment_id": state.current_best_experiment_id,
            "current_best_primary": state.current_best_primary,
            "completed_iterations": state.completed_iterations,
            "valid_comparisons": state.valid_comparisons,
            "stop_reason_code": state.stop_reason_code,
        },
        "selected_experiment_id": selected.get("experiment_id") if selected else "baseline",
        "seed_confirmation": store.read_root_json("seed_confirmation.json"),
        "checkpoint_sha256": checkpoint_sha256,
        "immutable_manifest_fingerprint": manifest["immutable_fingerprint"],
        "finalizer_code_sha256": normalized_file_sha256(Path(__file__)),
        "submission_code_sha256": normalized_file_sha256(
            Path(__file__).resolve().parents[1] / "submit.py"
        ),
        "target": str(target.resolve()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _acquire_finalization_lock(lock_path: Path) -> None:
    """Atomically create a complete PID lock under an OS-backed guard."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    guard_path = lock_path.with_name(f"{lock_path.name}.guard")
    with guard_path.open("a+b") as guard:
        guard.seek(0, os.SEEK_END)
        if guard.tell() == 0:
            guard.write(b"\0")
            guard.flush()
            os.fsync(guard.fileno())
        guard.seek(0)
        if os.name != "nt":  # pragma: no cover - Windows is the competition host.
            raise RuntimeError("finalization locking requires the Windows host")
        import msvcrt

        try:
            msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("lock arbitration is already active") from exc

        def unlock() -> None:
            msvcrt.locking(guard.fileno(), msvcrt.LK_UNLCK, 1)
        try:
            for _attempt in range(2):
                try:
                    descriptor = os.open(
                        lock_path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    )
                    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                        json.dump(
                            {
                                "pid": os.getpid(),
                                "created_at_epoch_seconds": time.time(),
                            },
                            handle,
                            sort_keys=True,
                        )
                        handle.flush()
                        os.fsync(handle.fileno())
                    return
                except FileExistsError as exc:
                    try:
                        raw = lock_path.read_text(encoding="utf-8").strip()
                        if not raw:
                            lock_path.unlink(missing_ok=True)
                            continue
                        value = json.loads(raw)
                        owner_pid = int(value["pid"] if isinstance(value, dict) else value)
                    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                        raise RuntimeError(
                            "finalization lock owner cannot be verified; refusing to guess"
                        ) from exc
                    if ArtifactStore._pid_is_running(owner_pid):
                        raise RuntimeError("another finalization transaction is active") from exc
                    lock_path.unlink(missing_ok=True)
            raise RuntimeError("could not recover the finalization transaction lock")
        finally:
            unlock()


def _require_finalizable(
    store: ArtifactStore,
    state: ResearchState,
    *,
    confirm_final_evaluation: bool,
) -> dict[str, Any] | None:
    """Enforce the one-way boundary between validation research and final test use."""
    if not confirm_final_evaluation:
        raise PermissionError(
            "final test access requires explicit confirm_final_evaluation=True"
        )
    if not state.stopped:
        raise RuntimeError("research must converge or exhaust its configured budget before finalization")
    allowed_stop_codes = {"plateau", "iteration_budget", "wall_clock_budget"}
    if state.stop_reason_code not in allowed_stop_codes:
        raise RuntimeError(
            "finalization is forbidden for stop reason "
            f"{state.stop_reason_code or 'unclassified'}"
        )
    selected = _selected_iteration(store, state.current_best_experiment_id)
    if selected is None:
        return None
    comparison = selected.get("comparison_validity") or {}
    if selected.get("decision") != "accepted":
        resolution = (store.read_root_json("promotion_resolutions.json") or {}).get(
            state.current_best_experiment_id,
            {},
        )
        if resolution.get("decision") != "accepted":
            raise RuntimeError("selected candidate was not accepted")
    if selected.get("config", {}).get("fidelity") != "full":
        raise RuntimeError("selected candidate is not full fidelity")
    if comparison.get("valid") is not True:
        raise RuntimeError("selected candidate lacks valid comparison evidence")
    confirmation = store.read_root_json("seed_confirmation.json")
    if not confirmation:
        raise RuntimeError("selected candidate has not completed three-seed confirmation")
    if confirmation.get("selected_experiment_id") != state.current_best_experiment_id:
        raise RuntimeError("seed confirmation does not belong to the selected candidate")
    if confirmation.get("seeds") != [0, 1, 2] or confirmation.get("submission_seed") != 0:
        raise RuntimeError("seed confirmation does not match the fixed final seed policy")
    if confirmation.get("confirmed") is not True:
        return None
    promotion = (store.read_root_json("promotion_confirmations.json") or {}).get(
        state.current_best_experiment_id
    )
    resolution = (store.read_root_json("promotion_resolutions.json") or {}).get(
        state.current_best_experiment_id
    )
    if not isinstance(promotion, dict) or not isinstance(resolution, dict):
        raise RuntimeError("confirmed candidate lacks durable promotion evidence")
    if resolution.get("decision") != "accepted":
        raise RuntimeError("confirmed candidate promotion was not accepted")
    if resolution.get("certificate") != promotion or confirmation != promotion:
        raise RuntimeError("seed evidence differs across promotion and final certificates")
    checkpoint_path = confirmation.get("submission_checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise RuntimeError("confirmed candidate has no seed-0 submission checkpoint")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise RuntimeError("confirmed seed-0 submission checkpoint is missing")
    try:
        checkpoint.resolve().relative_to(store.runs_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(
            "confirmed seed-0 submission checkpoint is outside run artifacts"
        ) from exc

    # The certificate must bind bundle CONTENT, not a mutable path: a bundle
    # could otherwise be swapped between certification and finalization.
    descriptor = confirmation.get("submission_bundle")
    if not isinstance(descriptor, dict) or not descriptor:
        raise RuntimeError(
            "seed certificate has no bundle descriptor; recertify under the "
            "current implementation before finalizing"
        )
    verify_bundle_descriptor(checkpoint, descriptor)

    selected_metadata = dict(selected.get("runner_metadata") or {})
    if (
        selected_metadata.get("model") != "fm_rank_ensemble"
        or selected_metadata.get("checkpoint_schema_version") != 2
    ):
        raise RuntimeError(
            "only a schema-version-2 inference-only ensemble can be finalized; "
            "legacy training checkpoints must be recertified"
        )
    if descriptor.get("validation_score_sha256") != selected_metadata.get(
        "validation_score_sha256"
    ):
        raise RuntimeError("certificate and experiment disagree on the validation hash")
    if descriptor.get("model_code_sha256") != selected_metadata.get("model_code_sha256"):
        raise RuntimeError("certificate and experiment disagree on model-code lineage")

    # Model code must not have drifted since certification.
    from .models.dispatch import run_candidate

    current_model_code = candidate_code_sha256(run_candidate)
    if selected_metadata.get("model_code_sha256") != current_model_code:
        raise RuntimeError(
            "model code changed since certification; recertify before finalizing "
            f"(certified {selected_metadata.get('model_code_sha256')}, "
            f"current {current_model_code})"
        )
    return {
        **selected,
        "runner_metadata": {
            **dict(selected.get("runner_metadata") or {}),
            "checkpoint_path": checkpoint_path,
        },
    }


def _selected_iteration(store: ArtifactStore, experiment_id: str) -> dict[str, Any] | None:
    if experiment_id == "baseline":
        return None
    for record in reversed(store.read_iterations()):
        if record.get("experiment_id") == experiment_id:
            return record
    raise ValueError(f"selected experiment is missing from iteration history: {experiment_id}")


def _preflight_selected_checkpoint(
    selected: dict[str, Any] | None,
    contract: BenchmarkContract,
    *,
    loaded_checkpoint: Any | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Replay the certified validation output before test access is recorded.

    This function may read validation rows and a fitted bundle. It must never
    call the finalization loader or materialize the test split.
    """
    if selected is None:
        return None
    metadata = dict(selected.get("runner_metadata") or {})
    if metadata.get("model") != "fm_rank_ensemble":
        # Legacy single-FM checkpoints are still validated structurally by
        # their writer. The competition champion path is the v2 ensemble.
        return {"model": str(metadata.get("model")), "validation_replay": "not_applicable"}

    from .data_boundary import load_research_splits
    from .metrics import evaluate_predictions
    from .models.ensemble_checkpoint import load_ensemble_checkpoint
    from .models.ensemble_fm import predict_loaded_ensemble_checkpoint
    from .runner import PreparedData

    if loaded_checkpoint is None:
        checkpoint_path = _resolve_bundle(metadata, selected)
        checkpoint = load_ensemble_checkpoint(checkpoint_path)
    else:
        checkpoint = loaded_checkpoint
        checkpoint_path = Path(checkpoint.path)
    expected_hash = metadata.get("validation_score_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise RuntimeError("selected experiment records no validation score hash")
    if checkpoint.manifest.get("validation_score_sha256") != expected_hash:
        raise RuntimeError("bundle manifest and experiment disagree on validation hash")

    rows = load_research_splits(
        contract.data_dir,
        (contract.validation_split,),
    )[contract.validation_split]
    replay = predict_loaded_ensemble_checkpoint(
        PreparedData((), rows),
        checkpoint,
    )
    actual_hash = _score_vector_sha256(replay.scores)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "bundle validation replay does not reproduce the certified scores: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    replay_primary = evaluate_predictions(
        replay.user_ids,
        replay.labels,
        replay.scores,
        split=contract.validation_split,
    ).primary
    recorded_primary = (selected.get("metrics") or {}).get("primary")
    if (
        isinstance(recorded_primary, bool)
        or not isinstance(recorded_primary, (int, float))
        or not math.isclose(
            replay_primary,
            float(recorded_primary),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            "bundle validation replay does not reproduce the recorded primary"
        )
    return {
        "model": "fm_rank_ensemble",
        "checkpoint_sha256": checkpoint_sha256 or _file_sha256(checkpoint_path),
        "validation_rows": len(rows),
        "validation_score_sha256": actual_hash,
        "validation_primary": replay_primary,
        "test_materialized": False,
    }


def _load_certified_selected_checkpoint(
    store: ArtifactStore,
    selected: dict[str, Any] | None,
) -> tuple[Any | None, str | None]:
    """Load one certified object used for both replay and final inference."""
    if selected is None:
        return None, None
    from .models.ensemble_checkpoint import load_ensemble_checkpoint_bytes

    metadata = dict(selected.get("runner_metadata") or {})
    confirmation = store.read_root_json("seed_confirmation.json") or {}
    raw_checkpoint_path = confirmation.get("submission_checkpoint_path")
    if not isinstance(raw_checkpoint_path, str) or not raw_checkpoint_path:
        raise RuntimeError("final certificate has no submission checkpoint path")
    if metadata.get("checkpoint_schema_version") != 2:
        raise RuntimeError("final certificate does not select a schema-version-2 bundle")
    checkpoint_path = Path(raw_checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("final certificate submission checkpoint is missing")
    descriptor = confirmation.get("submission_bundle")
    if not isinstance(descriptor, dict) or not descriptor:
        raise RuntimeError("final certificate has no bundle descriptor")
    expected_sha256 = descriptor.get("sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise RuntimeError("final certificate has no bundle content hash")
    content = checkpoint_path.read_bytes()
    if len(content) != descriptor.get("size_bytes"):
        raise RuntimeError("certified bundle size changed during snapshot")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise RuntimeError("certified bundle content changed during snapshot")
    checkpoint = load_ensemble_checkpoint_bytes(
        content,
        source_path=checkpoint_path,
    )
    return checkpoint, expected_sha256


def _write_selected_torch_submission(target: Path, record: dict[str, Any], contract: BenchmarkContract) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]

    from data import encode, load
    from submit import write_submission

    from .metrics import evaluate_predictions
    from .models.torch_fm import TorchFM, _predict

    checkpoint_path = Path(record.get("runner_metadata", {}).get("checkpoint_path", ""))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"selected PyTorch checkpoint is unavailable: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise RuntimeError("selected checkpoint has an invalid schema")
    splits = load(str(contract.data_dir))
    encoded, feature_dim = encode(splits)
    if checkpoint.get("feature_dim") != feature_dim:
        raise RuntimeError("selected checkpoint feature schema does not match final data")
    expected_config = record.get("config") or {}
    for key in ("loss", "embedding_dim", "learning_rate", "l2"):
        if key in expected_config and checkpoint["config"].get(key) != expected_config.get(key):
            raise RuntimeError(f"selected checkpoint config mismatch: {key}")
    model = TorchFM(feature_dim, int(checkpoint["config"].get("embedding_dim", 16)))
    model.load_state_dict(checkpoint["model_state"])
    test_x, test_y, test_users = encoded[contract.test_split]
    scores = _predict(model, torch.as_tensor(test_x, dtype=torch.long))
    write_submission(target, splits[contract.test_split], scores)
    return evaluate_predictions(
        test_users,
        test_y,
        scores.tolist(),
        split=contract.test_split,
        allow_test=True,
    ).as_dict()



def _load_finalization_rows(
    contract: BenchmarkContract,
    requested_splits: tuple[str, ...] | None = None,
) -> dict[str, list[tuple[Any, ...]]]:
    """Load 12-field train and test rows at the sanctioned test boundary.

    `data_boundary` deliberately refuses to materialize test rows -- that
    guarantee protects the research loop and must not be weakened. Finalization
    is the one authorised place test data may be read, so the richer loader
    lives here instead.
    """
    import csv
    import os

    root = os.fspath(contract.data_dir)
    meta: dict[str, tuple[str, str, str, int]] = {}
    with open(os.path.join(root, "video_features_basic_pure.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            uploaded = row.get("upload_dt", "").replace("-", "")
            meta[row["video_id"]] = (
                row["author_id"],
                row.get("music_id", "UNK"),
                row.get("tag", "UNK").split(",")[0],
                int(uploaded) if uploaded.isdigit() else 0,
            )
    unknown = ("UNK", "UNK", "UNK", 0)
    all_windows = {
        contract.train_split: (20220408, 20220421),
        contract.validation_split: (20220422, 20220428),
        contract.test_split: (20220429, 20220508),
    }
    requested = requested_splits or tuple(all_windows)
    if not requested or any(name not in all_windows for name in requested):
        raise ValueError("finalization requested an unknown or empty split set")
    windows = {name: all_windows[name] for name in requested}
    output: dict[str, list[tuple[Any, ...]]] = {name: [] for name in windows}
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with open(os.path.join(root, filename), encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                date = int(row["date"])
                split = next(
                    (name for name, (lo, hi) in windows.items() if lo <= date <= hi),
                    None,
                )
                if split is None:
                    continue
                author, music, tag, uploaded = meta.get(row["video_id"], unknown)
                output[split].append((
                    date, row["user_id"], row["video_id"], author, row["tab"],
                    float(row["duration_ms"]),
                    1 if row[contract.label] != "0" else 0,
                    int(row.get("hourmin") or 0) // 100,
                    float(row.get("play_time_ms") or 0.0),
                    music, tag, uploaded,
                ))
    return output


def _write_selected_ensemble_submission(
    target: Path,
    record: dict[str, Any],
    contract: BenchmarkContract,
    *,
    expected_checkpoint_sha256: str | None = None,
    loaded_checkpoint: Any | None = None,
) -> dict[str, Any]:
    """Score test by INFERENCE from the certified bundle. Never trains.

    The finalizer must not fit anything. It loads immutable member parameters and
    fitted encoders, transforms the target rows, and applies the validation-fixed
    weights. No optimizer, no early stopping, no weight fitting exists on this
    path -- see tests/test_no_test_leakage.py.
    """
    from submit import write_submission

    from .metrics import evaluate_predictions
    from .models.ensemble_fm import predict_loaded_ensemble_checkpoint
    from .runner import PreparedData

    metadata = record.get("runner_metadata") or {}
    if metadata.get("checkpoint_schema_version") != 2:
        raise RuntimeError("final inference requires a schema-version-2 bundle")
    if not isinstance(expected_checkpoint_sha256, str) or loaded_checkpoint is None:
        raise RuntimeError("final inference has no certified in-memory bundle")
    rows = _load_finalization_rows(contract, (contract.test_split,))

    output = predict_loaded_ensemble_checkpoint(
        PreparedData((), (), rows[contract.test_split]),
        loaded_checkpoint,
    )
    write_submission(target, rows[contract.test_split], output.scores)
    return evaluate_predictions(
        output.user_ids,
        output.labels,
        output.scores,
        split=contract.test_split,
        allow_test=True,
    ).as_dict()


def _score_vector_sha256(scores: Any) -> str:
    """Canonical dtype and byte order so the hash is machine-independent."""
    import numpy as np

    return hashlib.sha256(
        np.asarray(scores, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def _resolve_bundle(metadata: dict[str, Any], record: dict[str, Any]) -> Path:
    """Locate the certified bundle and refuse anything that is not a v2 bundle."""
    if metadata.get("checkpoint_schema_version") != 2:
        raise RuntimeError(
            "selected experiment has no schema-version-2 inference bundle; "
            "recertify in a clean artifact directory before finalizing"
        )
    raw = metadata.get("checkpoint_path")
    if not raw:
        raise RuntimeError("selected experiment records no checkpoint path")
    path = Path(str(raw))
    if not path.is_file():
        raise FileNotFoundError(f"certified bundle is missing: {path}")
    return path


def _make_official_baseline_submission(target: Path, contract: BenchmarkContract) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repository_root / "submit.py"),
        str(target),
        "--make",
        "--split",
        contract.test_split,
        "--data_dir",
        str(contract.data_dir),
    ]
    environment = dict(os.environ)
    # submit.py prints Chinese status text; force a portable encoding on Windows.
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=repository_root,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"official baseline submission failed: {completed.stderr.strip() or completed.stdout.strip()}")
