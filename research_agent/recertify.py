"""Focused, validation-only recertification of a prior winning screen."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract
from .controller import ExperimentController
from .finalize import (
    _acquire_finalization_lock,
    _load_certified_selected_checkpoint,
    _preflight_selected_checkpoint,
)
from .logger import ResearchLogger
from .manifest import IMMUTABLE_LINEAGE_FIELDS, ensure_run_manifest
from .models.ensemble_checkpoint import load_ensemble_checkpoint
from .reporter import MarkdownReporter
from .runner import CandidateCallable, ExperimentRunner
from .safety import ExperimentProposal, SafetyValidator
from .seed_validation import confirm_promotion_candidate, confirm_selected_candidate
from .state import ResearchState
from .store import ArtifactStore


def recertify_screen_candidate(
    source_store: ArtifactStore,
    destination_store: ArtifactStore,
    *,
    screen_experiment_id: str,
    candidate: CandidateCallable,
    contract: BenchmarkContract = BENCHMARK_CONTRACT,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize source capture and make the focused workflow resumable."""
    source_store.initialize()
    destination_store.initialize()
    if source_store.root.resolve() == destination_store.root.resolve():
        raise ValueError("recertification source and destination must be different stores")
    # Recertification must exclude finalization at *both* ends.  The destination
    # becomes terminal after its one configured iteration, before multi-seed
    # confirmation is complete; without the destination finalization mutex a
    # concurrent finalizer could cross the test boundary with the baseline.
    finalization_locks = sorted(
        {
            source_store.root / ".finalization.lock",
            destination_store.root / ".finalization.lock",
        },
        key=lambda path: str(path.resolve()).casefold(),
    )
    acquired_finalization_locks: list[Path] = []
    recertification_lock = destination_store.root / ".recertification.lock"
    recertification_acquired = False
    try:
        for lock in finalization_locks:
            _acquire_finalization_lock(lock)
            acquired_finalization_locks.append(lock)
        _acquire_finalization_lock(recertification_lock)
        recertification_acquired = True
        return _recertify_screen_candidate_locked(
            source_store,
            destination_store,
            screen_experiment_id=screen_experiment_id,
            candidate=candidate,
            contract=contract,
            environment=environment,
        )
    finally:
        if recertification_acquired:
            recertification_lock.unlink(missing_ok=True)
        for lock in reversed(acquired_finalization_locks):
            lock.unlink(missing_ok=True)


def _recertify_screen_candidate_locked(
    source_store: ArtifactStore,
    destination_store: ArtifactStore,
    *,
    screen_experiment_id: str,
    candidate: CandidateCallable,
    contract: BenchmarkContract,
    environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Run one full promotion and matched-seed confirmation in a fresh store.

    The source screen is selection evidence only. Its low-fidelity bundle is
    never reused. Every fitted parameter in the destination is trained again
    under the current code and immutable benchmark lineage, using train and
    validation only.
    """
    source_manifest = ensure_run_manifest(source_store, contract, create=False)
    source_finalization = source_store.read_root_json("finalization.json")
    if source_finalization is not None:
        raise RuntimeError(
            "recertification source has finalization/boundary state; only a "
            "validation-only source store is eligible"
        )
    source_records = source_store.read_iterations()
    source_state = _validate_source_run(source_store, source_records, contract=contract)
    source_record = _select_best_eligible_screen(
        source_records,
        screen_experiment_id,
        contract=contract,
    )
    _validate_screen_lineage(source_store, source_record, source_manifest)
    source_bundle_replay = _preflight_selected_checkpoint(source_record, contract)
    source_payload = json.dumps(
        source_record,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    source_evidence = {
        "source_artifact_root": str(source_store.root.resolve()),
        "source_manifest_fingerprint": source_manifest["immutable_fingerprint"],
        "source_experiment_id": screen_experiment_id,
        "source_record_sha256": hashlib.sha256(source_payload).hexdigest(),
        "source_validation_primary": float(source_record["metrics"]["primary"]),
        "source_bundle_replay": source_bundle_replay,
        "source_fidelity": "low",
        "source_stop_reason_code": source_state.stop_reason_code,
        "source_active_runtime_seconds": source_state.active_runtime_seconds,
        "source_contract_wall_clock_seconds": contract.max_wall_clock_seconds,
        "scope_disposition": (
            "new independent recertification run; the exhausted source budget "
            "was not extended or reset"
        ),
        "test_accessed_at_selection": False,
    }
    existing_manifest = destination_store.read_root_json("run_manifest.json")
    if existing_manifest is None:
        if (
            destination_store.read_root_json("state.json") is not None
            or destination_store.read_root_json("recertification_source.json") is not None
            or destination_store.read_iterations()
            or destination_store.run_experiment_ids()
        ):
            raise RuntimeError("recertification destination has artifacts but no manifest")
        ensure_run_manifest(
            destination_store,
            contract,
            create=True,
            environment=environment,
        )
        destination_store.write_root_json(
            "recertification_source.json",
            source_evidence,
        )
        logger = ResearchLogger(destination_store)
    else:
        ensure_run_manifest(destination_store, contract, create=False)
        persisted_source = destination_store.read_root_json("recertification_source.json")
        if persisted_source is None and not (
            destination_store.read_root_json("state.json") is not None
            or destination_store.read_iterations()
            or destination_store.run_experiment_ids()
        ):
            destination_store.write_root_json(
                "recertification_source.json",
                source_evidence,
            )
            persisted_source = source_evidence
        if persisted_source != source_evidence:
            raise RuntimeError("recertification resume source evidence changed")
        logger = ResearchLogger(destination_store)
    _ensure_recertification_intervention(
        destination_store,
        logger,
        screen_experiment_id=screen_experiment_id,
        source_record_sha256=source_evidence["source_record_sha256"],
    )
    config = {
        **dict(source_record["config"]),
        "fidelity": "full",
        "epochs": contract.full_max_epochs,
        "patience": contract.full_patience,
        "seed": 0,
    }
    proposal = ExperimentProposal(
        experiment_id="exp_001",
        parent_experiment_id="baseline",
        comparison_incumbent_id="baseline",
        hypothesis=str(source_record["hypothesis"]),
        rationale=(
            "Focused recertification of the highest validation-only screen from "
            f"{screen_experiment_id}; source record SHA-256 "
            f"{source_evidence['source_record_sha256']}."
        ),
        config=config,
        changed_factors=("loss",),
        runtime_budget_seconds=2400.0,
        research_direction_id="rank_ensemble",
        search_strategy="focused_recertification",
        search_region_id="region_rank_ensemble",
    )
    existing_records = destination_store.read_iterations()
    if len(existing_records) > 1:
        raise RuntimeError("focused recertification has unexpected iteration history")
    if existing_records:
        record = existing_records[0]
        if (
            record.get("experiment_id") != proposal.experiment_id
            or record.get("config") != proposal.config
            or record.get("comparison_incumbent_id") != "baseline"
        ):
            raise RuntimeError("recertification resume plan differs from persisted evidence")
        iteration_decision = str(record.get("decision"))
        iteration_metrics = dict(record.get("metrics") or {})
        _reconcile_destination_state_after_iteration(destination_store, record)
    runner = ExperimentRunner(logger, contract=contract)
    controller = ExperimentController(
        logger=logger,
        runner=runner,
        validator=SafetyValidator(max_runtime_seconds=2400.0),
        contract=contract,
        max_iterations=1,
        require_seed_confirmation=True,
    )
    if not existing_records:
        iteration = controller.run_iteration(proposal, candidate)
        iteration_decision = iteration.decision
        iteration_metrics = dict(iteration.metrics or {})
    if iteration_decision != "pending_confirmation":
        controller.checkpoint_active_runtime()
        MarkdownReporter(destination_store).write()
        return {
            "status": "not_promotable",
            "experiment_id": proposal.experiment_id,
            "decision": iteration_decision,
            "metrics": iteration_metrics,
        }

    record = destination_store.read_iterations()[-1]
    certificate = confirm_promotion_candidate(
        destination_store,
        runner,
        record,
        candidate=candidate,
        contract=contract,
    )
    resolution = controller.resolve_seed_confirmation(
        proposal.experiment_id,
        certificate.as_dict(),
    )
    final_certificate = None
    if resolution.decision == "accepted":
        final_certificate = confirm_selected_candidate(
            destination_store,
            runner,
            contract=contract,
        )
        if (
            final_certificate.selected_experiment_id != proposal.experiment_id
            or not final_certificate.confirmed
            or not final_certificate.submission_bundle
        ):
            raise RuntimeError(
                "accepted promotion did not produce a candidate-bound final certificate"
            )
    controller.checkpoint_active_runtime()

    replay_evidence = None
    if final_certificate is not None:
        selected = destination_store.read_iterations()[-1]
        selected = {
            **selected,
            "runner_metadata": {
                **dict(selected.get("runner_metadata") or {}),
                "checkpoint_path": final_certificate.submission_checkpoint_path,
            },
        }
        loaded_checkpoint, checkpoint_sha256 = _load_certified_selected_checkpoint(
            destination_store,
            selected,
        )
        replay_evidence = _preflight_selected_checkpoint(
            selected,
            contract,
            loaded_checkpoint=loaded_checkpoint,
            checkpoint_sha256=checkpoint_sha256,
        )
        destination_store.write_root_json(
            "bundle_replay.json",
            dict(replay_evidence or {}),
        )
    MarkdownReporter(destination_store).write()
    return {
        "status": "certified" if final_certificate is not None else "rejected",
        "experiment_id": proposal.experiment_id,
        "promotion_decision": resolution.decision,
        "candidate_scores": list(certificate.candidate_scores),
        "comparator_scores": list(certificate.comparator_scores),
        "mean_delta": certificate.mean_delta,
        "wins": certificate.wins,
        "submission_bundle": (
            dict(final_certificate.submission_bundle or {})
            if final_certificate is not None
            else {}
        ),
        "bundle_replay": replay_evidence,
    }


def _reconcile_destination_state_after_iteration(
    store: ArtifactStore,
    record: Mapping[str, Any],
) -> None:
    """Recover the two append-before-state crash windows without rerunning work."""
    record_state = record.get("state_after")
    if not isinstance(record_state, dict):
        raise RuntimeError("persisted recertification iteration has no state snapshot")
    resolutions = store.read_root_json("promotion_resolutions.json") or {}
    experiment_id = record.get("experiment_id")
    if not isinstance(experiment_id, str):
        raise RuntimeError("persisted recertification iteration has no experiment id")
    resolution = resolutions.get(experiment_id)
    expected = (
        resolution.get("state_after")
        if isinstance(resolution, dict)
        else record_state
    )
    if not isinstance(expected, dict):
        raise RuntimeError("persisted recertification resolution has no state snapshot")
    current = store.read_root_json("state.json")
    state_fields = (
        "completed_iterations",
        "current_best_experiment_id",
        "current_best_primary",
        "valid_comparisons",
        "stop_reason_code",
    )
    if isinstance(current, dict) and all(
        current.get(field) == expected.get(field) for field in state_fields
    ):
        return
    predecessor = record_state
    if isinstance(current, dict) and not all(
        current.get(field) == predecessor.get(field) for field in state_fields
    ):
        completed = current.get("completed_iterations")
        expected_completed = record_state.get("completed_iterations")
        if not (
            isinstance(completed, int)
            and isinstance(expected_completed, int)
            and completed + 1 == expected_completed
            and current.get("current_best_experiment_id")
            == record_state.get("current_best_experiment_id")
            and current.get("current_best_primary")
            == record_state.get("current_best_primary")
        ):
            raise RuntimeError("recertification destination state contradicts its evidence")
    store.write_root_json("state.json", expected)


def _ensure_recertification_intervention(
    store: ArtifactStore,
    logger: ResearchLogger,
    *,
    screen_experiment_id: str,
    source_record_sha256: str,
) -> None:
    """Make bootstrap audit logging recoverable across every write boundary."""
    description = (
        "Started a focused full-fidelity recertification of validation screen "
        f"{screen_experiment_id}; source record {source_record_sha256}."
    )
    reason = (
        "The autonomous source run exhausted its wall-clock budget after screening "
        "but before promotion; silently reopening terminal state would invalidate "
        "its budget evidence."
    )
    effect = (
        "No source checkpoint is reused. The selected core configuration is retrained "
        "at 40 epochs/patience 4 and compared on matched seeds 0/1/2."
    )
    marker = {
        "schema_version": 1,
        "experiment_id": "exp_001",
        "source_record_sha256": source_record_sha256,
        "manual_intervention_recorded": True,
    }
    existing_marker = store.read_root_json("recertification_initialized.json")
    if existing_marker is not None:
        if existing_marker != marker:
            raise RuntimeError("recertification initialization marker changed")
        return
    found = False
    if store.interventions_path.is_file():
        _recover_interrupted_intervention_tail(store)
        for item in store.read_interventions():
            if (
                item.get("experiment_id") == "exp_001"
                and item.get("description") == description
                and item.get("reason") == reason
                and item.get("effect") == effect
            ):
                found = True
                break
    if not found:
        logger.record_manual_intervention(
            description=description,
            reason=reason,
            effect=effect,
            experiment_id="exp_001",
        )
    store.write_root_json("recertification_initialized.json", marker)


def _recover_interrupted_intervention_tail(store: ArtifactStore) -> None:
    """Preserve and remove only a malformed final JSONL fragment."""
    content = store.interventions_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    if not lines:
        return
    try:
        json.loads(lines[-1].strip())
        return
    except json.JSONDecodeError:
        pass
    fragment = lines[-1]
    digest = hashlib.sha256(fragment.encode("utf-8")).hexdigest()
    store.write_root_json(
        "recertification_intervention_recovery.json",
        {
            "schema_version": 1,
            "reason": "interrupted final JSONL append",
            "fragment_sha256": digest,
            "fragment": fragment,
        },
    )
    ArtifactStore._atomic_write_text(store.interventions_path, "".join(lines[:-1]))


def _select_best_eligible_screen(
    records: list[dict[str, Any]],
    requested_id: str,
    *,
    contract: BenchmarkContract,
) -> dict[str, Any]:
    eligible = []
    for record in records:
        metadata = record.get("runner_metadata") or {}
        comparison = record.get("comparison_validity") or {}
        primary = (record.get("metrics") or {}).get("primary")
        if (
            record.get("decision") == "screened"
            and (record.get("config") or {}).get("fidelity") == "low"
            and metadata.get("model") == "fm_rank_ensemble"
            and record.get("comparison_incumbent_id") == "baseline"
            and comparison.get("selection_split") == contract.validation_split
            and comparison.get("reasons") == ["candidate is not full fidelity"]
            and not isinstance(primary, bool)
            and isinstance(primary, (int, float))
            and math.isfinite(float(primary))
        ):
            eligible.append(record)
    if not eligible:
        raise RuntimeError("source run has no eligible validation-only ensemble screen")
    selected = max(
        eligible,
        key=lambda item: (float(item["metrics"]["primary"]), str(item["experiment_id"])),
    )
    if selected.get("experiment_id") != requested_id:
        raise RuntimeError(
            f"requested screen {requested_id} is not the best eligible screen; "
            f"the evidence selects {selected.get('experiment_id')}"
        )
    return selected


def _validate_source_run(
    store: ArtifactStore,
    records: list[dict[str, Any]],
    *,
    contract: BenchmarkContract,
) -> ResearchState:
    payload = store.read_root_json("state.json")
    if payload is None:
        raise RuntimeError("recertification source has no terminal state")
    state = ResearchState.from_dict(payload)
    if not state.stopped or state.stop_reason_code not in {
        "plateau",
        "iteration_budget",
        "wall_clock_budget",
    }:
        raise RuntimeError("recertification source is not at an allowed terminal stop")
    if state.completed_iterations != len(records):
        raise RuntimeError("recertification source state and iteration history disagree")
    if records:
        last_state = records[-1].get("state_after") or {}
        for field in (
            "completed_iterations",
            "current_best_experiment_id",
            "current_best_primary",
        ):
            if last_state.get(field) != getattr(state, field):
                raise RuntimeError(
                    "recertification source terminal state does not match its history"
                )
        recorded_stop = last_state.get("stop_reason_code")
        if recorded_stop not in {None, state.stop_reason_code}:
            raise RuntimeError(
                "recertification source terminal stop contradicts its history"
            )
    live_statuses = []
    for status_path in store.runs_dir.glob("*/status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("recertification source has unreadable run status") from exc
        if status.get("status") in {"reserved", "running"}:
            live_statuses.append(status_path)
    locks = list(store.runs_dir.rglob(".reservation.lock"))
    if live_statuses or locks:
        raise RuntimeError("recertification source still has live or unresolved runs")
    return state


def _validate_screen_lineage(
    source_store: ArtifactStore,
    record: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> None:
    metadata = record.get("runner_metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("selected screen has no runner lineage")
    mismatches = [
        field
        for field in IMMUTABLE_LINEAGE_FIELDS
        if metadata.get(field) != source_manifest.get(field)
    ]
    if mismatches:
        raise RuntimeError(
            "selected screen lineage differs from its source manifest: "
            + ", ".join(sorted(mismatches))
        )
    model_hash = metadata.get("model_code_sha256")
    if (
        not isinstance(model_hash, str)
        or len(model_hash) != 64
        or any(character not in "0123456789abcdef" for character in model_hash)
    ):
        raise RuntimeError("selected screen has no valid model-code lineage")
    if (
        metadata.get("checkpoint_schema_version") != 2
        or metadata.get("checkpoint_model_kind") != "fm_rank_ensemble"
    ):
        raise RuntimeError("selected screen has no schema-v2 ensemble bundle evidence")
    checkpoint = metadata.get("checkpoint_path")
    if not isinstance(checkpoint, str):
        raise RuntimeError("selected screen has no checkpoint path")
    checkpoint_path = Path(checkpoint)
    try:
        checkpoint_path.resolve().relative_to(source_store.runs_dir.resolve())
    except ValueError as exc:
        raise RuntimeError("selected screen checkpoint is outside its source store") from exc
    bundle = load_ensemble_checkpoint(checkpoint_path)
    if bundle.manifest.get("validation_score_sha256") != metadata.get(
        "validation_score_sha256"
    ):
        raise RuntimeError("selected screen bundle and record disagree on validation hash")
    if bundle.manifest.get("config") != record.get("config"):
        raise RuntimeError("selected screen bundle and record disagree on configuration")
    if (
        bundle.manifest.get("seed") != metadata.get("seed")
        or bundle.manifest.get("seed") != (record.get("config") or {}).get("seed")
    ):
        raise RuntimeError("selected screen bundle and record disagree on seed")
    recorded_primary = (record.get("metrics") or {}).get("primary")
    bundle_primary = bundle.manifest.get("validation_primary")
    if (
        isinstance(recorded_primary, bool)
        or not isinstance(recorded_primary, (int, float))
        or isinstance(bundle_primary, bool)
        or not isinstance(bundle_primary, (int, float))
        or not math.isclose(
            float(bundle_primary),
            float(recorded_primary),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("selected screen bundle and record disagree on validation primary")
    bundle_lineage = bundle.manifest.get("lineage")
    if not isinstance(bundle_lineage, Mapping):
        raise RuntimeError("selected screen bundle has no lineage")
    lineage_fields = (*IMMUTABLE_LINEAGE_FIELDS, "model_code_sha256", "seed")
    bundle_mismatches = [
        field for field in lineage_fields if bundle_lineage.get(field) != metadata.get(field)
    ]
    if bundle_mismatches:
        raise RuntimeError(
            "selected screen bundle lineage differs from its record: "
            + ", ".join(sorted(bundle_mismatches))
        )
