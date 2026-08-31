"""Focused, validation-only recertification of a prior winning screen."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract
from .controller import ExperimentController
from .finalize import _preflight_selected_checkpoint
from .logger import ResearchLogger
from .manifest import ensure_run_manifest
from .reporter import MarkdownReporter
from .runner import CandidateCallable, ExperimentRunner
from .safety import ExperimentProposal, SafetyValidator
from .seed_validation import confirm_promotion_candidate, confirm_selected_candidate
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
    """Run one full promotion and matched-seed confirmation in a fresh store.

    The source screen is selection evidence only. Its low-fidelity bundle is
    never reused. Every fitted parameter in the destination is trained again
    under the current code and immutable benchmark lineage, using train and
    validation only.
    """
    if destination_store.read_root_json("state.json") is not None:
        raise RuntimeError("recertification destination already contains state")
    if destination_store.read_root_json("run_manifest.json") is not None:
        raise RuntimeError("recertification destination already contains a manifest")
    if destination_store.read_iterations() or destination_store.run_experiment_ids():
        raise RuntimeError("recertification destination is not empty")

    source_manifest = ensure_run_manifest(source_store, contract, create=False)
    source_finalization = source_store.read_root_json("finalization.json")
    if source_finalization is not None:
        raise RuntimeError(
            "recertification source has finalization/boundary state; only a "
            "validation-only source store is eligible"
        )
    source_record = _select_best_eligible_screen(
        source_store.read_iterations(),
        screen_experiment_id,
        contract=contract,
    )
    ensure_run_manifest(
        destination_store,
        contract,
        create=True,
        environment=environment,
    )
    logger = ResearchLogger(destination_store)
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
        "source_fidelity": "low",
        "test_accessed": False,
    }
    destination_store.write_root_json("recertification_source.json", source_evidence)
    logger.record_manual_intervention(
        description=(
            f"Started a focused full-fidelity recertification of validation screen "
            f"{screen_experiment_id} in a fresh artifact store."
        ),
        reason=(
            "The autonomous source run exhausted its wall-clock budget after "
            "screening but before promotion; silently reopening terminal state "
            "would invalidate its budget evidence."
        ),
        effect=(
            "No source checkpoint is reused. The selected core configuration is "
            "retrained at 40 epochs/patience 4 and compared on matched seeds 0/1/2."
        ),
        experiment_id="exp_001",
    )

    runner = ExperimentRunner(logger, contract=contract)
    controller = ExperimentController(
        logger=logger,
        runner=runner,
        validator=SafetyValidator(max_runtime_seconds=2400.0),
        contract=contract,
        max_iterations=1,
        require_seed_confirmation=True,
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
    iteration = controller.run_iteration(proposal, candidate)
    if iteration.decision != "pending_confirmation":
        controller.checkpoint_active_runtime()
        MarkdownReporter(destination_store).write()
        return {
            "status": "not_promotable",
            "experiment_id": iteration.experiment_id,
            "decision": iteration.decision,
            "metrics": dict(iteration.metrics or {}),
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
        iteration.experiment_id,
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
            final_certificate.selected_experiment_id != iteration.experiment_id
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
        replay_evidence = _preflight_selected_checkpoint(selected, contract)
        destination_store.write_root_json(
            "bundle_replay.json",
            dict(replay_evidence or {}),
        )
    MarkdownReporter(destination_store).write()
    return {
        "status": "certified" if final_certificate is not None else "rejected",
        "experiment_id": iteration.experiment_id,
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
