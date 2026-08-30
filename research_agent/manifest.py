"""Creation and fail-closed validation of an immutable research-run manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import BenchmarkContract
from .lineage import current_benchmark_lineage, lineage_fingerprint
from .store import ArtifactStore


IMMUTABLE_LINEAGE_FIELDS = (
    "data_sha256",
    "evaluator_sha256",
    "preprocessing_sha256",
    "staging_code_sha256",
    "feature_schema_sha256",
    "comparison_group_id",
)


def immutable_manifest_values(contract: BenchmarkContract) -> dict[str, Any]:
    lineage = current_benchmark_lineage(contract)
    values: dict[str, Any] = {
        "data_dir": str(Path(contract.data_dir).resolve()),
        "selection_split": contract.selection_split,
        "max_trials": contract.max_iterations,
        "max_wall_clock_seconds": contract.max_wall_clock_seconds,
        "full_max_epochs": contract.full_max_epochs,
        "full_patience": contract.full_patience,
        **lineage,
    }
    values["immutable_fingerprint"] = lineage_fingerprint(values)
    return values


def ensure_run_manifest(
    store: ArtifactStore,
    contract: BenchmarkContract,
    *,
    create: bool,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = immutable_manifest_values(contract)
    existing = store.read_root_json("run_manifest.json")
    if existing is None:
        if not create:
            raise RuntimeError("run manifest is missing; immutable inputs cannot be verified")
        manifest = {"schema_version": 2, **dict(environment or {}), **expected}
        store.write_root_json("run_manifest.json", manifest)
        return manifest

    missing = [field for field in expected if field not in existing]
    if missing:
        _validate_legacy_iteration_lineage(store, expected)
        existing = {**existing, "schema_version": 2, **expected}
        store.write_root_json("run_manifest.json", existing)
    mismatches = [
        field
        for field, expected_value in expected.items()
        if existing.get(field) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "immutable run inputs changed: " + ", ".join(sorted(mismatches))
        )
    return existing


def _validate_legacy_iteration_lineage(
    store: ArtifactStore,
    expected: Mapping[str, Any],
) -> None:
    for record in store.read_iterations():
        metadata = record.get("runner_metadata")
        if not isinstance(metadata, dict) or not metadata:
            continue
        mismatches = [
            field
            for field in IMMUTABLE_LINEAGE_FIELDS
            if metadata.get(field) is not None
            and metadata.get(field) != expected.get(field)
        ]
        if mismatches:
            raise RuntimeError(
                "legacy run evidence does not match current immutable inputs: "
                + ", ".join(sorted(mismatches))
            )
