"""Canonical hashes for immutable benchmark and preprocessing inputs."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import data as benchmark_data

from .contracts import BenchmarkContract


def normalized_file_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def candidate_code_sha256(candidate: Callable[..., Any]) -> str:
    """Hash the callable and every model source file it explicitly routes to."""
    source = inspect.getsourcefile(candidate)
    sources: list[str] = []
    if source and Path(source).is_file():
        sources.append(normalized_file_sha256(Path(source)))
    module = inspect.getmodule(candidate)
    for routed_path in getattr(module, "ROUTED_SOURCE_FILES", ()):
        path = Path(routed_path)
        if path.is_file():
            sources.append(normalized_file_sha256(path))
    if not sources:
        identity = (
            f"{getattr(candidate, '__module__', '')}:"
            f"{getattr(candidate, '__qualname__', '')}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return hashlib.sha256(":".join(sources).encode("ascii")).hexdigest()


def dataset_sha256(data_dir: Path) -> str:
    paths = [
        data_dir / "video_features_basic_pure.csv",
        data_dir / "log_standard_4_08_to_4_21_pure.csv",
        data_dir / "log_standard_4_22_to_5_08_pure.csv",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("benchmark dataset is incomplete: " + ", ".join(missing))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def current_benchmark_lineage(contract: BenchmarkContract) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    evaluator_sha256 = normalized_file_sha256(root / "evaluate.py")
    if evaluator_sha256 != contract.evaluator_sha256:
        raise RuntimeError(
            "actual evaluate.py does not match the organizer evaluator contract"
        )
    canonical_preprocessing_sha256 = normalized_file_sha256(root / "data.py")
    staging_code_sha256 = normalized_file_sha256(
        root / "research_agent" / "data_boundary.py"
    )
    preprocessing_sha256 = hashlib.sha256(
        f"{canonical_preprocessing_sha256}:{staging_code_sha256}".encode("ascii")
    ).hexdigest()
    feature_schema_sha256 = hashlib.sha256(
        json.dumps(list(benchmark_data.FIELDS), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    comparison = {
        "data_sha256": dataset_sha256(Path(contract.data_dir)),
        "evaluator_sha256": evaluator_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "staging_code_sha256": staging_code_sha256,
        "feature_schema_sha256": feature_schema_sha256,
    }
    comparison["comparison_group_id"] = hashlib.sha256(
        json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return comparison


def lineage_fingerprint(lineage: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(lineage), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def raw_file_sha256(path: Path) -> str:
    """Raw-byte hash for generated binary artifacts."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_descriptor(
    checkpoint_path: Path,
    metadata: Mapping[str, Any],
    *,
    store_root: Path | None = None,
) -> dict[str, Any]:
    """Identify a bundle by content, not by a mutable filesystem path.

    A path can be moved, replaced or rewritten between certification and
    finalization. Binding the certificate to size + SHA-256 + the recorded
    validation-score hash means a substituted bundle cannot be certified.
    """
    resolved = Path(checkpoint_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"bundle is missing: {resolved}")
    descriptor: dict[str, Any] = {
        "sha256": raw_file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
        "schema_version": metadata.get("checkpoint_schema_version"),
        "model_kind": metadata.get("checkpoint_model_kind"),
        "seed": metadata.get("seed"),
        "validation_score_sha256": metadata.get("validation_score_sha256"),
        "model_code_sha256": metadata.get("model_code_sha256"),
        "comparison_group_id": metadata.get("comparison_group_id"),
    }
    if store_root is not None:
        try:
            descriptor["path_relative_to_store"] = str(
                resolved.relative_to(Path(store_root).resolve())
            )
        except ValueError as exc:
            raise ValueError("bundle lies outside the artifact store") from exc
    return descriptor


def verify_bundle_descriptor(checkpoint_path: Path, descriptor: Mapping[str, Any]) -> None:
    """Fail closed if the bundle on disk is not the certified one."""
    resolved = Path(checkpoint_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"certified bundle is missing: {resolved}")
    if resolved.stat().st_size != descriptor.get("size_bytes"):
        raise RuntimeError("certified bundle size changed")
    if raw_file_sha256(resolved) != descriptor.get("sha256"):
        raise RuntimeError("certified bundle content changed")
    if descriptor.get("schema_version") != 2:
        raise RuntimeError("certified bundle is not a schema-version-2 inference bundle")
