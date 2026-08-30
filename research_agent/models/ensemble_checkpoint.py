"""Safe, self-describing persistence for a fitted rank ensemble.

The archive deliberately contains only non-object NumPy arrays. A scalar JSON
manifest names every other array and binds it to its dtype, shape, and digest;
loading fails closed if the archive and manifest do not agree exactly.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 2
MODEL_KIND = "fm_rank_ensemble"
_MANIFEST_KEY = "manifest_json"
_WEIGHTS_KEY = "weights"
_ALLOWED_STATE_KINDS = frozenset("biuf")
_ALLOWED_ENCODER_KINDS = frozenset("biufUS")


@dataclass(frozen=True)
class EnsembleCheckpoint:
    path: Path
    manifest: dict[str, Any]
    weights: np.ndarray
    states: tuple[dict[str, np.ndarray], ...]
    encoders: tuple[dict[str, np.ndarray], ...]


def _json_safe_copy(value: Any, *, context: str) -> Any:
    """Return a JSON-native copy, rejecting lossy or non-finite values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{context} keys must be non-empty strings")
            result[key] = _json_safe_copy(item, context=f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_safe_copy(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{context} contains non-JSON value {type(value).__name__}")


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _array_metadata(array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_digest(array),
    }


def _validated_array(
    value: Any,
    *,
    context: str,
    allowed_kinds: frozenset[str],
    allow_empty: bool,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.fields is not None:
        raise ValueError(f"{context} cannot use object or structured dtype")
    if array.dtype.kind not in allowed_kinds:
        raise ValueError(f"{context} has unsupported dtype {array.dtype}")
    if array.ndim > 8:
        raise ValueError(f"{context} has an unsupported number of dimensions")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{context} cannot be empty")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"{context} must contain only finite values")
    # ``np.ascontiguousarray`` promotes a scalar from ``()`` to ``(1,)``.
    # FM bias terms are intentionally scalar and their exact shape is part of
    # the checkpoint contract, so preserve zero-dimensional arrays.
    if array.ndim == 0:
        return array.copy(order="C")
    return np.ascontiguousarray(array).copy()


def _named_array_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    for name in value:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context} names must be non-empty strings")
    return value


def _add_arrays(
    archive_arrays: dict[str, np.ndarray],
    values: Mapping[str, Any],
    *,
    archive_prefix: str,
    context: str,
    allowed_kinds: frozenset[str],
    allow_empty: bool,
) -> dict[str, str]:
    references: dict[str, str] = {}
    for index, (logical_name, value) in enumerate(values.items()):
        archive_key = f"{archive_prefix}_{index}"
        archive_arrays[archive_key] = _validated_array(
            value,
            context=f"{context}.{logical_name}",
            allowed_kinds=allowed_kinds,
            allow_empty=allow_empty,
        )
        references[logical_name] = archive_key
    return references


def _require_int(value: Any, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _validate_state_shapes(descriptor: Mapping[str, Any], state: Mapping[str, np.ndarray]) -> None:
    feature_dim = _require_int(
        descriptor.get("feature_dim"), context="member feature_dim", minimum=1
    )
    embedding_dim = _require_int(
        descriptor.get("embedding_dim"), context="member embedding_dim", minimum=1
    )
    kind = descriptor.get("kind")
    if kind == "numpy":
        if set(state) != {"V", "W", "b"}:
            raise ValueError("numpy FM state must contain exactly V, W, and b")
        expected = {"V": (feature_dim, embedding_dim), "W": (feature_dim,), "b": ()}
    elif kind == "torch":
        if set(state) != {"embedding.weight", "linear.weight", "bias"}:
            raise ValueError(
                "torch FM state must contain exactly embedding.weight, linear.weight, and bias"
            )
        expected = {
            "embedding.weight": (feature_dim, embedding_dim),
            "linear.weight": (feature_dim, 1),
            "bias": (),
        }
    else:
        raise ValueError("ensemble member kind must be 'numpy' or 'torch'")
    for name, shape in expected.items():
        if state[name].shape != shape:
            raise ValueError(
                f"ensemble member state {name} has shape {state[name].shape}, expected {shape}"
            )


def _validate_weights(weights: np.ndarray, member_count: int) -> None:
    if weights.dtype != np.dtype(np.float64):
        raise ValueError("ensemble checkpoint weights must use float64")
    if weights.ndim != 1 or len(weights) != member_count:
        raise ValueError("ensemble checkpoint weights must be one-dimensional and aligned")
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("active ensemble weights must be positive and finite")
    if not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("active ensemble weights must sum to one")


def _validate_digest(value: Any, *, context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")


def _validate_member_descriptor(member: Mapping[str, Any]) -> None:
    groups = member.get("groups")
    if (
        not isinstance(groups, list)
        or any(not isinstance(group, str) or not group for group in groups)
        or len(groups) != len(set(groups))
    ):
        raise ValueError("ensemble member groups must contain unique non-empty strings")
    if member.get("loss") not in {"pointwise", "pairwise", "listwise"}:
        raise ValueError("ensemble member loss is unsupported")
    _require_int(member.get("feature_dim"), context="member feature_dim", minimum=1)
    _require_int(member.get("embedding_dim"), context="member embedding_dim", minimum=1)
    epochs_run = _require_int(
        member.get("epochs_run"), context="member epochs_run", minimum=1
    )
    best_epoch = _require_int(
        member.get("best_epoch"), context="member best_epoch", minimum=1
    )
    if best_epoch > epochs_run:
        raise ValueError("ensemble member best_epoch exceeds epochs_run")
    primary = member.get("primary")
    if (
        isinstance(primary, bool)
        or not isinstance(primary, (int, float))
        or not math.isfinite(float(primary))
        or not 0.0 <= float(primary) <= 1.0
    ):
        raise ValueError("ensemble member primary must be finite and in [0, 1]")
    encoder = member.get("encoder")
    if not isinstance(encoder, dict):
        raise ValueError("ensemble checkpoint member has no encoder descriptor")
    _json_safe_copy(encoder, context="member encoder")


def _descriptor_from_member(
    member: Mapping[str, Any], *, index: int, arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    raw_state = _named_array_mapping(member.get("state"), context=f"member {index} state")
    if not raw_state:
        raise ValueError("every active ensemble member needs a fitted state")
    state_arrays = _add_arrays(
        arrays,
        raw_state,
        archive_prefix=f"member_{index}_state",
        context=f"member {index} state",
        allowed_kinds=_ALLOWED_STATE_KINDS,
        allow_empty=False,
    )

    raw_encoder_manifest = member.get("encoder_manifest")
    if not isinstance(raw_encoder_manifest, Mapping):
        raise ValueError("every active ensemble member needs an encoder manifest")
    encoder = _json_safe_copy(raw_encoder_manifest, context=f"member {index} encoder")
    if "encoder_arrays" in encoder:
        raise ValueError("encoder_arrays is reserved for checkpoint array references")
    raw_encoder_arrays = _named_array_mapping(
        member.get("encoder_arrays"), context=f"member {index} encoder arrays"
    )
    encoder["encoder_arrays"] = _add_arrays(
        arrays,
        raw_encoder_arrays,
        archive_prefix=f"member_{index}_encoder",
        context=f"member {index} encoder",
        allowed_kinds=_ALLOWED_ENCODER_KINDS,
        allow_empty=True,
    )

    groups = member.get("groups")
    if (
        not isinstance(groups, (list, tuple))
        or any(not isinstance(group, str) or not group for group in groups)
        or len(groups) != len(set(groups))
    ):
        raise ValueError("ensemble member groups must contain unique non-empty strings")
    name = member.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("ensemble member name must be a non-empty string")
    loss = member.get("loss")
    if loss not in {"pointwise", "pairwise", "listwise"}:
        raise ValueError("ensemble member loss is unsupported")
    epochs_run = _require_int(member.get("epochs_run"), context="member epochs_run", minimum=1)
    best_epoch = _require_int(member.get("best_epoch"), context="member best_epoch", minimum=1)
    if best_epoch > epochs_run:
        raise ValueError("ensemble member best_epoch exceeds epochs_run")
    primary = member.get("primary")
    if (
        isinstance(primary, bool)
        or not isinstance(primary, (int, float))
        or not math.isfinite(float(primary))
        or not 0.0 <= float(primary) <= 1.0
    ):
        raise ValueError("ensemble member primary must be finite and in [0, 1]")
    descriptor = {
        "name": name,
        "kind": member.get("kind"),
        "groups": list(groups),
        "loss": loss,
        "embedding_dim": _require_int(member.get("embedding_dim"), context="member embedding_dim", minimum=1),
        "feature_dim": _require_int(member.get("feature_dim"), context="member feature_dim", minimum=1),
        "primary": float(primary),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "state_arrays": state_arrays,
        "encoder": encoder,
    }
    _validate_state_shapes(descriptor, {name: arrays[key] for name, key in state_arrays.items()})
    return descriptor


def write_ensemble_checkpoint(
    path: Path,
    *,
    seed: int,
    config: Mapping[str, Any],
    trained_members: Sequence[str],
    active_members: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    validation_score_sha256: str,
    validation_primary: float,
    lineage: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write fitted states and encoders in a non-pickle NPZ."""
    if not active_members:
        raise ValueError("ensemble checkpoint needs at least one active member")
    arrays: dict[str, np.ndarray] = {
        _WEIGHTS_KEY: _validated_array(
            np.asarray(weights, dtype=np.float64),
            context="ensemble weights",
            allowed_kinds=frozenset("f"),
            allow_empty=False,
        )
    }
    descriptors = [
        _descriptor_from_member(member, index=index, arrays=arrays)
        for index, member in enumerate(active_members)
    ]
    member_names = [descriptor["name"] for descriptor in descriptors]
    if len(member_names) != len(set(member_names)):
        raise ValueError("active ensemble member names must be unique")
    trained_names = list(trained_members)
    if (
        any(not isinstance(name, str) or not name for name in trained_names)
        or len(trained_names) != len(set(trained_names))
    ):
        raise ValueError("trained ensemble member names must be unique non-empty strings")
    if any(name not in trained_names for name in member_names):
        raise ValueError("active ensemble members must occur in trained_members")
    active_set = set(member_names)
    if member_names != [name for name in trained_names if name in active_set]:
        raise ValueError("active ensemble members must preserve trained member order")
    _validate_weights(arrays[_WEIGHTS_KEY], len(member_names))
    _validate_digest(validation_score_sha256, context="validation_score_sha256")
    if (
        isinstance(validation_primary, bool)
        or not isinstance(validation_primary, (int, float))
        or not math.isfinite(float(validation_primary))
        or not 0.0 <= float(validation_primary) <= 1.0
    ):
        raise ValueError("validation_primary must be finite and in [0, 1]")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "seed": _require_int(seed, context="seed", minimum=0),
        "config": _json_safe_copy(config, context="config"),
        "trained_members": trained_names,
        "active_members": member_names,
        "validation_score_sha256": validation_score_sha256,
        "validation_primary": float(validation_primary),
        # Provenance: binds this bundle to the exact inputs and code that made it.
        "lineage": _json_safe_copy(dict(lineage or {}), context="lineage"),
        "runtime": _json_safe_copy(dict(runtime or {}), context="runtime"),
        "members": descriptors,
        "array_metadata": {key: _array_metadata(array) for key, array in arrays.items()},
    }
    _validate_manifest(manifest)
    arrays[_MANIFEST_KEY] = np.asarray(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        # numpy stubs type the second positional as `bool`; **arrays is correct
        # at runtime and is how savez_compressed is designed to be called.
        np.savez_compressed(temporary_path, **arrays)  # type: ignore[arg-type]
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"checkpoint manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _manifest_from_array(array: np.ndarray) -> dict[str, Any]:
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError("checkpoint manifest must be a scalar string array")
    raw = array.item()
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    manifest = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    return manifest


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "model_kind", "seed", "config", "trained_members",
        "active_members", "validation_score_sha256", "validation_primary",
        "members", "array_metadata", "lineage", "runtime",
    }
    if set(manifest) != required:
        missing = sorted(required - set(manifest))
        extra = sorted(set(manifest) - required)
        raise ValueError(
            "ensemble checkpoint manifest fields do not match the schema"
            + (f"; missing={missing}" if missing else "")
            + (f"; unexpected={extra}" if extra else "")
        )
    for field in ("lineage", "runtime"):
        if not isinstance(manifest.get(field), dict):
            raise ValueError(f"ensemble checkpoint {field} must be an object")
        _json_safe_copy(manifest[field], context=field)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported ensemble checkpoint schema")
    if manifest.get("model_kind") != MODEL_KIND:
        raise ValueError("unexpected ensemble checkpoint model kind")
    _require_int(manifest.get("seed"), context="seed", minimum=0)
    if not isinstance(manifest.get("config"), dict):
        raise ValueError("ensemble checkpoint config must be an object")
    _json_safe_copy(manifest["config"], context="config")
    _validate_digest(
        manifest.get("validation_score_sha256"), context="validation_score_sha256"
    )
    validation_primary = manifest.get("validation_primary")
    if (
        isinstance(validation_primary, bool)
        or not isinstance(validation_primary, (int, float))
        or not math.isfinite(float(validation_primary))
        or not 0.0 <= float(validation_primary) <= 1.0
    ):
        raise ValueError("validation_primary must be finite and in [0, 1]")
    names = manifest.get("active_members")
    trained = manifest.get("trained_members")
    members = manifest.get("members")
    if not isinstance(names, list) or not names:
        raise ValueError("ensemble checkpoint has no active members")
    if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        raise ValueError("active ensemble member names must be unique non-empty strings")
    if not isinstance(trained, list) or any(
        not isinstance(name, str) or not name for name in trained
    ) or len(trained) != len(set(trained)):
        raise ValueError("trained ensemble member names must be unique non-empty strings")
    active_set = set(names)
    if names != [name for name in trained if name in active_set]:
        raise ValueError("active ensemble members must preserve trained member order")
    if not isinstance(members, list) or len(members) != len(names):
        raise ValueError("ensemble checkpoint members are misaligned")
    if [member.get("name") if isinstance(member, dict) else None for member in members] != names:
        raise ValueError("ensemble member descriptor order does not match weights")


def _references_from_member(member: Any) -> tuple[Mapping[str, str], Mapping[str, str]]:
    if not isinstance(member, dict):
        raise ValueError("ensemble member descriptor must be an object")
    required = {
        "name", "kind", "groups", "loss", "embedding_dim", "feature_dim",
        "primary", "epochs_run", "best_epoch", "state_arrays", "encoder",
    }
    if set(member) != required:
        raise ValueError("ensemble member descriptor fields do not match the schema")
    _validate_member_descriptor(member)
    state_refs = member.get("state_arrays")
    encoder = member.get("encoder")
    if not isinstance(state_refs, dict) or not state_refs:
        raise ValueError("ensemble checkpoint member has no state arrays")
    if not isinstance(encoder, dict) or not isinstance(encoder.get("encoder_arrays"), dict):
        raise ValueError("ensemble checkpoint member has no encoder descriptor")
    encoder_refs = encoder["encoder_arrays"]
    for context, references in (("state", state_refs), ("encoder", encoder_refs)):
        if any(not isinstance(name, str) or not name for name in references):
            raise ValueError(f"ensemble {context} names must be non-empty strings")
        if any(not isinstance(key, str) or not key for key in references.values()):
            raise ValueError(f"ensemble {context} array references must be non-empty strings")
    return state_refs, encoder_refs


def _verify_archive_array(
    archive: Any,
    key: str,
    metadata: Mapping[str, Any],
    *,
    allowed_kinds: frozenset[str],
    allow_empty: bool,
) -> np.ndarray:
    if key not in archive.files:
        raise ValueError(f"ensemble checkpoint is missing {key}")
    array = _validated_array(
        archive[key], context=f"checkpoint array {key}",
        allowed_kinds=allowed_kinds, allow_empty=allow_empty,
    )
    if set(metadata) != {"dtype", "shape", "sha256"}:
        raise ValueError(f"checkpoint metadata for {key} is malformed")
    if metadata.get("dtype") != array.dtype.str:
        raise ValueError(f"checkpoint array {key} dtype does not match its manifest")
    if metadata.get("shape") != list(array.shape):
        raise ValueError(f"checkpoint array {key} shape does not match its manifest")
    if metadata.get("sha256") != _array_digest(array):
        raise ValueError(f"checkpoint array {key} digest does not match its manifest")
    return array


def load_ensemble_checkpoint(path: Path) -> EnsembleCheckpoint:
    """Load and fully validate a checkpoint with pickle disabled."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if _MANIFEST_KEY not in archive.files or _WEIGHTS_KEY not in archive.files:
                raise ValueError("ensemble checkpoint is missing its manifest or weights")
            manifest = _manifest_from_array(archive[_MANIFEST_KEY])
            _validate_manifest(manifest)
            metadata = manifest["array_metadata"]
            if not isinstance(metadata, dict):
                raise ValueError("checkpoint array metadata must be an object")
            state_references: list[Mapping[str, str]] = []
            encoder_references: list[Mapping[str, str]] = []
            all_references = {_WEIGHTS_KEY}
            for member in manifest["members"]:
                state_refs, encoder_refs = _references_from_member(member)
                for archive_key in (*state_refs.values(), *encoder_refs.values()):
                    if archive_key in all_references:
                        raise ValueError("checkpoint array is referenced more than once")
                    all_references.add(archive_key)
                state_references.append(state_refs)
                encoder_references.append(encoder_refs)
            expected_archive_keys = all_references | {_MANIFEST_KEY}
            if set(archive.files) != expected_archive_keys:
                missing = expected_archive_keys - set(archive.files)
                extra = set(archive.files) - expected_archive_keys
                detail = []
                if missing:
                    detail.append("missing: " + ", ".join(sorted(missing)))
                if extra:
                    detail.append("unreferenced: " + ", ".join(sorted(extra)))
                raise ValueError("ensemble checkpoint array set mismatch (" + "; ".join(detail) + ")")
            if set(metadata) != all_references:
                raise ValueError("checkpoint array metadata does not match referenced arrays")

            weights = _verify_archive_array(
                archive, _WEIGHTS_KEY, metadata[_WEIGHTS_KEY],
                allowed_kinds=frozenset("f"), allow_empty=False,
            )
            states: list[dict[str, np.ndarray]] = []
            encoders: list[dict[str, np.ndarray]] = []
            for descriptor, state_refs, encoder_refs in zip(
                manifest["members"], state_references, encoder_references
            ):
                state = {
                    logical_name: _verify_archive_array(
                        archive, archive_key, metadata[archive_key],
                        allowed_kinds=_ALLOWED_STATE_KINDS, allow_empty=False,
                    )
                    for logical_name, archive_key in state_refs.items()
                }
                encoder_arrays = {
                    logical_name: _verify_archive_array(
                        archive, archive_key, metadata[archive_key],
                        allowed_kinds=_ALLOWED_ENCODER_KINDS, allow_empty=True,
                    )
                    for logical_name, archive_key in encoder_refs.items()
                }
                _validate_state_shapes(descriptor, state)
                states.append(state)
                encoders.append(encoder_arrays)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ensemble checkpoint: {exc}") from exc

    names = manifest["active_members"]
    _validate_weights(weights, len(names))
    return EnsembleCheckpoint(path, dict(manifest), weights.copy(), tuple(states), tuple(encoders))
