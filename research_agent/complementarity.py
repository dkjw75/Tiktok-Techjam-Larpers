"""Leakage-safe validation of an incumbent ensemble plus one candidate.

The evaluator deliberately has no persistence or data-loading behavior.  A caller
supplies four row-aligned validation vectors and persists the returned partition
fingerprint alongside its normal experiment evidence.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np

from .contracts import BENCHMARK_CONTRACT
from .metrics import evaluate_predictions


DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_PARTITION_SALT = "kuairand-complementarity-v1"


class ComplementarityValidationError(ValueError):
    """Raised when complementarity evidence cannot be produced safely."""


@dataclass(frozen=True)
class UserPartition:
    """A deterministic, user-disjoint fitting and reporting partition."""

    fit_user_ids: tuple[Any, ...]
    held_user_ids: tuple[Any, ...]
    fingerprint: str


@dataclass(frozen=True)
class ComplementarityResult:
    """Held-user evidence for adding one candidate to an incumbent ensemble."""

    incumbent_held_primary: float
    blended_held_primary: float
    ensemble_delta_if_added: float
    alpha: float
    fit_blended_primary: float
    rows: int
    users: int
    fit_rows: int
    fit_users: int
    held_rows: int
    held_users: int
    partition_fingerprint: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "incumbent_held_primary": self.incumbent_held_primary,
            "blended_held_primary": self.blended_held_primary,
            "ensemble_delta_if_added": self.ensemble_delta_if_added,
            "alpha": self.alpha,
            "fit_blended_primary": self.fit_blended_primary,
            "rows": self.rows,
            "users": self.users,
            "fit_rows": self.fit_rows,
            "fit_users": self.fit_users,
            "held_rows": self.held_rows,
            "held_users": self.held_users,
            "partition_fingerprint": self.partition_fingerprint,
        }


def deterministic_user_partition(
    user_ids: Sequence[Any],
    *,
    salt: str = DEFAULT_PARTITION_SALT,
) -> UserPartition:
    """Split unique users deterministically, independently of row order or labels."""
    if not isinstance(salt, str) or not salt:
        raise ComplementarityValidationError("partition salt must be a non-empty string")
    users = _validated_users(user_ids)
    by_token: dict[bytes, Any] = {}
    for user in users:
        token = _user_token(user)
        previous = by_token.setdefault(token, user)
        if previous != user:
            raise ComplementarityValidationError("user identity encoding collision")
    if len(by_token) < 2:
        raise ComplementarityValidationError(
            "complementarity evaluation requires at least two distinct users"
        )

    salt_bytes = salt.encode("utf-8")
    ordered = sorted(
        by_token,
        key=lambda token: (hashlib.sha256(salt_bytes + b"\0" + token).digest(), token),
    )
    fit_count = (len(ordered) + 1) // 2
    fit_tokens = ordered[:fit_count]
    held_tokens = ordered[fit_count:]
    fingerprint = _partition_fingerprint(salt_bytes, fit_tokens, held_tokens)
    return UserPartition(
        fit_user_ids=tuple(by_token[token] for token in fit_tokens),
        held_user_ids=tuple(by_token[token] for token in held_tokens),
        fingerprint=fingerprint,
    )


def evaluate_complementarity(
    user_ids: Sequence[Any],
    labels: Sequence[Any],
    incumbent_scores: Sequence[Any],
    candidate_scores: Sequence[Any],
    *,
    split: str = BENCHMARK_CONTRACT.selection_split,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    partition_salt: str = DEFAULT_PARTITION_SALT,
    expected_partition_fingerprint: str | None = None,
) -> ComplementarityResult:
    """Fit a coarse ``E + candidate`` blend on fit users and report held users.

    ``alpha`` is the candidate weight, so the evaluated score is
    ``(1 - alpha) * incumbent + alpha * candidate``.  Alpha selection calls the
    unchanged official primary metric only on fit users.  Held users are touched
    only after the winning alpha has been frozen.
    """
    if split != BENCHMARK_CONTRACT.selection_split:
        raise ComplementarityValidationError(
            "complementarity evaluation is validation-only; test is forbidden"
        )

    users = _validated_users(user_ids)
    label_values = _numeric_vector("labels", labels)
    incumbent_values = _numeric_vector("incumbent_scores", incumbent_scores)
    candidate_values = _numeric_vector("candidate_scores", candidate_scores)
    if not (
        len(users)
        == len(label_values)
        == len(incumbent_values)
        == len(candidate_values)
    ):
        raise ComplementarityValidationError(
            "user_ids, labels, incumbent_scores, and candidate_scores must be row-aligned"
        )
    if not users:
        raise ComplementarityValidationError("cannot evaluate empty predictions")
    if not np.isfinite(label_values).all() or not np.isin(label_values, (0.0, 1.0)).all():
        raise ComplementarityValidationError("labels must be finite binary long_view values")
    if not np.isfinite(incumbent_values).all() or not np.isfinite(candidate_values).all():
        raise ComplementarityValidationError("incumbent and candidate scores must be finite")

    alpha_values = _validated_alphas(alphas)
    partition = deterministic_user_partition(users, salt=partition_salt)
    if (
        expected_partition_fingerprint is not None
        and expected_partition_fingerprint != partition.fingerprint
    ):
        raise ComplementarityValidationError(
            "partition fingerprint does not match the persisted fitting partition"
        )

    fit_tokens = {_user_token(user) for user in partition.fit_user_ids}
    fit_mask = np.asarray([_user_token(user) in fit_tokens for user in users], dtype=bool)
    held_mask = ~fit_mask
    if not fit_mask.any() or not held_mask.any():
        raise ComplementarityValidationError("fit and held partitions must both contain rows")
    fit_users = [user for user, selected in zip(users, fit_mask) if selected]
    held_users = [user for user, selected in zip(users, held_mask) if selected]
    if set(fit_users) & set(held_users):
        raise ComplementarityValidationError("fit and held user partitions overlap")

    best_alpha: float | None = None
    best_fit_primary = -math.inf
    for alpha in alpha_values:
        blended = _blend(incumbent_values[fit_mask], candidate_values[fit_mask], alpha)
        fit_primary = _official_primary(
            fit_users, label_values[fit_mask], blended.tolist(), split=split
        )
        # Alphas are sorted. Strict improvement makes exact ties conservatively
        # retain the smaller candidate weight.
        if fit_primary > best_fit_primary:
            best_alpha = alpha
            best_fit_primary = fit_primary
    if best_alpha is None or not math.isfinite(best_fit_primary):
        raise ComplementarityValidationError("alpha selection produced no finite primary")

    incumbent_held = _official_primary(
        held_users,
        label_values[held_mask],
        incumbent_values[held_mask].tolist(),
        split=split,
    )
    blended_held = _official_primary(
        held_users,
        label_values[held_mask],
        _blend(
            incumbent_values[held_mask], candidate_values[held_mask], best_alpha
        ).tolist(),
        split=split,
    )
    delta = blended_held - incumbent_held
    if not all(math.isfinite(value) for value in (incumbent_held, blended_held, delta)):
        raise ComplementarityValidationError("official held metrics must be finite")

    return ComplementarityResult(
        incumbent_held_primary=float(incumbent_held),
        blended_held_primary=float(blended_held),
        ensemble_delta_if_added=float(delta),
        alpha=float(best_alpha),
        fit_blended_primary=float(best_fit_primary),
        rows=len(users),
        users=len(set(users)),
        fit_rows=int(fit_mask.sum()),
        fit_users=len(set(fit_users)),
        held_rows=int(held_mask.sum()),
        held_users=len(set(held_users)),
        partition_fingerprint=partition.fingerprint,
    )


def _validated_users(values: Sequence[Any]) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise ComplementarityValidationError("user_ids must be a one-dimensional vector")
    try:
        users = list(values)
    except TypeError as exc:
        raise ComplementarityValidationError("user_ids must be a one-dimensional vector") from exc
    for user in users:
        _user_token(user)
    return users


def _numeric_vector(name: str, values: Sequence[Any]) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ComplementarityValidationError(f"{name} must be one-dimensional")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ComplementarityValidationError(f"{name} must be numeric") from exc
    if array.ndim != 1:
        raise ComplementarityValidationError(f"{name} must be one-dimensional")
    return array


def _validated_alphas(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ComplementarityValidationError("alphas must be a numeric sequence")
    try:
        converted = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ComplementarityValidationError("alphas must be a numeric sequence") from exc
    if not converted:
        raise ComplementarityValidationError("alphas must not be empty")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in converted):
        raise ComplementarityValidationError("alphas must be finite and within [0, 1]")
    if len(set(converted)) != len(converted):
        raise ComplementarityValidationError("alphas must not contain duplicates")
    if 0.0 not in converted:
        raise ComplementarityValidationError("alphas must include 0.0 as the incumbent baseline")
    return tuple(sorted(converted))


def _blend(incumbent: np.ndarray, candidate: np.ndarray, alpha: float) -> np.ndarray:
    blended = (1.0 - alpha) * incumbent + alpha * candidate
    if not np.isfinite(blended).all():
        raise ComplementarityValidationError("blended scores must remain finite")
    return blended


def _official_primary(
    user_ids: Sequence[Any],
    labels: Sequence[Any],
    scores: Sequence[Any],
    *,
    split: str,
) -> float:
    result = evaluate_predictions(user_ids, labels, scores, split=split)
    if result.evaluator_sha256 != BENCHMARK_CONTRACT.evaluator_sha256:
        raise ComplementarityValidationError(
            "official evaluator fingerprint does not match the benchmark contract"
        )
    return result.primary


def _user_token(user: Any) -> bytes:
    if isinstance(user, bool):
        raise ComplementarityValidationError("boolean user IDs are not supported")
    if isinstance(user, Integral):
        return b"int:" + str(int(user)).encode("ascii")
    if isinstance(user, str):
        return b"str:" + user.encode("utf-8")
    raise ComplementarityValidationError(
        "user IDs must be integer or string scalars for deterministic fingerprinting"
    )


def _partition_fingerprint(
    salt: bytes, fit_tokens: Sequence[bytes], held_tokens: Sequence[bytes]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"complementarity-user-partition-v1\0")
    digest.update(len(salt).to_bytes(8, "big"))
    digest.update(salt)
    for name, tokens in ((b"fit", fit_tokens), (b"held", held_tokens)):
        digest.update(name + b"\0")
        digest.update(len(tokens).to_bytes(8, "big"))
        for token in tokens:
            digest.update(len(token).to_bytes(8, "big"))
            digest.update(token)
    return digest.hexdigest()
