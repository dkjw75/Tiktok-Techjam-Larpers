"""Within-user rank ensemble of diverse FM members.

Ported from the validated manual harness (`manual/exp_ensemble.py`). Members are
deliberately heterogeneous: three differ by feature set, two by training
objective. The measured gain comes from their disagreement, not from any single
member's accuracy -- the two weakest members are the ones that moved the blend
over the evidence threshold.

Every member consumes rows staged by `data_boundary`, which are produced by the
same canonical split logic as `data.py`. No raw CSV is read here.
"""
from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..metrics import evaluate_predictions
from ..runner import CandidateOutput, PreparedData
from .ensemble_checkpoint import (
    load_ensemble_checkpoint,
    write_ensemble_checkpoint,
)

# Canonical staged-row positions (see data_boundary.CANONICAL_ROW_WIDTH).
D_DATE, D_USER, D_VID, D_AUTH, D_TAB, D_DUR, D_Y, D_HOUR, D_PLAY, D_MUSIC, D_TAG, D_UP = range(12)

SMOOTH_PRIOR = 20.0
N_BINS = 20

# Member catalogue. Each entry is (feature groups, training objective).
# "" groups means the organizer's 5 base fields only.
MEMBERS: dict[str, tuple[str, str]] = {
    "fm": ("", "pointwise"),
    "watch": ("watch", "pointwise"),
    "item": ("item,author,ua", "pointwise"),
    "watchtime": ("watch,time", "pointwise"),
    "listwise": ("", "listwise"),
    "pairwise": ("", "pairwise"),
    "fm_k8": ("", "pointwise"),
    "fm_k32": ("", "pointwise"),
}
# Organizer measured embedding size as flat standalone (0.5895/0.5902/0.5887).
# A different k is still a different error surface, and decorrelation is what
# the blend pays for -- so these earn a slot as members, not as replacements.
MEMBER_K: dict[str, int] = {"fm_k8": 8, "fm_k32": 32}
DEFAULT_MEMBERS = ("fm", "watch", "item", "watchtime", "listwise", "pairwise")
# Named sets the Search Controller may select between as its one controlled factor.
MEMBER_SETS: dict[str, tuple[str, ...]] = {
    "core4": ("fm", "watch", "item", "listwise"),
    "core5": ("fm", "watch", "item", "watchtime", "listwise"),
    "core6": DEFAULT_MEMBERS,
}
# fm_k8 / fm_k32 are retained as MEMBERS for reproducing past evidence, but are
# NOT offered as search-space sets: measured at 16628s and 13361s per low-
# fidelity screen for a +0.00006 gain, they exhausted the budget before any
# promotion could run.
WEIGHT_STEP = 0.2


# --------------------------------------------------------------- feature build
def _runtime_versions() -> dict[str, str]:
    """Recorded so a replay mismatch can be attributed to a library change."""
    import platform
    import sys

    versions = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    try:
        import torch

        versions["torch_version"] = str(torch.__version__)
    except ModuleNotFoundError:
        versions["torch_version"] = "unavailable"
    return versions


def _watch_ratio(row: Sequence[Any]) -> float:
    return min(row[D_PLAY] / row[D_DUR], 3.0) if row[D_DUR] > 0 else 0.0


class _RateTable:
    """Smoothed positive rate; train rows see only strictly earlier dates."""

    def __init__(self, keyfn):
        self.keyfn = keyfn
        self.pos: collections.Counter = collections.Counter()
        self.n: collections.Counter = collections.Counter()
        self.total_pos = 0
        self.total_n = 0

    def observe(self, row):
        key = self.keyfn(row)
        self.n[key] += 1
        self.pos[key] += row[D_Y]
        self.total_n += 1
        self.total_pos += row[D_Y]

    def rate(self, row):
        prior = self.total_pos / self.total_n if self.total_n else 0.0
        key = self.keyfn(row)
        return (self.pos[key] + SMOOTH_PRIOR * prior) / (self.n[key] + SMOOTH_PRIOR)

    def count(self, row):
        return float(self.n[self.keyfn(row)])


class _MeanTable:
    def __init__(self, keyfn, valfn):
        self.keyfn, self.valfn = keyfn, valfn
        self.total: dict[Any, float] = collections.defaultdict(float)
        self.n: collections.Counter = collections.Counter()
        self.grand_total = 0.0
        self.grand_n = 0

    def observe(self, row):
        key, value = self.keyfn(row), self.valfn(row)
        self.total[key] += value
        self.n[key] += 1
        self.grand_total += value
        self.grand_n += 1

    def rate(self, row):
        prior = self.grand_total / self.grand_n if self.grand_n else 0.0
        key = self.keyfn(row)
        return (self.total[key] + SMOOTH_PRIOR * prior) / (self.n[key] + SMOOTH_PRIOR)


@dataclass
class _FittedEncoder:
    groups: tuple[str, ...]
    names: tuple[str, ...]
    tables: list[Any]
    kinds: tuple[str, ...]
    statistic_edges: tuple[np.ndarray, ...]
    duration_edges: np.ndarray
    vocabs: tuple[dict[str, int], ...]
    feature_dim: int


GROUPS = {
    "item": [("vid_rate", lambda: _RateTable(lambda r: r[D_VID]), "rate"),
             ("vid_cnt", lambda: _RateTable(lambda r: r[D_VID]), "count")],
    "author": [("auth_rate", lambda: _RateTable(lambda r: r[D_AUTH]), "rate"),
               ("auth_cnt", lambda: _RateTable(lambda r: r[D_AUTH]), "count")],
    "ua": [("ua_rate", lambda: _RateTable(lambda r: (r[D_USER], r[D_AUTH])), "rate"),
           ("ua_cnt", lambda: _RateTable(lambda r: (r[D_USER], r[D_AUTH])), "count")],
    "watch": [("u_watch", lambda: _MeanTable(lambda r: r[D_USER], _watch_ratio), "rate"),
              ("v_watch", lambda: _MeanTable(lambda r: r[D_VID], _watch_ratio), "rate")],
}


def _feature_specs(groups):
    return [spec for group in groups if group in GROUPS for spec in GROUPS[group]]


def _raw_fields(row, extra, *, groups, duration_edges):
    fields = [
        row[D_USER],
        row[D_VID],
        row[D_AUTH],
        row[D_TAB],
        str(int(np.searchsorted(duration_edges, row[D_DUR]))),
    ]
    fields += list(extra)
    if "time" in groups:
        age = row[D_DATE] - row[D_UP] if row[D_UP] else -1
        fields += [str(row[D_HOUR]), str(min(max(age, -1), 400) // 7)]
    return [str(value) for value in fields]


def _fit_member_encoder(train_rows, groups):
    """Fit every preprocessing object once and retain its exact inference state."""
    specs = _feature_specs(groups)
    names = tuple(name for name, _factory, _kind in specs)
    tables = [factory() for _name, factory, _kind in specs]
    kinds = tuple(kind for _name, _factory, kind in specs)

    def read(row):
        return [
            table.rate(row) if kind == "rate" else table.count(row)
            for table, kind in zip(tables, kinds)
        ]

    ordered = sorted(range(len(train_rows)), key=lambda index: train_rows[index][D_DATE])
    train_statistics: list[Any] = [None] * len(train_rows)
    i = 0
    while i < len(ordered):
        j = i
        while (
            j < len(ordered)
            and train_rows[ordered[j]][D_DATE] == train_rows[ordered[i]][D_DATE]
        ):
            j += 1
        for index in ordered[i:j]:
            train_statistics[index] = read(train_rows[index])
        for index in ordered[i:j]:
            for table in tables:
                table.observe(train_rows[index])
        i = j

    if names:
        statistics = np.asarray(train_statistics, dtype=np.float64)
        statistic_edges = tuple(
            np.unique(np.quantile(statistics[:, column], np.linspace(0, 1, N_BINS + 1)[1:-1]))
            for column in range(statistics.shape[1])
        )
        train_extra = [
            [
                str(int(np.searchsorted(statistic_edges[column], value)))
                for column, value in enumerate(row)
            ]
            for row in statistics
        ]
    else:
        statistic_edges = ()
        train_extra = [[] for _ in train_rows]
    duration_edges = np.quantile(
        [row[D_DUR] for row in train_rows], np.linspace(0, 1, 11)[1:-1]
    )
    raw_train = [
        _raw_fields(row, extra, groups=groups, duration_edges=duration_edges)
        for row, extra in zip(train_rows, train_extra)
    ]
    width = len(raw_train[0])
    vocabs: list[dict[str, int]] = [dict() for _ in range(width)]
    for values in raw_train:
        for column, value in enumerate(values):
            if value not in vocabs[column]:
                vocabs[column][value] = len(vocabs[column])
    feature_dim = sum(len(vocab) + 1 for vocab in vocabs)
    encoder = _FittedEncoder(
        groups=tuple(groups),
        names=names,
        tables=tables,
        kinds=kinds,
        statistic_edges=statistic_edges,
        duration_edges=np.asarray(duration_edges, dtype=np.float64),
        vocabs=tuple(vocabs),
        feature_dim=feature_dim,
    )
    return encoder, _transform_with_encoder(train_rows, encoder, extras=train_extra)


def _transform_with_encoder(rows, encoder: _FittedEncoder, *, extras=None):
    if extras is None:
        statistics = [
            [
                table.rate(row) if kind == "rate" else table.count(row)
                for table, kind in zip(encoder.tables, encoder.kinds)
            ]
            for row in rows
        ]
        extras = [
            [
                str(int(np.searchsorted(encoder.statistic_edges[column], value)))
                for column, value in enumerate(row)
            ]
            for row in statistics
        ] if encoder.names else [[] for _ in rows]
    offsets = np.cumsum(
        [0] + [len(vocab) + 1 for vocab in encoder.vocabs[:-1]]
    ).astype(np.int32)
    unknown = [len(vocab) for vocab in encoder.vocabs]
    matrix = np.empty((len(rows), len(encoder.vocabs)), dtype=np.int32)
    labels = np.empty(len(rows), dtype=np.float32)
    users = []
    for row_index, (row, extra) in enumerate(zip(rows, extras)):
        values = _raw_fields(
            row,
            extra,
            groups=encoder.groups,
            duration_edges=encoder.duration_edges,
        )
        for column, value in enumerate(values):
            matrix[row_index, column] = (
                encoder.vocabs[column].get(value, unknown[column]) + offsets[column]
            )
        labels[row_index] = row[D_Y]
        users.append(row[D_USER])
    return matrix, labels, users


def _encoder_checkpoint_parts(encoder: _FittedEncoder):
    arrays: dict[str, np.ndarray] = {
        "duration_edges": encoder.duration_edges,
    }
    manifest: dict[str, Any] = {
        "groups": list(encoder.groups),
        "feature_names": list(encoder.names),
        "kinds": list(encoder.kinds),
        "feature_dim": encoder.feature_dim,
        "duration_edges": "duration_edges",
        "statistic_edges": [],
        "vocabs": [],
        "tables": [],
    }
    for index, edges in enumerate(encoder.statistic_edges):
        name = f"statistic_edges_{index}"
        arrays[name] = np.asarray(edges, dtype=np.float64)
        manifest["statistic_edges"].append(name)
    for index, vocab in enumerate(encoder.vocabs):
        name = f"vocab_{index}"
        ordered = [token for token, _value in sorted(vocab.items(), key=lambda item: item[1])]
        arrays[name] = np.asarray(ordered, dtype=np.str_)
        manifest["vocabs"].append(name)
    for index, (table, feature_name) in enumerate(zip(encoder.tables, encoder.names)):
        counts = table.n
        keys = sorted(
            counts,
            key=lambda key: tuple(str(value) for value in key)
            if isinstance(key, tuple)
            else (str(key),),
        )
        key_width = len(keys[0]) if keys and isinstance(keys[0], tuple) else 1
        key_arrays = []
        for column in range(key_width):
            name = f"table_{index}_key_{column}"
            arrays[name] = np.asarray(
                [str(key[column] if isinstance(key, tuple) else key) for key in keys],
                dtype=np.str_,
            )
            key_arrays.append(name)
        count_name = f"table_{index}_count"
        value_name = f"table_{index}_value"
        arrays[count_name] = np.asarray([counts[key] for key in keys], dtype=np.int64)
        values = table.pos if isinstance(table, _RateTable) else table.total
        arrays[value_name] = np.asarray([values[key] for key in keys], dtype=np.float64)
        descriptor = {
            "feature_name": feature_name,
            "table_kind": "rate" if isinstance(table, _RateTable) else "mean",
            "key_width": key_width,
            "key_arrays": key_arrays,
            "count_array": count_name,
            "value_array": value_name,
        }
        if isinstance(table, _RateTable):
            descriptor.update(total_pos=table.total_pos, total_n=table.total_n)
        else:
            descriptor.update(grand_total=table.grand_total, grand_n=table.grand_n)
        manifest["tables"].append(descriptor)
    return manifest, arrays


def _encoder_from_checkpoint(manifest, arrays) -> _FittedEncoder:
    groups = tuple(str(value) for value in manifest["groups"])
    specs = _feature_specs(groups)
    expected_names = tuple(name for name, _factory, _kind in specs)
    if list(expected_names) != list(manifest["feature_names"]):
        raise RuntimeError("checkpoint encoder feature catalogue changed")
    tables = [factory() for _name, factory, _kind in specs]
    for table, descriptor in zip(tables, manifest["tables"]):
        key_columns = [np.asarray(arrays[name]).tolist() for name in descriptor["key_arrays"]]
        keys = [
            tuple(column[row] for column in key_columns)
            if int(descriptor["key_width"]) > 1
            else key_columns[0][row]
            for row in range(len(key_columns[0]))
        ]
        counts = np.asarray(arrays[descriptor["count_array"]]).tolist()
        values = np.asarray(arrays[descriptor["value_array"]]).tolist()
        for key, count, value in zip(keys, counts, values):
            table.n[key] = int(count)
            if isinstance(table, _RateTable):
                # positives are sums of 0/1 labels, so genuinely integral
                table.pos[key] = int(value)
            else:
                table.total[key] = float(value)
        if isinstance(table, _RateTable):
            table.total_pos = float(descriptor["total_pos"])
            table.total_n = int(descriptor["total_n"])
        else:
            table.grand_total = float(descriptor["grand_total"])
            table.grand_n = int(descriptor["grand_n"])
    vocabs = tuple(
        {str(token): index for index, token in enumerate(np.asarray(arrays[name]).tolist())}
        for name in manifest["vocabs"]
    )
    return _FittedEncoder(
        groups=groups,
        names=expected_names,
        tables=tables,
        kinds=tuple(str(value) for value in manifest["kinds"]),
        statistic_edges=tuple(
            np.asarray(arrays[name], dtype=np.float64)
            for name in manifest["statistic_edges"]
        ),
        duration_edges=np.asarray(arrays[manifest["duration_edges"]], dtype=np.float64),
        vocabs=vocabs,
        feature_dim=int(manifest["feature_dim"]),
    )


def _build_statistics(train_rows, valid_rows, groups):
    """Prequential for train, whole-train for validation. No row sees its own label."""
    specs = [spec for group in groups if group in GROUPS for spec in GROUPS[group]]
    if not specs:
        return [[] for _ in train_rows], [[] for _ in valid_rows], []

    names = [name for name, _f, _k in specs]
    tables = [factory() for _n, factory, _k in specs]
    kinds = [kind for _n, _f, kind in specs]

    def read(row):
        return [t.rate(row) if k == "rate" else t.count(row) for t, k in zip(tables, kinds)]

    ordered = sorted(range(len(train_rows)), key=lambda i: train_rows[i][D_DATE])
    out_train: list[Any] = [None] * len(train_rows)
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and train_rows[ordered[j]][D_DATE] == train_rows[ordered[i]][D_DATE]:
            j += 1
        for index in ordered[i:j]:
            out_train[index] = read(train_rows[index])
        for index in ordered[i:j]:
            for table in tables:
                table.observe(train_rows[index])
        i = j
    return out_train, [read(row) for row in valid_rows], names


def _bucketize(train_stats, valid_stats, names):
    if not names:
        return [[] for _ in train_stats], [[] for _ in valid_stats]
    train = np.asarray(train_stats, dtype=np.float64)
    edges = [np.unique(np.quantile(train[:, c], np.linspace(0, 1, N_BINS + 1)[1:-1]))
             for c in range(train.shape[1])]

    def apply(stats):
        arr = np.asarray(stats, dtype=np.float64)
        columns = [np.searchsorted(edges[c], arr[:, c]) for c in range(arr.shape[1])]
        return [[str(int(v)) for v in row] for row in zip(*columns)]

    return apply(train_stats), apply(valid_stats)


def _encode(train_rows, valid_rows, train_extra, valid_extra, groups):
    edges = np.quantile([x[D_DUR] for x in train_rows], np.linspace(0, 1, 11)[1:-1])
    use_time = "time" in groups

    def raw(x, extra):
        fields = [x[D_USER], x[D_VID], x[D_AUTH], x[D_TAB],
                  str(int(np.searchsorted(edges, x[D_DUR])))]
        fields += list(extra)
        if use_time:
            age = x[D_DATE] - x[D_UP] if x[D_UP] else -1
            fields += [str(x[D_HOUR]), str(min(max(age, -1), 400) // 7)]
        return fields

    width = len(raw(train_rows[0], train_extra[0]))
    vocabs: list[dict[str, int]] = [dict() for _ in range(width)]
    for x, extra in zip(train_rows, train_extra):
        for i, value in enumerate(raw(x, extra)):
            if value not in vocabs[i]:
                vocabs[i][value] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)

    def matrix(rows, extras):
        X = np.empty((len(rows), width), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        for j, (x, extra) in enumerate(zip(rows, extras)):
            for i, value in enumerate(raw(x, extra)):
                X[j, i] = vocabs[i].get(value, unk[i]) + offsets[i]
            y[j] = x[D_Y]
            users.append(x[D_USER])
        return X, y, users

    return matrix(train_rows, train_extra), matrix(valid_rows, valid_extra), int(sum(dims))


# ------------------------------------------------------------------- rank blend
def within_user_percentile(scores, users):
    """Percentile rank inside each user, ties averaged. Scale-free and leak-free."""
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users)
    out = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and users[order[end]] == users[order[start]]:
            end += 1
        idx = order[start:end]
        n = len(idx)
        if n == 1:
            out[idx[0]] = 0.5
        else:
            local = scores[idx]
            ranks = np.empty(n, dtype=np.float64)
            sub = np.argsort(local, kind="stable")
            i = 0
            while i < n:
                j = i
                while j + 1 < n and local[sub[j + 1]] == local[sub[i]]:
                    j += 1
                ranks[sub[i:j + 1]] = (i + j) / 2.0
                i = j + 1
            out[idx] = ranks / (n - 1)
        start = end
    return out


def _user_half_mask(users, parity):
    """Split by user, never by date: a date slice changes which users GAUC counts."""
    return np.asarray([int(u) % 2 == parity for u in users])


def _simplex(count, step):
    values = [round(v, 4) for v in np.arange(0.0, 1.0 + 1e-9, step)]

    def walk(remaining, slots):
        if slots == 1:
            if any(abs(v - remaining) < 1e-9 for v in values):
                yield (round(remaining, 4),)
            return
        for value in values:
            if value <= remaining + 1e-9:
                for tail in walk(round(remaining - value, 4), slots - 1):
                    yield (value,) + tail

    return list(walk(1.0, count))


@dataclass(frozen=True)
class _MemberResult:
    name: str
    scores: np.ndarray
    primary: float
    epochs_run: int
    stopped_by: str
    labels: np.ndarray
    users: list[Any]
    kind: str
    groups: tuple[str, ...]
    loss: str
    embedding_dim: int
    feature_dim: int
    best_epoch: int
    state: dict[str, np.ndarray]
    encoder: _FittedEncoder


# ------------------------------------------------------------------- candidate
def run_ensemble_fm_candidate(
    prepared: PreparedData,
    config: Mapping[str, Any],
    run_dir: Path,
) -> CandidateOutput:
    """Train every member, then blend their within-user ranks."""
    seed = int(config.get("seed", 0))
    epochs = int(config["epochs"])
    patience = int(config.get("patience", 4))
    names = tuple(
        config.get("members")
        or MEMBER_SETS.get(str(config.get("member_set", "core6")), DEFAULT_MEMBERS)
    )
    unknown = [name for name in names if name not in MEMBERS]
    if unknown:
        raise ValueError(f"unknown ensemble members: {', '.join(unknown)}")
    if len(names) < 2:
        raise ValueError("an ensemble needs at least two members")

    train_rows = list(prepared.train_rows)
    valid_rows = list(prepared.validation_rows)
    raw_prediction_rows = prepared.prediction_rows
    predict_rows = None if raw_prediction_rows is None else list(raw_prediction_rows)
    fixed = config.get("blend_weights")
    if predict_rows is not None and fixed is None:
        raise ValueError(
            "prediction scoring requires blend_weights selected on validation"
        )
    if train_rows and len(train_rows[0]) < 12:
        raise ValueError(
            "ensemble members need the 12-field canonical row; restage the inputs"
        )

    results: list[_MemberResult] = []
    users: list[Any] = []
    labels: np.ndarray | None = None
    for name in names:
        groups_spec, loss = MEMBERS[name]
        groups = [g for g in groups_spec.split(",") if g]
        member = _train_member(
            train_rows, valid_rows, groups, loss,
            seed=seed, epochs=epochs, patience=patience,
            k=MEMBER_K.get(name, 16),
            predict_rows=predict_rows,
        )
        results.append(member)
        if labels is None:
            labels, users = member.labels, member.users
        elif member.users != users:
            raise RuntimeError(f"member {name} produced misaligned validation rows")

    assert labels is not None
    ranks = np.vstack([within_user_percentile(item.scores, users) for item in results])

    user_array = np.asarray(users)
    if fixed is not None:
        # Finalization path: weights were selected on validation and are applied
        # unchanged. Never re-fit them on the split being predicted.
        if len(fixed) != len(names):
            raise ValueError("blend_weights must have one entry per member")
        best_weights = tuple(float(w) for w in fixed)
        blended = np.tensordot(np.asarray(best_weights), ranks, axes=(0, 0))
        best_fit = float("nan")
        held = float("nan")
    else:
        # Fit weights on half the users, hold the other half out, then report full.
        fit_mask = _user_half_mask(users, 0)
        held_mask = _user_half_mask(users, 1)
        best_weights, best_fit = (), -1.0
        for weights in _simplex(len(names), WEIGHT_STEP):
            blended = np.tensordot(np.asarray(weights), ranks, axes=(0, 0))
            fit = evaluate_predictions(
                list(user_array[fit_mask]), labels[fit_mask], blended[fit_mask]
            ).primary
            if fit > best_fit:
                best_fit, best_weights = fit, weights
        if not best_weights:
            raise RuntimeError("weight search produced no candidate weighting")
        blended = np.tensordot(np.asarray(best_weights), ranks, axes=(0, 0))
        held = evaluate_predictions(
            list(user_array[held_mask]), labels[held_mask], blended[held_mask]
        ).primary

    epochs_run = max(item.epochs_run for item in results)
    stopped_by = (
        "early_stopping"
        if all(item.stopped_by == "early_stopping" for item in results)
        else "max_epochs_truncated"
    )
    # Persist EVERY trained member with its exact weight. Pruning zero-weight
    # members and renormalizing was measurably not neutral: it changed the
    # replayed validation primary by 1.7e-4, so a pruned bundle could not
    # reproduce the score it certified. Exact reproduction outweighs bundle size.
    active = [
        (name, item, float(weight))
        for name, item, weight in zip(names, results, best_weights)
    ]
    active_weights = [float(weight) for _name, _item, weight in active]
    checkpoint_members = []
    for name, item, _weight in active:
        encoder_manifest, encoder_arrays = _encoder_checkpoint_parts(item.encoder)
        checkpoint_members.append(
            {
                "name": name,
                "kind": item.kind,
                "groups": item.groups,
                "loss": item.loss,
                "embedding_dim": item.embedding_dim,
                "feature_dim": item.feature_dim,
                "primary": item.primary,
                "epochs_run": item.epochs_run,
                "best_epoch": item.best_epoch,
                "state": item.state,
                "encoder_manifest": encoder_manifest,
                "encoder_arrays": encoder_arrays,
            }
        )
    checkpoint_path = run_dir / "ensemble_checkpoint_v2.npz"
    blended_list = [float(value) for value in blended]
    validation_score_sha256 = _score_vector_sha256(blended_list)
    write_ensemble_checkpoint(
        checkpoint_path,
        seed=seed,
        config=dict(config),
        trained_members=names,
        active_members=checkpoint_members,
        weights=active_weights,
        validation_score_sha256=validation_score_sha256,
        validation_primary=evaluate_predictions(
            list(users), [float(v) for v in labels], blended_list
        ).primary,
        lineage=prepared.lineage,
        runtime=_runtime_versions(),
    )
    return CandidateOutput(
        user_ids=users,
        labels=labels.tolist(),
        scores=blended.tolist(),
        metadata={
            "framework": "numpy+pytorch",
            "model": "fm_rank_ensemble",
            "members": list(names),
            "member_primaries": {item.name: item.primary for item in results},
            "member_epochs_run": {item.name: item.epochs_run for item in results},
            "blend_weights": dict(zip(names, [float(w) for w in best_weights])),
            "active_members": [name for name, _item, _weight in active],
            "checkpoint_schema_version": 2,
            "checkpoint_model_kind": "fm_rank_ensemble",
            "validation_score_sha256": validation_score_sha256,
            "blend_weight_fit_half_primary": float(best_fit),
            "blend_weight_held_half_primary": float(held),
            "blend_space": "within_user_percentile_rank",
            "scored_split": "prediction" if predict_rows is not None else "validation",
            "weight_selection": ("fixed from validation" if fixed is not None
                                 else "half of validation users, held half reported separately"),
            "epochs_run": epochs_run,
            "best_epoch": epochs_run,
            "stopped_by": stopped_by,
            "configured_epochs": epochs,
            "effective_patience": patience,
            "checkpoint_path": str(checkpoint_path),
        },
    )


def _train_member(train_rows, valid_rows, groups, loss, *, seed, epochs, patience,
                   k=16, predict_rows=None):
    """Train one member; stop on validation, score `predict_rows` if given."""
    encoder, (Xtr, ytr, _train_users) = _fit_member_encoder(train_rows, groups)
    Xva, yva, uva = _transform_with_encoder(valid_rows, encoder)
    dim = encoder.feature_dim
    target = None
    if predict_rows is not None:
        target = _transform_with_encoder(predict_rows, encoder)
    if loss == "pointwise":
        scores, primary, epochs_run, stopped, state, best_epoch = _train_numpy_fm(
            Xtr, ytr, Xva, yva, uva, dim,
            seed=seed, epochs=epochs, patience=patience, k=k, target=target,
        )
    else:
        scores, primary, epochs_run, stopped, state, best_epoch = _train_torch_member(
            Xtr, ytr, Xva, yva, uva, dim, loss,
            seed=seed, epochs=epochs, patience=patience, target=target,
        )
    return _MemberResult(
        name=f"{loss}:{','.join(groups) or 'base'}:k{k}",
        scores=scores,
        primary=primary,
        epochs_run=epochs_run,
        stopped_by=stopped,
        labels=target[1] if target is not None else yva,
        users=target[2] if target is not None else uva,
        kind="numpy" if loss == "pointwise" else "torch",
        groups=tuple(groups),
        loss=loss,
        embedding_dim=k if loss == "pointwise" else 16,
        feature_dim=dim,
        best_epoch=best_epoch,
        state=state,
        encoder=encoder,
    )


def _train_numpy_fm(Xtr, ytr, Xva, yva, uva, dim, *, seed, epochs, patience, k=16,
                    target=None):
    import baseline as B

    model = B.FM(dim, k=k, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best, state, bad, run, best_epoch = -1.0, None, 0, 0, 0
    for _ in range(epochs):
        run += 1
        order = rng.permutation(len(ytr))
        for i in range(0, len(order), 8192):
            model.step(Xtr[order[i:i + 8192]], ytr[order[i:i + 8192]])
        primary = evaluate_predictions(uva, yva, model.predict(Xva)).primary
        if primary > best + 1e-5:
            best, bad = primary, 0
            state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            best_epoch = run
        else:
            bad += 1
            if bad >= patience:
                break
    if state is None:
        raise RuntimeError("member produced no valid validation state")
    model.V, model.W, model.b = state
    stopped = "early_stopping" if run < epochs else "max_epochs_truncated"
    # `best` and the restored state both come from validation only.
    scored = model.predict(target[0] if target is not None else Xva)
    fitted_state = {
        "V": np.asarray(model.V).copy(),
        "W": np.asarray(model.W).copy(),
        "b": np.asarray(model.b).copy(),
    }
    return scored, best, run, stopped, fitted_state, best_epoch


def _train_torch_member(Xtr, ytr, Xva, yva, uva, dim, loss, *, seed, epochs, patience,
                        target=None):
    """Reuse the approved torch FM training rules on a pre-encoded matrix."""
    import torch

    from .torch_fm import TorchFM, _pair_groups, _listwise_groups, _predict, _train_epoch

    torch.manual_seed(seed)
    model = TorchFM(dim, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    train_x = torch.as_tensor(Xtr, dtype=torch.long)
    train_y = torch.as_tensor(ytr, dtype=torch.float32)
    valid_x = torch.as_tensor(Xva, dtype=torch.long)
    pair_groups = _pair_groups(_row_users(Xtr), ytr) if loss == "pairwise" else None
    listwise_groups = _listwise_groups(_row_users(Xtr), ytr) if loss == "listwise" else None

    best, state, bad, run, best_epoch = -1.0, None, 0, 0, 0
    for epoch in range(1, epochs + 1):
        run = epoch
        model.train()
        _train_epoch(
            model, optimizer, train_x, train_y,
            batch_size=8192, seed=seed + epoch, loss_name=loss,
            pair_groups=pair_groups, listwise_groups=listwise_groups,
            listwise_temperature=1.0, pointwise_weight=0.0,
        )
        scores = _predict(model, valid_x)
        primary = evaluate_predictions(uva, yva, scores.tolist()).primary
        if primary > best + 1e-5:
            best, bad = primary, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = run
        else:
            bad += 1
            if bad >= patience:
                break
    if state is None:
        raise RuntimeError("member produced no valid validation state")
    model.load_state_dict(state)
    stopped = "early_stopping" if run < epochs else "max_epochs_truncated"
    scoring_x = (
        torch.as_tensor(target[0], dtype=torch.long) if target is not None else valid_x
    )
    fitted_state = {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in model.state_dict().items()
    }
    return _predict(model, scoring_x), best, run, stopped, fitted_state, best_epoch


def predict_ensemble_checkpoint(
    prepared: PreparedData,
    checkpoint_path: Path,
) -> CandidateOutput:
    """Run inference from fitted member parameters; no optimizer is constructed."""
    checkpoint = load_ensemble_checkpoint(checkpoint_path)
    target_rows = list(
        prepared.prediction_rows
        if prepared.prediction_rows is not None
        else prepared.validation_rows
    )
    member_scores: list[np.ndarray] = []
    users: list[Any] | None = None
    labels: np.ndarray | None = None
    for descriptor, state, encoder_arrays in zip(
        checkpoint.manifest["members"], checkpoint.states, checkpoint.encoders
    ):
        encoder_manifest = dict(descriptor["encoder"])
        encoder_manifest.pop("encoder_arrays", None)
        encoder = _encoder_from_checkpoint(encoder_manifest, encoder_arrays)
        Xtarget, ytarget, utarget = _transform_with_encoder(
            target_rows, encoder
        )
        feature_dim = encoder.feature_dim
        if feature_dim != int(descriptor["feature_dim"]):
            raise RuntimeError("ensemble checkpoint feature dimension changed")
        if descriptor["kind"] == "numpy":
            import baseline as baseline_model

            model = baseline_model.FM(
                feature_dim,
                k=int(descriptor["embedding_dim"]),
                lr=0.001,
                seed=int(checkpoint.manifest["seed"]),
            )
            model.V = np.asarray(state["V"], dtype=np.float32).copy()
            model.W = np.asarray(state["W"], dtype=np.float32).copy()
            model.b = np.float32(np.asarray(state["b"]).item())
            scores = np.asarray(model.predict(Xtarget), dtype=np.float64)
        elif descriptor["kind"] == "torch":
            import torch

            from .torch_fm import TorchFM, _predict

            model = TorchFM(feature_dim, int(descriptor["embedding_dim"]))
            model.load_state_dict(
                {name: torch.as_tensor(value) for name, value in state.items()},
                strict=True,
            )
            scores = np.asarray(
                _predict(model, torch.as_tensor(Xtarget, dtype=torch.long)),
                dtype=np.float64,
            )
        else:
            raise RuntimeError("ensemble checkpoint contains an unknown member kind")
        member_scores.append(scores)
        if users is None:
            users, labels = utarget, ytarget
        elif users != utarget:
            raise RuntimeError("ensemble checkpoint members produced misaligned rows")
    assert users is not None and labels is not None
    ranks = np.vstack(
        [within_user_percentile(scores, users) for scores in member_scores]
    )
    blended = np.tensordot(checkpoint.weights, ranks, axes=(0, 0))
    return CandidateOutput(
        user_ids=users,
        labels=labels.tolist(),
        scores=blended.tolist(),
        metadata={
            "model": "fm_rank_ensemble",
            "checkpoint_schema_version": 2,
            "checkpoint_model_kind": "fm_rank_ensemble",
            "active_members": list(checkpoint.manifest["active_members"]),
            "validation_score_sha256": checkpoint.manifest[
                "validation_score_sha256"
            ],
            "inference_only": True,
        },
    )


def _score_vector_sha256(scores: Sequence[float]) -> str:
    values = np.asarray(scores, dtype="<f8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _row_users(X):
    """User field is column 0 of the encoded matrix; ids are already offset-unique."""
    return X[:, 0].tolist()
