"""Research-only data staging that preserves organizer source files unchanged."""
from __future__ import annotations

import csv
import os
import gzip
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from data import LABEL, SPLITS


CANONICAL_ROW_WIDTH = 12


def load_research_splits(
    data_dir: str | Path,
    requested_splits: Sequence[str] = ("train", "valid"),
) -> dict[str, list[tuple[Any, ...]]]:
    """Materialize only requested canonical rows, discarding test-dated rows immediately."""
    requested = tuple(requested_splits)
    unknown = sorted(set(requested) - set(SPLITS))
    if unknown:
        raise ValueError(f"unknown split(s): {', '.join(unknown)}")
    if not requested or "test" in requested:
        raise ValueError("research staging accepts train/validation splits only")

    root = os.fspath(data_dir)
    video_meta: dict[str, tuple[str, str, str, int]] = {}
    with open(os.path.join(root, "video_features_basic_pure.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # Only author_id is required; the rest are optional enrichment used
            # by ensemble members, so a minimal metadata file still loads.
            uploaded = row.get("upload_dt", "").replace("-", "")
            video_meta[row["video_id"]] = (
                row["author_id"],
                row.get("music_id", "UNK"),
                row.get("tag", "UNK").split(",")[0],
                int(uploaded) if uploaded.isdigit() else 0,
            )
    unknown_meta = ("UNK", "UNK", "UNK", 0)

    output: dict[str, list[tuple[Any, ...]]] = {name: [] for name in requested}
    sources = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    for filename in sources:
        with open(os.path.join(root, filename), encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                date = int(row["date"])
                split = next(
                    (
                        name
                        for name in requested
                        if SPLITS[name][0] <= date <= SPLITS[name][1]
                    ),
                    None,
                )
                if split is None:
                    continue
                author, music, tag, uploaded = video_meta.get(
                    row["video_id"], unknown_meta
                )
                output[split].append(
                    (
                        date,
                        row["user_id"],
                        row["video_id"],
                        author,
                        row["tab"],
                        float(row["duration_ms"]),
                        1 if row[LABEL] != "0" else 0,
                        # Fields 7-11 are appended, never inserted, so data.py's
                        # positional encode() keeps working unchanged.
                        int(row.get("hourmin") or 0) // 100,
                        float(row.get("play_time_ms") or 0.0),
                        music,
                        tag,
                        uploaded,
                    )
                )
    return output


def stage_research_splits(
    data_dir: str | Path,
    destination: Path,
    *,
    train_split: str,
    validation_split: str,
    source_data_sha256: str,
    staging_code_sha256: str,
) -> Path:
    """Atomically stage a worker input that contains no raw data path or test rows."""
    if destination.exists():
        try:
            load_staged_splits(
                destination,
                source_data_sha256=source_data_sha256,
                staging_code_sha256=staging_code_sha256,
            )
            return destination
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    splits = load_research_splits(data_dir, (train_split, validation_split))
    payload = {
        "schema_version": 2,
        "source_data_sha256": source_data_sha256,
        "staging_code_sha256": staging_code_sha256,
        "splits": splits,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_staged_splits(
    path: Path,
    *,
    source_data_sha256: str,
    staging_code_sha256: str,
) -> Mapping[str, Sequence[Any]]:
    """Read a locally generated train/validation-only stage artifact."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("staged research input has an unsupported schema")
    if payload.get("source_data_sha256") != source_data_sha256:
        raise ValueError("staged research input does not match the source dataset")
    if payload.get("staging_code_sha256") != staging_code_sha256:
        raise ValueError("staged research input was created by different staging code")
    value = payload.get("splits")
    if not isinstance(value, dict) or set(value) != {"train", "valid"}:
        raise ValueError("staged research input must contain exactly train and valid")
    if not all(isinstance(rows, list) for rows in value.values()):
        raise ValueError("staged research split rows must be lists")
    for split, rows in value.items():
        lower, upper = SPLITS[split]
        for row in rows:
            if not isinstance(row, list) or len(row) != CANONICAL_ROW_WIDTH:
                raise ValueError("staged research row does not match the canonical schema")
            if not isinstance(row[0], int) or not lower <= row[0] <= upper:
                raise ValueError(f"staged {split} input contains a row outside its date boundary")
            if row[6] not in {0, 1}:
                raise ValueError("staged research labels must be binary")
    return value
