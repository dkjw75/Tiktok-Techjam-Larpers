"""Subprocess entrypoint for one production candidate trial."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .data_boundary import load_staged_splits
from .runner import (
    APPROVED_CANDIDATES,
    CandidateOutput,
    PreparedData,
    candidate_output_to_json,
    classify_exception,
    import_callable,
    validate_candidate_alignment,
)


def run_job(job_path: Path) -> int:
    run_dir = job_path.parent
    result_path = run_dir / "worker_result.json"
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        required = {
            "candidate_key",
            "config",
            "run_dir",
            "prepared_data_path",
            "train_split",
            "validation_split",
            "source_data_sha256",
            "staging_code_sha256",
        }
        missing = sorted(required - set(job))
        if missing:
            raise ValueError(f"worker job missing fields: {', '.join(missing)}")
        declared_run_dir = Path(job["run_dir"]).resolve()
        if declared_run_dir != run_dir.resolve():
            raise ValueError("worker job run_dir does not match its containing attempt")
        run_dir = declared_run_dir
        train_split = str(job["train_split"])
        validation_split = str(job["validation_split"])
        splits = load_staged_splits(
            Path(job["prepared_data_path"]).resolve(),
            source_data_sha256=str(job["source_data_sha256"]),
            staging_code_sha256=str(job["staging_code_sha256"]),
        )
        job_lineage = job.get("lineage")
        prepared = PreparedData(
            train_rows=splits[train_split],
            validation_rows=splits[validation_split],
            lineage=dict(job_lineage) if isinstance(job_lineage, dict) else None,
        )
        candidate_key = str(job["candidate_key"])
        candidate_path = APPROVED_CANDIDATES.get(candidate_key)
        if candidate_path is None:
            raise ValueError(f"unapproved candidate key: {candidate_key!r}")
        candidate = import_callable(candidate_path)
        output = candidate(prepared, dict(job["config"]), run_dir)
        if not isinstance(output, CandidateOutput):
            raise TypeError(f"candidate returned {type(output).__name__}, expected CandidateOutput")
        validate_candidate_alignment(output, prepared)
        output = CandidateOutput(
            output.user_ids,
            output.labels,
            output.scores,
            {**dict(output.metadata), "worker_pid": os.getpid(), "candidate_key": candidate_key},
        )
        _atomic_write_json(run_dir / "candidate_output.json", candidate_output_to_json(output))
        _atomic_write_json(result_path, {"status": "completed", "failure_kind": None, "error": None})
        return 0
    except BaseException as exc:
        _atomic_write_json(
            result_path,
            {
                "status": "failed",
                "failure_kind": classify_exception(exc),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 2


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated research candidate")
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_job(args.job))


if __name__ == "__main__":
    main()
