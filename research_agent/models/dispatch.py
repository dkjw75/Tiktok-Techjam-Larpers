"""Single importable entry point that routes a config to its candidate model.

The runner may execute candidates in a worker process by import path, so this
must be a module-level function rather than a closure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..runner import CandidateOutput, PreparedData

# A router's own hash would never change when a model changes, so the runner
# folds these into model_code_sha256 as well.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROUTED_SOURCE_FILES = (
    str(Path(__file__).with_name("ensemble_fm.py")),
    str(Path(__file__).with_name("ensemble_checkpoint.py")),
    str(Path(__file__).with_name("torch_fm.py")),
    str(Path(__file__).parents[1] / "metrics.py"),
    str(_REPOSITORY_ROOT / "baseline.py"),
)


def run_candidate(
    prepared: PreparedData,
    config: Mapping[str, Any],
    run_dir: Path,
) -> CandidateOutput:
    if str(config.get("loss")) == "ensemble":
        from .ensemble_fm import run_ensemble_fm_candidate

        return run_ensemble_fm_candidate(prepared, config, run_dir)
    from .torch_fm import run_torch_fm_candidate

    return run_torch_fm_candidate(prepared, config, run_dir)
