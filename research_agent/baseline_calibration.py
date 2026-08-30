"""Three-seed validation-only reference calibration for the competition harness."""
from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract
from .logger import ResearchLogger
from .manifest import ensure_run_manifest
from .metrics import evaluate_predictions
from .models.organizer_fm import run_organizer_fm_candidate
from .models.torch_fm import run_torch_fm_candidate
from .runner import CandidateCallable, ExperimentRunner
from .search import SearchController
from .store import ArtifactStore


def run_reference_calibration(
    store: ArtifactStore,
    *,
    contract: BenchmarkContract = BENCHMARK_CONTRACT,
) -> dict[str, Any]:
    """Measure organizer and PyTorch pointwise references on seeds 0, 1, and 2."""
    ensure_run_manifest(store, contract, create=store.read_root_json("run_manifest.json") is None)
    logger = ResearchLogger(store)
    runner = ExperimentRunner(logger, contract=contract)
    references = {
        "organizer_fm": run_organizer_fm_candidate,
        "torch_pointwise_fm": run_torch_fm_candidate,
    }
    results: dict[str, Any] = {}
    for name, candidate in references.items():
        seeds = [
            _run_reference_seed(
                runner,
                name=name,
                candidate=candidate,
                seed=seed,
                contract=contract,
            )
            for seed in (0, 1, 2)
        ]
        for lineage_field in ("comparison_group_id", "data_sha256", "model_code_sha256"):
            values = {str(item[lineage_field]) for item in seeds}
            if len(values) != 1:
                raise RuntimeError(
                    f"{name} reference seeds have mixed {lineage_field} lineage"
                )
        results[name] = {
            "seeds": seeds,
            "mean": {
                metric: mean(float(item[metric]) for item in seeds)
                for metric in ("GAUC", "nDCG@5", "primary", "runtime_seconds")
            },
            "std": {
                metric: pstdev(float(item[metric]) for item in seeds)
                for metric in ("GAUC", "nDCG@5", "primary")
            },
        }
    organizer_mean = float(results["organizer_fm"]["mean"]["primary"])
    summary = {
        "selection_split": contract.validation_split,
        "seeds": [0, 1, 2],
        "full_fidelity": {
            "max_epochs": contract.full_max_epochs,
            "patience": contract.full_patience,
        },
        "references": results,
        "minimum_evidence_threshold": contract.improvement_threshold,
        "operational_target_delta": 0.003,
        "operational_target_primary": organizer_mean + 0.003,
    }
    store.write_root_json("baseline_calibration.json", summary)
    return summary


def _run_reference_seed(
    runner: ExperimentRunner,
    *,
    name: str,
    candidate: CandidateCallable,
    seed: int,
    contract: BenchmarkContract,
) -> dict[str, Any]:
    config = {
        **dict(SearchController.BASELINE_CONFIG),
        "loss": "pointwise",
        "seed": seed,
        "fidelity": "full",
        "epochs": contract.full_max_epochs,
        "patience": contract.full_patience,
    }
    result = runner.run(
        experiment_id=f"reference-{name}-seed-{seed}",
        hypothesis=f"Freeze the {name} validation reference at seed {seed}.",
        config=config,
        candidate=candidate,
        timeout_seconds=600.0,
    )
    if result.status != "completed" or result.output is None:
        raise RuntimeError(f"{name} seed {seed} failed: {result.error or result.status}")
    metadata = result.output.metadata
    if metadata.get("stopped_by") != "early_stopping":
        raise RuntimeError(f"{name} seed {seed} did not produce comparable early-stop evidence")
    metrics = evaluate_predictions(
        result.output.user_ids,
        result.output.labels,
        result.output.scores,
        split=contract.validation_split,
    ).as_dict()
    values: dict[str, Any] = {
        "seed": seed,
        "GAUC": float(metrics["GAUC"]),
        "nDCG@5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
        "runtime_seconds": result.runtime_seconds,
        "epochs_run": metadata.get("epochs_run"),
        "best_epoch": metadata.get("best_epoch"),
        "stopped_by": metadata.get("stopped_by"),
        "comparison_group_id": metadata.get("comparison_group_id"),
        "data_sha256": metadata.get("data_sha256"),
        "model_code_sha256": metadata.get("model_code_sha256"),
    }
    if not all(math.isfinite(float(values[key])) for key in ("GAUC", "nDCG@5", "primary")):
        raise RuntimeError(f"{name} seed {seed} produced non-finite reference metrics")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze three-seed validation references")
    parser.add_argument("--artifact-dir", default="runs_baseline_calibration")
    parser.add_argument("--data-dir", default=str(BENCHMARK_CONTRACT.data_dir))
    args = parser.parse_args()
    contract = replace(BENCHMARK_CONTRACT, data_dir=Path(args.data_dir))
    summary = run_reference_calibration(
        ArtifactStore(args.artifact_dir),
        contract=contract,
    )
    print(summary)


if __name__ == "__main__":
    main()
