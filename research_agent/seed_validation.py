"""Independent multi-seed confirmation for the final validation-selected candidate."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract
from .metrics import evaluate_predictions
from .runner import CandidateCallable, ExperimentRunner
from .state import ResearchState
from .lineage import bundle_descriptor
from .store import ArtifactStore


@dataclass(frozen=True)
class SeedConfirmationResult:
    selected_experiment_id: str
    comparator_experiment_id: str
    seeds: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    comparator_scores: tuple[float, ...]
    mean_delta: float
    wins: int
    confirmed: bool
    submission_checkpoint_path: str | None
    comparison_mode: str
    submission_bundle: Mapping[str, Any] | None = None
    candidate_comparison_groups: tuple[str, ...] = ()
    comparator_comparison_groups: tuple[str, ...] = ()
    confirmation_attempts: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_experiment_id": self.selected_experiment_id,
            "comparator_experiment_id": self.comparator_experiment_id,
            "seeds": list(self.seeds),
            "candidate_scores": list(self.candidate_scores),
            "comparator_scores": list(self.comparator_scores),
            "mean_delta": self.mean_delta,
            "wins": self.wins,
            "confirmed": self.confirmed,
            "submission_seed": self.seeds[0],
            "submission_checkpoint_path": self.submission_checkpoint_path,
            "submission_bundle": dict(self.submission_bundle or {}),
            "comparison_mode": self.comparison_mode,
            "candidate_comparison_groups": list(self.candidate_comparison_groups),
            "comparator_comparison_groups": list(self.comparator_comparison_groups),
            "confirmation_attempts": self.confirmation_attempts,
        }


def confirm_selected_candidate(
    store: ArtifactStore,
    runner: ExperimentRunner,
    *,
    contract: BenchmarkContract = BENCHMARK_CONTRACT,
    seeds: Sequence[int] = (0, 1, 2),
    timeout_seconds: float = 2400.0,
) -> SeedConfirmationResult:
    """Confirm a selected configuration without reading the test split."""
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if normalized_seeds != (0, 1, 2):
        raise ValueError("final seed policy is fixed to seeds 0, 1, and 2")
    state = ResearchState.from_dict(store.read_root_json("state.json") or {})
    if state.stop_reason_code not in {"plateau", "iteration_budget", "wall_clock_budget"}:
        raise RuntimeError("seed confirmation requires a valid terminal research state")
    selected = _iteration(store, state.current_best_experiment_id)
    if selected is None:
        result = SeedConfirmationResult(
            selected_experiment_id="baseline",
            comparator_experiment_id="baseline",
            seeds=normalized_seeds,
            candidate_scores=(state.current_best_primary,) * 3,
            comparator_scores=(state.current_best_primary,) * 3,
            mean_delta=0.0,
            wins=0,
            confirmed=True,
            submission_checkpoint_path=None,
            comparison_mode="organizer_baseline_selected",
        )
        store.write_root_json("seed_confirmation.json", result.as_dict())
        return result
    if selected.get("decision") != "accepted" or not (selected.get("comparison_validity") or {}).get("valid"):
        resolution = (store.read_root_json("promotion_resolutions.json") or {}).get(
            state.current_best_experiment_id,
            {},
        )
        if resolution.get("decision") != "accepted":
            raise RuntimeError("selected candidate lacks accepted comparison evidence")

    promotion = (store.read_root_json("promotion_confirmations.json") or {}).get(
        state.current_best_experiment_id
    )
    if promotion and promotion.get("confirmed") is True:
        result = SeedConfirmationResult(
            selected_experiment_id=state.current_best_experiment_id,
            comparator_experiment_id=str(promotion["comparator_experiment_id"]),
            seeds=tuple(int(seed) for seed in promotion["seeds"]),
            candidate_scores=tuple(float(score) for score in promotion["candidate_scores"]),
            comparator_scores=tuple(float(score) for score in promotion["comparator_scores"]),
            mean_delta=float(promotion["mean_delta"]),
            wins=int(promotion["wins"]),
            confirmed=True,
            submission_checkpoint_path=str(
                promotion.get("submission_checkpoint_path") or ""
            ) or None,
            submission_bundle=(
                dict(promotion["submission_bundle"])
                if isinstance(promotion.get("submission_bundle"), Mapping)
                else None
            ),
            comparison_mode=str(promotion["comparison_mode"]),
            candidate_comparison_groups=tuple(
                str(value) for value in promotion.get("candidate_comparison_groups", [])
            ),
            comparator_comparison_groups=tuple(
                str(value) for value in promotion.get("comparator_comparison_groups", [])
            ),
            confirmation_attempts=int(promotion.get("confirmation_attempts", 0)),
        )
        store.write_root_json("seed_confirmation.json", result.as_dict())
        return result

    from .models.torch_fm import run_torch_fm_candidate

    candidate_scores, checkpoints, _candidate_groups, descriptors = _run_configuration(
        runner,
        selected,
        role="candidate",
        seeds=normalized_seeds,
        timeout_seconds=timeout_seconds,
        candidate=run_torch_fm_candidate,
        require_checkpoint=True,
    )
    parent_id = str(selected.get("comparison_incumbent_id") or "baseline")
    parent = _iteration(store, parent_id)
    if parent is None:
        comparator_scores = (0.6016,) * len(normalized_seeds)
        comparison_mode = "published_organizer_baseline_mean"
    else:
        comparator_scores, _, _comparator_groups, _ = _run_configuration(
            runner,
            parent,
            role="comparator",
            seeds=normalized_seeds,
            timeout_seconds=timeout_seconds,
            candidate=run_torch_fm_candidate,
            require_checkpoint=False,
        )
        comparison_mode = "matched_seed_parent"

    deltas = tuple(candidate - comparator for candidate, comparator in zip(candidate_scores, comparator_scores))
    mean_delta = mean(deltas)
    wins = sum(delta > 0.0 for delta in deltas)
    confirmed = (
        mean_delta > contract.improvement_threshold
        and not math.isclose(
            mean_delta,
            contract.improvement_threshold,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and wins >= 2
    )
    result = SeedConfirmationResult(
        selected_experiment_id=state.current_best_experiment_id,
        comparator_experiment_id=parent_id,
        seeds=normalized_seeds,
        candidate_scores=candidate_scores,
        comparator_scores=comparator_scores,
        mean_delta=mean_delta,
        wins=wins,
        confirmed=confirmed,
        submission_checkpoint_path=checkpoints[0] if confirmed else None,
        submission_bundle=descriptors[0] if confirmed and descriptors else None,
        comparison_mode=comparison_mode,
    )
    store.write_root_json("seed_confirmation.json", result.as_dict())
    return result


def confirm_promotion_candidate(
    store: ArtifactStore,
    runner: ExperimentRunner,
    record: Mapping[str, Any],
    *,
    candidate: CandidateCallable,
    contract: BenchmarkContract = BENCHMARK_CONTRACT,
    seeds: Sequence[int] = (0, 1, 2),
    timeout_seconds: float = 2400.0,
) -> SeedConfirmationResult:
    """Confirm a provisional candidate against matched incumbent seed evidence."""
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if normalized_seeds != (0, 1, 2):
        raise ValueError("promotion seed policy is fixed to seeds 0, 1, and 2")
    selected_id = str(record["experiment_id"])
    existing = (store.read_root_json("promotion_confirmations.json") or {}).get(selected_id)
    if existing:
        return SeedConfirmationResult(
            selected_experiment_id=selected_id,
            comparator_experiment_id=str(existing["comparator_experiment_id"]),
            seeds=normalized_seeds,
            candidate_scores=tuple(float(value) for value in existing["candidate_scores"]),
            comparator_scores=tuple(float(value) for value in existing["comparator_scores"]),
            mean_delta=float(existing["mean_delta"]),
            wins=int(existing["wins"]),
            confirmed=bool(existing["confirmed"]),
            submission_checkpoint_path=existing.get("submission_checkpoint_path"),
            submission_bundle=(
                dict(existing["submission_bundle"])
                if isinstance(existing.get("submission_bundle"), Mapping)
                else None
            ),
            comparison_mode=str(existing["comparison_mode"]),
            candidate_comparison_groups=tuple(
                str(value) for value in existing.get("candidate_comparison_groups", [])
            ),
            comparator_comparison_groups=tuple(
                str(value) for value in existing.get("comparator_comparison_groups", [])
            ),
            confirmation_attempts=int(existing.get("confirmation_attempts", 0)),
        )

    candidate_scores, checkpoints, candidate_groups, descriptors = _run_configuration(
        runner,
        record,
        role="promotion-candidate",
        seeds=normalized_seeds,
        timeout_seconds=timeout_seconds,
        candidate=candidate,
        require_checkpoint=True,
    )
    comparator_id = str(record.get("comparison_incumbent_id") or "baseline")
    if comparator_id == "baseline":
        from .models.organizer_fm import run_organizer_fm_candidate
        from .search import SearchController

        baseline_record = {
            "experiment_id": "baseline",
            "config": {
                **dict(SearchController.BASELINE_CONFIG),
                "fidelity": "full",
                "epochs": contract.full_max_epochs,
                "patience": contract.full_patience,
            },
        }
        comparator_scores, _, comparator_groups, _ = _run_configuration(
            runner,
            baseline_record,
            role="promotion-comparator",
            seeds=normalized_seeds,
            timeout_seconds=timeout_seconds,
            candidate=run_organizer_fm_candidate,
            require_checkpoint=False,
        )
        comparison_mode = "matched_seed_organizer_baseline"
        confirmation_attempts = 6
    else:
        incumbent = (store.read_root_json("promotion_confirmations.json") or {}).get(
            comparator_id
        )
        if not incumbent or incumbent.get("confirmed") is not True:
            raise RuntimeError("incumbent lacks reusable matched-seed promotion evidence")
        if incumbent.get("seeds") != [0, 1, 2]:
            raise RuntimeError("incumbent seed evidence does not match the fixed seed policy")
        comparator_scores = tuple(float(value) for value in incumbent["candidate_scores"])
        comparator_groups = tuple(
            str(value) for value in incumbent.get("candidate_comparison_groups", [])
        )
        comparison_mode = "reused_matched_seed_incumbent"
        confirmation_attempts = 3

    if candidate_groups != comparator_groups or len(candidate_groups) != len(normalized_seeds):
        raise RuntimeError("candidate and incumbent seed runs have different comparison lineage")

    deltas = tuple(
        candidate_score - comparator_score
        for candidate_score, comparator_score in zip(candidate_scores, comparator_scores)
    )
    mean_delta = mean(deltas)
    wins = sum(delta > contract.improvement_threshold for delta in deltas)
    confirmed = (
        mean_delta > contract.improvement_threshold
        and not math.isclose(
            mean_delta,
            contract.improvement_threshold,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and wins >= 2
    )
    result = SeedConfirmationResult(
        selected_experiment_id=selected_id,
        comparator_experiment_id=comparator_id,
        seeds=normalized_seeds,
        candidate_scores=candidate_scores,
        comparator_scores=comparator_scores,
        mean_delta=mean_delta,
        wins=wins,
        confirmed=confirmed,
        submission_checkpoint_path=checkpoints[0] if confirmed else None,
        submission_bundle=descriptors[0] if confirmed and descriptors else None,
        comparison_mode=comparison_mode,
        candidate_comparison_groups=candidate_groups,
        comparator_comparison_groups=comparator_groups,
        confirmation_attempts=confirmation_attempts,
    )
    confirmations = store.read_root_json("promotion_confirmations.json") or {}
    confirmations[selected_id] = result.as_dict()
    store.write_root_json("promotion_confirmations.json", confirmations)
    return result


def _run_configuration(
    runner: ExperimentRunner,
    record: Mapping[str, Any],
    *,
    role: str,
    seeds: tuple[int, ...],
    timeout_seconds: float,
    candidate: CandidateCallable,
    require_checkpoint: bool,
) -> tuple[
    tuple[float, ...], tuple[str, ...], tuple[str, ...], tuple[dict[str, Any], ...]
]:
    config = dict(record["config"])
    config.update({"fidelity": "full", "epochs": 40, "patience": 4})
    scores: list[float] = []
    checkpoints: list[str] = []
    descriptors: list[dict[str, Any]] = []
    comparison_groups: list[str] = []
    selected_id = str(record["experiment_id"])
    for seed in seeds:
        config["seed"] = seed
        result = runner.run(
            experiment_id=f"{selected_id}-{role}-seed-{seed}",
            hypothesis=f"Final matched-seed confirmation for {selected_id} ({role}, seed {seed}).",
            config=config,
            candidate=candidate,
            timeout_seconds=timeout_seconds,
        )
        if result.status != "completed" or result.output is None:
            raise RuntimeError(
                f"seed confirmation failed for {selected_id} seed {seed}: {result.error or result.status}"
            )
        metadata = result.output.metadata
        if metadata.get("stopped_by") != "early_stopping":
            raise RuntimeError(
                f"seed confirmation run for {selected_id} seed {seed} is truncated"
            )
        if metadata.get("configured_epochs") != 40 or metadata.get("effective_patience") != 4:
            raise RuntimeError("seed confirmation run did not use the full stopping budget")
        comparison_group = metadata.get("comparison_group_id")
        if not isinstance(comparison_group, str) or len(comparison_group) != 64:
            raise RuntimeError("seed confirmation run lacks comparison lineage")
        metrics = evaluate_predictions(
            result.output.user_ids,
            result.output.labels,
            result.output.scores,
            split="valid",
        ).as_dict()
        score = float(metrics["primary"])
        if not math.isfinite(score):
            raise RuntimeError(f"seed confirmation produced a non-finite score for seed {seed}")
        checkpoint = Path(str(result.output.metadata.get("checkpoint_path", "")))
        if require_checkpoint and not checkpoint.is_file():
            raise RuntimeError(f"seed confirmation checkpoint is missing for seed {seed}")
        scores.append(score)
        checkpoints.append(str(checkpoint) if checkpoint.is_file() else "")
        if checkpoint.is_file():
            # Bind the certificate to bundle CONTENT; a path alone is mutable.
            descriptors.append(
                bundle_descriptor(checkpoint, dict(result.output.metadata))
            )
        else:
            descriptors.append({})
        comparison_groups.append(comparison_group)
    return tuple(scores), tuple(checkpoints), tuple(comparison_groups), tuple(descriptors)


def _iteration(store: ArtifactStore, experiment_id: str) -> dict[str, Any] | None:
    if experiment_id == "baseline":
        return None
    for record in reversed(store.read_iterations()):
        if record.get("experiment_id") == experiment_id:
            return record
    raise RuntimeError(f"experiment is missing from research history: {experiment_id}")
