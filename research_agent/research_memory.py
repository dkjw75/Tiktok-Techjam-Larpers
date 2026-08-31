"""Deterministic compatibility views over the append-only research evidence."""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract
from .store import ArtifactStore


# Published organizer benchmark reference. Matched-seed calibrated means are
# reported separately and never silently substituted for this contract value.
CONTRACT_REFERENCE_PRIMARY = 0.6016
# Frozen three-seed organizer calibration persisted by the audited benchmark
# run.  A run-local matched certificate, when present, still takes precedence.
FROZEN_CALIBRATED_BASELINE_PRIMARY = 0.6014399038727732
FROZEN_CALIBRATED_BASELINE_GAUC = 0.6671971984010782
FROZEN_CALIBRATED_BASELINE_NDCG5 = 0.5356826093444684


LEDGER_FIELDS = (
    "experiment_id",
    "parent_champion_id",
    "hypothesis",
    "research_family",
    "allocation_status",
    "changed_factors",
    "full_config",
    "feature_groups",
    "model",
    "loss",
    "seed",
    "fidelity",
    "epochs_run",
    "best_epoch",
    "stopping_reason",
    "GAUC",
    "nDCG@5",
    "primary",
    "delta_vs_baseline",
    "delta_vs_champion",
    "standalone_rank",
    "ensemble_delta_if_added",
    "runtime_seconds",
    "initial_decision",
    "resolved_decision",
    "seed_mean_primary",
    "seed_std_primary",
    "seed_wins",
    "error",
    "recovery",
    "artifact_paths",
)


class ResearchMemoryProjector:
    """Materialize requested memory files without becoming a second truth store."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        contract: BenchmarkContract = BENCHMARK_CONTRACT,
    ) -> None:
        self.store = store
        self.contract = contract

    def write_all(self) -> None:
        self.store.initialize()
        records = self.store.read_iterations()
        resolutions = self.store.read_root_json("promotion_resolutions.json") or {}
        state = self.store.read_root_json("state.json") or {}
        self.store.write_root_json(
            "research_state.json",
            self._state_view(state, records, resolutions),
        )
        self._atomic_write(
            self.store.root / "experiment_ledger.csv",
            self._ledger_csv(records, resolutions),
        )
        self._atomic_write(
            self.store.root / "research_lessons.md",
            self._lessons_markdown(records, resolutions),
        )

    def _state_view(
        self,
        state: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
        resolutions: Mapping[str, Any],
    ) -> dict[str, Any]:
        champion_id = str(state.get("current_best_experiment_id") or "baseline")
        champion_record = next(
            (item for item in reversed(records) if item.get("experiment_id") == champion_id),
            None,
        )
        champion_record_metrics = dict((champion_record or {}).get("metrics") or {})
        latest = records[-1] if records else {}
        regions = self._region_statuses(records, resolutions)
        active_region = str(latest.get("search_region_id") or "unassigned")
        baseline_seed_scores = self._baseline_seed_scores(resolutions)
        baseline_primary = self._baseline_primary(baseline_seed_scores)
        final_certificate = self.store.read_root_json("seed_confirmation.json") or {}
        champion_resolution = resolutions.get(champion_id)
        champion_certificate = (
            dict(champion_resolution.get("certificate") or {})
            if isinstance(champion_resolution, Mapping)
            else {}
        )
        champion_seed_scores = [
            float(value) for value in champion_certificate.get("candidate_scores", [])
        ]
        return {
            "schema_version": 1,
            "source_of_truth": {
                "state": "state.json",
                "iterations": "iterations.jsonl",
                "promotion_resolutions": "promotion_resolutions.json",
            },
            "current_champion": champion_id,
            "champion_status": {
                "promotion_decision": (
                    champion_resolution.get("decision")
                    if isinstance(champion_resolution, Mapping)
                    else ("organizer_baseline" if champion_id == "baseline" else None)
                ),
                "terminal_research_state": state.get("stop_reason_code") is not None,
                "final_seed_certificate_present": bool(final_certificate),
                "final_seed_certificate_matches_champion": (
                    final_certificate.get("selected_experiment_id") == champion_id
                    and final_certificate.get("confirmed") is True
                ),
                # Only finalize._require_finalizable may assert certification;
                # this read model deliberately cannot bless mutable bundles.
                "certified_for_finalization": False,
                "certification_note": "not asserted by a compatibility projection; run finalization preflight",
            },
            "champion_metrics": {
                "selected_run": champion_record_metrics,
                "seed_confirmed_primary_mean": (
                    mean(champion_seed_scores)
                    if champion_seed_scores
                    else float(
                        state.get("current_best_primary", CONTRACT_REFERENCE_PRIMARY)
                    )
                ),
                "seed_confirmed_primary_std": (
                    pstdev(champion_seed_scores)
                    if len(champion_seed_scores) > 1
                    else None
                ),
                "seed_scores": champion_seed_scores,
            },
            "baseline_metrics": {
                "contract_reference_primary": CONTRACT_REFERENCE_PRIMARY,
                "frozen_calibrated_primary": baseline_primary,
                "matched_seed_mean_primary": (
                    mean(baseline_seed_scores) if baseline_seed_scores else None
                ),
                "matched_seed_scores": baseline_seed_scores,
                "matched_seed_evidence_present": bool(baseline_seed_scores),
            },
            "completed_experiment_count": int(state.get("completed_iterations", len(records))),
            "elapsed_research_seconds": float(state.get("elapsed_seconds", 0.0)),
            "active_runtime_seconds": float(state.get("active_runtime_seconds", 0.0)),
            "consecutive_non_improvements": int(
                state.get("consecutive_non_improvements", 0)
            ),
            "current_research_hypothesis": latest.get("hypothesis"),
            "active_research_region": active_region,
            "research_regions": regions,
            "rejected_or_dead_regions": sorted(
                region for region, status in regions.items() if status == "ABANDON"
            ),
            "stop_reason_code": state.get("stop_reason_code"),
            "stop_reason": state.get("stop_reason"),
        }

    def _ledger_csv(
        self,
        records: Sequence[Mapping[str, Any]],
        resolutions: Mapping[str, Any],
    ) -> str:
        baseline_seed_scores = self._baseline_seed_scores(resolutions)
        baseline_primary = self._baseline_primary(baseline_seed_scores)
        primaries = sorted(
            {
                float((record.get("metrics") or {})["primary"])
                for record in records
                if isinstance((record.get("metrics") or {}).get("primary"), (int, float))
                and not isinstance((record.get("metrics") or {}).get("primary"), bool)
            },
            reverse=True,
        )
        rank = {value: index + 1 for index, value in enumerate(primaries)}
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for record in records:
            experiment_id = str(record.get("experiment_id") or "")
            metrics = dict(record.get("metrics") or {})
            metadata = dict(record.get("runner_metadata") or {})
            config = dict(record.get("config") or {})
            resolution = resolutions.get(experiment_id)
            certificate = (
                dict(resolution.get("certificate") or {})
                if isinstance(resolution, Mapping)
                else {}
            )
            scores = [float(value) for value in certificate.get("candidate_scores", [])]
            primary = metrics.get("primary")
            resolved_delta = (
                certificate.get("mean_delta")
                if certificate.get("mean_delta") is not None
                else record.get("delta_primary")
            )
            row = {
                "experiment_id": experiment_id,
                "parent_champion_id": record.get("comparison_incumbent_id"),
                "hypothesis": record.get("hypothesis"),
                "research_family": record.get("direction_id"),
                "allocation_status": self._allocation_status(record, resolution),
                "changed_factors": json.dumps(record.get("changed_factors") or []),
                "full_config": json.dumps(config, sort_keys=True, separators=(",", ":")),
                "feature_groups": json.dumps(metadata.get("members") or []),
                "model": metadata.get("model"),
                "loss": config.get("loss"),
                "seed": metadata.get("seed", config.get("seed")),
                "fidelity": config.get("fidelity"),
                "epochs_run": metadata.get("epochs_run"),
                "best_epoch": metadata.get("best_epoch"),
                "stopping_reason": metadata.get("stopped_by"),
                "GAUC": metrics.get("GAUC"),
                "nDCG@5": metrics.get("nDCG@5"),
                "primary": primary,
                "delta_vs_baseline": (
                    float(primary) - baseline_primary
                    if isinstance(primary, (int, float)) and not isinstance(primary, bool)
                    else None
                ),
                "delta_vs_champion": resolved_delta,
                "standalone_rank": rank.get(float(primary)) if isinstance(primary, (int, float)) else None,
                "ensemble_delta_if_added": metadata.get("ensemble_delta_if_added"),
                "runtime_seconds": record.get("runtime_seconds"),
                "initial_decision": record.get("decision"),
                "resolved_decision": (
                    resolution.get("decision")
                    if isinstance(resolution, Mapping)
                    else record.get("decision")
                ),
                "seed_mean_primary": mean(scores) if scores else None,
                "seed_std_primary": pstdev(scores) if len(scores) > 1 else None,
                "seed_wins": certificate.get("wins"),
                "error": record.get("error"),
                "recovery": record.get("recovery"),
                "artifact_paths": json.dumps(
                    {
                        "checkpoint": metadata.get("checkpoint_path"),
                        "code_diff": record.get("code_diff_path"),
                    },
                    sort_keys=True,
                ),
            }
            writer.writerow(row)
        return output.getvalue()

    def _lessons_markdown(
        self,
        records: Sequence[Mapping[str, Any]],
        resolutions: Mapping[str, Any],
    ) -> str:
        lines = [
            "# Research Lessons",
            "",
            "Generated from append-only iteration and promotion evidence; unavailable fields are stated, not invented.",
            "",
        ]
        by_id = {str(item.get("experiment_id")): item for item in records}
        for record in records:
            experiment_id = str(record.get("experiment_id") or "unknown")
            metrics = dict(record.get("metrics") or {})
            metadata = dict(record.get("runner_metadata") or {})
            resolution = resolutions.get(experiment_id)
            decision = (
                resolution.get("decision")
                if isinstance(resolution, Mapping)
                else record.get("decision")
            )
            validity = dict(record.get("comparison_validity") or {})
            certificate = (
                dict(resolution.get("certificate") or {})
                if isinstance(resolution, Mapping)
                else {}
            )
            scores = [float(value) for value in certificate.get("candidate_scores", [])]
            delta = certificate.get("mean_delta", record.get("delta_primary"))
            evidence_primary = mean(scores) if scores else metrics.get("primary", "unavailable")
            parent_id = str(record.get("comparison_incumbent_id") or "baseline")
            parent_metrics = dict((by_id.get(parent_id) or {}).get("metrics") or {})
            parent_gauc = parent_metrics.get("GAUC", FROZEN_CALIBRATED_BASELINE_GAUC)
            parent_ndcg = parent_metrics.get("nDCG@5", FROZEN_CALIBRATED_BASELINE_NDCG5)
            gauc_delta = self._metric_delta(metrics.get("GAUC"), parent_gauc)
            ndcg_delta = self._metric_delta(metrics.get("nDCG@5"), parent_ndcg)
            seed_std = pstdev(scores) if len(scores) > 1 else None
            complementarity = metadata.get("ensemble_delta_if_added")
            confidence = self._confidence(record, resolution)
            lines.extend(
                [
                    f"## {experiment_id}",
                    "",
                    f"- Hypothesis: {record.get('hypothesis', 'unavailable')}",
                    f"- Evidence: GAUC={metrics.get('GAUC', 'unavailable')}; nDCG@5={metrics.get('nDCG@5', 'unavailable')}; decision_primary={evidence_primary}; seed_std={pstdev(scores) if len(scores) > 1 else 'unavailable'}; delta={delta if delta is not None else 'unavailable'}.",
                    f"- Reflection: GAUC improved={self._improved(gauc_delta)} (delta={gauc_delta if gauc_delta is not None else 'unavailable'}); nDCG@5 improved={self._improved(ndcg_delta)} (delta={ndcg_delta if ndcg_delta is not None else 'unavailable'}); primary improved={self._improved(delta)}; improvement larger than seed std={self._larger_than_variance(delta, seed_std)}; complementarity delta={complementarity if complementarity is not None else 'unavailable'}.",
                    f"- Result: {decision}; comparable={validity.get('valid', False)}; stopping={metadata.get('stopped_by', 'unavailable')}.",
                    f"- Interpretation: {self._interpretation(record, decision)}",
                    f"- Confidence: {confidence}",
                    f"- Implication: {self._implication(record, decision)}",
                    f"- Next recommended experiment: {self._next_recommendation(record, decision)}",
                    "",
                ]
            )
            if record.get("error"):
                lines.insert(-1, f"- Error/recovery: {record.get('error')} / {record.get('recovery', 'none')}")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _allocation_status(record: Mapping[str, Any], resolution: Any) -> str:
        decision = (
            resolution.get("decision")
            if isinstance(resolution, Mapping)
            else record.get("decision")
        )
        if decision == "accepted":
            return "EXPLOIT"
        if (
            decision == "screened"
            and isinstance(record.get("delta_primary"), (int, float))
            and not isinstance(record.get("delta_primary"), bool)
            and float(record["delta_primary"]) <= 0.0
        ):
            return "ABANDON"
        if decision in {"rejected", "invalid", "failed", "critic_rejected"}:
            return "ABANDON"
        return "EXPLORE"

    @staticmethod
    def _metric_delta(value: Any, reference: Any) -> float | None:
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(reference, (int, float))
            and not isinstance(reference, bool)
        ):
            return float(value) - float(reference)
        return None

    @staticmethod
    def _improved(delta: Any) -> str:
        if not isinstance(delta, (int, float)) or isinstance(delta, bool):
            return "unavailable"
        return "yes" if float(delta) > 0.0 else "no"

    @staticmethod
    def _larger_than_variance(delta: Any, seed_std: float | None) -> str:
        if (
            not isinstance(delta, (int, float))
            or isinstance(delta, bool)
            or seed_std is None
        ):
            return "unavailable"
        return "yes" if abs(float(delta)) > seed_std else "no"

    def _region_statuses(
        self,
        records: Sequence[Mapping[str, Any]],
        resolutions: Mapping[str, Any],
    ) -> dict[str, str]:
        grouped: dict[str, list[str]] = {}
        for record in records:
            region = str(record.get("search_region_id") or "unassigned")
            grouped.setdefault(region, []).append(
                self._allocation_status(
                    record,
                    resolutions.get(str(record.get("experiment_id"))),
                )
            )
        result: dict[str, str] = {}
        for region, statuses in grouped.items():
            if "EXPLOIT" in statuses:
                result[region] = "EXPLOIT"
            elif len(statuses) >= 3 and all(status == "ABANDON" for status in statuses[-3:]):
                result[region] = "ABANDON"
            else:
                result[region] = "EXPLORE"
        return result

    @staticmethod
    def _baseline_seed_scores(resolutions: Mapping[str, Any]) -> list[float]:
        for resolution in reversed(list(resolutions.values())):
            if not isinstance(resolution, Mapping):
                continue
            certificate = resolution.get("certificate") or {}
            if certificate.get("comparator_experiment_id") == "baseline":
                return [float(value) for value in certificate.get("comparator_scores", [])]
        return []

    @staticmethod
    def _baseline_primary(scores: Sequence[float]) -> float:
        return mean(scores) if scores else FROZEN_CALIBRATED_BASELINE_PRIMARY

    @staticmethod
    def _confidence(record: Mapping[str, Any], resolution: Any) -> str:
        if isinstance(resolution, Mapping):
            certificate = resolution.get("certificate") or {}
            if certificate.get("confirmed") is True:
                return "high — matched seeds 0/1/2"
            return "medium — matched-seed rejection"
        if (record.get("config") or {}).get("fidelity") == "full":
            return "medium — full fidelity, not seed-confirmed"
        return "low — screening evidence only"

    @staticmethod
    def _interpretation(record: Mapping[str, Any], decision: Any) -> str:
        if record.get("error"):
            return "The run failed; this is infrastructure evidence, not a model-quality result."
        if decision == "accepted":
            return "The hypothesis survived comparable full-fidelity and seed-confirmation gates."
        if decision == "screened":
            return "The result ranks the idea for promotion only; it cannot establish a champion."
        return "The tested configuration did not clear the applicable promotion gate."

    @staticmethod
    def _implication(record: Mapping[str, Any], decision: Any) -> str:
        family = record.get("direction_id") or "this research family"
        if decision == "accepted":
            return f"Exploit {family} locally while preserving one-factor comparisons."
        if decision in {"rejected", "invalid"}:
            return f"Do not repeat this exact {family} configuration without a new mechanism."
        return f"Retain {family} as unresolved until comparable evidence exists."

    @staticmethod
    def _next_recommendation(record: Mapping[str, Any], decision: Any) -> str:
        if decision == "accepted":
            return "Test one bounded, evidence-backed refinement or freeze for certification."
        if decision == "screened":
            return "Promote only if it is among the best eligible screens and budget remains."
        return "Review separate GAUC/nDCG movement and choose a materially different hypothesis."

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def render_cycle_summary(
    *,
    cycle: int,
    hypothesis: str,
    record: Mapping[str, Any],
    state: Mapping[str, Any],
    contract: BenchmarkContract = BENCHMARK_CONTRACT,
    seed_status: str = "not required",
    lesson: str = "See research_lessons.md",
    next_hypothesis: str = "Evidence review required",
    baseline_primary: float = FROZEN_CALIBRATED_BASELINE_PRIMARY,
) -> str:
    metrics = dict(record.get("metrics") or {})
    metadata = dict(record.get("runner_metadata") or {})
    primary = metrics.get("primary")
    delta = (
        float(primary) - baseline_primary
        if isinstance(primary, (int, float)) and not isinstance(primary, bool)
        else None
    )
    remaining_trials = max(0, contract.max_iterations - int(state.get("completed_iterations", 0)))
    remaining_seconds = max(
        0.0,
        contract.max_wall_clock_seconds - float(state.get("elapsed_seconds", 0.0)),
    )
    fields = {
        "CYCLE": cycle,
        "HYPOTHESIS": hypothesis,
        "CHANGE": record.get("changed_factors") or [],
        "FIDELITY": (record.get("config") or {}).get("fidelity", "unavailable"),
        "PRIMARY": primary if primary is not None else "unavailable",
        "GAUC": metrics.get("GAUC", "unavailable"),
        "NDCG@5": metrics.get("nDCG@5", "unavailable"),
        "DELTA VS BASELINE": delta if delta is not None else "unavailable",
        "DELTA VS CHAMPION": record.get("delta_primary", "unavailable"),
        "ENSEMBLE DELTA": metadata.get("ensemble_delta_if_added", "not measured"),
        "SEED STATUS": seed_status,
        "DECISION": record.get("decision", "unavailable"),
        "LESSON": lesson,
        "NEXT HYPOTHESIS": next_hypothesis,
        "BUDGET REMAINING": f"{remaining_trials} trials; {remaining_seconds:.1f}s active runtime",
    }
    return "\n".join(f"{key}: {value}" for key, value in fields.items())
