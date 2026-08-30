"""Renders human-readable reports only from structured run records."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .store import ArtifactStore


class MarkdownReporter:
    """Generates the readable log from the append-only artifact store."""

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def write(self, destination: str | Path | None = None) -> Path:
        target = Path(destination) if destination else self.store.root / "research_log.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(), encoding="utf-8")
        return target

    def render(self) -> str:
        lines = ["# Research Run Log", ""]
        lines.extend(self._render_run_summary())
        iterations = self.store.read_iterations()
        resolutions = self.store.read_root_json("promotion_resolutions.json") or {}
        if not iterations:
            lines.extend(["No completed iterations have been recorded.", ""])
        for record in iterations:
            lines.extend(
                self._render_iteration(
                    record,
                    resolutions.get(str(record.get("experiment_id"))),
                )
            )
        lines.extend(self._render_interventions())
        lines.extend(self._render_final_summary())
        return "\n".join(lines).rstrip() + "\n"

    def _render_run_summary(self) -> list[str]:
        state = self.store.read_root_json("state.json") or {}
        manifest = self.store.read_root_json("run_manifest.json") or {}
        events = self.store.read_events()
        input_tokens = 0
        output_tokens = 0
        fallback_count = 0
        for event in events:
            details = event.get("details") or {}
            usage = details.get("usage") or {}
            input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            if details.get("degraded") is True:
                fallback_count += 1
        return [
            "## Run Summary",
            "",
            f"- Stop code: {state.get('stop_reason_code', 'not stopped')}",
            f"- Stop reason: {state.get('stop_reason', 'not stopped')}",
            f"- Research cycles: {state.get('completed_cycles', 0)}",
            f"- Logical trials: {state.get('completed_iterations', 0)}",
            f"- Runner attempts: {state.get('runner_attempts', 0)}",
            f"- Valid full comparisons: {state.get('valid_comparisons', 0)}",
            f"- Elapsed seconds: {state.get('elapsed_seconds', 0)}",
            f"- LLM input tokens: {input_tokens}",
            f"- LLM output tokens: {output_tokens}",
            f"- Deterministic planner fallbacks: {fallback_count}",
            f"- Python: {manifest.get('python_version', 'unavailable')}",
            f"- Platform: {manifest.get('platform', 'unavailable')}",
            f"- Torch: {manifest.get('torch_version', 'unavailable')}",
            f"- CUDA available: {manifest.get('cuda_available', 'unavailable')}",
            f"- Evaluator SHA-256: {manifest.get('evaluator_sha256', 'unavailable')}",
            f"- Dataset SHA-256: {manifest.get('data_sha256', 'unavailable')}",
            f"- Screen promotion width: {state.get('screen_promotion_width', 1)}",
            f"- Last pause reason: {state.get('last_pause_reason', 'none')}",
            "",
        ]

    def _render_iteration(
        self,
        record: dict[str, Any],
        resolution: Any = None,
    ) -> list[str]:
        metrics = record.get("metrics", {})
        effective_decision = (
            resolution.get("decision")
            if isinstance(resolution, dict)
            else record["decision"]
        )
        lines = [
            f"## {record['experiment_id']}",
            "",
            f"- Decision: {effective_decision}",
            f"- Hypothesis: {record['hypothesis']}",
            f"- Rationale: {record['rationale']}",
            f"- GAUC: {metrics.get('GAUC', 'unavailable')}",
            f"- nDCG@5: {metrics.get('nDCG@5', 'unavailable')}",
            f"- Primary: {metrics.get('primary', 'unavailable')}",
            f"- Delta primary: {record.get('delta_primary', 'unavailable')}",
            f"- Runtime seconds: {record.get('runtime_seconds', 'unavailable')}",
            f"- Comparison incumbent: {record.get('comparison_incumbent_id', 'unavailable')}",
        ]
        if effective_decision != record["decision"]:
            lines.append(f"- Initial decision: {record['decision']}")
            certificate = resolution.get("certificate") or {}
            lines.extend(
                [
                    f"- Seed confirmation mean delta: {certificate.get('mean_delta', 'unavailable')}",
                    f"- Seed confirmation wins: {certificate.get('wins', 'unavailable')}",
                    f"- Seed confirmation attempts: {certificate.get('confirmation_attempts', 'unavailable')}",
                    f"- Candidate comparison groups: {certificate.get('candidate_comparison_groups', [])}",
                    f"- Comparator comparison groups: {certificate.get('comparator_comparison_groups', [])}",
                ]
            )
        validity = record.get("comparison_validity") or {}
        if validity:
            lines.extend(
                [
                    f"- Comparison valid: {validity.get('valid', False)}",
                    f"- Comparison validity reasons: {validity.get('reasons', [])}",
                ]
            )
        metadata = record.get("runner_metadata") or {}
        if metadata:
            lines.extend(
                [
                    f"- Termination: {metadata.get('stopped_by', 'unavailable')}",
                    f"- Epochs run: {metadata.get('epochs_run', 'unavailable')}",
                    f"- Seed: {metadata.get('seed', 'unavailable')}",
                    f"- Comparison group: {metadata.get('comparison_group_id', 'unavailable')}",
                    f"- Model code SHA-256: {metadata.get('model_code_sha256', 'unavailable')}",
                ]
            )
        if record.get("error"):
            lines.append(f"- Error: {record['error']}")
        if record.get("recovery"):
            lines.append(f"- Recovery: {record['recovery']}")
        lines.append("")
        return lines

    def _render_interventions(self) -> list[str]:
        interventions = self.store.read_interventions()
        lines = ["## Manual Intervention Summary", "", f"Manual interventions: {len(interventions)}", ""]
        for index, item in enumerate(interventions, start=1):
            lines.extend(
                [
                    f"{index}. {item['timestamp']} — {item.get('experiment_id') or 'run-wide'}",
                    f"   - Action: {item['description']}",
                    f"   - Reason: {item['reason']}",
                    f"   - Effect: {item['effect']}",
                    "",
                ]
            )
        return lines

    def _render_final_summary(self) -> list[str]:
        summary = self.store.read_root_json("final_summary.json")
        if not summary:
            return []
        lines = ["## Final Summary", ""]
        confirmation = self.store.read_root_json("seed_confirmation.json") or {}
        for key in (
            "seeds",
            "candidate_scores",
            "comparator_scores",
            "mean_delta",
            "wins",
            "confirmed",
            "comparison_mode",
            "candidate_comparison_groups",
            "comparator_comparison_groups",
            "confirmation_attempts",
        ):
            lines.append(f"- seed_{key}: {confirmation.get(key, 'unavailable')}")
        for key in ("selected_experiment_id", "selection_primary", "test_GAUC", "test_nDCG@5", "test_primary", "submission_path", "submission_checked"):
            lines.append(f"- {key}: {summary.get(key, 'unavailable')}")
        lines.append("")
        return lines
