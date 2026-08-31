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
        lines.extend(self._render_proposals())
        iterations = self.store.read_iterations()
        if not iterations:
            lines.extend(["No completed iterations have been recorded.", ""])
        for record in iterations:
            lines.extend(self._render_iteration(record))
        lines.extend(self._render_interventions())
        lines.extend(self._render_final_summary())
        return "\n".join(lines).rstrip() + "\n"

    def _render_proposals(self) -> list[str]:
        """Render the LLM decision trail as one clearly separated proposal block."""
        events = self.store.read_events()
        proposals: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] | None = None
        for event in events:
            if event["action"] in {"llm_broad_hypothesis_proposed", "research_direction_proposed"}:
                current = [event]
                proposals.append(current)
            elif current is not None:
                current.append(event)

        if not proposals:
            return []

        lines = ["## LLM Proposals", ""]
        for number, events_for_proposal in enumerate(proposals, start=1):
            proposal_event = events_for_proposal[0]
            proposal = proposal_event["details"]
            if proposal_event["action"] == "research_direction_proposed":
                lines.extend(self._render_tool_direction(number, proposal))
                for event in events_for_proposal[1:]:
                    lines.extend(self._render_proposal_event(event))
                lines.append("")
                continue
            lines.extend(
                [
                    f"### Proposal {number}",
                    "",
                    f"- Hypothesis: {proposal.get('hypothesis', 'unavailable')}",
                    f"- Why: {proposal.get('rationale', 'unavailable')}",
                    f"- Controlled change: {proposal.get('controlled_change', 'unavailable')}",
                    f"- Area: {proposal.get('area', 'unavailable')}",
                    f"- Model family: {proposal.get('model_family', 'unavailable')}",
                    f"- Leakage checks: {'; '.join(proposal.get('leakage_risks', [])) or 'none recorded'}",
                ]
            )
            for event in events_for_proposal[1:]:
                lines.extend(self._render_proposal_event(event))
            lines.append("")
        return lines

    @staticmethod
    def _render_tool_direction(number: int, direction: dict[str, Any]) -> list[str]:
        space = direction.get("search_space", {})
        return [
            f"### Proposal {number}",
            "",
            f"- Hypothesis: {direction.get('hypothesis', 'unavailable')}",
            f"- Why: {direction.get('rationale', 'unavailable')}",
            f"- Research tool: {direction.get('direction_id', 'unavailable')}",
            f"- Search strategy: {direction.get('strategy', 'unavailable')}",
            f"- Search space: {space or 'unavailable'}",
            f"- Evaluation budget: {direction.get('evaluation_budget', 'unavailable')}",
        ]

    @staticmethod
    def _render_proposal_event(event: dict[str, Any]) -> list[str]:
        """Keep the outcome of each proposal visible without dumping raw JSON."""
        action, details = event["action"], event.get("details", {})
        if action == "llm_critic_completed":
            return [f"- Critic: {details.get('decision', 'unavailable')} — {details.get('rationale', '')}"]
        if action == "llm_specialist_consulted":
            finding = details.get("finding", {})
            return [
                f"- Specialist ({details.get('role', 'unavailable')}): "
                f"{finding.get('finding', 'no finding recorded')} "
                f"→ {finding.get('recommended_tool_id', 'no tool recorded')}"
            ]
        if action == "human_review_required":
            return [f"- Human review required: {details.get('reason', 'no reason recorded')}"]
        if action == "generated_candidate_rejected":
            return [f"- Candidate code rejected before execution: {details.get('error', 'unavailable')}"]
        if action == "isolated_candidate_verifier_completed":
            return [f"- Candidate verifier: {details.get('decision', 'unavailable')} — {details.get('rationale', '')}"]
        if action == "isolated_candidate_static_check_passed":
            return ["- Deterministic isolated-source safety check passed."]
        if action == "isolated_candidate_synthetic_contract_completed":
            return ["- Synthetic typed-input preflight passed (including string user IDs)."]
        if action == "isolated_candidate_preflight_completed":
            return ["- Real-data candidate preflight passed."]
        if action == "isolated_candidate_preflight_failed":
            return [f"- Real-data candidate preflight failed: {details.get('error', 'unavailable')}"]
        if action == "isolated_candidate_rejected":
            return [f"- Candidate rejected before training: {details.get('reason', 'unavailable')}"]
        if action == "candidate_recovery_decided":
            return [f"- Candidate recovery decision: {details.get('decision', 'unavailable')} — {details.get('rationale', '')}"]
        if action == "candidate_abandoned":
            return [f"- Candidate abandoned: {details.get('reason', 'no reason recorded')}"]
        if action == "coding_subagent_completed":
            return [f"- Candidate implementation prepared: {details.get('workspace', 'unavailable')}"]
        if action == "evidence_reviewed":
            return [f"- Evidence review: {details.get('action', 'unavailable')} — {details.get('rationale', '')}"]
        if action in {"proposal_critic_reviewed", "promotion_critic_reviewed"}:
            status = "approved" if details.get("approved") else "rejected"
            return [f"- Safety critic ({action}): {status}; reasons: {', '.join(details.get('reasons', [])) or 'none'}"]
        return []

    def _render_iteration(self, record: dict[str, Any]) -> list[str]:
        metrics = record.get("metrics", {})
        lines = [
            f"## {record['experiment_id']}",
            "",
            f"- Decision: {record['decision']}",
            f"- Hypothesis: {record['hypothesis']}",
            f"- Rationale: {record['rationale']}",
            f"- GAUC: {metrics.get('GAUC', 'unavailable')}",
            f"- nDCG@5: {metrics.get('nDCG@5', 'unavailable')}",
            f"- Primary: {metrics.get('primary', 'unavailable')}",
            f"- Delta primary: {record.get('delta_primary', 'unavailable')}",
            f"- Runtime seconds: {record.get('runtime_seconds', 'unavailable')}",
        ]
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
        for key in ("selected_experiment_id", "selection_primary", "test_GAUC", "test_nDCG@5", "test_primary", "submission_path", "submission_checked"):
            lines.append(f"- {key}: {summary.get(key, 'unavailable')}")
        lines.append("")
        return lines
