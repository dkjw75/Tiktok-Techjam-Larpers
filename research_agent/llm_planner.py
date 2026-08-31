"""OpenAI-powered high-level hypothesis planning; no model code or data access."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .planner import ResearchDirection, ResearchPlanner
from .state import ResearchState


class LLMPlanningError(RuntimeError):
    """A planning failure must pause the run rather than fall back silently."""


@dataclass(frozen=True)
class OpenAIResponsesClient:
    api_key: str
    model: str
    endpoint: str = "https://api.openai.com/v1/responses"
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    sleep: Callable[[float], None] = time.sleep

    def create_json(self, instructions: str, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["direction_id", "hypothesis", "rationale", "strategy"],
            "properties": {
                "direction_id": {"type": "string", "enum": list(_APPROVED_DIRECTIONS)},
                "hypothesis": {"type": "string", "minLength": 20},
                "rationale": {"type": "string", "minLength": 20},
                "strategy": {"type": "string", "enum": ["exploration", "local_refinement", "diverse_restart"]},
            },
        }
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": prompt,
            "text": {"format": {"type": "json_schema", "name": "research_direction", "strict": True, "schema": schema}},
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urlopen(request, timeout=60) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.max_attempts:
                    raise LLMPlanningError(
                        f"OpenAI planning request failed after {attempt} attempt(s): HTTP {exc.code}"
                    ) from exc
            except (URLError, TimeoutError) as exc:
                if attempt == self.max_attempts:
                    raise LLMPlanningError(
                        f"OpenAI planning request failed after {attempt} attempt(s): {exc}"
                    ) from exc
            self.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

        if raw.get("status") == "incomplete":
            details = raw.get("incomplete_details") or {}
            raise LLMPlanningError(f"OpenAI response was incomplete: {details}")
        if raw.get("status") in {"failed", "cancelled"}:
            raise LLMPlanningError(f"OpenAI response ended with status {raw.get('status')}")
        text = _extract_output_text(raw)
        try:
            return json.loads(text), {
                "model": self.model,
                "usage": raw.get("usage", {}),
                "response_id": raw.get("id"),
                "status": raw.get("status"),
            }
        except json.JSONDecodeError as exc:
            raise LLMPlanningError("OpenAI response was not valid structured JSON") from exc


def _extract_output_text(raw: Mapping[str, Any]) -> str:
    """Extract structured text from the wire-format Responses API payload."""
    output_text = raw.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    refusal: str | None = None
    text_parts: list[str] = []
    output = raw.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                part_type = part.get("type")
                if part_type == "refusal" and isinstance(part.get("refusal"), str):
                    refusal = part["refusal"]
                if part_type == "output_text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
    if refusal:
        raise LLMPlanningError(f"OpenAI planning request was refused: {refusal}")
    if text_parts:
        return "".join(text_parts)
    raise LLMPlanningError("OpenAI response did not include structured output text")


class OpenAIPlanner:
    """The real research agent: an LLM chooses a safe investigation direction."""

    def __init__(self, client: OpenAIResponsesClient) -> None:
        self.client = client
        self.last_metadata: dict[str, Any] = {}

    @classmethod
    def from_environment(cls) -> "OpenAIPlanner":
        load_dotenv()
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("OPENAI_MODEL", "gpt-5").strip()
        if not key:
            raise LLMPlanningError("OPENAI_API_KEY is not set; add it to .env before starting the LLM agent")
        return cls(OpenAIResponsesClient(api_key=key, model=model))

    def propose(self, history: Sequence[Mapping[str, Any]], state: ResearchState) -> ResearchDirection:
        response, metadata = self.client.create_json(_instructions(), _prompt(history, state))
        if not isinstance(response, Mapping):
            raise LLMPlanningError("LLM structured response must be an object")
        direction_id = response.get("direction_id")
        if not isinstance(direction_id, str):
            raise LLMPlanningError("LLM direction_id must be a string")
        template = _APPROVED_DIRECTIONS.get(direction_id)
        if template is None:
            raise LLMPlanningError("LLM proposed an unsupported direction")
        hypothesis = str(response.get("hypothesis", "")).strip()
        rationale = str(response.get("rationale", "")).strip()
        strategy = str(response.get("strategy", "")).strip()
        if strategy not in {"exploration", "local_refinement", "diverse_restart"}:
            raise LLMPlanningError("LLM proposed an unsupported search strategy")
        if len(hypothesis) < 20 or len(rationale) < 20:
            raise LLMPlanningError("LLM proposal lacks a sufficiently specific hypothesis or rationale")
        self.last_metadata = {**metadata, "raw_proposal": response}
        return ResearchDirection(
            direction_id=direction_id,
            hypothesis=hypothesis,
            rationale=rationale,
            search_space=template["search_space"],
            success_evidence="Validation primary improves by more than 0.002 over the accepted parent.",
            evaluation_budget={
                "low_epochs": 4,
                "low_patience": 2,
                "full_epochs": 40,
                "full_patience": 4,
            },
            strategy=strategy,
        )


class ResilientPlanner:
    """Use a logged deterministic planner only after an operational LLM failure."""

    def __init__(self, primary: ResearchPlanner, fallback: ResearchPlanner) -> None:
        self.primary = primary
        self.fallback = fallback
        self.last_metadata: dict[str, Any] = {}

    def propose(
        self,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
    ) -> ResearchDirection:
        try:
            direction = self.primary.propose(history, state)
        except LLMPlanningError as exc:
            direction = self.fallback.propose(history, state)
            self.last_metadata = {
                "planner_mode": "deterministic_fallback",
                "degraded": True,
                "error": str(exc),
            }
            return direction
        self.last_metadata = {
            **dict(getattr(self.primary, "last_metadata", {})),
            "planner_mode": "llm",
            "degraded": False,
        }
        return direction


_APPROVED_DIRECTIONS: Mapping[str, Mapping[str, Any]] = {
    "rank_ensemble": {
        "search_space": {
            "loss": ["ensemble"],
            "member_set": ["core4", "core5", "core6"],
        }
    },
    "listwise_fm_ranking": {
        "search_space": {
            "loss": ["listwise"],
            "objective_variant": ["t1", "t05", "t1_bce25"],
        }
    },
}


def load_dotenv(path: str = ".env") -> None:
    """Load simple local secrets without adding a dependency."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


def _instructions() -> str:
    return (
        "You are the research-planning layer for a KuaiRand-Pure recommender experiment agent. "
        "Generate one clear hypothesis and rationale from the supplied evidence. You select the high-level "
        "approved direction only; never select numeric hyperparameters, access data, write code, use test labels, "
        "or propose external data. After a plateau, choose a direction meaningfully different from recent failures."
    )


def _prompt(history: Sequence[Mapping[str, Any]], state: ResearchState) -> str:
    evidence_review = next(
        (
            item.get("evidence_review")
            for item in reversed(history)
            if item.get("record_type") == "evidence_review"
        ),
        None,
    )
    experiment_history = [
        item for item in history if item.get("record_type") != "evidence_review"
    ]
    compact = [
        {key: item.get(key) for key in ("experiment_id", "hypothesis", "direction_id", "metrics", "decision", "error")}
        for item in experiment_history[-8:]
    ]
    return json.dumps(
        {
            "benchmark": {"selection_split": "valid", "objective": "primary=(GAUC+nDCG@5)/2", "improvement_threshold": 0.002},
            "state": state.as_dict(),
            "approved_directions": list(_APPROVED_DIRECTIONS),
            "recent_evidence": compact,
            "structured_evidence_review": evidence_review,
            "request": "Choose exactly one approved direction and explain a distinct, evidence-based hypothesis.",
        },
        sort_keys=True,
    )
