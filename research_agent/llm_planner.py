"""OpenAI-powered high-level hypothesis planning; no model code or data access."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .planner import ResearchDirection
from .state import ResearchState


class LLMPlanningError(RuntimeError):
    """A planning failure must pause the run rather than fall back silently."""


DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_RETRIES = 2


@dataclass(frozen=True)
class OpenAIResponsesClient:
    api_key: str
    model: str
    endpoint: str = "https://api.openai.com/v1/responses"
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES

    def create_json(self, instructions: str, prompt: str, *, schema: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = schema or {
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
        retry_errors: list[str] = []
        last_exception: Exception | None = None
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                last_exception = exc
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                error = f"HTTP {exc.code}: {detail}"
                retryable = exc.code == 408 or exc.code == 429 or 500 <= exc.code < 600
            except (URLError, TimeoutError, OSError) as exc:
                last_exception = exc
                error = str(exc)
                retryable = True
            retry_errors.append(error)
            if not retryable or attempt == attempts:
                raise LLMPlanningError(
                    f"OpenAI planning request failed after {attempt} attempt(s): {error}"
                ) from last_exception
            time.sleep(min(2 ** (attempt - 1), 4))
        text = raw.get("output_text") or _extract_output_text(raw)
        if not isinstance(text, str):
            raise LLMPlanningError("OpenAI response did not include output_text")
        try:
            return json.loads(text), {
                "model": self.model,
                "usage": raw.get("usage", {}),
                "response_id": raw.get("id"),
                "request_attempts": attempt,
                "transient_request_errors": retry_errors,
            }
        except json.JSONDecodeError as exc:
            raise LLMPlanningError("OpenAI response was not valid structured JSON") from exc


def _extract_output_text(raw: Mapping[str, Any]) -> str | None:
    """Support Responses payloads that return structured output content."""
    for item in raw.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    return None


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
        try:
            timeout_seconds = float(os.environ.get("OPENAI_REQUEST_TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS)))
            max_retries = int(os.environ.get("OPENAI_REQUEST_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
        except ValueError as exc:
            raise LLMPlanningError("OPENAI_REQUEST_TIMEOUT_SECONDS must be a number and OPENAI_REQUEST_MAX_RETRIES must be an integer") from exc
        if timeout_seconds <= 0 or max_retries < 0:
            raise LLMPlanningError("OPENAI_REQUEST_TIMEOUT_SECONDS must be positive and OPENAI_REQUEST_MAX_RETRIES cannot be negative")
        return cls(OpenAIResponsesClient(api_key=key, model=model, timeout_seconds=timeout_seconds, max_retries=max_retries))

    def propose(self, history: Sequence[Mapping[str, Any]], state: ResearchState) -> ResearchDirection:
        response, metadata = self.client.create_json(_instructions(), _prompt(history, state))
        direction_id = response.get("direction_id")
        template = _APPROVED_DIRECTIONS.get(direction_id)
        if template is None:
            raise LLMPlanningError("LLM proposed an unsupported direction")
        hypothesis = str(response.get("hypothesis", "")).strip()
        rationale = str(response.get("rationale", "")).strip()
        strategy = str(response.get("strategy", "")).strip()
        if len(hypothesis) < 20 or len(rationale) < 20:
            raise LLMPlanningError("LLM proposal lacks a sufficiently specific hypothesis or rationale")
        self.last_metadata = {**metadata, "raw_proposal": response}
        return ResearchDirection(
            direction_id=direction_id,
            hypothesis=hypothesis,
            rationale=rationale,
            search_space=template["search_space"],
            success_evidence="Validation primary improves by more than 0.002 over the accepted parent.",
            evaluation_budget={"low_epochs": 4, "full_epochs": 40, "patience": 4},
            strategy=strategy,
        )


_APPROVED_DIRECTIONS: Mapping[str, Mapping[str, Any]] = {
    "pointwise_fm_optimization": {"search_space": {"loss": ["pointwise"], "learning_rate": [0.0005, 0.001, 0.002], "l2": [0.0, 1e-6, 1e-5]}},
    "pairwise_fm_ranking": {"search_space": {"loss": ["pairwise"], "learning_rate": [0.0005, 0.001], "l2": [0.0, 1e-6]}},
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
    compact = [
        {key: item.get(key) for key in ("experiment_id", "hypothesis", "direction_id", "metrics", "decision", "error")}
        for item in history[-8:]
    ]
    return json.dumps(
        {
            "benchmark": {"selection_split": "valid", "objective": "primary=(GAUC+nDCG@5)/2", "improvement_threshold": 0.002},
            "state": state.as_dict(),
            "approved_directions": list(_APPROVED_DIRECTIONS),
            "recent_evidence": compact,
            "request": "Choose exactly one approved direction and explain a distinct, evidence-based hypothesis.",
        },
        sort_keys=True,
    )
