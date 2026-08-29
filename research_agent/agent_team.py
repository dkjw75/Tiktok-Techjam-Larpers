"""LLM roles for broad proposal, critique, coding, and evidence review.

Each role is an independent structured call.  Generated candidate source is
kept in an experiment artifact, never written into the shared repository.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .llm_planner import LLMPlanningError, OpenAIResponsesClient
from .runner import CandidateCallable


@dataclass(frozen=True)
class BroadProposal:
    hypothesis: str
    rationale: str
    area: str
    controlled_change: str
    model_family: str
    requires_human_review: bool
    leakage_risks: tuple[str, ...]


@dataclass(frozen=True)
class Critique:
    decision: str  # approved, needs_human_review, rejected
    rationale: str


class LLMResearchTeam:
    """Specialist LLM roles; deterministic checks remain the final authority."""

    def __init__(self, client: OpenAIResponsesClient) -> None:
        self.client = client

    def propose(self, history: Sequence[Mapping[str, Any]], state: Mapping[str, Any]) -> tuple[BroadProposal, dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["hypothesis", "rationale", "area", "controlled_change", "model_family", "requires_human_review", "leakage_risks"], "properties": {
            "hypothesis": {"type": "string"}, "rationale": {"type": "string"},
            "area": {"type": "string", "enum": ["training", "feature", "sampling", "multitask", "model", "evaluation"]},
            "controlled_change": {"type": "string"}, "model_family": {"type": "string"},
            "requires_human_review": {"type": "boolean"}, "leakage_risks": {"type": "array", "items": {"type": "string"}},
        }}
        prompt = json.dumps({"state": state, "recent_experiments": list(history[-8:]), "rules": "Use only the rows and fields already returned by data.py; do not request extra KuaiRand files, external datasets, raw CSV access, or validation/test labels. One main change only. Prefer an automatically runnable change within the existing PyTorch FM before proposing a new model family/dependency, which requires review."}, default=str)
        data, meta = self.client.create_json("You are the research-planning specialist. Propose one evidence-based experiment, not numeric hyperparameters.", prompt, schema=schema)
        return BroadProposal(data["hypothesis"], data["rationale"], data["area"], data["controlled_change"], data["model_family"], bool(data["requires_human_review"]), tuple(data["leakage_risks"])), meta

    def critique(self, proposal: BroadProposal) -> tuple[Critique, dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["decision", "rationale"], "properties": {"decision": {"type": "string", "enum": ["approved", "needs_human_review", "rejected"]}, "rationale": {"type": "string"}}}
        data, meta = self.client.create_json("You are a benchmark-safety critic. Reject leakage, test usage, external data, or unrelated multi-change proposals. Require review for new dependencies or model families.", json.dumps(proposal.__dict__), schema=schema)
        return Critique(data["decision"], data["rationale"]), meta

    def code(self, proposal: BroadProposal) -> tuple[str, dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["source"], "properties": {"source": {"type": "string"}}}
        instruction = "You are an isolated candidate-coding specialist. Return plain Python source only (no Markdown): exactly a run_candidate(prepared, config, run_dir) function. No imports, files, network, subprocesses, eval/exec, test data, or external data. Use only injected run_torch_fm_candidate, torch, np, and the supplied arguments."
        data, meta = self.client.create_json(instruction, json.dumps(proposal.__dict__), schema=schema)
        return data["source"], meta

    def review(self, history: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["decision", "rationale"], "properties": {"decision": {"type": "string", "enum": ["refine", "explore", "restart"]}, "rationale": {"type": "string"}}}
        return self.client.create_json("You are the evidence-review specialist. Interpret GAUC and nDCG@5 and choose the next high-level action.", json.dumps(list(history[-8:])), schema=schema)


def build_isolated_candidate(source: str, workspace: Path) -> CandidateCallable:
    """Validate generated code and execute it with no imports or filesystem builtins."""
    source = _plain_source(source)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "candidate.py").write_text(source, encoding="utf-8")
    tree = ast.parse(source, filename="candidate.py", mode="exec")
    forbidden = (ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.Raise, ast.Global, ast.Nonlocal)
    for node in ast.walk(tree):
        if isinstance(node, forbidden) or (isinstance(node, ast.Name) and node.id in {"open", "exec", "eval", "__import__", "compile", "input"}) or (isinstance(node, ast.Attribute) and node.attr.startswith("__")):
            raise LLMPlanningError("generated candidate failed isolated-code safety validation")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_candidate"]
    if len(functions) != 1:
        raise LLMPlanningError("generated candidate must define exactly one run_candidate function")
    import numpy as np
    import torch
    from .models.torch_fm import run_torch_fm_candidate
    scope: dict[str, Any] = {"__builtins__": {"len": len, "range": range, "int": int, "float": float, "min": min, "max": max, "sum": sum, "dict": dict, "list": list, "tuple": tuple}, "run_torch_fm_candidate": run_torch_fm_candidate, "torch": torch, "np": np}
    exec(compile(tree, "candidate.py", "exec"), scope, scope)
    return scope["run_candidate"]


def _plain_source(source: str) -> str:
    """Remove an accidental Markdown code fence without accepting other prose."""
    stripped = source.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return "\n".join(stripped.splitlines()[1:-1]).strip() + "\n"
    return source
