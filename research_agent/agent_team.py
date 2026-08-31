"""LLM roles for broad proposal, critique, coding, and evidence review.

Each role is an independent structured call.  Generated candidate source is
kept in an experiment artifact, never written into the shared repository.
"""
from __future__ import annotations

import ast
import builtins
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .architecture_context import ArchitectureContext
from .llm_planner import LLMPlanningError, OpenAIResponsesClient
from .planner import ResearchDirection
from .research_tools import ResearchToolCatalog
from .state import ResearchState
from .search import SearchSpaceExhausted
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
    implementation_surface: str = "isolated_candidate"
    revisit_rationale: str = ""


@dataclass(frozen=True)
class Critique:
    decision: str  # approved, needs_human_review, rejected
    rationale: str


@dataclass(frozen=True)
class GeneratedCandidate:
    """Candidate source plus its serializable, logged training configuration."""

    source: str
    config_patch: Mapping[str, Any]


class LLMResearchTeam:
    """Specialist LLM roles; deterministic checks remain the final authority."""

    def __init__(self, client: OpenAIResponsesClient, architecture: ArchitectureContext | None = None) -> None:
        self.client = client
        self.architecture = architecture

    def _instruction(self, role: str) -> str:
        """Give every specialist the same current architecture, not stale copies."""
        if self.architecture is None:
            return role
        return f"{role}\n\n{self.architecture.prompt_context()}"

    def propose(
        self,
        history: Sequence[Mapping[str, Any]],
        state: Mapping[str, Any],
        capabilities: Sequence[Mapping[str, Any]] = (),
        research_memory: Mapping[str, Any] | None = None,
    ) -> tuple[BroadProposal, dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["hypothesis", "rationale", "area", "controlled_change", "model_family", "requires_human_review", "leakage_risks", "implementation_surface", "revisit_rationale"], "properties": {
            "hypothesis": {"type": "string"}, "rationale": {"type": "string"},
            "area": {"type": "string", "enum": ["training", "feature", "sampling", "multitask", "model", "evaluation"]},
            "controlled_change": {"type": "string"}, "model_family": {"type": "string"},
            "requires_human_review": {"type": "boolean"}, "leakage_risks": {"type": "array", "items": {"type": "string"}},
            "implementation_surface": {"type": "string", "enum": ["isolated_candidate", "human_review"]},
            "revisit_rationale": {"type": "string"},
        }}
        prompt = json.dumps({"state": state, "recent_experiments": list(history[-8:]), "cross_run_research_memory": dict(research_memory or {}), "verified_capabilities": list(capabilities[-12:]), "rules": "Use only the rows and fields already returned by data.py; do not request extra KuaiRand files, external datasets, raw CSV access, or validation/test labels. One main change only. Use the cross-run evidence: do not simply repeat a method that repeatedly failed. It is not a forbidden-method list: you may revisit it only when revisit_rationale states the material difference from prior attempts; otherwise leave revisit_rationale empty. Use isolated_candidate for every ordinary in-scope FM experiment: it may implement its own PyTorch model, training loop, ranking objective, sampling, and leakage-safe features using only prepared rows. Select human_review only for a new dependency, a substantially different model family, external data, or work outside the benchmark contract."}, default=str)
        data, meta = self.client.create_json(self._instruction("You are the research-planning specialist. Propose one evidence-based experiment, not numeric hyperparameters."), prompt, schema=schema)
        return BroadProposal(data["hypothesis"], data["rationale"], data["area"], data["controlled_change"], data["model_family"], bool(data["requires_human_review"]), tuple(data["leakage_risks"]), str(data.get("implementation_surface", "isolated_candidate")), str(data.get("revisit_rationale", ""))), meta

    def choose_research_tool(
        self,
        history: Sequence[Mapping[str, Any]],
        state: Mapping[str, Any],
        tools: ResearchToolCatalog,
        specialist_findings: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[ResearchDirection, dict[str, Any]]:
        """Let the LLM choose what to investigate, never exact trial values."""
        schema = {"type": "object", "additionalProperties": False, "required": ["tool_id", "hypothesis", "rationale"], "properties": {
            "tool_id": {"type": "string", "enum": [item["tool_id"] for item in tools.prompt_records()]},
            "hypothesis": {"type": "string"},
            "rationale": {"type": "string"},
        }}
        prompt = json.dumps(
            {
                "current_state": dict(state),
                "recent_experiments": list(history[-12:]),
                "available_research_tools": tools.prompt_records(),
                "specialist_findings": list(specialist_findings),
                "decision_rule": "Choose the high-level tool and explain why it is the best next research direction. Do not select exact hyperparameter values; the search controller does that. If evidence is weak or plateaued, prefer a diverse tool/region.",
            },
            default=str,
        )
        data, meta = self.client.create_json(
            self._instruction("You are the research-orchestrator agent. Choose the next available research tool using the current evidence and architecture guidance."),
            prompt,
            schema=schema,
        )
        return tools.direction(data["tool_id"], hypothesis=data["hypothesis"], rationale=data["rationale"]), meta

    def consult_specialists(
        self,
        history: Sequence[Mapping[str, Any]],
        state: Mapping[str, Any],
        tools: ResearchToolCatalog,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Collect independent, evidence-bound advice before coordination."""
        schema = {"type": "object", "additionalProperties": False, "required": ["area", "finding", "recommended_tool_id"], "properties": {
            "area": {"type": "string"}, "finding": {"type": "string"},
            "recommended_tool_id": {"type": "string", "enum": [item["tool_id"] for item in tools.prompt_records()]},
        }}
        evidence = json.dumps({"state": dict(state), "recent_experiments": list(history[-12:]), "available_research_tools": tools.prompt_records()}, default=str)
        findings, metadata = [], []
        for role in (
            "feature and data specialist", "model architecture specialist", "training specialist",
            "sampling specialist", "evaluation specialist",
        ):
            data, meta = self.client.create_json(
                self._instruction(
                    f"You are the {role}. Inspect only the supplied evidence. State one concise finding and recommend the most relevant available research tool. Do not select numeric settings, access data, or invent an unsupported tool."
                ),
                evidence,
                schema=schema,
            )
            findings.append(data)
            metadata.append({"role": role, **meta})
        return findings, metadata

    def critique(self, proposal: BroadProposal) -> tuple[Critique, dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["decision", "rationale"], "properties": {"decision": {"type": "string", "enum": ["approved", "needs_human_review", "rejected"]}, "rationale": {"type": "string"}}}
        data, meta = self.client.create_json(self._instruction("You are a benchmark-safety critic. Reject leakage, test usage, external data, or unrelated multi-change proposals. Require review for new dependencies or model families."), json.dumps(proposal.__dict__), schema=schema)
        return Critique(data["decision"], data["rationale"]), meta

    def code(self, proposal: BroadProposal, *, failure: str = "") -> tuple[GeneratedCandidate, dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["source", "config_patch"], "properties": {
            "source": {"type": "string"},
            "config_patch": {"type": "object", "additionalProperties": False, "required": ["loss", "learning_rate", "l2", "embedding_dim", "batch_size", "epochs", "patience", "extension_name"], "properties": {
                "loss": {"type": "string", "enum": ["pointwise", "pairwise", "custom"]},
                "learning_rate": {"type": "number"}, "l2": {"type": "number"},
                "embedding_dim": {"type": "integer"}, "batch_size": {"type": "integer"},
                "epochs": {"type": "integer"}, "patience": {"type": "integer"},
                "extension_name": {"type": "string"},
            }},
        }}
        instruction = "You are an autonomous but isolated candidate-coding specialist. Return Python source defining exactly one public run_candidate(prepared, config, run_dir) function and a serializable config_patch. This is a complete in-scope PyTorch candidate, not a hook: it may define helper functions/classes and implement its own FM-family model, training loop, ranking objective, sampler, and leakage-safe feature logic. prepared exposes only train_rows and validation_rows from data.py; test rows do not exist. torch, np, CandidateOutput, and deterministic config values are injected by the host, so do not import anything. Return CandidateOutput(user_ids, labels, scores, metadata); scores must align to every validation row. Do not use files (including run_dir), network, subprocesses, eval/exec, raw CSVs, external data, test data/labels, or unapproved dependencies. Do not modify baseline.py, evaluate.py, data.py, or shared code. Keep one declared main change and use PyTorch only."
        data, meta = self.client.create_json(
            self._instruction(instruction),
            json.dumps({"proposal": proposal.__dict__, "previous_failure": failure}),
            schema=schema,
        )
        return GeneratedCandidate(data["source"], data["config_patch"]), meta

    def review(self, history: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["decision", "rationale"], "properties": {"decision": {"type": "string", "enum": ["refine", "explore", "restart"]}, "rationale": {"type": "string"}}}
        return self.client.create_json(self._instruction("You are the evidence-review specialist. Interpret GAUC and nDCG@5 and choose the next high-level action."), json.dumps(list(history[-8:])), schema=schema)

    def verify_capability(self, proposal: BroadProposal, source: str) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["decision", "rationale"], "properties": {"decision": {"type": "string", "enum": ["verified", "rejected", "needs_human_review"]}, "rationale": {"type": "string"}}}
        prompt = json.dumps({"proposal": proposal.__dict__, "source": source})
        return self.client.create_json(self._instruction("You are an isolated-candidate safety verifier. Verify that the source is an in-scope PyTorch FM-family candidate which implements the stated single change using only prepared train/validation rows, returns CandidateOutput, and contains no imports, filesystem, network, subprocess, test data, external data, unapproved dependency, or protected-file behavior. Full model and training-loop implementations are allowed. Reject only unsafe, invalid, or genuinely out-of-contract work; do not reject it because it is not a registered hook."), prompt, schema=schema)

    def decide_capability_recovery(self, proposal: BroadProposal, failure: str) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["decision", "rationale"], "properties": {
            "decision": {"type": "string", "enum": ["repair", "abandon"]}, "rationale": {"type": "string"},
        }}
        prompt = json.dumps({"proposal": proposal.__dict__, "failure": failure})
        return self.client.create_json(self._instruction("You are the research-orchestrator recovery specialist. Decide whether to repair this failed in-scope capability once or abandon it and seek a different hypothesis. Prefer repair only when the failure is clearly implementation-related. The only valid interfaces are sampler(labels, seed), loss_function(logits, labels), and feature_transform(train_features, valid_features, feature_dim); do not request unavailable scores, epochs, raw rows, files, or model internals."), prompt, schema=schema)


class LLMResearchPlanner:
    """Adapter that exposes the LLM's high-level choices to the search loop."""

    def __init__(self, team: LLMResearchTeam, tools: ResearchToolCatalog | None = None) -> None:
        self.team = team
        self.tools = tools or ResearchToolCatalog()
        self.last_metadata: dict[str, Any] | None = None
        self.last_specialist_metadata: list[dict[str, Any]] = []

    def propose(self, history: Sequence[Mapping[str, Any]], state: ResearchState) -> ResearchDirection:
        if not self.tools.prompt_records():
            raise SearchSpaceExhausted("all available laboratory tools are exhausted")
        findings, metadata = self.team.consult_specialists(history, state.as_dict(), self.tools)
        self.last_specialist_metadata = [
            {**item, "finding": finding} for item, finding in zip(metadata, findings)
        ]
        direction, self.last_metadata = self.team.choose_research_tool(
            history, state.as_dict(), self.tools, specialist_findings=findings,
        )
        return direction

    def mark_exhausted(self, direction_id: str) -> None:
        self.tools.exclude(direction_id)


def build_isolated_candidate(source: str, workspace: Path) -> CandidateCallable:
    """Build a full candidate in a workspace with a deliberately tiny runtime surface."""
    source = _plain_source(source)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "candidate.py").write_text(source, encoding="utf-8")
    tree = ast.parse(source, filename="candidate.py", mode="exec")
    forbidden = (ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.Raise, ast.Global, ast.Nonlocal)
    forbidden_attributes = {"load", "save", "download", "from_file", "system", "popen", "run", "cuda"}
    for node in ast.walk(tree):
        if (
            isinstance(node, forbidden)
            or (isinstance(node, ast.Name) and node.id in {"open", "exec", "eval", "__import__", "compile", "input"})
            or (isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in forbidden_attributes))
        ):
            raise LLMPlanningError("generated candidate failed isolated-code safety validation")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_candidate"]
    if len(functions) != 1:
        raise LLMPlanningError("generated candidate must define exactly one run_candidate function")
    arguments = functions[0].args
    if [argument.arg for argument in arguments.args] != ["prepared", "config", "run_dir"] or arguments.vararg or arguments.kwarg:
        raise LLMPlanningError("run_candidate must accept exactly (prepared, config, run_dir)")
    import numpy as np
    import torch
    from .runner import CandidateOutput
    from .models.torch_fm import run_torch_fm_extension
    # The legacy extension remains injected only so existing saved candidates can
    # be replayed.  New candidates are never required to call it.
    scope: dict[str, Any] = {"__builtins__": {"__build_class__": builtins.__build_class__, "object": object, "super": super, "len": len, "range": range, "int": int, "float": float, "min": min, "max": max, "sum": sum, "dict": dict, "list": list, "tuple": tuple, "abs": abs, "bool": bool, "isinstance": isinstance, "enumerate": enumerate, "zip": zip}, "__name__": "isolated_candidate", "run_torch_fm_extension": run_torch_fm_extension, "CandidateOutput": CandidateOutput, "torch": torch, "np": np}
    exec(compile(tree, "candidate.py", "exec"), scope, scope)
    return scope["run_candidate"]


def _plain_source(source: str) -> str:
    """Remove an accidental Markdown code fence without accepting other prose."""
    stripped = source.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return "\n".join(stripped.splitlines()[1:-1]).strip() + "\n"
    return source
