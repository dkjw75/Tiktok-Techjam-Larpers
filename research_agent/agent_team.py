"""LLM roles and a constrained full-candidate execution boundary."""
from __future__ import annotations

import ast
import builtins
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .architecture_context import ArchitectureContext
from .llm_planner import OpenAIResponsesClient
from .runner import CandidateCallable, CandidateOutput


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
    decision: str
    rationale: str


@dataclass(frozen=True)
class GeneratedCandidate:
    source: str
    config_patch: Mapping[str, Any]


class LLMResearchTeam:
    """LLM reasoning roles; source safety and execution remain deterministic."""

    def __init__(self, client: OpenAIResponsesClient, architecture: ArchitectureContext | None = None) -> None:
        self.client, self.architecture = client, architecture

    def _instruction(self, role: str) -> str:
        return role if self.architecture is None else f"{role}\n\n{self.architecture.prompt_context()}"

    def propose(self, history: Sequence[Mapping[str, Any]], state: Mapping[str, Any], research_memory: Mapping[str, Any] | None = None) -> tuple[BroadProposal, dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["hypothesis", "rationale", "area", "controlled_change", "model_family", "requires_human_review", "leakage_risks", "revisit_rationale"], "properties": {
            "hypothesis": {"type": "string"}, "rationale": {"type": "string"}, "area": {"type": "string", "enum": ["training", "feature", "sampling", "multitask", "model", "evaluation"]}, "controlled_change": {"type": "string"}, "model_family": {"type": "string"}, "requires_human_review": {"type": "boolean"}, "leakage_risks": {"type": "array", "items": {"type": "string"}}, "revisit_rationale": {"type": "string"},
        }}
        rules = "Propose one controlled in-scope FM-family experiment. It may write a complete PyTorch model and training loop using only canonical prepared arrays and train/validation user IDs. Never request raw CSV access, external data, test data/labels, network access, a new dependency, or protected-file changes. New model families require human review. Prior negative results are evidence, not bans; state a material difference before revisiting one."
        data, meta = self.client.create_json(self._instruction("You are the research-planning specialist."), json.dumps({"state": dict(state), "recent_experiments": list(history[-12:]), "cross_run_memory": dict(research_memory or {}), "rules": rules}, default=str), schema=schema)
        return BroadProposal(data["hypothesis"], data["rationale"], data["area"], data["controlled_change"], data["model_family"], bool(data["requires_human_review"]), tuple(data["leakage_risks"]), revisit_rationale=data["revisit_rationale"]), meta

    def critique(self, proposal: BroadProposal) -> tuple[Critique, dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["decision", "rationale"], "properties": {"decision": {"type": "string", "enum": ["approved", "needs_human_review", "rejected"]}, "rationale": {"type": "string"}}}
        data, meta = self.client.create_json(self._instruction("You are a benchmark-safety critic. Reject leakage, test use, external data, unapproved dependencies, model-family changes, and multi-change proposals."), json.dumps(proposal.__dict__), schema=schema)
        return Critique(data["decision"], data["rationale"]), meta

    def code(self, proposal: BroadProposal, *, failure: str = "") -> tuple[GeneratedCandidate, dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["source", "config_patch"], "properties": {"source": {"type": "string"}, "config_patch": {"type": "object", "additionalProperties": False, "required": ["learning_rate", "l2", "embedding_dim", "batch_size", "epochs", "patience", "extension_name"], "properties": {"learning_rate": {"type": "number"}, "l2": {"type": "number"}, "embedding_dim": {"type": "integer"}, "batch_size": {"type": "integer"}, "epochs": {"type": "integer"}, "patience": {"type": "integer"}, "extension_name": {"type": "string"}}}}}
        instruction = "Return plain Python source defining run_candidate(prepared, config, run_dir), plus a small serializable config_patch. This is a full isolated PyTorch candidate: helpers, classes, try/except, raise ValueError for local validation, with torch.no_grad(), and `del run_dir` are allowed. prepared supplies canonical train_features, validation_features, train_labels, validation_labels, train_user_ids, validation_user_ids, train_rows, validation_rows, feature_dim, and field_names; data already came through data.py. torch, np, CPU_DEVICE, validation_output, and CandidateOutput are injected—do not import. Use CPU_DEVICE; do not inspect or select CUDA. Return `validation_output(prepared, scores, metadata)` so user IDs and labels always align to validation rows. Do not use run_dir or any filesystem, network, subprocess, eval/exec, raw data path, external data, test data/labels, or unapproved dependency. Keep exactly one declared main change."
        data, meta = self.client.create_json(self._instruction(instruction), json.dumps({"proposal": proposal.__dict__, "previous_failure": failure}), schema=schema)
        return GeneratedCandidate(data["source"], data["config_patch"]), meta

    def review(self, history: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["decision", "rationale"], "properties": {"decision": {"type": "string", "enum": ["refine", "explore", "restart"]}, "rationale": {"type": "string"}}}
        return self.client.create_json(self._instruction("You are the evidence-review specialist. Interpret validation GAUC and nDCG@5 and choose a high-level next action."), json.dumps(list(history[-12:])), schema=schema)

    def decide_candidate_recovery(self, proposal: BroadProposal, failure: str) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["decision", "rationale"], "properties": {"decision": {"type": "string", "enum": ["repair", "abandon"]}, "rationale": {"type": "string"}}}
        data, meta = self.client.create_json(self._instruction("You are the recovery specialist. Repair once only when the error is a local source/runtime defect. Do not fall back to hooks; full candidates have the prepared arrays and user IDs they need. Abandon only unsafe or genuinely out-of-contract work."), json.dumps({"proposal": proposal.__dict__, "failure": failure}), schema=schema)
        return data, meta


def build_isolated_candidate(source: str, workspace: Path) -> CandidateCallable:
    """Persist and compile one full candidate without imports or I/O primitives."""
    source = _plain_source(source)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "candidate.py").write_text(source, encoding="utf-8")
    tree = ast.parse(source, filename="candidate.py", mode="exec")
    forbidden_nodes = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
    forbidden_names = {"open", "exec", "eval", "__import__", "compile", "input", "breakpoint"}
    forbidden_attributes = {"load", "save", "download", "from_file", "system", "popen", "run", "cuda", "read", "write", "unlink", "mkdir", "rmdir", "remove", "rename", "replace", "chdir"}
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"candidate uses prohibited syntax {type(node).__name__} at line {node.lineno}")
        if isinstance(node, ast.Delete) and any(not isinstance(target, ast.Name) or target.id != "run_dir" for target in node.targets):
            raise ValueError(f"candidate may only delete the unused run_dir argument (line {node.lineno})")
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            raise ValueError(f"candidate uses prohibited name {node.id!r} at line {node.lineno}")
        if isinstance(node, ast.Attribute) and ((node.attr.startswith("__") and node.attr != "__init__") or node.attr in forbidden_attributes):
            raise ValueError(f"candidate uses prohibited attribute {node.attr!r} at line {node.lineno}")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_candidate"]
    if len(functions) != 1:
        raise ValueError("candidate must define exactly one run_candidate function")
    arguments = functions[0].args
    if [argument.arg for argument in arguments.args] != ["prepared", "config", "run_dir"] or arguments.vararg or arguments.kwarg:
        raise ValueError("run_candidate must accept exactly (prepared, config, run_dir)")
    import numpy as np
    import torch
    safe_builtins = {"__build_class__": builtins.__build_class__, "object": object, "Exception": Exception, "ValueError": ValueError, "RuntimeError": RuntimeError, "super": super, "len": len, "range": range, "int": int, "float": float, "str": str, "min": min, "max": max, "sum": sum, "all": all, "any": any, "sorted": sorted, "dict": dict, "list": list, "tuple": tuple, "set": set, "abs": abs, "bool": bool, "isinstance": isinstance, "enumerate": enumerate, "zip": zip, "getattr": getattr, "hasattr": hasattr, "vars": vars}
    def validation_output(prepared: Any, scores: Any, metadata: Mapping[str, Any] | None = None) -> CandidateOutput:
        return CandidateOutput(prepared.validation_user_ids, prepared.validation_labels, scores, dict(metadata or {}))
    scope: dict[str, Any] = {"__builtins__": safe_builtins, "__name__": "isolated_candidate", "CandidateOutput": CandidateOutput, "validation_output": validation_output, "CPU_DEVICE": torch.device("cpu"), "torch": torch, "np": np}
    exec(compile(tree, "candidate.py", "exec"), scope, scope)
    return scope["run_candidate"]


def _plain_source(source: str) -> str:
    stripped = source.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return "\n".join(stripped.splitlines()[1:-1]).strip() + "\n"
    return source
