"""Deterministic safety checks for proposed research experiments."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract


@dataclass(frozen=True)
class ExperimentProposal:
    """A declarative candidate that can be checked before any training begins."""

    experiment_id: str
    hypothesis: str
    rationale: str
    config: Mapping[str, Any]
    changed_factors: tuple[str, ...]
    parent_experiment_id: str | None = None
    comparison_incumbent_id: str | None = None
    training_split: str = "train"
    selection_split: str = "valid"
    external_data_sources: tuple[str, ...] = ()
    uses_test_labels: bool = False
    loads_raw_csv: bool = False
    modified_files: tuple[str, ...] = ()
    requested_dependencies: tuple[str, ...] = ()
    model_family: str = "fm"
    human_reviewed: bool = False
    runtime_budget_seconds: float = 600.0
    research_direction_id: str | None = None
    search_strategy: str = ""
    search_region_id: str = ""

    def config_fingerprint(self) -> str:
        """Stable representation used to reject exact duplicate candidates."""
        return json.dumps(dict(self.config), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class SafetyReport:
    passed: bool
    violations: tuple[str, ...] = ()


class SafetyValidator:
    """Enforces benchmark and project rules before a runner is invoked."""

    def __init__(
        self,
        *,
        contract: BenchmarkContract = BENCHMARK_CONTRACT,
        max_runtime_seconds: float = 600.0,
        allowed_dependencies: frozenset[str] = frozenset({"numpy", "torch"}),
        approved_model_families: frozenset[str] = frozenset({"fm"}),
    ) -> None:
        self.contract = contract
        self.max_runtime_seconds = max_runtime_seconds
        self.allowed_dependencies = allowed_dependencies
        self.approved_model_families = approved_model_families

    def validate(
        self,
        proposal: ExperimentProposal,
        *,
        historical_configs: Sequence[Mapping[str, Any]] = (),
    ) -> SafetyReport:
        violations: list[str] = []
        if not proposal.hypothesis.strip():
            violations.append("proposal must state a hypothesis")
        if not proposal.rationale.strip():
            violations.append("proposal must state a rationale")
        if len(proposal.changed_factors) != 1:
            violations.append("proposal must change exactly one main factor")
        if proposal.training_split != self.contract.train_split:
            violations.append("training must use only the train split")
        if proposal.selection_split != self.contract.selection_split:
            violations.append("candidate selection must use only the valid split")
        if proposal.external_data_sources:
            violations.append("external datasets are not permitted")
        if proposal.uses_test_labels:
            violations.append("test labels must not be used for optimization")
        if proposal.loads_raw_csv:
            violations.append("controllers, runners, and models must use data.py rather than raw CSV files")
        protected = set(proposal.modified_files) & set(self.contract.protected_modules)
        if protected:
            violations.append(f"protected benchmark files may not be modified: {', '.join(sorted(protected))}")
        unknown_dependencies = set(proposal.requested_dependencies) - self.allowed_dependencies
        if unknown_dependencies:
            violations.append(
                "unapproved dependencies requested: " + ", ".join(sorted(unknown_dependencies))
            )
        if proposal.model_family not in self.approved_model_families and not proposal.human_reviewed:
            violations.append("a substantially different model family requires human review")
        if not 0 < proposal.runtime_budget_seconds <= self.max_runtime_seconds:
            violations.append(
                f"runtime budget must be greater than 0 and at most {self.max_runtime_seconds:g} seconds"
            )
        if proposal.config_fingerprint() in {self._fingerprint(config) for config in historical_configs}:
            violations.append("proposal duplicates an existing experiment configuration")
        return SafetyReport(passed=not violations, violations=tuple(violations))

    @staticmethod
    def _fingerprint(config: Mapping[str, Any]) -> str:
        return json.dumps(dict(config), sort_keys=True, separators=(",", ":"), default=str)
