"""Dependency-free search controller for selecting exact trial configurations."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .planner import ResearchDirection
from .safety import ExperimentProposal
from .state import ResearchState


@dataclass(frozen=True)
class SearchState:
    status: str = "BOOTSTRAP"
    region_id: str = "region_01"
    strategy: str = "exploration"
    fidelity: str = "low"


class SearchController:
    """Selects exact one-factor trials after the planner selects a direction."""

    BASELINE_CONFIG: Mapping[str, Any] = {
        "loss": "pointwise",
        "learning_rate": 0.001,
        "l2": 1e-6,
        "embedding_dim": 16,
        "batch_size": 8192,
        "listwise_temperature": 1.0,
        "pointwise_weight": 0.0,
        "objective_variant": "t1",
        "member_set": "core6",
        "seed": 0,
    }

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed

    def propose_trial(
        self,
        direction: ResearchDirection,
        state: ResearchState,
        history: Sequence[Mapping[str, Any]],
        *,
        search_state: SearchState | None = None,
        reserved_ids: Sequence[str] = (),
    ) -> ExperimentProposal:
        search_state = search_state or SearchState(strategy=direction.strategy)
        champion_id, champion_config = self._champion_config(state, history)
        direction_loss = list(direction.search_space.get("loss", [champion_config["loss"]]))
        if len(direction_loss) != 1:
            raise ValueError("each approved direction must define exactly one loss")
        parent_id, parent_config = self._direction_parent_config(
            direction.direction_id,
            direction_loss[0],
            champion_id,
            champion_config,
            history,
        )
        active_factor = self._choose_active_factor(
            direction,
            history,
            state,
            parent_config=parent_config,
            direction_loss=direction_loss[0],
        )
        value = self._choose_value(
            direction,
            active_factor,
            history,
            state,
            baseline=parent_config.get(active_factor),
        )
        config = dict(parent_config)
        config[active_factor] = value
        config["epochs"] = int(direction.evaluation_budget["low_epochs"])
        config["patience"] = int(direction.evaluation_budget.get("low_patience", 2))
        config["fidelity"] = search_state.fidelity
        experiment_id = self._next_experiment_id(history, reserved_ids)
        hypothesis = direction.hypothesis
        rationale = (
            f"Search Controller selected {active_factor} from direction "
            f"{direction.direction_id} using {search_state.strategy}."
        )
        return ExperimentProposal(
            experiment_id=experiment_id,
            parent_experiment_id=parent_id,
            comparison_incumbent_id=state.current_best_experiment_id,
            hypothesis=hypothesis,
            rationale=rationale,
            config=config,
            changed_factors=(active_factor,),
            runtime_budget_seconds=2400.0,
            research_direction_id=direction.direction_id,
            search_strategy=search_state.strategy,
            search_region_id=search_state.region_id,
        )

    def propose_batch(
        self,
        direction: ResearchDirection,
        state: ResearchState,
        history: Sequence[Mapping[str, Any]],
        *,
        count: int = 3,
        search_state: SearchState | None = None,
        reserved_ids: Sequence[str] = (),
    ) -> tuple[ExperimentProposal, ...]:
        """Generate proposals for planning only; execution should request one at a time.

        Later configurations in a direction may legitimately descend from an
        evaluated earlier screen. They must not descend from proposals that
        have not run, so the autonomous loop calls this with ``count=1`` and
        refreshes history after every execution.
        """
        if count <= 0:
            raise ValueError("batch count must be positive")
        if count != 1:
            raise ValueError("screen proposals must be generated sequentially after evidence")
        working_history = list(history)
        proposals: list[ExperimentProposal] = []
        fingerprints = {
            self._config_fingerprint(item.get("config", {}))
            for item in working_history
            if item.get("config")
        }
        attempts = 0
        while len(proposals) < count and attempts < count * 4:
            attempts += 1
            proposal = self.propose_trial(
                direction,
                state,
                working_history,
                search_state=search_state,
                reserved_ids=reserved_ids,
            )
            fingerprint = proposal.config_fingerprint()
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            proposals.append(proposal)
        return tuple(proposals)

    def _choose_active_factor(
        self,
        direction: ResearchDirection,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
        *,
        parent_config: Mapping[str, Any],
        direction_loss: Any,
    ) -> str:
        if parent_config.get("loss") != direction_loss:
            return "loss"
        candidates = [key for key in direction.search_space if key not in {"loss"}]
        if not candidates:
            raise ValueError(f"direction has no searchable factor: {direction.direction_id}")
        counts = {
            factor: sum(factor in item.get("changed_factors", []) for item in history)
            for factor in candidates
        }
        minimum = min(counts.values())
        tied = sorted(factor for factor, count in counts.items() if count == minimum)
        return random.Random(self.seed + state.completed_iterations).choice(tied)

    def _choose_value(
        self,
        direction: ResearchDirection,
        factor: str,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
        *,
        baseline: Any,
    ) -> Any:
        values = list(direction.search_space[factor])
        candidates = [value for value in values if value != baseline] or values
        seen = {
            item.get("config", {}).get(factor)
            for item in history
            if item.get("direction_id") == direction.direction_id
        }
        unseen = [value for value in candidates if value not in seen] or candidates
        return random.Random(self.seed + state.completed_iterations + len(history)).choice(unseen)

    def _champion_config(
        self,
        state: ResearchState,
        history: Sequence[Mapping[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        parent_id = state.current_best_experiment_id
        if parent_id != "baseline":
            for item in reversed(history):
                if item.get("experiment_id") == parent_id and isinstance(item.get("config"), Mapping):
                    return parent_id, dict(item["config"])
            raise ValueError(f"champion configuration is missing from history: {parent_id}")
        return parent_id, dict(self.BASELINE_CONFIG)

    @staticmethod
    def _direction_parent_config(
        direction_id: str,
        direction_loss: Any,
        champion_id: str,
        champion_config: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Use direction-local configuration ancestry without changing the comparator."""
        for item in reversed(history):
            config = item.get("config")
            if (
                item.get("direction_id") == direction_id
                and bool(item.get("metrics"))
                and item.get("decision")
                in {"screened", "accepted", "rejected", "inconclusive"}
                and isinstance(config, Mapping)
                and config.get("loss") == direction_loss
            ):
                return str(item.get("experiment_id") or champion_id), dict(config)
        return champion_id, dict(champion_config)

    @staticmethod
    def _config_fingerprint(config: Mapping[str, Any]) -> str:
        import json

        return json.dumps(dict(config), sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _next_experiment_id(
        history: Sequence[Mapping[str, Any]],
        reserved_ids: Sequence[str] = (),
    ) -> str:
        existing = {str(item.get("experiment_id", "")) for item in history} | set(reserved_ids)
        number = 1
        while f"exp_{number:03d}" in existing:
            number += 1
        return f"exp_{number:03d}"
