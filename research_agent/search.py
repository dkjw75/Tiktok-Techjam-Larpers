"""Dependency-free search controller for selecting exact trial configurations."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .planner import ResearchDirection
from .safety import ExperimentProposal
from .state import ResearchState


class SearchSpaceExhausted(RuntimeError):
    """A direction has no valid low-fidelity configuration left to run."""


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
    ) -> ExperimentProposal:
        search_state = search_state or SearchState(strategy=direction.strategy)
        choices = self._untried_choices(direction, state, history)
        if not choices:
            raise SearchSpaceExhausted(f"all low-fidelity configurations were already evaluated for {direction.direction_id}")
        config, changed_factors, algorithm = self._select_choice(choices, direction, history, state)
        config["epochs"] = int(direction.evaluation_budget["low_epochs"])
        config["fidelity"] = search_state.fidelity
        experiment_id = self._next_experiment_id(history)
        hypothesis = direction.hypothesis
        rationale = (
            f"Search Controller selected {', '.join(changed_factors)} from direction "
            f"{direction.direction_id} using {algorithm}."
        )
        return ExperimentProposal(
            experiment_id=experiment_id,
            parent_experiment_id=state.current_best_experiment_id,
            hypothesis=hypothesis,
            rationale=rationale,
            config=config,
            changed_factors=changed_factors,
            runtime_budget_seconds=600.0,
            research_direction_id=direction.direction_id,
            search_strategy=algorithm,
            search_region_id=search_state.region_id,
        )

    def _untried_choices(
        self,
        direction: ResearchDirection,
        state: ResearchState,
        history: Sequence[Mapping[str, Any]],
    ) -> list[tuple[dict[str, Any], tuple[str, ...], str]]:
        fixed = self._fixed_values(direction)
        base = {**self.BASELINE_CONFIG, **fixed}
        choices: list[tuple[dict[str, Any], tuple[str, ...], str]] = []
        defining = tuple(key for key, value in fixed.items() if self.BASELINE_CONFIG.get(key) != value)
        if defining:
            choices.append((dict(base), defining, "direction_bootstrap"))
        for factor, values in direction.search_space.items():
            if len(values) <= 1:
                continue
            for value in values:
                if value == base.get(factor):
                    continue
                config = {**base, factor: value}
                choices.append((config, (factor,), "diverse_exploration"))
        joint = self._joint_refinement(direction, base, history, state)
        if joint is not None:
            choices.append((joint, ("joint_training_refinement",), "joint_refinement"))
        return [choice for choice in choices if not self._was_run(choice[0], direction.direction_id, history)]

    def _fixed_values(self, direction: ResearchDirection) -> dict[str, Any]:
        """Tool-defined singleton values are contracts, not legacy defaults."""
        return {
            factor: values[0]
            for factor, values in direction.search_space.items()
            if len(values) == 1
        }

    def _joint_refinement(
        self,
        direction: ResearchDirection,
        base: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
    ) -> dict[str, Any] | None:
        """Combine the two strongest single-factor signals only when promising."""
        scored = [
            item for item in history
            if item.get("direction_id") == direction.direction_id
            and item.get("config", {}).get("fidelity") == "low"
            and isinstance(item.get("metrics", {}).get("primary"), (int, float))
        ]
        if len(scored) < 3 or max(float(item["metrics"]["primary"]) for item in scored) < state.current_best_primary - 0.002:
            return None
        tunable = [factor for factor, values in direction.search_space.items() if len(values) > 1]
        if len(tunable) < 2:
            return None
        config = dict(base)
        changed = 0
        for factor in tunable:
            factor_trials = [item for item in scored if item.get("config", {}).get(factor) != base[factor]]
            if not factor_trials:
                continue
            best = max(factor_trials, key=lambda item: float(item["metrics"]["primary"]))
            value = best["config"][factor]
            config[factor] = value
            changed += value != base[factor]
        return config if changed >= 2 else None

    @staticmethod
    def _was_run(config: Mapping[str, Any], direction_id: str, history: Sequence[Mapping[str, Any]]) -> bool:
        keys = ("loss", "learning_rate", "l2", "embedding_dim", "batch_size", "seed")
        return any(
            item.get("direction_id") == direction_id
            and item.get("config", {}).get("fidelity") == "low"
            and all(item.get("config", {}).get(key) == config.get(key) for key in keys)
            for item in history
        )

    def _select_choice(
        self,
        choices: Sequence[tuple[dict[str, Any], tuple[str, ...], str]],
        direction: ResearchDirection,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
    ) -> tuple[dict[str, Any], tuple[str, ...], str]:
        bootstrap = [choice for choice in choices if choice[2] == "direction_bootstrap"]
        if bootstrap:
            return bootstrap[0]
        observations = self._direction_observations(direction.direction_id, history)
        if len(observations) < 6:
            return random.Random(self.seed + state.completed_iterations).choice(list(choices))
        numeric = all(isinstance(value, (int, float)) for config, _, _ in choices for value in (config["learning_rate"], config["l2"]))
        if numeric:
            return max(choices, key=lambda choice: self._config_ucb(choice[0], observations))[:2] + ("bayesian_optimization",)
        return max(choices, key=lambda choice: self._config_tpe(choice[0], observations))[:2] + ("tpe",)

    @staticmethod
    def _direction_observations(direction_id: str, history: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [
            item for item in history
            if item.get("direction_id") == direction_id
            and item.get("config", {}).get("fidelity") == "low"
            and isinstance(item.get("metrics", {}).get("primary"), (int, float))
        ]

    @staticmethod
    def _config_ucb(config: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> float:
        distances = [
            abs(math.log10(float(config["learning_rate"])) - math.log10(float(item["config"]["learning_rate"])))
            + abs(math.log10(float(config["l2"]) + 1e-12) - math.log10(float(item["config"]["l2"]) + 1e-12))
            for item in observations
        ]
        weights = [math.exp(-distance) for distance in distances]
        mean = sum(weight * float(item["metrics"]["primary"]) for weight, item in zip(weights, observations)) / sum(weights)
        return mean + 0.01 * min(distances)

    @staticmethod
    def _config_tpe(config: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> float:
        ranked = sorted(observations, key=lambda item: float(item["metrics"]["primary"]), reverse=True)
        cutoff = max(1, math.ceil(len(ranked) / 3))
        good, bad = ranked[:cutoff], ranked[cutoff:]
        matches = lambda rows: sum(
            item["config"].get("learning_rate") == config["learning_rate"] and item["config"].get("l2") == config["l2"]
            for item in rows
        )
        return (1 + matches(good)) / (1 + matches(bad))

    def _choose_value(
        self,
        direction: ResearchDirection,
        factor: str,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
    ) -> tuple[Any, str]:
        values = list(direction.search_space[factor])
        baseline = self.BASELINE_CONFIG[factor]
        candidates = [value for value in values if value != baseline] or values
        seen = {
            item.get("config", {}).get(factor)
            for item in history
            if item.get("direction_id") == direction.direction_id
        }
        unseen = [value for value in candidates if value not in seen] or candidates
        observations = self._observations(direction.direction_id, factor, history)
        if len(observations) < 6:
            return random.Random(self.seed + state.completed_iterations + len(history)).choice(unseen), "diverse_exploration"
        if all(isinstance(value, (int, float)) for value in candidates):
            return self._bayesian_choice(unseen, observations), "bayesian_optimization"
        return self._tpe_choice(unseen, observations), "tpe"

    @staticmethod
    def _observations(direction_id: str, factor: str, history: Sequence[Mapping[str, Any]]) -> list[tuple[Any, float]]:
        return [
            (item.get("config", {}).get(factor), float(item["metrics"]["primary"]))
            for item in history
            if item.get("direction_id") == direction_id
            and item.get("config", {}).get(factor) is not None
            and isinstance(item.get("metrics", {}).get("primary"), (int, float))
        ]

    @staticmethod
    def _bayesian_choice(candidates: Sequence[Any], observations: Sequence[tuple[Any, float]]) -> Any:
        """Small dependency-free UCB surrogate for discrete numeric choices."""
        values = [float(value) for value, _ in observations]
        span = max(max(values) - min(values), 1e-9)
        def acquisition(candidate: Any) -> tuple[float, float]:
            distances = [abs(float(candidate) - value) / span for value in values]
            weights = [math.exp(-distance * 4.0) for distance in distances]
            mean = sum(weight * score for weight, (_, score) in zip(weights, observations)) / sum(weights)
            return mean + 0.01 * min(distances), min(distances)
        return max(candidates, key=acquisition)

    @staticmethod
    def _tpe_choice(candidates: Sequence[Any], observations: Sequence[tuple[Any, float]]) -> Any:
        """Prefer categories over-represented among the best third of trials."""
        ranked = sorted(observations, key=lambda item: item[1], reverse=True)
        cutoff = max(1, math.ceil(len(ranked) / 3))
        good, bad = ranked[:cutoff], ranked[cutoff:]
        def ratio(candidate: Any) -> tuple[float, float]:
            good_count = 1 + sum(value == candidate for value, _ in good)
            bad_count = 1 + sum(value == candidate for value, _ in bad)
            return good_count / bad_count, -sum(value == candidate for value, _ in observations)
        return max(candidates, key=ratio)

    @staticmethod
    def _next_experiment_id(history: Sequence[Mapping[str, Any]]) -> str:
        existing = {str(item.get("experiment_id", "")) for item in history}
        number = 1
        while f"exp_{number:03d}" in existing:
            number += 1
        return f"exp_{number:03d}"
