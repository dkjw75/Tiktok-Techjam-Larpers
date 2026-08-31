"""The verified research tools an LLM may select during an autonomous run.

The catalogue describes *laboratory equipment*, not a sequence of hypotheses.
The LLM chooses a tool and supplies the scientific rationale; the search and
fidelity components select and execute valid exact configurations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .planner import ResearchDirection


@dataclass(frozen=True)
class ResearchTool:
    tool_id: str
    description: str
    search_space: Mapping[str, tuple[Any, ...]]
    low_epochs: int
    full_epochs: int
    strategy: str

    def as_prompt_record(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "description": self.description,
            "search_strategy": self.strategy,
            "low_fidelity_epochs": self.low_epochs,
            "full_fidelity_epochs": self.full_epochs,
            "searchable_parameters": {key: list(values) for key, values in self.search_space.items()},
        }


class ResearchToolCatalog:
    """Only exposes tools backed by tested project code and PyTorch FM support."""

    def __init__(self) -> None:
        self._excluded: set[str] = set()
        self._tools = {
            "fm_training_search": ResearchTool(
                tool_id="fm_training_search",
                description="Explore one PyTorch FM training setting at a time using diverse low-fidelity trials, then promote promising candidates to full training.",
                search_space={"loss": ("pointwise",), "learning_rate": (0.00025, 0.0005, 0.001, 0.002, 0.004), "l2": (0.0, 1e-7, 1e-6, 1e-5, 1e-4)},
                low_epochs=4,
                full_epochs=40,
                strategy="diverse_search",
            ),
            "fm_pairwise_search": ResearchTool(
                tool_id="fm_pairwise_search",
                description="Test the existing PyTorch FM with its implemented within-user pairwise loss; vary only one supported training setting per trial.",
                search_space={"loss": ("pairwise",), "learning_rate": (0.00025, 0.0005, 0.001, 0.002, 0.004), "l2": (0.0, 1e-7, 1e-6, 1e-5, 1e-4)},
                low_epochs=4,
                full_epochs=40,
                strategy="diverse_search",
            ),
        }

    def prompt_records(self) -> list[dict[str, Any]]:
        return [tool.as_prompt_record() for tool_id, tool in self._tools.items() if tool_id not in self._excluded]

    def exclude(self, tool_id: str) -> None:
        if tool_id in self._tools:
            self._excluded.add(tool_id)

    def direction(self, tool_id: str, *, hypothesis: str, rationale: str) -> ResearchDirection:
        tool = self._tools.get(tool_id)
        if tool is None:
            raise ValueError(f"LLM selected unavailable research tool: {tool_id}")
        return ResearchDirection(
            direction_id=tool.tool_id,
            hypothesis=hypothesis,
            rationale=rationale,
            search_space=tool.search_space,
            success_evidence="Validation primary improves by more than 0.002 over the accepted parent.",
            evaluation_budget={"low_epochs": tool.low_epochs, "full_epochs": tool.full_epochs, "patience": 4},
            strategy=tool.strategy,
        )
