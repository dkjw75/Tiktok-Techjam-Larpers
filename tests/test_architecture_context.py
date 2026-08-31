from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_agent.agent_team import LLMResearchPlanner, LLMResearchTeam
from research_agent.architecture_context import ArchitectureContext
from research_agent.contracts import BENCHMARK_CONTRACT
from research_agent.state import ResearchState


class RecordingClient:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    def create_json(self, instruction, _prompt, *, schema):
        self.instructions.append(instruction)
        return {
            "hypothesis": "Test a safe direction.",
            "rationale": "The architecture requests controlled research.",
            "area": "training",
            "controlled_change": "One change.",
            "model_family": "fm",
            "requires_human_review": False,
            "leakage_risks": [],
        }, {"model": "fake"}


class ToolSelectingClient(RecordingClient):
    def create_json(self, instruction, _prompt, *, schema):
        self.instructions.append(instruction)
        return {
            "tool_id": "fm_pairwise_search",
            "hypothesis": "Pairwise training may improve ranking.",
            "rationale": "Ranking evidence favours a ranking-aligned objective.",
        }, {"model": "fake"}


class ArchitectureContextTests(unittest.TestCase):
    def test_live_document_and_fixed_boundaries_are_injected_into_planner(self):
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "agent-architecture.md"
            document.write_text("# Architecture\n\nUse diverse research directions.", encoding="utf-8")
            context = ArchitectureContext.from_file(document, BENCHMARK_CONTRACT)
            client = RecordingClient()

            LLMResearchTeam(client, context).propose([], {})

            instruction = client.instructions[0]
            self.assertIn("Use diverse research directions.", instruction)
            self.assertIn("Use only KuaiRand-Pure through data.py", instruction)
            self.assertEqual(len(context.source_sha256), 64)

    def test_llm_chooses_a_verified_tool_but_not_exact_trial_values(self):
        client = ToolSelectingClient()
        planner = LLMResearchPlanner(LLMResearchTeam(client))

        direction = planner.propose([], ResearchState())

        self.assertEqual(direction.direction_id, "fm_pairwise_search")
        self.assertEqual(direction.search_space["loss"], ("pairwise",))
        self.assertNotIn("learning_rate", direction.hypothesis)
