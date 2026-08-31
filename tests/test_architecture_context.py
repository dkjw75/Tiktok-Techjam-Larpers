from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_agent.agent_team import LLMResearchTeam
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
            "revisit_rationale": "",
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
