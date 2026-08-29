from __future__ import annotations

import unittest

from research_agent.llm_planner import LLMPlanningError, OpenAIPlanner, _extract_output_text
from research_agent.state import ResearchState


class FakeClient:
    def create_json(self, _instructions, _prompt):
        return (
            {
                "direction_id": "pairwise_fm_ranking",
                "hypothesis": "Pairwise training may better optimise within-user ranking than the baseline loss.",
                "rationale": "The validation objective is ranking-oriented, so test this distinct alignment hypothesis.",
                "strategy": "exploration",
            },
            {"model": "fake", "usage": {"input_tokens": 12}, "response_id": "test-response"},
        )


class LLMPlannerTests(unittest.TestCase):
    def test_structured_responses_output_is_extracted(self):
        raw = {"output": [{"content": [{"type": "output_text", "text": '{"result":"ok"}'}]}]}
        self.assertEqual(_extract_output_text(raw), '{"result":"ok"}')

    def test_llm_response_supplies_the_hypothesis_and_is_logged_as_metadata(self):
        planner = OpenAIPlanner(FakeClient())
        direction = planner.propose([], ResearchState())
        self.assertEqual(direction.direction_id, "pairwise_fm_ranking")
        self.assertIn("Pairwise training", direction.hypothesis)
        self.assertEqual(planner.last_metadata["model"], "fake")

    def test_unsupported_llm_direction_is_rejected(self):
        class UnsafeClient:
            def create_json(self, _instructions, _prompt):
                return ({"direction_id": "external_data", "hypothesis": "x" * 30, "rationale": "x" * 30, "strategy": "exploration"}, {})
        with self.assertRaises(LLMPlanningError):
            OpenAIPlanner(UnsafeClient()).propose([], ResearchState())
