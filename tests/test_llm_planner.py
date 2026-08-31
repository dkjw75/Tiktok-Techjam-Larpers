from __future__ import annotations

import unittest
from unittest.mock import patch

from research_agent.llm_planner import LLMPlanningError, OpenAIPlanner, OpenAIResponsesClient, _extract_output_text
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
    def test_request_retries_a_temporary_timeout_and_records_the_recovery(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"id":"response-1","output_text":"{\\"direction_id\\":\\"pairwise_fm_ranking\\"}"}'

        client = OpenAIResponsesClient(api_key="test", model="test", timeout_seconds=123, max_retries=2)
        with patch("research_agent.llm_planner.urlopen", side_effect=[TimeoutError("slow response"), Response()]) as open_mock, patch("research_agent.llm_planner.time.sleep") as sleep_mock:
            payload, metadata = client.create_json("instruction", "prompt")

        self.assertEqual(payload["direction_id"], "pairwise_fm_ranking")
        self.assertEqual(metadata["request_attempts"], 2)
        self.assertEqual(metadata["transient_request_errors"], ["slow response"])
        self.assertEqual(open_mock.call_count, 2)
        self.assertEqual(open_mock.call_args.kwargs["timeout"], 123)
        sleep_mock.assert_called_once_with(1)

    def test_request_reports_exhausted_retries(self):
        client = OpenAIResponsesClient(api_key="test", model="test", max_retries=1)
        with patch("research_agent.llm_planner.urlopen", side_effect=TimeoutError("slow response")), patch("research_agent.llm_planner.time.sleep"):
            with self.assertRaisesRegex(LLMPlanningError, "after 2 attempt\\(s\\)"):
                client.create_json("instruction", "prompt")

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
