from __future__ import annotations

import unittest
from unittest.mock import patch

from research_agent.llm_planner import (
    LLMPlanningError,
    OpenAIPlanner,
    OpenAIResponsesClient,
    ResilientPlanner,
    _extract_output_text,
)
from research_agent.planner import EvidencePlanner
from research_agent.state import ResearchState


class FakeClient:
    def create_json(self, _instructions, _prompt):
        return (
            {
                "direction_id": "listwise_fm_ranking",
                "hypothesis": "Listwise training may better optimise within-user ranking than the baseline loss.",
                "rationale": "The validation objective is ranking-oriented, so test this distinct alignment hypothesis.",
                "strategy": "exploration",
            },
            {"model": "fake", "usage": {"input_tokens": 12}, "response_id": "test-response"},
        )


class LLMPlannerTests(unittest.TestCase):
    def test_wire_format_output_is_extracted_from_response_items(self):
        payload = {
            "id": "resp_123",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"direction_id":"listwise_fm_ranking"}',
                        }
                    ],
                }
            ],
        }
        self.assertEqual(
            _extract_output_text(payload),
            '{"direction_id":"listwise_fm_ranking"}',
        )

    def test_refusal_is_reported_instead_of_being_treated_as_missing_text(self):
        payload = {
            "output": [
                {"type": "message", "content": [{"type": "refusal", "refusal": "cannot comply"}]}
            ]
        }
        with self.assertRaisesRegex(LLMPlanningError, "refused"):
            _extract_output_text(payload)

    def test_retryable_transport_failure_is_retried(self):
        response_payload = {
            "id": "resp_123",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                '{"direction_id":"listwise_fm_ranking",'
                                '"hypothesis":"Listwise training should improve ranking quality.",'
                                '"rationale":"The objective rewards within-user ranking performance.",'
                                '"strategy":"exploration"}'
                            ),
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                import json

                return json.dumps(response_payload).encode("utf-8")

        calls = [TimeoutError("slow"), FakeResponse()]

        def fake_urlopen(*_args, **_kwargs):
            result = calls.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        sleeps = []
        client = OpenAIResponsesClient(
            api_key="test",
            model="test-model",
            max_attempts=2,
            backoff_seconds=0.25,
            sleep=sleeps.append,
        )
        with patch("research_agent.llm_planner.urlopen", side_effect=fake_urlopen):
            result, metadata = client.create_json("instructions", "prompt")

        self.assertEqual(result["direction_id"], "listwise_fm_ranking")
        self.assertEqual(metadata["response_id"], "resp_123")
        self.assertEqual(sleeps, [0.25])

    def test_llm_response_supplies_the_hypothesis_and_is_logged_as_metadata(self):
        planner = OpenAIPlanner(FakeClient())
        direction = planner.propose([], ResearchState())
        self.assertEqual(direction.direction_id, "listwise_fm_ranking")
        self.assertIn("Listwise training", direction.hypothesis)
        self.assertEqual(planner.last_metadata["model"], "fake")

    def test_unsupported_llm_direction_is_rejected(self):
        class UnsafeClient:
            def create_json(self, _instructions, _prompt):
                return ({"direction_id": "external_data", "hypothesis": "x" * 30, "rationale": "x" * 30, "strategy": "exploration"}, {})
        with self.assertRaises(LLMPlanningError):
            OpenAIPlanner(UnsafeClient()).propose([], ResearchState())

    def test_operational_failure_uses_explicit_logged_fallback(self):
        class BrokenPlanner:
            def propose(self, _history, _state):
                raise LLMPlanningError("temporary outage")

        planner = ResilientPlanner(BrokenPlanner(), EvidencePlanner(seed=0))
        direction = planner.propose([], ResearchState())

        self.assertTrue(direction.direction_id)
        self.assertEqual(planner.last_metadata["planner_mode"], "deterministic_fallback")
        self.assertTrue(planner.last_metadata["degraded"])
        self.assertIn("temporary outage", planner.last_metadata["error"])
