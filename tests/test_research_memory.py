from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_agent.research_memory import ResearchMemory


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class ResearchMemoryTests(unittest.TestCase):
    def test_bootstrap_is_append_only_and_summarizes_weak_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_run = root / "runs_autonomous_01"
            _write_jsonl(old_run / "iterations.jsonl", [
                {"experiment_id": "exp_002", "hypothesis": "pairwise idea", "rationale": "test", "config": {"extension_name": "pairwise_v1"}, "changed_factors": ["loss"], "decision": "rejected", "metrics": {"primary": 0.59}, "delta_primary": -0.01},
                {"experiment_id": "exp_003", "hypothesis": "pairwise idea again", "rationale": "test", "config": {"extension_name": "pairwise_v1"}, "changed_factors": ["loss"], "decision": "rejected", "metrics": {"primary": 0.58}, "delta_primary": -0.02},
            ])
            _write_jsonl(old_run / "experiments.jsonl", [
                {"timestamp": "now", "action": "candidate_failure_recorded", "experiment_id": "exp_004", "details": {"attempt_id": "a1", "failure_class": "runtime_error", "reason": "bad candidate"}},
            ])
            current = root / "runs_autonomous_02"
            memory = ResearchMemory(root)

            first = memory.bootstrap(exclude_run=current)
            second = memory.bootstrap(exclude_run=current)
            summary = memory.planner_summary()

            self.assertEqual(first["imported_records"], 3)
            self.assertEqual(second["imported_records"], 0)
            self.assertEqual(len(memory.records()), 3)
            self.assertEqual(summary["method_evidence"][0]["method"], "pairwise_v1")
            self.assertEqual(summary["method_evidence"][0]["status"], "deprioritized")
            self.assertEqual(summary["recent_implementation_failures"][0]["failure_class"], "runtime_error")

    def test_new_iteration_is_available_without_reimporting_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = ResearchMemory(Path(directory))
            record = {"experiment_id": "exp_001", "hypothesis": "new", "rationale": "why", "config": {"loss": "custom"}, "decision": "screened", "metrics": {"primary": 0.6}, "delta_primary": -0.001}

            self.assertTrue(memory.append_iteration(record, source_run="runs_new"))
            self.assertFalse(memory.append_iteration(record, source_run="runs_new"))
            self.assertEqual(memory.planner_summary()["recent_evidence"][0]["experiment_id"], "exp_001")
