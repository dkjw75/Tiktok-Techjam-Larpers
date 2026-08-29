from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_agent.agent_team import BroadProposal, build_isolated_candidate
from research_agent.llm_planner import LLMPlanningError


class CandidateWorkspaceTests(unittest.TestCase):
    def test_safe_generated_candidate_is_saved_in_its_workspace(self):
        source = "def run_candidate(prepared, config, run_dir):\n    return run_torch_fm_candidate(prepared, config, run_dir)\n"
        with tempfile.TemporaryDirectory() as directory:
            candidate = build_isolated_candidate(source, Path(directory))
            self.assertTrue((Path(directory) / "candidate.py").exists())
            self.assertTrue(callable(candidate))

    def test_generated_candidate_cannot_import_or_open_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(LLMPlanningError):
                build_isolated_candidate("import os\ndef run_candidate(prepared, config, run_dir):\n return None\n", Path(directory))

    def test_markdown_fenced_source_is_normalized_before_validation(self):
        source = "```python\ndef run_candidate(prepared, config, run_dir):\n    return run_torch_fm_candidate(prepared, config, run_dir)\n```"
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(callable(build_isolated_candidate(source, Path(directory))))
