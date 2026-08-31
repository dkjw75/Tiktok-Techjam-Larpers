from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_agent.agent_team import BroadProposal, build_isolated_candidate
from research_agent.llm_planner import LLMPlanningError


class CandidateWorkspaceTests(unittest.TestCase):
    def test_safe_generated_candidate_is_saved_in_its_workspace(self):
        source = "def run_candidate(prepared, config, run_dir):\n    return CandidateOutput([], [], [], {'framework': 'pytorch'})\n"
        with tempfile.TemporaryDirectory() as directory:
            candidate = build_isolated_candidate(source, Path(directory))
            self.assertTrue((Path(directory) / "candidate.py").exists())
            self.assertTrue(callable(candidate))

    def test_generated_candidate_cannot_import_or_open_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(LLMPlanningError):
                build_isolated_candidate("import os\ndef run_candidate(prepared, config, run_dir):\n return None\n", Path(directory))

    def test_generated_candidate_cannot_use_torch_file_loading(self):
        source = "def run_candidate(prepared, config, run_dir):\n    return torch.load('model.pt')\n"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(LLMPlanningError):
                build_isolated_candidate(source, Path(directory))

    def test_markdown_fenced_source_is_normalized_before_validation(self):
        source = "```python\ndef run_candidate(prepared, config, run_dir):\n    return CandidateOutput([], [], [], {'framework': 'pytorch'})\n```"
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(callable(build_isolated_candidate(source, Path(directory))))

    def test_custom_loss_extension_is_allowed_without_imports(self):
        source = """def run_candidate(prepared, config, run_dir):
    def focal(logits, labels):
        base = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        return ((1.0 - torch.exp(-base)) * base).mean()
    return run_torch_fm_extension(prepared, config, run_dir, loss_function=focal)
"""
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(callable(build_isolated_candidate(source, Path(directory))))

    def test_full_pytorch_candidate_does_not_need_the_legacy_hook(self):
        source = """class LinearHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, values):
        return values * self.weight

def run_candidate(prepared, config, run_dir):
    values = torch.tensor([0.25 for _ in prepared.validation_rows])
    return CandidateOutput([row[0] for row in prepared.validation_rows], [row[1] for row in prepared.validation_rows], values.tolist(), {'framework': 'pytorch', 'candidate_type': 'full'})
"""
        with tempfile.TemporaryDirectory() as directory:
            candidate = build_isolated_candidate(source, Path(directory))
            self.assertTrue(callable(candidate))
