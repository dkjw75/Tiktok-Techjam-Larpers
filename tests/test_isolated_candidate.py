from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_agent.agent_team import build_isolated_candidate


class IsolatedCandidateTests(unittest.TestCase):
    def test_full_candidate_allows_safe_control_flow_and_module_classes(self):
        source = """class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, values):
        return values

def run_candidate(prepared, config, run_dir):
    try:
        with torch.no_grad():
            scores = torch.zeros(len(prepared.validation_labels)).numpy()
    except Exception:
        scores = np.zeros(len(prepared.validation_labels))
    return CandidateOutput(prepared.validation_user_ids, prepared.validation_labels, scores)
"""
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(callable(build_isolated_candidate(source, Path(directory))))

    def test_full_candidate_still_rejects_file_access(self):
        source = "def run_candidate(prepared, config, run_dir):\n    return open('data.csv')\n"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                build_isolated_candidate(source, Path(directory))

    def test_candidate_can_ignore_run_directory_and_raise_local_validation_errors(self):
        source = """def run_candidate(prepared, config, run_dir):
    del run_dir
    if len(prepared.validation_labels) < 0:
        raise ValueError('unreachable local validation')
    return validation_output(prepared, np.zeros(len(prepared.validation_labels)))
"""
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(callable(build_isolated_candidate(source, Path(directory))))
