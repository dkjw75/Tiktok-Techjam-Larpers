from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research_agent.agent_team import build_isolated_candidate
from research_agent.broad_loop import BroadAutonomousLoop


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

    def test_candidate_can_delete_local_temporary_values(self):
        source = """def run_candidate(prepared, config, run_dir):
    del run_dir
    temporary = np.zeros(len(prepared.validation_labels))
    del temporary
    return validation_output(prepared, np.zeros(len(prepared.validation_labels)))
"""
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(callable(build_isolated_candidate(source, Path(directory))))

    def test_candidate_can_use_explicit_categorical_field_contract(self):
        source = """class TinyFM(torch.nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, 1)
    def forward(self, field_ids):
        return self.embedding(field_ids).sum(dim=1).squeeze(1)

def run_candidate(prepared, config, run_dir):
    del run_dir
    train_ids = feature_ids(prepared, 'train')
    valid_ids = feature_ids(prepared, 'validation')
    valid_users = grouping_user_ids(prepared, 'validation')
    if train_ids.shape[1] != prepared.field_count:
        raise ValueError('wrong categorical field width')
    model = TinyFM(prepared.vocabulary_size)
    scores = model(valid_ids).detach().numpy()
    if len(valid_users) != scores.shape[0]:
        raise ValueError('wrong opaque user-ID alignment')
    return validation_output(prepared, scores)
"""
        with tempfile.TemporaryDirectory() as directory:
            candidate = build_isolated_candidate(source, Path(directory))
            prepared = SimpleNamespace(
                train_features=np.array([[0, 1, 2, 3, 4]], dtype=np.int32),
                validation_features=np.array([[0, 1, 2, 3, 4]], dtype=np.int32),
                validation_user_ids=["user"],
                validation_labels=[1],
                train_user_ids=["user"],
                vocabulary_size=5,
                field_count=5,
            )
            output = candidate(prepared, {}, Path(directory))
            self.assertEqual(len(output.scores), 1)

    def test_synthetic_contract_preflight_handles_string_user_ids(self):
        source = """def run_candidate(prepared, config, run_dir):
    del run_dir
    train_ids = feature_ids(prepared, 'train')
    labels = binary_labels(prepared, 'train')
    users = grouping_user_ids(prepared, 'train')
    if row_count(prepared, 'train') != labels.shape[0] or len(users) != labels.shape[0]:
        raise ValueError('misaligned candidate inputs')
    scores = feature_ids(prepared, 'validation').sum(dim=1).numpy()
    return validation_output(prepared, scores)
"""
        with tempfile.TemporaryDirectory() as directory:
            candidate = build_isolated_candidate(source, Path(directory))
            BroadAutonomousLoop._synthetic_contract_preflight(candidate, {"batch_size": 8})
