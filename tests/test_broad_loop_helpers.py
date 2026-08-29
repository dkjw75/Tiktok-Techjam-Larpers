import unittest

from research_agent.broad_loop import _normalize_model_family


class BroadLoopHelperTests(unittest.TestCase):
    def test_unchanged_baseline_prose_normalizes_to_fm(self):
        self.assertEqual(_normalize_model_family("Unchanged baseline model family"), "fm")
