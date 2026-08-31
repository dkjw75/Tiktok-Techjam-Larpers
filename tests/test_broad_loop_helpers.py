import unittest

from research_agent.broad_loop import _candidate_config, _lock_candidate_config, _normalize_model_family


class BroadLoopHelperTests(unittest.TestCase):
    def test_unchanged_baseline_prose_normalizes_to_fm(self):
        self.assertEqual(_normalize_model_family("Unchanged baseline model family"), "fm")
        self.assertEqual(_normalize_model_family("Existing baseline model family; objective-only change."), "fm")

    def test_candidate_configuration_records_the_declared_controlled_settings(self):
        config = _candidate_config({"learning_rate": 0.1, "embedding_dim": 128, "extension_name": "test"})
        locked = _lock_candidate_config(config, {"hook_type": "sampler"})

        self.assertEqual(config["learning_rate"], 0.1)
        self.assertEqual(config["embedding_dim"], 128)
        self.assertEqual(locked["learning_rate"], 0.1)
        self.assertEqual(locked["embedding_dim"], 128)
        self.assertEqual(locked["loss"], "pointwise")
        self.assertEqual(locked["_locked_settings"]["batch_size"], 8192)
