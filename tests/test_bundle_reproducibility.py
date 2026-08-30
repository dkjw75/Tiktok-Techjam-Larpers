"""The inference bundle must reproduce the certified model on its own.

The 10/10 reproducibility requirement is that the champion can be reconstructed
by loading its immutable bundle -- no retraining, no rebuilt vocabularies, no
recomputed statistics. These tests are the executable statement of that.
"""
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_agent.models.ensemble_checkpoint import load_ensemble_checkpoint
from research_agent.models.ensemble_fm import (
    predict_ensemble_checkpoint,
    run_ensemble_fm_candidate,
)
from research_agent.runner import PreparedData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_no_test_leakage import CONFIG, synthetic  # noqa: E402

LINEAGE = {
    "data_sha256": "a" * 64,
    "evaluator_sha256": "b" * 64,
    "preprocessing_sha256": "c" * 64,
    "staging_code_sha256": "d" * 64,
    "feature_schema_sha256": "e" * 64,
    "comparison_group_id": "f" * 64,
    "model_code_sha256": "0" * 64,
}


def score_hash(scores) -> str:
    """Canonical dtype and byte order, so the hash is machine-independent."""
    return hashlib.sha256(np.asarray(scores, dtype="<f8").tobytes(order="C")).hexdigest()


class BundleReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.train, self.valid, self.predict = synthetic()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output = run_ensemble_fm_candidate(
            PreparedData(self.train, self.valid, lineage=LINEAGE),
            CONFIG,
            Path(self._tmp.name),
        )
        self.bundle = Path(self.output.metadata["checkpoint_path"])

    def test_bundle_replays_validation_scores_bit_for_bit(self):
        replay = predict_ensemble_checkpoint(
            PreparedData((), self.valid), self.bundle
        )
        np.testing.assert_allclose(
            np.asarray(replay.scores, dtype=float),
            np.asarray(self.output.scores, dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(
            score_hash(replay.scores),
            self.output.metadata["validation_score_sha256"],
        )

    def test_replay_needs_no_training_rows(self):
        """Inference must come from persisted encoders, not refitted ones."""
        replay = predict_ensemble_checkpoint(
            PreparedData((), self.valid), self.bundle
        )
        self.assertEqual(len(replay.scores), len(self.valid))

    def test_bundle_reproduces_in_a_fresh_process(self):
        """Guards against state that only exists in the training interpreter."""
        script = (
            "import hashlib,sys,numpy as np;"
            "sys.path.insert(0, r'.');"
            "sys.path.insert(0, r'tests');"
            "from test_no_test_leakage import synthetic;"
            "from research_agent.models.ensemble_fm import predict_ensemble_checkpoint;"
            "from research_agent.runner import PreparedData;"
            "_t,v,_p = synthetic();"
            f"o = predict_ensemble_checkpoint(PreparedData((), v), r'{self.bundle}');"
            "print(hashlib.sha256(np.asarray(o.scores,dtype='<f8')"
            ".tobytes(order='C')).hexdigest())"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertEqual(
            completed.stdout.strip().splitlines()[-1],
            self.output.metadata["validation_score_sha256"],
        )

    def test_bundle_records_lineage_and_runtime(self):
        manifest = load_ensemble_checkpoint(self.bundle).manifest
        self.assertEqual(manifest["lineage"], LINEAGE)
        for field in ("python_version", "numpy_version", "torch_version"):
            self.assertIn(field, manifest["runtime"])

    def test_bundle_without_lineage_records_an_empty_mapping_not_a_lie(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = run_ensemble_fm_candidate(
                PreparedData(self.train, self.valid), CONFIG, Path(tmp)
            )
            manifest = load_ensemble_checkpoint(
                Path(output.metadata["checkpoint_path"])
            ).manifest
        self.assertEqual(manifest["lineage"], {})

    def test_mutating_a_bundle_array_is_detected(self):
        raw = dict(np.load(self.bundle, allow_pickle=False))
        victim = next(
            key for key, value in raw.items()
            if value.dtype.kind == "f" and value.size > 1
        )
        raw[victim] = raw[victim] + np.float32(1.0)
        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "tampered.npz"
            np.savez_compressed(tampered, **raw)
            with self.assertRaises(Exception):
                load_ensemble_checkpoint(tampered)


class LineagePropagationTests(unittest.TestCase):
    def test_prepared_data_defaults_lineage_to_none(self):
        prepared = PreparedData([], [])
        self.assertIsNone(prepared.lineage)
        self.assertIsNone(prepared.prediction_rows)


if __name__ == "__main__":
    unittest.main()
