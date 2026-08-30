"""Exercise the finalization path end to end without touching the test split.

Validation stands in as a fake prediction split. This proves the submission
machinery -- inference from the bundle, row alignment, schema, determinism,
restart safety -- works *before* the single real test evaluation is spent.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_agent.models.ensemble_fm import (
    predict_ensemble_checkpoint,
    run_ensemble_fm_candidate,
)
from research_agent.runner import PreparedData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_no_test_leakage import CONFIG, synthetic  # noqa: E402


class ShadowFinalizationTests(unittest.TestCase):
    """The prediction split here is validation, so no real test row is involved."""

    def setUp(self):
        self.train, self.valid, self.shadow = synthetic()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output = run_ensemble_fm_candidate(
            PreparedData(self.train, self.valid), CONFIG, Path(self._tmp.name)
        )
        self.bundle = Path(self.output.metadata["checkpoint_path"])

    def test_shadow_prediction_covers_every_row_in_order(self):
        result = predict_ensemble_checkpoint(
            PreparedData((), self.valid, self.shadow), self.bundle
        )
        self.assertEqual(len(result.scores), len(self.shadow))
        self.assertEqual(len(result.user_ids), len(self.shadow))
        # Row order must follow the prediction split exactly; the submission
        # contract is positional.
        self.assertEqual(list(result.user_ids), [row[1] for row in self.shadow])

    def test_shadow_scores_are_finite(self):
        result = predict_ensemble_checkpoint(
            PreparedData((), self.valid, self.shadow), self.bundle
        )
        scores = np.asarray(result.scores, dtype=float)
        self.assertTrue(np.isfinite(scores).all(), "submission scores must be finite")

    def test_shadow_prediction_is_deterministic_across_repeats(self):
        """A restarted finalization must produce the identical submission."""
        first = predict_ensemble_checkpoint(
            PreparedData((), self.valid, self.shadow), self.bundle
        )
        second = predict_ensemble_checkpoint(
            PreparedData((), self.valid, self.shadow), self.bundle
        )
        np.testing.assert_allclose(
            np.asarray(first.scores, dtype=float),
            np.asarray(second.scores, dtype=float),
            rtol=0.0,
            atol=0.0,
        )

    def test_shadow_path_never_reads_prediction_labels(self):
        flipped = [tuple(r[:6]) + (1 - r[6],) + tuple(r[7:]) for r in self.shadow]
        base = predict_ensemble_checkpoint(
            PreparedData((), self.valid, self.shadow), self.bundle
        )
        altered = predict_ensemble_checkpoint(
            PreparedData((), self.valid, flipped), self.bundle
        )
        np.testing.assert_allclose(
            np.asarray(base.scores, dtype=float),
            np.asarray(altered.scores, dtype=float),
            rtol=0.0,
            atol=0.0,
            err_msg="shadow finalization leaked prediction labels",
        )

    def test_submission_rows_align_with_the_prediction_split(self):
        """Mirror submit.py's alignment contract without importing test data."""
        result = predict_ensemble_checkpoint(
            PreparedData((), self.valid, self.shadow), self.bundle
        )
        rows = [
            (index, row[1], row[2], float(score))
            for index, (row, score) in enumerate(zip(self.shadow, result.scores))
        ]
        self.assertEqual([r[0] for r in rows], list(range(len(self.shadow))))
        self.assertEqual([r[1] for r in rows], [row[1] for row in self.shadow])
        self.assertEqual([r[2] for r in rows], [row[2] for row in self.shadow])
        self.assertTrue(all(np.isfinite(r[3]) for r in rows))


if __name__ == "__main__":
    unittest.main()
