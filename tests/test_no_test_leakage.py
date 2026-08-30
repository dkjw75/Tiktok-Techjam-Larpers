"""The prediction split must not influence anything that is fitted.

This is the guard for a real defect: an earlier finalization path passed the test
rows in as the *validation* argument, so every ensemble member early-stopped
against test labels. These tests make that class of mistake fail loudly.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_agent.models.ensemble_fm import run_ensemble_fm_candidate
from research_agent.runner import PreparedData


def row(date, user, video, author, label, play=500.0):
    return (date, user, video, author, "1", 1000.0, label, 9, play, "m", "t", 20220401)


def synthetic(n_users=40, days=6):
    """Small but structurally valid: mixed labels, several days, repeat items."""
    train, valid, predict = [], [], []
    for u in range(n_users):
        user = f"u{u}"
        for d in range(days):
            for v in range(4):
                label = 1 if (u + v + d) % 3 == 0 else 0
                train.append(row(20220408 + d, user, f"v{v}", f"a{v % 3}", label,
                                 play=100.0 * ((u + v) % 5)))
        for v in range(4):
            label = 1 if (u + v) % 2 == 0 else 0
            valid.append(row(20220422, user, f"v{v}", f"a{v % 3}", label,
                             play=100.0 * ((u + v) % 4)))
            predict.append(row(20220429, user, f"v{v}", f"a{v % 3}", label,
                               play=100.0 * ((u + v) % 6)))
    return train, valid, predict


CONFIG = {
    "seed": 0,
    "epochs": 3,
    "patience": 2,
    "members": ["fm", "watch"],
    "blend_weights": [0.5, 0.5],
}


def run(train, valid, predict):
    with tempfile.TemporaryDirectory() as tmp:
        return run_ensemble_fm_candidate(
            PreparedData(train, valid, predict), CONFIG, Path(tmp)
        )


class TestLabelIndependenceTests(unittest.TestCase):
    def test_prediction_split_requires_validation_fixed_weights(self):
        train, valid, predict = synthetic()
        unsafe = {key: value for key, value in CONFIG.items() if key != "blend_weights"}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "requires blend_weights"):
                run_ensemble_fm_candidate(
                    PreparedData(train, valid, predict), unsafe, Path(tmp)
                )

    def test_prediction_split_labels_cannot_change_predictions(self):
        """Flipping every prediction-split label must not move a single score."""
        train, valid, predict = synthetic()
        baseline = run(train, valid, predict)

        flipped = [tuple(r[:6]) + (1 - r[6],) + tuple(r[7:]) for r in predict]
        self.assertNotEqual(
            [r[6] for r in predict], [r[6] for r in flipped], "labels did not flip"
        )
        altered = run(train, valid, flipped)

        np.testing.assert_allclose(
            np.asarray(baseline.scores, dtype=float),
            np.asarray(altered.scores, dtype=float),
            rtol=0.0,
            atol=0.0,
            err_msg="prediction-split labels leaked into the scores",
        )

    def test_prediction_split_labels_cannot_change_epochs_or_members(self):
        train, valid, predict = synthetic()
        baseline = run(train, valid, predict).metadata
        flipped = [tuple(r[:6]) + (1 - r[6],) + tuple(r[7:]) for r in predict]
        altered = run(train, valid, flipped).metadata

        self.assertEqual(baseline["member_epochs_run"], altered["member_epochs_run"])
        self.assertEqual(baseline["member_primaries"], altered["member_primaries"])
        self.assertEqual(baseline["epochs_run"], altered["epochs_run"])

    def test_fixed_weights_are_not_refitted_on_the_prediction_split(self):
        train, valid, predict = synthetic()
        metadata = run(train, valid, predict).metadata
        self.assertEqual(metadata["weight_selection"], "fixed from validation")
        self.assertEqual(metadata["blend_weights"], {"fm": 0.5, "watch": 0.5})
        self.assertEqual(metadata["scored_split"], "prediction")

    def test_member_scores_reported_are_validation_not_prediction(self):
        """`member_primaries` must describe validation, the split used to select."""
        train, valid, predict = synthetic()
        baseline = run(train, valid, predict).metadata
        # Same train/valid, completely different prediction rows: the reported
        # member quality must be identical because it is a validation quantity.
        other = [tuple(r[:6]) + (1 - r[6],) + tuple(r[7:]) for r in predict]
        altered = run(train, valid, other).metadata
        self.assertEqual(baseline["member_primaries"], altered["member_primaries"])

    def test_without_a_prediction_split_scoring_falls_back_to_validation(self):
        train, valid, _predict = synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            output = run_ensemble_fm_candidate(
                PreparedData(train, valid), CONFIG, Path(tmp)
            )
        self.assertEqual(output.metadata["scored_split"], "validation")
        self.assertEqual(len(output.scores), len(valid))


if __name__ == "__main__":
    unittest.main()
