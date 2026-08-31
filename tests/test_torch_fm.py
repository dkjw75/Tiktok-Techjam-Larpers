import tempfile
import unittest

import numpy as np
import torch

from research_agent.models.torch_fm import TorchFM, _apply_feature_transform, run_torch_fm_candidate
from research_agent.runner import PreparedData


def row(user, video, label):
    return (20220408, str(user), str(video), "author", "1", 10_000.0, label)


class TorchFMTests(unittest.TestCase):
    def test_forward_shape(self):
        model = TorchFM(feature_dim=20, embedding_dim=4)
        self.assertEqual(model(torch.tensor([[1, 2], [3, 4]])).shape, (2,))

    def test_pointwise_and_pairwise_candidates_produce_validation_scores(self):
        prepared = PreparedData(
            train_rows=[row("u1", "v1", 1), row("u1", "v2", 0), row("u2", "v1", 0), row("u2", "v2", 1)],
            validation_rows=[row("u1", "v1", 1), row("u1", "v2", 0), row("u2", "v1", 0), row("u2", "v2", 1)],
        )
        with tempfile.TemporaryDirectory() as directory:
            for loss in ("pointwise", "pairwise"):
                output = run_torch_fm_candidate(
                    prepared,
                    {"loss": loss, "learning_rate": 0.01, "l2": 0.0, "epochs": 2, "batch_size": 2, "seed": 0},
                    __import__("pathlib").Path(directory),
                )
                self.assertEqual(len(output.scores), 4)
                self.assertTrue((__import__("pathlib").Path(directory) / "epoch_metrics.csv").exists())
                self.assertTrue((__import__("pathlib").Path(directory) / "checkpoint.pt").exists())

    def test_feature_transform_can_add_valid_categorical_cross_features(self):
        train = np.array([[0, 1], [2, 3]], dtype=np.int64)
        valid = np.array([[1, 2]], dtype=np.int64)

        def transform(train_features, valid_features, feature_dim):
            add_cross = lambda values: np.concatenate(
                [values, ((values[:, :1] * 7 + values[:, 1:2]) % 4) + feature_dim], axis=1
            )
            return add_cross(train_features), add_cross(valid_features), feature_dim + 4

        train_out, valid_out, feature_dim = _apply_feature_transform(transform, train, valid, 4)

        self.assertEqual(train_out.shape, (2, 3))
        self.assertEqual(valid_out.shape, (1, 3))
        self.assertEqual(feature_dim, 8)
