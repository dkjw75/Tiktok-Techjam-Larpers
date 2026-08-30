import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from research_agent.models.torch_fm import (
    TorchFM,
    _listwise_batches,
    _listwise_groups,
    _listwise_objective,
    _listwise_softmax_loss,
    run_torch_fm_candidate,
)
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
                    Path(directory),
                )
                self.assertEqual(len(output.scores), 4)
                self.assertTrue((Path(directory) / "epoch_metrics.csv").exists())
                self.assertTrue((Path(directory) / "checkpoint.pt").exists())

    def test_listwise_softmax_loss_matches_formula_and_rewards_correct_order(self):
        labels = torch.tensor([1.0, 0.0, 0.0, 1.0])
        scores = torch.tensor([2.0, 0.0, 1.0, -1.0])
        loss = _listwise_softmax_loss(scores, labels, (2, 2), temperature=1.0)
        expected = (
            -F.log_softmax(scores[:2], dim=0)[0]
            - F.log_softmax(scores[2:], dim=0)[1]
        ) / 2
        self.assertAlmostEqual(float(loss), float(expected), places=7)

        correctly_ranked = _listwise_softmax_loss(
            torch.tensor([3.0, -1.0]), torch.tensor([1.0, 0.0]), (2,), temperature=1.0
        )
        incorrectly_ranked = _listwise_softmax_loss(
            torch.tensor([-1.0, 3.0]), torch.tensor([1.0, 0.0]), (2,), temperature=1.0
        )
        self.assertLess(float(correctly_ranked), float(incorrectly_ranked))

    def test_mixed_listwise_objective_uses_requested_weighting(self):
        scores = torch.tensor([1.5, -0.5, 0.25])
        labels = torch.tensor([1.0, 0.0, 1.0])
        listwise = _listwise_softmax_loss(scores, labels, (3,), temperature=0.5)
        pointwise = F.binary_cross_entropy_with_logits(scores, labels)
        mixed = _listwise_objective(
            scores,
            labels,
            (3,),
            temperature=0.5,
            pointwise_weight=0.25,
        )
        self.assertAlmostEqual(float(mixed), float(0.75 * listwise + 0.25 * pointwise), places=7)

    def test_listwise_groups_keep_complete_slates_and_skip_degenerate_users(self):
        groups = _listwise_groups(
            ["u1", "u2", "u1", "u3", "u2", "u3"],
            [1, 0, 0, 1, 0, 1],
        )
        self.assertEqual([group.tolist() for group in groups], [[0, 2]])

    def test_listwise_batches_are_seeded_and_never_split_a_user(self):
        groups = [np.asarray([0, 1]), np.asarray([2, 3, 4]), np.asarray([5])]
        first = [(indices.tolist(), sizes) for indices, sizes in _listwise_batches(groups, batch_size=3, seed=7)]
        second = [(indices.tolist(), sizes) for indices, sizes in _listwise_batches(groups, batch_size=3, seed=7)]
        self.assertEqual(first, second)
        self.assertEqual(sorted(index for indices, _ in first for index in indices), list(range(6)))
        self.assertTrue(all(sum(sizes) == len(indices) for indices, sizes in first))
        self.assertTrue(all(size in {1, 2, 3} for _, sizes in first for size in sizes))

    def test_listwise_candidate_is_deterministic_and_records_objective_metadata(self):
        prepared = PreparedData(
            train_rows=[row("u1", "v1", 1), row("u1", "v2", 0), row("u2", "v1", 0), row("u2", "v2", 1)],
            validation_rows=[row("u1", "v1", 1), row("u1", "v2", 0), row("u2", "v1", 0), row("u2", "v2", 1)],
        )
        config = {
            "loss": "listwise",
            "listwise_temperature": 0.5,
            "pointwise_weight": 0.25,
            "learning_rate": 0.01,
            "l2": 0.0,
            "epochs": 2,
            "batch_size": 2,
            "seed": 11,
        }
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = run_torch_fm_candidate(prepared, config, Path(first_dir))
            second = run_torch_fm_candidate(prepared, config, Path(second_dir))
        self.assertEqual(first.scores, second.scores)
        self.assertEqual(first.metadata["listwise_temperature"], 0.5)
        self.assertEqual(first.metadata["pointwise_weight"], 0.25)
        self.assertEqual(first.metadata["listwise_user_groups"], 2)

    def test_named_listwise_variants_map_to_the_three_bounded_objectives(self):
        prepared = PreparedData(
            train_rows=[row("u1", "v1", 1), row("u1", "v2", 0)],
            validation_rows=[row("u1", "v1", 1), row("u1", "v2", 0)],
        )
        expected = {
            "t1": (1.0, 0.0),
            "t05": (0.5, 0.0),
            "t1_bce25": (1.0, 0.25),
        }
        for variant, parameters in expected.items():
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                output = run_torch_fm_candidate(
                    prepared,
                    {
                        "loss": "listwise",
                        "objective_variant": variant,
                        "learning_rate": 0.01,
                        "l2": 0.0,
                        "epochs": 1,
                        "batch_size": 2,
                        "seed": 0,
                    },
                    Path(directory),
                )
                self.assertEqual(
                    (
                        output.metadata["listwise_temperature"],
                        output.metadata["pointwise_weight"],
                    ),
                    parameters,
                )

    def test_listwise_parameters_and_groups_are_validated(self):
        prepared = PreparedData(
            train_rows=[row("u1", "v1", 1), row("u1", "v2", 0)],
            validation_rows=[row("u1", "v1", 1), row("u1", "v2", 0)],
        )
        base = {
            "loss": "listwise",
            "learning_rate": 0.01,
            "l2": 0.0,
            "epochs": 1,
            "batch_size": 2,
            "seed": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            for update in (
                {"listwise_temperature": 0.75},
                {"listwise_temperature": None},
                {"pointwise_weight": 0.5},
            ):
                with self.subTest(update=update), self.assertRaises(ValueError):
                    run_torch_fm_candidate(prepared, {**base, **update}, Path(directory))

        no_positive = PreparedData(
            train_rows=[row("u1", "v1", 0), row("u1", "v2", 0)],
            validation_rows=prepared.validation_rows,
        )
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            run_torch_fm_candidate(no_positive, base, Path(directory))
