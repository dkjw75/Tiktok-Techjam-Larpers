from __future__ import annotations

import unittest

from research_agent.complementarity import (
    ComplementarityValidationError,
    deterministic_user_partition,
    evaluate_complementarity,
)


class ComplementarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.users = [user for user in range(10, 16) for _ in range(4)]
        self.incumbent = [value for _user in range(6) for value in (4.0, 3.0, 2.0, 1.0)]
        self.candidate = [value for _user in range(6) for value in (1.0, 2.0, 3.0, 4.0)]
        self.partition = deterministic_user_partition(self.users)

    def _labels(self, *, fit_candidate_wins: bool, held_candidate_wins: bool) -> list[int]:
        fit_users = set(self.partition.fit_user_ids)
        labels: list[int] = []
        for user in range(10, 16):
            candidate_wins = (
                fit_candidate_wins if user in fit_users else held_candidate_wins
            )
            labels.extend((0, 0, 1, 1) if candidate_wins else (1, 1, 0, 0))
        return labels

    def test_changing_held_labels_cannot_change_selected_alpha(self) -> None:
        first = evaluate_complementarity(
            self.users,
            self._labels(fit_candidate_wins=False, held_candidate_wins=False),
            self.incumbent,
            self.candidate,
        )
        second = evaluate_complementarity(
            self.users,
            self._labels(fit_candidate_wins=False, held_candidate_wins=True),
            self.incumbent,
            self.candidate,
        )

        self.assertEqual(first.alpha, 0.0)
        self.assertEqual(second.alpha, first.alpha)
        self.assertEqual(second.partition_fingerprint, first.partition_fingerprint)
        self.assertNotEqual(second.blended_held_primary, first.blended_held_primary)

    def test_changing_fit_labels_can_change_selected_alpha(self) -> None:
        incumbent_fit = evaluate_complementarity(
            self.users,
            self._labels(fit_candidate_wins=False, held_candidate_wins=False),
            self.incumbent,
            self.candidate,
        )
        candidate_fit = evaluate_complementarity(
            self.users,
            self._labels(fit_candidate_wins=True, held_candidate_wins=False),
            self.incumbent,
            self.candidate,
        )

        self.assertEqual(incumbent_fit.alpha, 0.0)
        self.assertNotEqual(candidate_fit.alpha, incumbent_fit.alpha)
        self.assertGreater(candidate_fit.alpha, 0.5)

    def test_user_partitions_are_disjoint_deterministic_and_fingerprinted(self) -> None:
        repeated = deterministic_user_partition(list(reversed(self.users)))

        self.assertEqual(repeated, self.partition)
        self.assertTrue(set(self.partition.fit_user_ids).isdisjoint(
            self.partition.held_user_ids
        ))
        self.assertEqual(
            set(self.partition.fit_user_ids) | set(self.partition.held_user_ids),
            set(self.users),
        )
        self.assertEqual(len(self.partition.fingerprint), 64)

        result = evaluate_complementarity(
            self.users,
            self._labels(fit_candidate_wins=False, held_candidate_wins=True),
            self.incumbent,
            self.candidate,
            expected_partition_fingerprint=self.partition.fingerprint,
        )
        self.assertEqual(result.fit_users + result.held_users, result.users)
        self.assertEqual(result.fit_rows + result.held_rows, result.rows)
        self.assertAlmostEqual(
            result.ensemble_delta_if_added,
            result.blended_held_primary - result.incumbent_held_primary,
        )

    def test_fails_closed_on_test_split_or_partition_drift(self) -> None:
        labels = self._labels(fit_candidate_wins=False, held_candidate_wins=False)
        with self.assertRaisesRegex(ComplementarityValidationError, "validation"):
            evaluate_complementarity(
                self.users, labels, self.incumbent, self.candidate, split="test"
            )
        with self.assertRaisesRegex(ComplementarityValidationError, "fingerprint"):
            evaluate_complementarity(
                self.users,
                labels,
                self.incumbent,
                self.candidate,
                expected_partition_fingerprint="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
