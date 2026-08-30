import unittest


from research_agent.models.din import PAD, build_sequences


def row(date, user, video, author, label):
    # canonical 12-field staged row
    return (date, user, video, author, "1", 1000.0, label, 9, 0.0, "m", "t", 0)


class DINSequenceTests(unittest.TestCase):
    def setUp(self):
        # One user, four train days, then a validation row.
        self.train = [
            row(20220408, "u1", "v1", "a1", 1),
            row(20220409, "u1", "v2", "a2", 0),
            row(20220410, "u1", "v3", "a1", 1),
            row(20220411, "u1", "v4", "a3", 0),
            row(20220408, "u2", "v5", "a4", 1),
        ]
        self.valid = [row(20220422, "u1", "v2", "a2", 1)]

    def test_training_row_never_sees_its_own_day_or_later(self):
        train, _valid, _sizes = build_sequences(self.train, self.valid, max_len=10)
        # Rows are encoded in input order; row 3 is u1 on 20220411.
        history = train["hist_v"][3]
        present = {int(v) for v in history if v != PAD}

        # It may see the three strictly-earlier days...
        self.assertEqual(len(present), 3)
        # ...and must not contain the candidate it is being scored on.
        self.assertNotIn(int(train["cand_v"][3]), present)

    def test_first_interaction_has_empty_history(self):
        train, _valid, _sizes = build_sequences(self.train, self.valid, max_len=10)
        self.assertTrue(all(v == PAD for v in train["hist_v"][0]))

    def test_history_is_truncated_to_the_most_recent_events(self):
        train, _valid, _sizes = build_sequences(self.train, self.valid, max_len=2)
        history = [int(v) for v in train["hist_v"][3] if v != PAD]
        self.assertEqual(len(history), 2)
        # The two kept events must be the latest two, not the earliest.
        earliest = int(train["cand_v"][0])
        self.assertNotIn(earliest, history)

    def test_validation_history_comes_only_from_train_rows(self):
        train, valid, _sizes = build_sequences(self.train, self.valid, max_len=10)
        seen = {int(v) for v in valid["hist_v"][0] if v != PAD}
        train_videos = {int(v) for v in train["cand_v"]}
        self.assertTrue(seen)
        self.assertTrue(seen.issubset(train_videos))

    def test_users_absent_from_train_get_empty_history_not_a_crash(self):
        _train, valid, _sizes = build_sequences(
            self.train, [row(20220422, "brand_new_user", "v1", "a1", 0)], max_len=5
        )
        self.assertTrue(all(v == PAD for v in valid["hist_v"][0]))

    def test_unknown_validation_items_fall_back_to_pad(self):
        _train, valid, _sizes = build_sequences(
            self.train, [row(20220422, "u1", "UNSEEN_VIDEO", "UNSEEN_AUTHOR", 0)], max_len=5
        )
        self.assertEqual(int(valid["cand_v"][0]), PAD)


class DINForwardTests(unittest.TestCase):
    def test_empty_history_produces_finite_scores(self):
        """A user with no prior events makes the attention row fully masked."""
        import torch

        from research_agent.models.din import DIN

        model = DIN({"video": 5, "author": 5, "tab": 3}, embedding_dim=4, hidden=8)
        scores = model(
            torch.tensor([1, 2]),
            torch.tensor([1, 2]),
            torch.tensor([1, 1]),
            torch.tensor([[0, 0], [1, 2]]),   # first row: no history at all
            torch.tensor([[0, 0], [1, 2]]),
        )
        self.assertTrue(torch.isfinite(scores).all(), scores)


if __name__ == "__main__":
    unittest.main()
