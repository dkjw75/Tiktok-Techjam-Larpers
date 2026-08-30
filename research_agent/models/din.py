"""Deep Interest Network candidate (Zhou et al., KDD 2018).

Why DIN rather than a sequential encoder: this benchmark ranks *within* a user's
slate, so any representation that is constant across that user's candidates is
inert -- a single user vector cannot reorder anything. DIN's target attention
produces a *candidate-conditioned* user representation, which is precisely the
property the metric rewards.

Leakage discipline: a training row dated d sees only that user's interactions
strictly before d; a validation row sees the user's complete train history and
nothing from validation. Day granularity avoids same-day ordering ambiguity.
Test rows are never available here -- the data boundary does not stage them.
"""
from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]
from torch.nn import functional as F  # type: ignore[import-not-found]

from ..metrics import evaluate_predictions
from ..runner import CandidateOutput, PreparedData

D_DATE, D_USER, D_VID, D_AUTH, D_TAB, D_DUR, D_Y, D_HOUR, D_PLAY, D_MUSIC, D_TAG, D_UP = range(12)

PAD = 0  # index 0 of every vocabulary is reserved for padding


def _vocab(values) -> dict[Any, int]:
    """Build a 1-based vocabulary; 0 stays reserved for PAD."""
    mapping: dict[Any, int] = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def build_sequences(
    train_rows: Sequence[Sequence[Any]],
    valid_rows: Sequence[Sequence[Any]],
    *,
    max_len: int,
):
    """Encode candidates and strictly-past behaviour sequences.

    Returns encoded train/valid tensors plus vocabulary sizes. The only source of
    history is the train split, so no validation label can reach a feature.
    """
    video_vocab = _vocab(row[D_VID] for row in train_rows)
    author_vocab = _vocab(row[D_AUTH] for row in train_rows)
    tab_vocab = _vocab(row[D_TAB] for row in train_rows)

    # Per-user chronological train behaviour, day-bucketed.
    by_user: dict[Any, list[tuple[int, int, int]]] = collections.defaultdict(list)
    for row in train_rows:
        by_user[row[D_USER]].append(
            (row[D_DATE], video_vocab[row[D_VID]], author_vocab[row[D_AUTH]])
        )
    for events in by_user.values():
        events.sort(key=lambda item: item[0])

    def history_before(user, cutoff_date):
        """Most recent `max_len` events strictly before `cutoff_date`."""
        events = by_user.get(user, ())
        picked = [event for event in events if event[0] < cutoff_date]
        return picked[-max_len:]

    def encode(rows, *, cutoff_is_row_date):
        count = len(rows)
        cand_v = np.zeros(count, dtype=np.int64)
        cand_a = np.zeros(count, dtype=np.int64)
        cand_t = np.zeros(count, dtype=np.int64)
        hist_v = np.zeros((count, max_len), dtype=np.int64)
        hist_a = np.zeros((count, max_len), dtype=np.int64)
        labels = np.zeros(count, dtype=np.float32)
        users = []
        for index, row in enumerate(rows):
            cand_v[index] = video_vocab.get(row[D_VID], PAD)
            cand_a[index] = author_vocab.get(row[D_AUTH], PAD)
            cand_t[index] = tab_vocab.get(row[D_TAB], PAD)
            events = (
                history_before(row[D_USER], row[D_DATE])
                if cutoff_is_row_date
                else by_user.get(row[D_USER], ())[-max_len:]
            )
            if events:
                hist_v[index, : len(events)] = [event[1] for event in events]
                hist_a[index, : len(events)] = [event[2] for event in events]
            labels[index] = row[D_Y]
            users.append(row[D_USER])
        return {
            "cand_v": cand_v, "cand_a": cand_a, "cand_t": cand_t,
            "hist_v": hist_v, "hist_a": hist_a,
            "y": labels, "users": users,
        }

    train = encode(train_rows, cutoff_is_row_date=True)
    valid = encode(valid_rows, cutoff_is_row_date=False)
    sizes = {
        "video": len(video_vocab) + 1,
        "author": len(author_vocab) + 1,
        "tab": len(tab_vocab) + 1,
    }
    return train, valid, sizes


class DIN(nn.Module):
    """Target-attention CTR model over a padded behaviour sequence."""

    def __init__(self, sizes: Mapping[str, int], embedding_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.video = nn.Embedding(sizes["video"], embedding_dim, padding_idx=PAD)
        self.author = nn.Embedding(sizes["author"], embedding_dim, padding_idx=PAD)
        self.tab = nn.Embedding(sizes["tab"], embedding_dim, padding_idx=PAD)
        for embedding in (self.video, self.author, self.tab):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                embedding.weight[PAD].zero_()

        item_dim = embedding_dim * 2                      # video + author
        # DIN local activation unit: [h, c, h-c, h*c] -> scalar weight.
        self.attention = nn.Sequential(
            nn.Linear(item_dim * 4, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(item_dim * 2 + embedding_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, cand_v, cand_a, cand_t, hist_v, hist_a):
        candidate = torch.cat([self.video(cand_v), self.author(cand_a)], dim=-1)
        history = torch.cat([self.video(hist_v), self.author(hist_a)], dim=-1)
        mask = (hist_v != PAD).unsqueeze(-1)

        expanded = candidate.unsqueeze(1).expand_as(history)
        weights = self.attention(
            torch.cat([history, expanded, history - expanded, history * expanded], dim=-1)
        )
        weights = weights.masked_fill(~mask, float("-inf"))
        # A user with no prior history has an all-masked row; softmax would be
        # NaN, so fall back to a zero interest vector for them.
        empty = ~mask.any(dim=1, keepdim=True)
        weights = torch.softmax(weights.masked_fill(empty.expand_as(weights), 0.0), dim=1)
        interest = (history * weights * mask).sum(dim=1)
        interest = interest.masked_fill(empty.squeeze(1), 0.0)

        return self.mlp(
            torch.cat([candidate, interest, self.tab(cand_t)], dim=-1)
        ).squeeze(-1)


def run_din_candidate(
    prepared: PreparedData,
    config: Mapping[str, Any],
    run_dir: Path,
) -> CandidateOutput:
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    max_len = int(config.get("max_len", 20))
    epochs = int(config["epochs"])
    patience = int(config.get("patience", 4))
    batch_size = int(config.get("batch_size", 4096))

    train, valid, sizes = build_sequences(
        list(prepared.train_rows), list(prepared.validation_rows), max_len=max_len
    )
    model = DIN(sizes, int(config.get("embedding_dim", 16)), int(config.get("hidden", 64)))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("l2", 1e-6)),
    )

    def tensors(split):
        return tuple(
            torch.as_tensor(split[key])
            for key in ("cand_v", "cand_a", "cand_t", "hist_v", "hist_a")
        )

    train_inputs = tensors(train)
    train_y = torch.as_tensor(train["y"])
    valid_inputs = tensors(valid)

    best, best_state, bad, epochs_run = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        order = torch.randperm(len(train_y), generator=torch.Generator().manual_seed(seed + epoch))
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model(*[item[index] for item in train_inputs]), train_y[index]
            )
            loss.backward()
            optimizer.step()
        scores = _predict(model, valid_inputs, batch_size)
        primary = evaluate_predictions(valid["users"], valid["y"], scores.tolist()).primary
        if primary > best + 1e-5:
            best, bad = primary, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        raise RuntimeError("DIN candidate produced no valid validation state")
    model.load_state_dict(best_state)
    scores = _predict(model, valid_inputs, batch_size)
    checkpoint = run_dir / "din_checkpoint.pt"
    torch.save({"model_state": best_state, "sizes": dict(sizes), "config": dict(config)}, checkpoint)
    return CandidateOutput(
        user_ids=valid["users"],
        labels=valid["y"].tolist(),
        scores=scores.tolist(),
        metadata={
            "framework": "pytorch",
            "model": "din",
            "max_len": max_len,
            "epochs_run": epochs_run,
            "best_epoch": epochs_run - bad,
            "stopped_by": "early_stopping" if epochs_run < epochs else "max_epochs_truncated",
            "configured_epochs": epochs,
            "effective_patience": patience,
            "best_primary": best,
            "checkpoint_path": str(checkpoint),
        },
    )


@torch.no_grad()
def _predict(model, inputs, batch_size):
    model.eval()
    total = len(inputs[0])
    out = []
    for start in range(0, total, batch_size):
        out.append(model(*[item[start : start + batch_size] for item in inputs]))
    return torch.cat(out).numpy()
