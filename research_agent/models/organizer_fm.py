"""Validation-only adapter for the unchanged organizer FM baseline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from baseline import FM
from data import encode

from ..metrics import evaluate_predictions
from ..runner import CandidateOutput, PreparedData


def run_organizer_fm_candidate(
    prepared: PreparedData,
    config: Mapping[str, Any],
    _run_dir: Path,
) -> CandidateOutput:
    """Run the organizer's FM training rule without loading or scoring test."""
    seed = int(config.get("seed", 0))
    epochs = int(config.get("epochs", 40))
    patience = int(config.get("patience", 4))
    batch_size = int(config.get("batch_size", 8192))
    encoded, feature_dim = encode(
        {"train": prepared.train_rows, "valid": prepared.validation_rows}
    )
    train_x, train_y, _ = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    model = FM(
        feature_dim,
        k=int(config.get("embedding_dim", 16)),
        lr=float(config.get("learning_rate", 0.001)),
        l2=float(config.get("l2", 1e-6)),
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    best = float("-inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    epochs_run = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_y))
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            model.step(train_x[index], train_y[index])
        score = float(
            evaluate_predictions(
                valid_users,
                valid_y,
                model.predict(valid_x),
            ).primary
        )
        epochs_run = epoch
        if score > best + 1e-5:
            best = score
            best_epoch = epoch
            bad_epochs = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("organizer FM did not produce a validation state")
    model.V, model.W, model.b = best_state
    stopped_by = "early_stopping" if epochs_run < epochs else "max_epochs_truncated"
    return CandidateOutput(
        valid_users,
        valid_y,
        model.predict(valid_x),
        {
            "framework": "numpy",
            "model": "organizer_fm",
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "stopped_by": stopped_by,
            "configured_epochs": epochs,
            "effective_patience": patience,
        },
    )
