"""PyTorch Factorization Machine candidates for approved research directions."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import encode

from ..metrics import evaluate_predictions
from ..runner import CandidateOutput, PreparedData


class TorchFM(nn.Module):
    """Second-order FM matching the official baseline's feature formulation."""

    def __init__(self, feature_dim: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(feature_dim, embedding_dim)
        self.linear = nn.Embedding(feature_dim, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(features)
        summed = embedded.sum(dim=1)
        interactions = 0.5 * ((summed.square()).sum(dim=1) - embedded.square().sum(dim=(1, 2)))
        return self.bias + self.linear(features).squeeze(-1).sum(dim=1) + interactions


@dataclass(frozen=True)
class TrainingSummary:
    best_epoch: int
    best_primary: float
    best_metrics: Mapping[str, Any]


def run_torch_fm_candidate(
    prepared: PreparedData,
    config: Mapping[str, Any],
    run_dir: Path,
) -> CandidateOutput:
    """Train a validation-only FM candidate using data prepared by data.py."""
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    encoded, feature_dim = encode({"train": prepared.train_rows, "valid": prepared.validation_rows})
    x_train, y_train, train_users = encoded["train"]
    x_valid, y_valid, valid_users = encoded["valid"]

    model = TorchFM(feature_dim, int(config.get("embedding_dim", 16)))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["l2"]),
    )
    train_x = torch.as_tensor(x_train, dtype=torch.long)
    train_y = torch.as_tensor(y_train, dtype=torch.float32)
    valid_x = torch.as_tensor(x_valid, dtype=torch.long)
    batch_size = int(config.get("batch_size", 8192))
    epochs = int(config["epochs"])
    patience = int(config.get("patience", 3))
    loss_name = str(config["loss"])
    pair_groups = _pair_groups(train_users, y_train) if loss_name == "pairwise" else None
    if loss_name not in {"pointwise", "pairwise"}:
        raise ValueError(f"unsupported loss: {loss_name}")
    if pair_groups is not None and not pair_groups:
        raise ValueError("pairwise loss requires at least one user with positive and negative training rows")

    best_primary = float("-inf")
    best_metrics: Mapping[str, Any] = {}
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad_epochs = 0
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = _train_epoch(
            model,
            optimizer,
            train_x,
            train_y,
            batch_size=batch_size,
            seed=seed + epoch,
            loss_name=loss_name,
            pair_groups=pair_groups,
        )
        scores = _predict(model, valid_x)
        metrics = evaluate_predictions(valid_users, y_valid, scores).as_dict()
        epoch_rows.append({"epoch": epoch, "loss": epoch_loss, **metrics})
        if float(metrics["primary"]) > best_primary + 1e-5:
            best_primary = float(metrics["primary"])
            best_metrics = metrics
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("candidate did not produce a valid validation state")
    model.load_state_dict(best_state)
    _write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": best_state,
            "feature_dim": feature_dim,
            "config": dict(config),
            "summary": {"best_epoch": best_epoch, "best_primary": best_primary, "metrics": dict(best_metrics)},
        },
        checkpoint_path,
    )
    final_scores = _predict(model, valid_x)
    return CandidateOutput(
        user_ids=valid_users,
        labels=y_valid,
        scores=final_scores,
        metadata={
            "framework": "pytorch",
            "model": "fm",
            "loss": loss_name,
            "best_epoch": best_epoch,
            "best_metrics": dict(best_metrics),
            "checkpoint_path": str(checkpoint_path),
        },
    )


def _train_epoch(
    model: TorchFM,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    loss_name: str,
    pair_groups: Sequence[tuple[np.ndarray, np.ndarray]] | None,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    if loss_name == "pointwise":
        order = torch.randperm(len(labels), generator=generator)
        if not isinstance(order, torch.Tensor) or order.ndim != 1 or len(order) == 0:
            raise ValueError("custom sampler must return a non-empty one-dimensional torch tensor")
        order = order.to(dtype=torch.long)
        if int(order.min()) < 0 or int(order.max()) >= len(labels):
            raise ValueError("custom sampler returned an index outside the training split")
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(features[index])
            loss = F.binary_cross_entropy_with_logits(logits, labels[index])
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError("custom loss must return one finite scalar torch tensor")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    else:
        assert pair_groups is not None
        rng = np.random.default_rng(seed)
        steps = math.ceil(len(labels) / batch_size)
        for _ in range(steps):
            selected = rng.integers(len(pair_groups), size=batch_size)
            positives = np.fromiter(
                (groups[0][rng.integers(len(groups[0]))] for groups in (pair_groups[i] for i in selected)),
                dtype=np.int64,
                count=batch_size,
            )
            negatives = np.fromiter(
                (groups[1][rng.integers(len(groups[1]))] for groups in (pair_groups[i] for i in selected)),
                dtype=np.int64,
                count=batch_size,
            )
            optimizer.zero_grad()
            difference = model(features[torch.from_numpy(positives)]) - model(features[torch.from_numpy(negatives)])
            loss = -F.logsigmoid(difference).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _pair_groups(users: Sequence[Any], labels: Sequence[Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped: dict[Any, tuple[list[int], list[int]]] = {}
    for index, (user, label) in enumerate(zip(users, labels)):
        positive, negative = grouped.setdefault(user, ([], []))
        (positive if int(label) == 1 else negative).append(index)
    return [
        (np.asarray(positive, dtype=np.int64), np.asarray(negative, dtype=np.int64))
        for positive, negative in grouped.values()
        if positive and negative
    ]


def _predict(model: TorchFM, features: torch.Tensor, batch_size: int = 200_000) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            output.append(model(features[start : start + batch_size]).cpu().numpy())
    return np.concatenate(output)


def _write_epoch_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ("epoch", "loss", "GAUC", "nDCG@5", "primary", "users", "rows", "evaluator_sha256")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
