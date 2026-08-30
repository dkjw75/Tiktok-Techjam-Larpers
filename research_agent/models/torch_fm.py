"""PyTorch Factorization Machine candidates for approved research directions."""
from __future__ import annotations

import csv
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]
from torch.nn import functional as F  # type: ignore[import-not-found]

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
    epochs_run: int
    stopped_by: str
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
    listwise_groups = _listwise_groups(train_users, y_train) if loss_name == "listwise" else None
    if loss_name not in {"pointwise", "pairwise", "listwise"}:
        raise ValueError(f"unsupported loss: {loss_name}")
    if pair_groups is not None and not pair_groups:
        raise ValueError("pairwise loss requires at least one user with positive and negative training rows")
    listwise_temperature = 1.0
    pointwise_weight = 0.0
    if loss_name == "listwise":
        variant = str(config.get("objective_variant", "custom"))
        presets = {
            "t1": (1.0, 0.0),
            "t05": (0.5, 0.0),
            "t1_bce25": (1.0, 0.25),
        }
        if variant != "custom" and variant not in presets:
            raise ValueError(f"unsupported listwise objective_variant: {variant}")
        if variant in presets:
            listwise_temperature, pointwise_weight = presets[variant]
        else:
            try:
                listwise_temperature = float(config.get("listwise_temperature", 1.0))
                pointwise_weight = float(config.get("pointwise_weight", 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError("listwise loss parameters must be numeric") from exc
        if listwise_temperature not in {0.5, 1.0}:
            raise ValueError("listwise_temperature must be 0.5 or 1.0")
        if pointwise_weight not in {0.0, 0.25}:
            raise ValueError("pointwise_weight must be 0 or 0.25")
        if not listwise_groups:
            raise ValueError(
                "listwise loss requires at least one mixed-label training user"
            )

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
            listwise_groups=listwise_groups,
            listwise_temperature=listwise_temperature,
            pointwise_weight=pointwise_weight,
        )
        scores = _predict(model, valid_x)
        metrics = evaluate_predictions(valid_users, y_valid, scores.tolist()).as_dict()
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
    epochs_run = len(epoch_rows)
    stopped_by = "early_stopping" if epochs_run < epochs else "max_epochs_truncated"
    model.load_state_dict(best_state)
    _write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": best_state,
            "feature_dim": feature_dim,
            "config": dict(config),
            "summary": {
                "best_epoch": best_epoch,
                "epochs_run": epochs_run,
                "stopped_by": stopped_by,
                "best_primary": best_primary,
                "metrics": dict(best_metrics),
            },
        },
        checkpoint_path,
    )
    final_scores = _predict(model, valid_x)
    return CandidateOutput(
        user_ids=valid_users,
        labels=y_valid,
        scores=final_scores.tolist(),
        metadata={
            "framework": "pytorch",
            "model": "fm",
            "loss": loss_name,
            "objective_variant": config.get("objective_variant") if loss_name == "listwise" else None,
            "listwise_temperature": listwise_temperature if loss_name == "listwise" else None,
            "pointwise_weight": pointwise_weight if loss_name == "listwise" else None,
            "listwise_user_groups": len(listwise_groups) if listwise_groups is not None else None,
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "stopped_by": stopped_by,
            "configured_epochs": epochs,
            "effective_patience": patience,
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
    listwise_groups: Sequence[np.ndarray] | None = None,
    listwise_temperature: float = 1.0,
    pointwise_weight: float = 0.0,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    if loss_name == "pointwise":
        order = torch.randperm(len(labels), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(features[index]), labels[index])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    elif loss_name == "pairwise":
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
    else:
        assert listwise_groups is not None
        total_loss = 0.0
        total_groups = 0
        for index, group_sizes in _listwise_batches(listwise_groups, batch_size=batch_size, seed=seed):
            optimizer.zero_grad()
            batch_scores = model(features[index])
            batch_labels = labels[index]
            loss = _listwise_objective(
                batch_scores,
                batch_labels,
                group_sizes,
                temperature=listwise_temperature,
                pointwise_weight=pointwise_weight,
            )
            loss.backward()
            optimizer.step()
            group_count = len(group_sizes)
            total_loss += float(loss.detach()) * group_count
            total_groups += group_count
        return total_loss / total_groups
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


def _listwise_groups(users: Sequence[Any], labels: Sequence[Any]) -> list[np.ndarray]:
    """Return complete slates for users with both positive and negative rows."""
    if len(users) != len(labels):
        raise ValueError("users and labels must have the same length")
    grouped: dict[Any, tuple[list[int], bool, bool]] = {}
    for index, (user, label) in enumerate(zip(users, labels)):
        binary_label = float(label)
        if binary_label not in {0.0, 1.0}:
            raise ValueError("listwise labels must be binary")
        indices, has_positive, has_negative = grouped.setdefault(
            user,
            ([], False, False),
        )
        indices.append(index)
        grouped[user] = (
            indices,
            has_positive or binary_label == 1.0,
            has_negative or binary_label == 0.0,
        )
    return [
        np.asarray(indices, dtype=np.int64)
        for indices, has_positive, has_negative in grouped.values()
        if has_positive and has_negative
    ]


def _listwise_batches(
    groups: Sequence[np.ndarray],
    *,
    batch_size: int,
    seed: int,
) -> Iterator[tuple[torch.Tensor, tuple[int, ...]]]:
    """Pack whole user slates into deterministic, approximately row-sized batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = np.random.default_rng(seed).permutation(len(groups))
    selected: list[np.ndarray] = []
    selected_rows = 0
    for position in order:
        group = groups[int(position)]
        if selected and selected_rows + len(group) > batch_size:
            yield torch.from_numpy(np.concatenate(selected)), tuple(len(item) for item in selected)
            selected = []
            selected_rows = 0
        selected.append(group)
        selected_rows += len(group)
    if selected:
        yield torch.from_numpy(np.concatenate(selected)), tuple(len(item) for item in selected)


def _listwise_objective(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    *,
    temperature: float,
    pointwise_weight: float,
) -> torch.Tensor:
    """Combine per-user listwise loss with an optional pointwise stabilizer."""
    if pointwise_weight not in {0.0, 0.25}:
        raise ValueError("pointwise_weight must be 0 or 0.25")
    listwise = _listwise_softmax_loss(scores, labels, group_sizes, temperature=temperature)
    if pointwise_weight == 0.0:
        return listwise
    pointwise = F.binary_cross_entropy_with_logits(scores, labels)
    return (1.0 - pointwise_weight) * listwise + pointwise_weight * pointwise


def _listwise_softmax_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    *,
    temperature: float,
) -> torch.Tensor:
    """Mean per-user positive log-softmax loss over complete exposed slates."""
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if sum(group_sizes) != len(scores) or len(scores) != len(labels):
        raise ValueError("group sizes must exactly partition scores and labels")
    if not bool(torch.all((labels == 0) | (labels == 1))):
        raise ValueError("listwise labels must be binary")
    losses: list[torch.Tensor] = []
    start = 0
    for size in group_sizes:
        if size <= 0:
            raise ValueError("listwise groups must not be empty")
        end = start + size
        positive = labels[start:end] == 1
        if not bool(positive.any()):
            raise ValueError("every listwise group must contain a positive row")
        log_probabilities = F.log_softmax(scores[start:end] / temperature, dim=0)
        losses.append(-log_probabilities[positive].mean())
        start = end
    if not losses:
        raise ValueError("listwise loss requires at least one user group")
    return torch.stack(losses).mean()


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
