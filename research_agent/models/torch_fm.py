"""PyTorch Factorization Machine candidates for approved research directions."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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


def run_torch_fm_extension(
    prepared: PreparedData,
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    sampler: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    feature_transform: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray, int]] | None = None,
) -> CandidateOutput:
    """Run a generated in-scope FM extension through the same safe pipeline.

    Hooks live only in memory; the checkpoint contains only the serializable
    configuration and model weights.  This prevents a generated local function
    from being serialized or later mistaken for a reusable baseline setting.
    """
    _validate_locked_config(config)
    return _run_torch_fm(
        prepared,
        config,
        run_dir,
        loss_function=loss_function,
        sampler=sampler,
        feature_transform=feature_transform,
    )


def _run_torch_fm(
    prepared: PreparedData,
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    sampler: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    feature_transform: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray, int]] | None = None,
) -> CandidateOutput:
    """Shared implementation for fixed and generated in-scope FM methods."""
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    encoded, feature_dim = encode({"train": prepared.train_rows, "valid": prepared.validation_rows})
    x_train, y_train, train_users = encoded["train"]
    x_valid, y_valid, valid_users = encoded["valid"]
    if feature_transform is not None:
        x_train, x_valid, feature_dim = _apply_feature_transform(
            feature_transform, x_train, x_valid, feature_dim
        )

    model = TorchFM(feature_dim, int(config.get("embedding_dim", 16)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["l2"]))
    train_x = torch.as_tensor(x_train, dtype=torch.long)
    train_y = torch.as_tensor(y_train, dtype=torch.float32)
    valid_x = torch.as_tensor(x_valid, dtype=torch.long)
    batch_size = int(config.get("batch_size", 8192))
    epochs = int(config["epochs"])
    patience = int(config.get("patience", 3))
    loss_name = str(config["loss"])
    pair_groups = _pair_groups(train_users, y_train) if loss_name == "pairwise" else None
    if loss_name not in {"pointwise", "pairwise", "custom"}:
        raise ValueError(f"unsupported loss: {loss_name}")
    if loss_name == "custom" and loss_function is None:
        raise ValueError("custom loss requires an in-memory loss_function hook")
    if pair_groups is not None and not pair_groups:
        raise ValueError("pairwise loss requires at least one user with positive and negative training rows")

    best_primary, best_metrics, best_state, best_epoch, bad_epochs = float("-inf"), {}, None, 0, 0
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = _train_epoch(
            model, optimizer, train_x, train_y, batch_size=batch_size, seed=seed + epoch,
            loss_name=loss_name, pair_groups=pair_groups, loss_function=loss_function, sampler=sampler,
        )
        scores = _predict(model, valid_x)
        metrics = evaluate_predictions(valid_users, y_valid, scores).as_dict()
        epoch_rows.append({"epoch": epoch, "loss": epoch_loss, **metrics})
        if float(metrics["primary"]) > best_primary + 1e-5:
            best_primary, best_metrics = float(metrics["primary"]), metrics
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            best_epoch, bad_epochs = epoch, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("candidate did not produce a valid validation state")
    model.load_state_dict(best_state)
    _write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save({"model_state": best_state, "feature_dim": feature_dim, "config": dict(config), "summary": {"best_epoch": best_epoch, "best_primary": best_primary, "metrics": dict(best_metrics)}}, checkpoint_path)
    return CandidateOutput(user_ids=valid_users, labels=y_valid, scores=_predict(model, valid_x), metadata={"framework": "pytorch", "model": "fm", "loss": loss_name, "extension_name": config.get("extension_name", "none"), "best_epoch": best_epoch, "best_metrics": dict(best_metrics), "checkpoint_path": str(checkpoint_path)})


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
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    sampler: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    if loss_name in {"pointwise", "custom"}:
        order = sampler(labels, seed) if sampler is not None else torch.randperm(len(labels), generator=generator)
        if not isinstance(order, torch.Tensor) or order.ndim != 1 or len(order) == 0:
            raise ValueError("custom sampler must return a non-empty one-dimensional torch tensor")
        order = order.to(dtype=torch.long)
        if int(order.min()) < 0 or int(order.max()) >= len(labels):
            raise ValueError("custom sampler returned an index outside the training split")
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(features[index])
            loss = F.binary_cross_entropy_with_logits(logits, labels[index]) if loss_name == "pointwise" else loss_function(logits, labels[index])
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


def _apply_feature_transform(
    transform: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray, int]],
    train_features: np.ndarray,
    valid_features: np.ndarray,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate an LLM-authored, data.py-only categorical feature transform."""
    result = transform(train_features.copy(), valid_features.copy(), int(feature_dim))
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("feature_transform must return (train_features, valid_features, feature_dim)")
    train_out, valid_out, transformed_dim = result
    if not isinstance(transformed_dim, (int, np.integer)) or int(transformed_dim) < feature_dim:
        raise ValueError("feature_transform must retain the original feature ID range")
    if not isinstance(train_out, np.ndarray) or not isinstance(valid_out, np.ndarray):
        raise ValueError("feature_transform must return NumPy feature arrays")
    if train_out.ndim != 2 or valid_out.ndim != 2 or len(train_out) != len(train_features) or len(valid_out) != len(valid_features):
        raise ValueError("feature_transform must preserve row counts with two-dimensional features")
    if train_out.shape[1] != valid_out.shape[1] or not np.issubdtype(train_out.dtype, np.integer) or not np.issubdtype(valid_out.dtype, np.integer):
        raise ValueError("feature_transform must return matching integer feature matrices")
    if train_out.size and (train_out.min() < 0 or train_out.max() >= transformed_dim):
        raise ValueError("feature_transform returned train feature IDs outside its declared range")
    if valid_out.size and (valid_out.min() < 0 or valid_out.max() >= transformed_dim):
        raise ValueError("feature_transform returned validation feature IDs outside its declared range")
    return train_out.astype(np.int64, copy=False), valid_out.astype(np.int64, copy=False), int(transformed_dim)


def _validate_locked_config(config: Mapping[str, Any]) -> None:
    """Prevent generated source from silently changing parity-controlled settings."""
    if "_locked_settings" not in config:
        raise ValueError("candidate configuration is missing its locked-settings record")
    locked = config["_locked_settings"]
    if not isinstance(locked, Mapping):
        raise ValueError("candidate configuration is missing its locked-settings record")
    required = {"loss", "learning_rate", "l2", "embedding_dim", "batch_size", "seed", "epochs", "patience"}
    if not required.issubset(locked):
        raise ValueError("candidate locked-settings record is incomplete")
    changed = [name for name, value in locked.items() if config.get(name) != value]
    if changed:
        raise ValueError("candidate changed parity-locked settings: " + ", ".join(sorted(changed)))


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
