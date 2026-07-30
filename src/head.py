"""Linear-head training on cached CLS features.

For each (seed, layer) an independent linear head ``768 -> 150`` (Xavier-uniform
weight, zero bias) is trained on the cached CLS features of that layer while the
backbone stays frozen. Validation accuracy selects the checkpoint (tie-break:
lower validation NLL); the test split is evaluated exactly once with the
selected checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .seeding import seed_all


class LinearHead(nn.Module):
    """Linear classifier head with Xavier-uniform weights and zero bias."""

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@torch.no_grad()
def evaluate(head: nn.Module, features: torch.Tensor, labels: torch.Tensor, batch_size: int = 512) -> tuple[torch.Tensor, float, float]:
    """Return (logits (N,C), accuracy, mean_nll)."""
    head.eval()
    logits = []
    for start in range(0, features.shape[0], batch_size):
        logits.append(head(features[start : start + batch_size]))
    logits = torch.cat(logits, dim=0)
    loss = F.cross_entropy(logits, labels, reduction="mean")
    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean().item()
    return logits, acc, loss.item()


def train_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    seed: int,
    lr: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    grad_clip: float,
    in_dim: int,
    n_classes: int,
    device: torch.device,
    ckpt_dir: Path,
    mlflow_log=None,
    save_checkpoints: bool = True,
) -> dict:
    """Train one linear head; save every-epoch checkpoints; select best by val acc.

    Returns ``{"val_history": [...], "best_epoch": int, "best_val_acc": float,
    "best_val_nll": float}``. When ``save_checkpoints`` is False (lr smoke test),
    no epoch checkpoints are written and ``ckpt_dir`` is not required to persist.
    """
    if save_checkpoints:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Reset RNG so every layer under the same seed shares init + shuffle (paired).
    seed_all(seed)

    head = LinearHead(in_dim, n_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(train_x, train_y)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

    best = {"best_epoch": -1, "best_val_acc": -1.0, "best_val_nll": float("inf")}
    val_history = []

    for epoch in range(1, epochs + 1):
        head.train()
        for xb, yb in loader:
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()

        _, val_acc, val_nll = evaluate(head, val_x, val_y)
        val_history.append({"epoch": epoch, "validation_accuracy": val_acc, "validation_nll": val_nll})

        # save every-epoch checkpoint
        if save_checkpoints:
            torch.save(head.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")

        if mlflow_log is not None:
            mlflow_log(epoch, val_acc, val_nll)

        improved = (val_acc > best["best_val_acc"]) or (
            val_acc == best["best_val_acc"] and val_nll < best["best_val_nll"]
        )
        if improved:
            best = {"best_epoch": epoch, "best_val_acc": val_acc, "best_val_nll": val_nll}

    if save_checkpoints:
        with open(ckpt_dir / "best_checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

    return {"val_history": val_history, **best}


@torch.no_grad()
def predict_with_checkpoint(
    ckpt_path: Path,
    features: torch.Tensor,
    in_dim: int,
    n_classes: int,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Load a saved checkpoint and return logits (N, C) in float32 on CPU."""
    head = LinearHead(in_dim, n_classes).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    head.load_state_dict(state)
    head.eval()
    logits = []
    for start in range(0, features.shape[0], batch_size):
        logits.append(head(features[start : start + batch_size]).cpu())
    return torch.cat(logits, dim=0)
