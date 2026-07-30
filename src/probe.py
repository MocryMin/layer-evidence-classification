"""General probe training supporting head variants and optimizers (EXP-002).

Used by the diagnostic campaign to compare plain/LN/ablation heads and
AdamW / LBFGS / SGD optimisers on cached CLS features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .seeding import seed_all


@torch.no_grad()
def eval_probe(head: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1024) -> tuple[float, float]:
    head.eval()
    logits = []
    for s in range(0, x.shape[0], batch_size):
        logits.append(head(x[s : s + batch_size]))
    logits = torch.cat(logits, dim=0)
    loss = F.cross_entropy(logits, y, reduction="mean").item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return acc, loss


def _make_optimizer(name: str, params, lr: float, weight_decay: float, lbfgs_cfg: dict | None = None):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if name == "lbfgs":
        cfg = lbfgs_cfg or {}
        # LBFGS ignores weight_decay in this wrapper (no decoupled decay); kept simple
        return torch.optim.LBFGS(
            params,
            lr=cfg.get("lr", 1.0),
            max_iter=cfg.get("max_iter", 100),
            history_size=cfg.get("history_size", 50),
            line_search_fn=cfg.get("line_search_fn", "strong_wolfe"),
        )
    raise ValueError(f"unknown optimizer {name}")


def train_probe(
    head: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    optimizer: str,
    lr: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    grad_clip: float,
    seed: int,
    device: torch.device,
    lbfgs_cfg: dict | None = None,
) -> dict:
    """Train ``head``; return val history + best (acc, tie-break lower nll)."""
    seed_all(seed)
    head = head.to(device)
    opt = _make_optimizer(optimizer, head.parameters(), lr, weight_decay, lbfgs_cfg)
    is_lbfgs = optimizer.lower() == "lbfgs"

    best = {"best_epoch": -1, "best_val_acc": -1.0, "best_val_nll": float("inf")}
    history = []

    if is_lbfgs:
        # full-batch LBFGS
        for epoch in range(1, epochs + 1):

            def closure():
                opt.zero_grad()
                loss = F.cross_entropy(head(train_x), train_y)
                loss.backward()
                return loss

            head.train()
            opt.step(closure)
            acc, nll = eval_probe(head, val_x, val_y)
            history.append({"epoch": epoch, "val_acc": acc, "val_nll": nll})
            if (acc > best["best_val_acc"]) or (acc == best["best_val_acc"] and nll < best["best_val_nll"]):
                best = {"best_epoch": epoch, "best_val_acc": acc, "best_val_nll": nll}
    else:
        ds = TensorDataset(train_x, train_y)
        g = torch.Generator().manual_seed(seed)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
        for epoch in range(1, epochs + 1):
            head.train()
            for xb, yb in loader:
                loss = F.cross_entropy(head(xb), yb)
                opt.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
                opt.step()
            acc, nll = eval_probe(head, val_x, val_y)
            history.append({"epoch": epoch, "val_acc": acc, "val_nll": nll})
            if (acc > best["best_val_acc"]) or (acc == best["best_val_acc"] and nll < best["best_val_nll"]):
                best = {"best_epoch": epoch, "best_val_acc": acc, "best_val_nll": nll}

    return {"history": history, **best, "final_val_acc": history[-1]["val_acc"], "final_val_nll": history[-1]["val_nll"]}
