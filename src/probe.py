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


def train_probe_fullbatch_es(
    head: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    min_epochs: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
    device: torch.device,
) -> dict:
    """Full-batch probe training with early stopping on validation accuracy.

    Each epoch performs exactly one gradient step on the full training set
    (no mini-batching, no shuffle). Early stopping: after ``min_epochs``, stop
    if validation accuracy has not improved by ``>= min_delta`` for ``patience``
    consecutive epochs. ``max_epochs`` is a hard cap; a run that reaches it
    without early stopping is tagged as non-converged (inefficient under this
    compute budget).

    Checkpoint selection follows the EXP-001 convention: the epoch with the
    highest validation accuracy, ties broken by lower validation NLL. The
    best head state is retained in memory (CPU) for one-time test evaluation.

    Returns ``{"history", "best_epoch", "best_val_acc", "best_val_nll",
    "final_val_acc", "final_val_nll", "converged", "stop_reason",
    "best_state"}``.
    """
    seed_all(seed)
    head = head.to(device)
    opt = _make_optimizer("adamw", head.parameters(), lr, weight_decay)

    best = {"best_epoch": -1, "best_val_acc": -1.0, "best_val_nll": float("inf")}
    best_state: dict | None = None
    history: list[dict] = []
    epochs_since_improve = 0
    best_for_patience = -1.0  # val-acc reference for the min_delta patience test
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        head.train()
        opt.zero_grad()
        logits = head(train_x)
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
        opt.step()

        val_acc, val_nll = eval_probe(head, val_x, val_y)
        history.append({
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_acc": val_acc,
            "val_nll": val_nll,
        })

        # checkpoint selection: max val_acc, tie-break lower val_nll
        ckpt_improved = (val_acc > best["best_val_acc"]) or (
            val_acc == best["best_val_acc"] and val_nll < best["best_val_nll"]
        )
        if ckpt_improved:
            best = {"best_epoch": epoch, "best_val_acc": val_acc, "best_val_nll": val_nll}
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

        # early-stopping patience: reset only on a min_delta improvement
        if val_acc > best_for_patience + min_delta:
            best_for_patience = val_acc
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch >= min_epochs and epochs_since_improve >= patience:
            stopped_early = True
            break

    return {
        "history": history,
        "best_epoch": best["best_epoch"],
        "best_val_acc": best["best_val_acc"],
        "best_val_nll": best["best_val_nll"],
        "final_val_acc": history[-1]["val_acc"],
        "final_val_nll": history[-1]["val_nll"],
        "converged": stopped_early,
        "stop_reason": "early_stop" if stopped_early else "max_epochs",
        "best_state": best_state,
    }
