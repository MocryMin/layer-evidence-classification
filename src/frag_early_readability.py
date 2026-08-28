"""Pure analysis/probe helpers for the fragmented Llama early-readability run."""
from __future__ import annotations

import copy
import statistics
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FAMILIES = ("single_block", "canonical_prefix")


def path_specs(n_layers: int = 28) -> list[dict[str, Any]]:
    specs = []
    for layer in range(1, n_layers + 1):
        specs.append({
            "family": "single_block",
            "layer": layer,
            "path": [layer],
            "path_id": f"single_L{layer:02d}",
        })
    for layer in range(1, n_layers + 1):
        specs.append({
            "family": "canonical_prefix",
            "layer": layer,
            "path": list(range(1, layer + 1)),
            "path_id": f"prefix_L{layer:02d}",
        })
    return specs


def variance_stats(features: torch.Tensor) -> dict[str, float | int | bool]:
    """EXP-002-compatible last-token variance statistics without a dense covariance."""
    x = features.float()
    if x.ndim != 2:
        raise ValueError("features must have shape [N,D]")
    inter = x.std(dim=0, unbiased=False)
    within = x.std(dim=1, unbiased=False)
    norms = x.norm(dim=1)
    inter_mean = float(inter.mean().item())
    within_mean = float(within.mean().item())
    return {
        "n_samples": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "mean_norm": float(norms.mean().item()),
        "std_norm": float(norms.std(unbiased=False).item()),
        "mean_abs": float(x.abs().mean().item()),
        "inter_sample_std_mean": inter_mean,
        "inter_sample_std_median": float(inter.median().item()),
        "inter_sample_std_min": float(inter.min().item()),
        "inter_sample_std_max": float(inter.max().item()),
        "within_sample_std_mean": within_mean,
        "inter_to_within_std_ratio": inter_mean / max(within_mean, 1e-12),
        "exp002_collapsed": inter_mean < 1e-3,
    }


class PlainProbe(nn.Module):
    def __init__(self, in_dim: int = 3072, n_classes: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LNPlainProbe(nn.Module):
    def __init__(self, in_dim: int = 3072, n_classes: int = 5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, n_classes, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.ln(x))


def build_probe(family: str, in_dim: int = 3072) -> nn.Module:
    return {"plain": PlainProbe, "ln_plain": LNPlainProbe}[family](in_dim, 5)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


@torch.no_grad()
def evaluate_probe(
    head: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float, torch.Tensor]:
    head.eval()
    logits = head(x)
    valid = masked_logits(logits, mask)
    return (
        float((valid.argmax(dim=1) == y).float().mean().item()),
        float(F.cross_entropy(valid, y).item()),
        valid.detach().cpu(),
    )


def train_adamw_smoke(
    *,
    family: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_mask: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_mask: torch.Tensor,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if train_x.is_cuda:
        torch.cuda.manual_seed_all(seed)
    head = build_probe(family, train_x.shape[1]).to(train_x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    best_accuracy = -1.0
    best_nll = float("inf")
    best_epoch = 0
    best_state = None
    patience_reference = -1.0
    stale = 0
    history = []
    failure = None
    for epoch in range(1, max_epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = masked_logits(head(train_x), train_mask)
        loss = F.cross_entropy(logits, train_y)
        if not torch.isfinite(loss):
            failure = f"non-finite training loss at epoch {epoch}"
            break
        loss.backward()
        optimizer.step()
        val_accuracy, val_nll, _ = evaluate_probe(head, val_x, val_y, val_mask)
        if not np.isfinite(val_nll):
            failure = f"non-finite validation NLL at epoch {epoch}"
            break
        history.append({"epoch": epoch, "val_accuracy": val_accuracy, "val_nll": val_nll})
        if val_accuracy > best_accuracy or (
            val_accuracy == best_accuracy and val_nll < best_nll
        ):
            best_accuracy = val_accuracy
            best_nll = val_nll
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in head.state_dict().items()})
        if val_accuracy > patience_reference + min_delta:
            patience_reference = val_accuracy
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    if best_state is None:
        return {
            "best_accuracy": -1.0,
            "best_nll": float("inf"),
            "best_epoch": 0,
            "epochs_run": len(history),
            "stopped_early": True,
            "failure": failure or "no finite validation result",
            "best_state": None,
            "history": history,
        }
    return {
        "best_accuracy": best_accuracy,
        "best_nll": best_nll,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "stopped_early": len(history) < max_epochs,
        "failure": failure,
        "best_state": best_state,
        "history": history,
    }


def train_adamw_fixed(
    *,
    family: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_mask: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    eval_mask: torch.Tensor,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if train_x.is_cuda:
        torch.cuda.manual_seed_all(seed)
    head = build_probe(family, train_x.shape[1]).to(train_x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(masked_logits(head(train_x), train_mask), train_y)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss during fixed AdamW fit")
        loss.backward()
        optimizer.step()
    train_accuracy, train_nll, _ = evaluate_probe(head, train_x, train_y, train_mask)
    eval_accuracy, eval_nll, eval_logits = evaluate_probe(head, eval_x, eval_y, eval_mask)
    return {
        "train_accuracy": train_accuracy,
        "train_nll": train_nll,
        "eval_accuracy": eval_accuracy,
        "eval_nll": eval_nll,
        "eval_logits": eval_logits,
    }


def select_smoke_value(
    rows: Sequence[dict[str, Any]], value_name: str
) -> tuple[float, dict[str, Any]]:
    values = sorted({float(row[value_name]) for row in rows})
    summaries = {}
    for value in values:
        selected = [row for row in rows if float(row[value_name]) == value]
        summaries[value] = {
            "mean_accuracy": float(statistics.mean(row["best_accuracy"] for row in selected)),
            "median_best_epoch": int(round(statistics.median(row["best_epoch"] for row in selected))),
            "n": len(selected),
        }
    best = max(values, key=lambda value: (summaries[value]["mean_accuracy"], -value))
    return best, {str(value): summaries[value] for value in values}


def expand_ridge_decision(clf: Any, x: np.ndarray, n_classes: int = 5) -> np.ndarray:
    raw = clf.decision_function(x)
    if raw.ndim == 1:
        raw = np.stack([-raw, raw], axis=1)
    out = np.full((len(x), n_classes), -1e30, dtype=np.float64)
    out[:, np.asarray(clf.classes_, dtype=np.int64)] = raw
    return out


def masked_numpy_accuracy(
    decision: np.ndarray, labels: np.ndarray, choice_counts: np.ndarray
) -> float:
    masked = decision.copy()
    for index, count in enumerate(choice_counts):
        masked[index, int(count):] = -1e30
    return float((masked.argmax(axis=1) == labels).mean())


def fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    alpha: float,
) -> tuple[Any, np.ndarray]:
    from sklearn.linear_model import RidgeClassifier

    clf = RidgeClassifier(
        alpha=float(alpha), fit_intercept=True, solver="lsqr", tol=1e-5
    )
    clf.fit(train_x, train_y)
    return clf, expand_ridge_decision(clf, eval_x, 5)
