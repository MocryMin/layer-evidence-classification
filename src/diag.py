"""Diagnostic routines for EXP-20260729-002.

- ``layerwise_feature_stats``: per-layer variance / anisotropy statistics (the
  core of the variance-collapse finding).
- ``lr_grid_probe``: unrestricted lr sweep of the plain linear probe.
- ``head_ablation``: plain / LN / norm-only / affine-only across layers.
- ``optimizer_comparison``: AdamW / LBFGS / SGD on the plain linear probe.
- ``cache_variant``: re-cache features for a different model path or prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .heads import build_head
from .probe import train_probe


# --------------------------------------------------------------------------- #
# Feature statistics (variance collapse / anisotropy)
# --------------------------------------------------------------------------- #
def feature_stats_layer(feat: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    """Per-layer CLS statistics. ``feat``: (N, D), ``labels``: (N,)."""
    feat = np.asarray(feat, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    N, D = feat.shape

    # inter-sample (across samples) per-dim std
    inter_std_per_dim = feat.std(axis=0)  # (D,)
    # within-sample (across dims) per-sample std
    within_std_per_sample = feat.std(axis=1)  # (N,)

    # class signal: between-class var / within-class var (averaged over dims)
    class_means = np.array([feat[labels == c].mean(axis=0) for c in range(n_classes)])  # (C, D)
    overall_mean = feat.mean(axis=0)
    between = ((class_means - overall_mean) ** 2).mean(axis=0).mean()
    within = np.mean([feat[labels == c].var(axis=0).mean() for c in range(n_classes)])
    class_signal_ratio = float(between / (within + 1e-12))

    # anisotropy via covariance eigenvalues (participation ratio)
    cov = np.cov(feat, rowvar=False)  # (D, D)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0, None)
    s = eig.sum()
    top1 = float(eig[-1] / (s + 1e-12))
    top10 = float(eig[-10:].sum() / (s + 1e-12))
    participation_ratio = float((s ** 2) / ((eig ** 2).sum() + 1e-12))  # in [1, D]

    return {
        "n_samples": int(N),
        "dim": int(D),
        "mean_norm": float(np.linalg.norm(feat, axis=1).mean()),
        "mean_abs": float(np.abs(feat).mean()),
        "inter_sample_std_mean": float(inter_std_per_dim.mean()),
        "inter_sample_std_median": float(np.median(inter_std_per_dim)),
        "inter_sample_std_min": float(inter_std_per_dim.min()),
        "inter_sample_std_max": float(inter_std_per_dim.max()),
        "within_sample_std_mean": float(within_std_per_sample.mean()),
        "class_signal_ratio": class_signal_ratio,
        "between_class_var": float(between),
        "within_class_var": float(within),
        "participation_ratio": participation_ratio,
        "top1_var_frac": top1,
        "top10_var_frac": top10,
    }


def layerwise_feature_stats(cache: dict, n_classes: int, split: str = "train") -> list[dict]:
    """Compute stats for every layer of a cached split. ``cache['hidden']: (N, L, D)``."""
    hidden = cache["hidden"]
    if hasattr(hidden, "cpu"):
        hidden = hidden.cpu().numpy()
    labels = cache["labels"]
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()
    L = hidden.shape[1]
    rows = []
    for l in range(L):
        st = feature_stats_layer(hidden[:, l, :], labels, n_classes)
        st["layer"] = l + 1
        st["split"] = split
        rows.append(st)
    return rows


def save_stats_json(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


# --------------------------------------------------------------------------- #
# Probe grids
# --------------------------------------------------------------------------- #
def _slice_layer(hidden: torch.Tensor, layer: int) -> torch.Tensor:
    return hidden[:, layer - 1, :]


def lr_grid_probe(
    train_cache: dict, val_cache: dict, lrs: list[float], layers: list[int],
    head_type: str, optimizer: str, epochs: int, batch_size: int,
    weight_decay: float, grad_clip: float, seed: int, device: torch.device,
    in_dim: int, n_classes: int, lbfgs_cfg: dict | None = None,
    log_every: int = 1,
) -> dict:
    """Run a lr x layer grid; return {(layer, lr): result}."""
    train_h = train_cache["hidden"]
    val_h = val_cache["hidden"]
    train_y = train_cache["labels"].to(device)
    val_y = val_cache["labels"].to(device)
    out = {}
    for layer in layers:
        tx = _slice_layer(train_h, layer)
        vx = _slice_layer(val_h, layer)
        for lr in lrs:
            head = build_head(head_type, in_dim, n_classes)
            res = train_probe(
                head, tx, train_y, vx, val_y,
                optimizer=optimizer, lr=lr, epochs=epochs, batch_size=batch_size,
                weight_decay=weight_decay, grad_clip=grad_clip, seed=seed,
                device=device, lbfgs_cfg=lbfgs_cfg,
            )
            out[f"layer_{layer}_lr_{lr:g}"] = {
                "layer": layer, "lr": lr,
                "best_val_acc": res["best_val_acc"], "best_val_nll": res["best_val_nll"],
                "best_epoch": res["best_epoch"],
                "final_val_acc": res["final_val_acc"], "final_val_nll": res["final_val_nll"],
            }
            print(f"    {head_type}/{optimizer} layer={layer} lr={lr:g}: "
                  f"best={res['best_val_acc']:.4f} final={res['final_val_acc']:.4f}")
    return out


def head_ablation(
    train_cache: dict, val_cache: dict, head_types: list[str], layers: list[int],
    lr: float, optimizer: str, epochs: int, batch_size: int, weight_decay: float,
    grad_clip: float, seed: int, device: torch.device, in_dim: int, n_classes: int,
    lbfgs_cfg: dict | None = None,
) -> dict:
    out = {}
    train_h = train_cache["hidden"]; val_h = val_cache["hidden"]
    train_y = train_cache["labels"].to(device); val_y = val_cache["labels"].to(device)
    for head_type in head_types:
        for layer in layers:
            tx = _slice_layer(train_h, layer); vx = _slice_layer(val_h, layer)
            head = build_head(head_type, in_dim, n_classes)
            res = train_probe(
                head, tx, train_y, vx, val_y, optimizer=optimizer, lr=lr, epochs=epochs,
                batch_size=batch_size, weight_decay=weight_decay, grad_clip=grad_clip,
                seed=seed, device=device, lbfgs_cfg=lbfgs_cfg,
            )
            out[f"{head_type}_layer_{layer}"] = {
                "head_type": head_type, "layer": layer,
                "best_val_acc": res["best_val_acc"], "best_val_nll": res["best_val_nll"],
                "final_val_acc": res["final_val_acc"],
            }
            print(f"    {head_type} layer={layer}: best={res['best_val_acc']:.4f}")
    return out


def optimizer_comparison(
    train_cache: dict, val_cache: dict, optimizers: list[str], layers: list[int],
    lr: float, epochs: int, batch_size: int, weight_decay: float, grad_clip: float,
    seed: int, device: torch.device, in_dim: int, n_classes: int,
    lbfgs_cfg: dict | None = None,
) -> dict:
    out = {}
    train_h = train_cache["hidden"]; val_h = val_cache["hidden"]
    train_y = train_cache["labels"].to(device); val_y = val_cache["labels"].to(device)
    for opt in optimizers:
        for layer in layers:
            tx = _slice_layer(train_h, layer); vx = _slice_layer(val_h, layer)
            head = build_head("plain", in_dim, n_classes)
            res = train_probe(
                head, tx, train_y, vx, val_y, optimizer=opt, lr=lr, epochs=epochs,
                batch_size=batch_size, weight_decay=weight_decay, grad_clip=grad_clip,
                seed=seed, device=device, lbfgs_cfg=lbfgs_cfg,
            )
            out[f"{opt}_layer_{layer}"] = {
                "optimizer": opt, "layer": layer,
                "best_val_acc": res["best_val_acc"], "best_val_nll": res["best_val_nll"],
                "final_val_acc": res["final_val_acc"],
            }
            print(f"    {opt} layer={layer}: best={res['best_val_acc']:.4f}")
    return out


# --------------------------------------------------------------------------- #
# Variant caching (different model path or prompt)
# --------------------------------------------------------------------------- #
def cache_variant(
    model_path: str, prompt: str, dataset_root, dataset_config, drop_oos_label,
    n_classes, max_length, truncation, n_layers, hidden_state_offset, cls_index,
    cache_dtype, device, out_dir, split_seed: int = 0,
) -> dict:
    """Cache 3 splits for a given backbone + prompt; write safetensors + manifest."""
    from transformers import AutoModel, AutoTokenizer
    from .data import build_label_maps, load_split, tokenise_split
    from .cache import cache_split, save_cache, write_manifest

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label2id, id2label, in_scope_ids = build_label_maps(dataset_root, dataset_config, drop_oos_label)
    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path), dtype=torch.float32)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    manifest = {"model_path": str(model_path), "prompt": prompt, "splits": {}}
    for split in ["train", "validation", "test"]:
        ds = load_split(dataset_root, dataset_config, split, drop_oos_label, in_scope_ids)
        tok_out = tokenise_split(ds, tok, prompt, max_length, truncation)
        cache = cache_split(
            model, tok, tok_out, device, batch_size=128,
            n_layers=n_layers, hidden_state_offset=hidden_state_offset, cls_index=cls_index,
        )
        if not torch.isfinite(cache["hidden"]).all():
            raise RuntimeError(f"NaN/Inf in cached hidden for split {split} ({model_path})")
        entry = save_cache(cache, out_dir / f"{split}_hidden.safetensors", cache_dtype)
        manifest["splits"][split] = {**entry, "n_samples": cache["hidden"].shape[0]}
        print(f"    cached {split}: {cache['hidden'].shape}")
    write_manifest(manifest, out_dir / "cache_manifest.json")
    return manifest


def load_variant_cache(out_dir, device, dtype=torch.float32) -> dict:
    from .cache import load_cache

    out = {}
    for split in ["train", "validation", "test"]:
        out[split] = load_cache(Path(out_dir) / f"{split}_hidden.safetensors", device=device, dtype=dtype)
    return out
