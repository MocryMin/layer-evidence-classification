"""EXP-20260729-002 task 3a follow-up: transformed-feature stats per head.

Task 3a's table reports the *raw* inter-sample std (one column shared by all
heads), but each head is itself a feature transform: LN normalises+affines,
norm_only normalises, affine_only scales+shifts. This re-trains the four heads
(identical config: AdamW lr=1e-2, seed 17, 100ep, batch 256, wd=0.01,
grad_clip=1.0) and computes, with the *trained* transformation parameters, the
per-layer inter-sample std (plus participation ratio and top-1 var frac) of the
transformed train features. This shows how each head reshapes the collapsed
geometry - the mechanism behind "why LN works, norm_only / affine_only don't".

plain = identity (transformed stats == raw). norm_only has no learnable params
(fixed normalisation). ln / affine_only use their trained gamma/beta.

Usage:
    python -u scripts/ln_ablation_transformed_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.diag import _slice_layer, feature_stats_layer  # noqa: E402
from src.heads import AffineOnlyHead, LNHead, NormOnlyHead, PlainHead  # noqa: E402
from src.seeding import enable_determinism, seed_all  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "artifacts/EXP-20260729-001/cache"
OUT_DIR = PROJECT_ROOT / "artifacts/EXP-20260729-002/03a_ln_ablation"
LR = 1e-2
EPOCHS = 100
BATCH = 256
WD = 0.01
GRAD_CLIP = 1.0
SEED = 17
IN_DIM = 768
N_CLASSES = 150
LAYERS = list(range(1, 13))
HEADS = {"plain": PlainHead, "ln": LNHead, "norm_only": NormOnlyHead, "affine_only": AffineOnlyHead}


def _load(device):
    from src.cache import load_cache
    out = {}
    for split in ["train", "validation"]:
        out[split] = load_cache(CACHE_DIR / f"{split}_hidden.safetensors", device=device, dtype=torch.float32)
    return out


def train_head(head, tx, ty, vx, vy, device):
    """Identical to src/probe.train_probe (AdamW branch): seed 17, same loader generator."""
    from torch.utils.data import DataLoader, TensorDataset
    seed_all(SEED)
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    ds = TensorDataset(tx, ty)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=g)
    best = {"epoch": -1, "val_acc": -1.0, "val_nll": float("inf")}
    for epoch in range(1, EPOCHS + 1):
        head.train()
        for xb, yb in loader:
            loss = F.cross_entropy(head(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
            opt.step()
        head.eval()
        with torch.no_grad():
            logits = head(vx)
            val_nll = F.cross_entropy(logits, vy, reduction="mean").item()
            val_acc = (logits.argmax(1) == vy).float().mean().item()
        if (val_acc > best["val_acc"]) or (val_acc == best["val_acc"] and val_nll < best["val_nll"]):
            best = {"epoch": epoch, "val_acc": val_acc, "val_nll": val_nll}
    return best


@torch.no_grad()
def transform_features(head_type, head, x):
    """Apply the head's feature transform (before the classifier) to x."""
    if head_type == "plain":
        return x
    if head_type in ("ln", "norm_only"):
        return head.ln(x)
    if head_type == "affine_only":
        return head.gamma * x + head.beta
    raise ValueError(head_type)


def main():
    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ln_ablation_transformed_stats] lr={LR} seed={SEED} epochs={EPOCHS} device={device}")
    c = _load(device)
    tr_h, va_h = c["train"]["hidden"], c["validation"]["hidden"]
    tr_y, va_y = c["train"]["labels"].to(device), c["validation"]["labels"].to(device)
    labels_np = c["train"]["labels"].numpy()

    results = {"config": {"lr": LR, "epochs": EPOCHS, "batch_size": BATCH, "weight_decay": WD,
                          "grad_clip": GRAD_CLIP, "seed": SEED, "cache": str(CACHE_DIR)},
               "description": "Per-head transformed-feature stats (train CLS, frozen base, "
                              "instruction prompt). plain = raw (identity). Trained with the "
                              "same config as 03a; stats use the trained gamma/beta for ln and "
                              "affine_only, the fixed normalisation for norm_only.",
               "per_head": {}}
    for hname, hcls in HEADS.items():
        results["per_head"][hname] = {}
        print(f"\n=== {hname} ===")
        print(f"{'L':>3} | {'best_val_acc':>12} | {'inter_std':>10} | {'part_ratio':>10} | {'top1':>6}")
        for layer in LAYERS:
            tx = _slice_layer(tr_h, layer); vx = _slice_layer(va_h, layer)
            head = hcls(IN_DIM, N_CLASSES).to(device)
            best = train_head(head, tx, tr_y, vx, va_y, device)
            head.eval()
            tf = transform_features(hname, head, tx).cpu().numpy()
            st = feature_stats_layer(tf, labels_np, N_CLASSES)
            entry = {"layer": layer, "best_val_acc": best["val_acc"], "best_epoch": best["epoch"],
                     "inter_sample_std": st["inter_sample_std_mean"],
                     "participation_ratio": st["participation_ratio"],
                     "top1_var_frac": st["top1_var_frac"],
                     "class_signal_ratio": st["class_signal_ratio"],
                     "mean_norm": st["mean_norm"]}
            results["per_head"][hname][str(layer)] = entry
            print(f"{layer:>3} | {best['val_acc']:>12.4f} | {st['inter_sample_std_mean']:>10.6f} | "
                  f"{st['participation_ratio']:>10.3f} | {st['top1_var_frac']:>6.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "transformed_feature_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
