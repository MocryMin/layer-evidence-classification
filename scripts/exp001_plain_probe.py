"""EXP-20260729-001: post-hoc run of the paused plain-probe mainline.

The lr smoke test (``smoke_lr_results.json``) recorded the collapse on layers
{1,6,12} only. The full mainline (12 layers x 10 seeds, plain linear probe at
the smoke-selected lr=1e-3) was never run because EXP-001 was paused. This script
runs it now so the experiment report has the full per-layer collapse curve
(validation accuracy, test accuracy at the best-val checkpoint, per-epoch val
history) across all 12 layers and all 10 seeds.

It mirrors ``src/head.py::train_head`` exactly (LinearHead, AdamW, seed_all,
same shuffle generator, grad_clip, best-by-val-acc tie-break lower nll) so the
seed-17 layers {1,6,12} reproduce ``smoke_lr_results.json``; the only addition
is in-memory best-state tracking + a one-time test eval at the best-val epoch.

Config: resolved ``run_config.yaml`` (lr=1e-3, wd=0.01, epochs=100, batch=256,
grad_clip=1.0, head=linear_with_bias, frozen backbone).

Usage:
    python -u scripts/exp001_plain_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.head import LinearHead, evaluate  # noqa: E402
from src.seeding import enable_determinism, seed_all  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "artifacts/EXP-20260729-001/cache"
OUT_DIR = PROJECT_ROOT / "artifacts/EXP-20260729-001/plain_probe_mainline"
LR = 1e-3
EPOCHS = 100
BATCH = 256
WD = 0.01
GRAD_CLIP = 1.0
IN_DIM = 768
N_CLASSES = 150
SEEDS = [17, 29, 43, 59, 71, 89, 101, 127, 149, 173]
LAYERS = list(range(1, 13))


def _load(device):
    from src.cache import load_cache
    out = {}
    for split in ["train", "validation", "test"]:
        out[split] = load_cache(CACHE_DIR / f"{split}_hidden.safetensors", device=device, dtype=torch.float32)
    return out


def train_one(tx, ty, vx, vy, tex, tey, seed, device):
    """Mirror head.train_head; add best-state tracking + test eval at best epoch."""
    seed_all(seed)
    head = LinearHead(IN_DIM, N_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    ds = TensorDataset(tx, ty)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=g)
    best = {"epoch": -1, "val_acc": -1.0, "val_nll": float("inf")}
    best_state = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        head.train()
        for xb, yb in loader:
            loss = F.cross_entropy(head(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
            opt.step()
        _, val_acc, val_nll = evaluate(head, vx, vy)
        history.append({"epoch": epoch, "validation_accuracy": val_acc, "validation_nll": val_nll})
        improved = (val_acc > best["val_acc"]) or (val_acc == best["val_acc"] and val_nll < best["val_nll"])
        if improved:
            best = {"epoch": epoch, "val_acc": val_acc, "val_nll": val_nll}
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    _, test_acc, test_nll = evaluate(head, tex, tey)
    return {
        "best_epoch": best["epoch"], "best_val_acc": best["val_acc"], "best_val_nll": best["val_nll"],
        "test_acc": test_acc, "test_nll": test_nll, "val_history": history,
    }


def main():
    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp001_plain_probe] lr={LR} epochs={EPOCHS} batch={BATCH} wd={WD} "
          f"grad_clip={GRAD_CLIP} device={device} seeds={SEEDS}")
    c = _load(device)
    tr_h, va_h, te_h = c["train"]["hidden"], c["validation"]["hidden"], c["test"]["hidden"]
    tr_y, va_y, te_y = (c[s]["labels"].to(device) for s in ["train", "validation", "test"])

    # smoke-test consistency check (seed 17, layers 1/6/12 must match smoke_lr_results.json)
    smoke_ref = {1: 0.06666666269302368, 6: 0.028333332389593124, 12: 0.6069999933242798}

    results = {"config": {"lr": LR, "epochs": EPOCHS, "batch_size": BATCH, "weight_decay": WD,
                          "grad_clip": GRAD_CLIP, "head": "linear_with_bias (Xavier, zero bias)",
                          "optimizer": "AdamW", "seeds": SEEDS, "layers": LAYERS,
                          "cache": str(CACHE_DIR), "note": "post-hoc run of the paused EXP-001 mainline"},
               "per_layer_seed": {}, "per_layer_summary": {}}
    for layer in LAYERS:
        tx = tr_h[:, layer - 1, :].contiguous()
        vx = va_h[:, layer - 1, :].contiguous()
        tex = te_h[:, layer - 1, :].contiguous()
        per_seed = {}
        for seed in SEEDS:
            r = train_one(tx, tr_y, vx, va_y, tex, te_y, seed, device)
            per_seed[str(seed)] = {k: v for k, v in r.items() if k != "val_history"}
            per_seed[str(seed)]["val_history"] = r["val_history"]
            tag = ""
            if seed == 17 and layer in smoke_ref:
                ok = abs(r["best_val_acc"] - smoke_ref[layer]) < 1e-4
                tag = f"  [smoke-check {'OK' if ok else 'MISMATCH'}: {r['best_val_acc']:.4f} vs {smoke_ref[layer]:.4f}]"
            print(f"  layer {layer:2d} seed {seed:3d}: best_val={r['best_val_acc']:.4f}@{r['best_epoch']} "
                  f"test={r['test_acc']:.4f}{tag}")
        results["per_layer_seed"][str(layer)] = per_seed
        va = [per_seed[s]["best_val_acc"] for s in map(str, SEEDS)]
        ta = [per_seed[s]["test_acc"] for s in map(str, SEEDS)]
        results["per_layer_summary"][str(layer)] = {
            "layer": layer,
            "val_acc_mean": float(np.mean(va)), "val_acc_std": float(np.std(va)),
            "val_acc_min": float(np.min(va)), "val_acc_max": float(np.max(va)),
            "test_acc_mean": float(np.mean(ta)), "test_acc_std": float(np.std(ta)),
            "test_acc_min": float(np.min(ta)), "test_acc_max": float(np.max(ta)),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved {OUT_DIR/'results.json'}")
    print("\n=== per-layer summary (mean over 10 seeds) ===")
    print(f"{'layer':>5} | {'val_acc':>16} | {'test_acc':>16}")
    for layer in LAYERS:
        s = results["per_layer_summary"][str(layer)]
        print(f"{layer:>5} | {s['val_acc_mean']:.4f} ± {s['val_acc_std']:.4f} | {s['test_acc_mean']:.4f} ± {s['test_acc_std']:.4f}")


if __name__ == "__main__":
    main()
