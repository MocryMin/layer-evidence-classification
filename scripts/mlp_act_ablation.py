"""EXP-20260729-002 task 05: activation ablation on the raw-feature MLP probe.

Task 04 found the ReLU MLP dies to 100% dead units within epoch 1 (uniform
predictions, loss -> ln(150)). This isolates the activation function: the same
structure ``z = W2 * act(W1 x + b1) + b2`` (both layers bias-carrying,
919r+150 params) with act in {none, relu, leaky, gelu}. ``none`` is a pure
2-layer linear (cannot dead-lock); ``leaky`` (slope 0.01) and ``gelu`` keep a
gradient through the negative interval.

Main matrix: act x r=128 x layers 1-12. Capacity check: L6 x r in {64,128,256}.
Training identical to task 04 / 03a (lr=1e-2, seed 17, 100ep, batch 256,
wd=0.01, grad_clip=1.0, AdamW). Records per-epoch train loss / val acc and the
final negative-interval fraction (h <= 0, a dead-fraction analogue for relu;
informational for leaky/gelu).

Usage:
    python -u scripts/mlp_act_ablation.py
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

from src.config import PROJECT_ROOT  # noqa: E402
from src.diag import _slice_layer  # noqa: E402
from src.heads import ActHead  # noqa: E402
from src.seeding import enable_determinism, seed_all  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "artifacts/EXP-20260729-001/cache"
OUT_DIR = PROJECT_ROOT / "artifacts/EXP-20260729-002/05_act_ablation"
LR = 1e-2
EPOCHS = 100
BATCH = 256
WD = 0.01
GRAD_CLIP = 1.0
SEED = 17
IN_DIM = 768
N_CLASSES = 150
ACTS = ["none", "relu", "leaky", "gelu"]
HIDDEN = [64, 128, 256]
LAYERS = list(range(1, 13))
LN150 = float(np.log(150))  # uniform-prediction loss


def _load(device):
    from src.cache import load_cache
    out = {}
    for split in ["train", "validation"]:
        out[split] = load_cache(CACHE_DIR / f"{split}_hidden.safetensors", device=device, dtype=torch.float32)
    return out


def run_act(train_cache, val_cache, act, r, layer, device):
    """Train ActHead; return per-epoch history + summary (train loss + neg frac tracked)."""
    seed_all(SEED)
    head = ActHead(IN_DIM, N_CLASSES, hidden=r, act=act).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    tx = _slice_layer(train_cache["hidden"], layer)
    vx = _slice_layer(val_cache["hidden"], layer)
    ty = train_cache["labels"].to(device)
    vy = val_cache["labels"].to(device)

    best = {"epoch": -1, "val_acc": -1.0, "val_nll": float("inf")}
    history = []
    for ep in range(1, EPOCHS + 1):
        head.train(); losses = []
        for s in range(0, tx.shape[0], BATCH):
            xb, yb = tx[s:s + BATCH], ty[s:s + BATCH]
            loss = F.cross_entropy(head(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
            opt.step(); losses.append(loss.item())
        head.eval()
        with torch.no_grad():
            logits = head(vx)
            val_nll = F.cross_entropy(logits, vy, reduction="mean").item()
            val_acc = (logits.argmax(1) == vy).float().mean().item()
        tloss = float(np.mean(losses))
        history.append({"epoch": ep, "train_loss": tloss, "val_acc": val_acc, "val_nll": val_nll})
        if (val_acc > best["val_acc"]) or (val_acc == best["val_acc"] and val_nll < best["val_nll"]):
            best = {"epoch": ep, "val_acc": val_acc, "val_nll": val_nll}
    # final train loss + negative-interval fraction on the train set
    with torch.no_grad():
        h = head.fc1(tx)
        neg_frac = float((h <= 0).float().mean())
        final_train_loss = float(F.cross_entropy(head(tx), ty, reduction="mean").item())
    summary = {
        "act": act, "r": r, "layer": layer, "lr": LR,
        "n_params": 768 * r + r + 150 * r + 150,
        "best_val_acc": best["val_acc"], "best_val_nll": best["val_nll"], "best_epoch": best["epoch"],
        "final_val_acc": history[-1]["val_acc"],
        "final_train_loss": final_train_loss, "at_uniform": bool(abs(final_train_loss - LN150) < 1e-3),
        "neg_frac": neg_frac, "n_epochs": len(history),
    }
    return summary, history


def main():
    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mlp_act_ablation] lr={LR} epochs={EPOCHS} batch={BATCH} wd={WD} seed={SEED} device={device}")
    print(f"  structure z=W2*act(W1 x + b1)+b2, params=919r+150 (r=128: 117,782 vs plain 115,350, +2.1%)")
    print(f"  acts={ACTS} ln(150)={LN150:.4f}")
    c = _load(device)

    results = {
        "description": "Activation ablation on the raw-feature MLP probe. Same structure "
                       "z=W2*act(W1 x + b1)+b2 (both biases, 919r+150 params) with act in "
                       "{none, relu, leaky, gelu}; lr=1e-2, seed 17, 100ep, batch 256, wd=0.01, "
                       "grad_clip=1.0, AdamW. Task 04's ReLU MLP had fc2 bias-free (919r); the "
                       "relu row here has b2 for structure parity (150 extra params, 0.13%).",
        "config": {"lr": LR, "epochs": EPOCHS, "batch_size": BATCH, "weight_decay": WD,
                   "grad_clip": GRAD_CLIP, "seed": SEED, "acts": ACTS, "hidden": HIDDEN,
                   "layers": LAYERS, "uniform_loss": LN150, "cache": str(CACHE_DIR)},
        "main": {}, "capacity": {},
    }

    print("\n=== main matrix (r=128, all 12 layers) ===")
    for act in ACTS:
        for layer in LAYERS:
            s, h = run_act(c["train"], c["validation"], act, 128, layer, device)
            results["main"][f"{act}_r128_layer_{layer}"] = s
            flag = "UNIFORM" if s["at_uniform"] else f"neg={s['neg_frac']:.2f}"
            print(f"  {act:5s} L{layer:2d}: best={s['best_val_acc']:.4f}@{s['best_epoch']} "
                  f"final_loss={s['final_train_loss']:.4f} [{flag}]")

    print("\n=== capacity check (L6, r in {64,128,256}) ===")
    for act in ACTS:
        for r in HIDDEN:
            s, _ = run_act(c["train"], c["validation"], act, r, 6, device)
            results["capacity"][f"{act}_r{r}_L6"] = s
            flag = "UNIFORM" if s["at_uniform"] else f"neg={s['neg_frac']:.2f}"
            print(f"  {act:5s} r={r:3d} L6: best={s['best_val_acc']:.4f}@{s['best_epoch']} "
                  f"final_loss={s['final_train_loss']:.4f} [{flag}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "act_ablation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
