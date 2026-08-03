"""EXP-20260729-002 task 04 follow-up: dead-ReLU diagnosis.

The raw-feature MLP collapses to uniform predictions (loss = ln(150)). This
diagnostic checks whether pervasive dead ReLUs accompany/explain the collapse:

A. Per-dimension feature stats (frozen-base CLS, with-prompt, train split):
   mu_i = mean over samples, delta_i = std over samples. Report the share of
   dims with mu < 0, mu + delta < 0 (almost-always-negative dims), mu - delta > 0
   (almost-always-positive), and the shape of the mu distribution. This tests
   whether the pre-activation W1 x + b1 is biased negative by the feature
   geometry (b1 = 0 at init, W1 ~ N(0, 1/768)).

B. Initial dead fraction of the first ReLU layer per layer (r=128, seed 17):
   frac(h <= 0) with h = W1 x + b1 before any training.

C. Training trajectory on L6 / r=128: dead fraction, train loss, val acc per
   epoch - does the dead fraction rise monotonically to 100% as the loss
   converges to ln(150) (uniform attractor)?

Usage:
    python -u scripts/mlp_dead_relu.py
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
from src.heads import MLPHead  # noqa: E402
from src.seeding import enable_determinism, seed_all  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "artifacts/EXP-20260729-001/cache"
OUT_DIR = PROJECT_ROOT / "artifacts/EXP-20260729-002/04_mlp_probe"
LR = 1e-2
EPOCHS = 100
BATCH = 256
WD = 0.01
GRAD_CLIP = 1.0
SEED = 17
IN_DIM = 768
N_CLASSES = 150
TRACK_LAYERS = [6, 12]
TRACK_R = 128


def _load(device):
    from src.cache import load_cache
    out = {}
    for split in ["train", "validation"]:
        out[split] = load_cache(CACHE_DIR / f"{split}_hidden.safetensors", device=device, dtype=torch.float32)
    return out


def feature_dim_stats(hidden_layer: torch.Tensor) -> dict:
    """Per-dim mu/delta of a layer's CLS; report negative-dim shares."""
    x = hidden_layer.float()
    N, D = x.shape
    mu = x.mean(dim=0)          # (D,)
    delta = x.std(dim=0)        # (D,)
    hist = torch.histc(mu, bins=10, min=mu.min().item(), max=mu.max().item()).tolist()
    return {
        "n_dims": int(D),
        "mu_min": float(mu.min()), "mu_max": float(mu.max()), "mu_mean": float(mu.mean()),
        "pct_mu_neg": float((mu < 0).float().mean()),
        "pct_mu_plus_delta_neg": float(((mu + delta) < 0).float().mean()),   # almost always < 0
        "pct_mu_minus_delta_pos": float(((mu - delta) > 0).float().mean()),  # almost always > 0
        "pct_mu_abs_gt_delta": float((mu.abs() > delta).float().mean()),     # |mu| > delta => sign-stable dim
        "delta_mean": float(delta.mean()),
        "mu_histogram_10bins": hist,
    }


@torch.no_grad()
def initial_dead_fraction(head: torch.nn.Module, x: torch.Tensor) -> float:
    """Fraction of first-layer units with pre-activation <= 0 (dead ReLU)."""
    h = head.fc1(x)
    return float((h <= 0).float().mean())


def main():
    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mlp_dead_relu] device={device} track_layers={TRACK_LAYERS} track_r={TRACK_R}")
    c = _load(device)
    tr_h = c["train"]["hidden"]
    va_h = c["validation"]["hidden"]
    tr_y = c["train"]["labels"].to(device)
    va_y = c["validation"]["labels"].to(device)

    results = {"description": "dead-ReLU diagnosis for the raw-feature MLP (uniform-prediction collapse)",
               "config": {"lr": LR, "epochs": EPOCHS, "batch_size": BATCH, "weight_decay": WD,
                          "grad_clip": GRAD_CLIP, "seed": SEED, "track_layers": TRACK_LAYERS,
                          "track_r": TRACK_R, "cache": str(CACHE_DIR)},
               "feature_dim_stats": {}, "initial_dead": {}, "trajectory": {}}

    # A. per-dim feature stats for all layers
    print("\n=== A. per-dim feature stats (train CLS) ===")
    hdr = f"{'L':>3} | {'pct mu<0':>9} | {'pct mu+d<0':>11} | {'pct mu-d>0':>11} | {'pct |mu|>d':>10} | {'mu range':>16} | {'delta_mean':>10}"
    print(hdr); print("-" * len(hdr))
    for L in range(1, 13):
        st = feature_dim_stats(_slice_layer(tr_h, L))
        results["feature_dim_stats"][str(L)] = st
        print(f"{L:>3} | {st['pct_mu_neg']:>9.3f} | {st['pct_mu_plus_delta_neg']:>11.3f} | "
              f"{st['pct_mu_minus_delta_pos']:>11.3f} | {st['pct_mu_abs_gt_delta']:>10.3f} | "
              f"{st['mu_min']:.4f}..{st['mu_max']:.4f} | {st['delta_mean']:>10.6f}")

    # B. initial dead fraction per layer
    print("\n=== B. initial dead-ReLU fraction (r=128, before training) ===")
    for L in range(1, 13):
        seed_all(SEED)
        head = MLPHead(IN_DIM, N_CLASSES, hidden=TRACK_R).to(device)
        x = _slice_layer(tr_h, L)
        frac = initial_dead_fraction(head, x)
        results["initial_dead"][str(L)] = frac
        print(f"  L{L:2d}: dead={frac:.3f}")

    # C. trajectory on L6 and L12
    print(f"\n=== C. dead-fraction trajectory (r={TRACK_R}, raw features) ===")
    for L in TRACK_LAYERS:
        seed_all(SEED)
        head = MLPHead(IN_DIM, N_CLASSES, hidden=TRACK_R).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
        tx = _slice_layer(tr_h, L); vx = _slice_layer(va_h, L)
        traj = []
        for ep in range(1, EPOCHS + 1):
            head.train(); losses = []
            for s in range(0, tx.shape[0], BATCH):
                xb, yb = tx[s:s+BATCH], tr_y[s:s+BATCH]
                loss = F.cross_entropy(head(xb), yb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
                opt.step(); losses.append(loss.item())
            head.eval()
            with torch.no_grad():
                h = head.fc1(tx)
                dead = float((h <= 0).float().mean())
                acc = (head(vx).argmax(1) == va_y).float().mean().item()
            tloss = float(np.mean(losses))
            traj.append({"epoch": ep, "train_loss": tloss, "dead_frac": dead, "val_acc": acc})
            if ep in (1, 2, 3, 5, 10, 20, 50, 100):
                print(f"  L{L} ep{ep:3d}: loss={tloss:.4f} dead={dead:.3f} val_acc={acc:.4f}")
        results["trajectory"][str(L)] = traj

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "dead_relu_diagnosis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
