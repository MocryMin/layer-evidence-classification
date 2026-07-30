"""EXP-20260729-002 task 03g: RidgeClassifier alpha grid.

Sweep alpha over [1e-6 .. 100] (powers of 10) on plain CLS for every layer, on a
given frozen-backbone cache. Motivation: the prior ridge control (03f) only tried
alpha=10 (+ a 5-point layer-6 sweep) and found Ridge does NOT rescue frozen-base
mid layers because alpha=10 is mis-scaled vs the tiny X^T X eigenvalues. This
grid extends alpha down to 1e-6 to test whether *any* alpha rescues the mid
layers, and contrasts the frozen base with the fine-tuned backbone (both under
the pure-utterance, no-instruction prompt).

Usage:
    python -u scripts/ridge_alpha_grid.py \
        --cache-dir artifacts/EXP-20260729-002/03b_no_prompt/cache \
        --label base_noprompt \
        --out artifacts/EXP-20260729-002/03g_ridge_alpha_grid/ridge_base_noprompt.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.cache import load_cache  # noqa: E402

ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def _load_split(cache_dir: Path, split: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Load one split's cache; return (hidden (N, L, D) float32, labels (N,))."""
    c = load_cache(cache_dir / f"{split}_hidden.safetensors", device="cpu", dtype=torch.float32)
    hidden = c["hidden"].cpu().numpy().astype(np.float32)  # (N, L, D)
    labels = c["labels"].cpu().numpy().astype(np.int64)
    return hidden, labels


def run_grid(cache_dir: Path, layers: list[int], alphas: list[float]) -> dict:
    from sklearn.linear_model import RidgeClassifier

    device = torch.device("cpu")
    tr_h, tr_y = _load_split(cache_dir, "train", device)
    va_h, va_y = _load_split(cache_dir, "validation", device)
    te_h, te_y = _load_split(cache_dir, "test", device)
    n_classes = int(tr_y.max()) + 1

    out = {
        "cache_dir": str(cache_dir),
        "alphas": alphas,
        "layers": layers,
        "n_train": int(tr_h.shape[0]),
        "n_val": int(va_h.shape[0]),
        "n_test": int(te_h.shape[0]),
        "n_classes": n_classes,
        "per_layer": {},
    }

    for layer in layers:
        Xtr = tr_h[:, layer - 1, :]
        Xva = va_h[:, layer - 1, :]
        Xte = te_h[:, layer - 1, :]
        row = {"layer": layer, "by_alpha": {}}
        for alpha in alphas:
            clf = RidgeClassifier(alpha=alpha, fit_intercept=True, solver="svd")
            clf.fit(Xtr, tr_y)
            val_acc = float((clf.predict(Xva) == va_y).mean())
            test_acc = float((clf.predict(Xte) == te_y).mean())
            row["by_alpha"][f"{alpha:g}"] = {
                "alpha": alpha, "val_acc": val_acc, "test_acc": test_acc,
            }
        # best alpha by val acc (tie-break: lower alpha = less regularisation)
        best = max(row["by_alpha"].values(), key=lambda r: (r["val_acc"], -r["alpha"]))
        row["best_alpha"] = best["alpha"]
        row["best_val_acc"] = best["val_acc"]
        row["best_test_acc"] = best["test_acc"]
        out["per_layer"][str(layer)] = row
        curve = ", ".join(f"a={a:g}:{row['by_alpha'][f'{a:g}']['val_acc']:.3f}" for a in alphas)
        print(f"  layer {layer:2d}: best alpha={best['alpha']:g} "
              f"val={best['val_acc']:.4f} test={best['test_acc']:.4f} | {curve}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--label", required=True, type=str)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(1, 13)))
    args = ap.parse_args()

    print(f"[ridge_alpha_grid] label={args.label} cache={args.cache_dir}")
    print(f"  alphas={ALPHAS}")
    res = run_grid(args.cache_dir, args.layers, ALPHAS)
    res["label"] = args.label
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
