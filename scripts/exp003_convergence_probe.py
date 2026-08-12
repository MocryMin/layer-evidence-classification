"""EXP-20260810-003 convergence probe: does the gradient probe converge given more compute?

EXP-003's main run used max_ep=1000 (parallel cap with convergence); centered
plain was non-converged in 110/120 runs and LN in 82/120. This probe answers:
given a larger epoch budget, does the run converge, and at what epoch?

Design: one long run per (family, layer) at seed 17 with early stopping
DISABLED (patience > max_epochs), recording the full per-epoch validation
history. A single long run is equivalent to any shorter max_ep budget M, since
the early-stopping rule and checkpoint selection only read the history prefix
[1..M]. Offline analysis then derives:

- best_val_acc under each budget M in {1000, 2000, 5000, 10000, 20000}
  (max val acc over the first M epochs);
- the simulated early-stop epoch (min_ep=100, patience=100, min_delta=1e-4 —
  the exact EXP-003 rule) within 20000 epochs;
- the first epoch reaching val_acc >= 0.7 / 0.8 / 0.9;
- one-time test accuracy of the best checkpoint (convergence probe reports
  test once per run for reference).

Only seed 17 is used: EXP-003 showed cross-seed std < 0.0015 for every layer
of both gradient families, so convergence behaviour is seed-representative.

Usage:
    python -u scripts/exp003_convergence_probe.py
    python -u scripts/exp003_convergence_probe.py --max-epochs 10000 --layers 6 11
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

from src.config import PROJECT_ROOT  # noqa: E402
from src.exp003 import compute_train_means, load_caches  # noqa: E402
from src.heads import build_head  # noqa: E402
from src.probe import train_probe_fullbatch_es  # noqa: E402
from src.seeding import enable_determinism  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "artifacts/EXP-20260729-001/cache"
OUT_DIR = PROJECT_ROOT / "artifacts/EXP-20260810-003/convergence_probe"
SEED = 17
N_CLASSES = 150
LAYERS = list(range(1, 13))
FAMILIES = ["centered_plain", "ln_plain"]
BUDGETS = [1000, 2000, 5000, 10000, 20000]
THRESHOLDS = [0.7, 0.8, 0.9]
# EXP-003 early-stopping rule (plan §3.1)
MIN_EPOCHS = 100
PATIENCE = 100
MIN_DELTA = 1e-4


def simulate_early_stop(val_acc: np.ndarray) -> int | None:
    """Simulate the EXP-003 early-stopping rule on a val-acc history.

    Rule: after min_epochs, stop when val_acc has not improved by >= min_delta
    for `patience` consecutive epochs. Returns the stop epoch (1-based), or
    None if it never triggers within the history length.
    """
    best = -1.0
    since_improve = 0
    for i, a in enumerate(val_acc, start=1):
        if a > best + MIN_DELTA:
            best = a
            since_improve = 0
        else:
            since_improve += 1
        if i >= MIN_EPOCHS and since_improve >= PATIENCE:
            return i
    return None


def first_epoch_at(val_acc: np.ndarray, threshold: float) -> int | None:
    """First epoch (1-based) with val_acc >= threshold, or None."""
    idx = np.argmax(val_acc >= threshold)
    if val_acc[idx] >= threshold:
        return int(idx) + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=20000)
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[conv-probe] max_epochs={args.max_epochs} layers={args.layers} "
          f"families={args.families} seed={args.seed} device={device}")

    caches = load_caches(CACHE_DIR, device)
    tr_h = caches["train"]["hidden"]
    va_h = caches["validation"]["hidden"]
    tr_y = caches["train"]["labels"].to(device)
    va_y = caches["validation"]["labels"].to(device)
    te_h = caches["test"]["hidden"]
    te_y = caches["test"]["labels"].to(device)
    train_means = compute_train_means(tr_h)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "probe": "convergence (max_ep sweep via single long run + history replay)",
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "budgets": BUDGETS,
        "thresholds": THRESHOLDS,
        "early_stop_rule": {"min_epochs": MIN_EPOCHS, "patience": PATIENCE,
                            "min_delta": MIN_DELTA, "monitor": "val_acc"},
        "note": "early stopping disabled in training (patience > max_epochs); "
                "budget results and stop epochs are replayed from the full history",
        "per_family": {},
    }

    for fam in args.families:
        per_layer = {}
        histories = {}
        for layer in args.layers:
            tx = tr_h[:, layer - 1, :].contiguous()
            vx = va_h[:, layer - 1, :].contiguous()
            tex = te_h[:, layer - 1, :].contiguous()
            if fam == "centered_plain":
                mu = train_means[layer - 1]
                tx, vx, tex = tx - mu, vx - mu, tex - mu

            head = build_head("plain" if fam == "centered_plain" else "ln",
                              in_dim=768, n_classes=N_CLASSES)
            res = train_probe_fullbatch_es(
                head=head, train_x=tx, train_y=tr_y, val_x=vx, val_y=va_y,
                lr=1e-2, weight_decay=0.0, grad_clip=0.0,
                min_epochs=MIN_EPOCHS, max_epochs=args.max_epochs,
                patience=args.max_epochs + 1,  # effectively disabled
                min_delta=MIN_DELTA, seed=args.seed, device=device,
            )
            val_acc = np.array([h["val_acc"] for h in res["history"]], dtype=float)

            # one-time test eval of the best checkpoint
            head.load_state_dict(res["best_state"])
            head.to(device).eval()
            with torch.no_grad():
                logits = torch.cat([head(tex[s:s + 1024]) for s in range(0, tex.shape[0], 1024)], dim=0)
            test_acc = float((logits.argmax(1) == te_y).float().mean().item())

            budgets = {str(M): float(val_acc[:M].max()) for M in BUDGETS}
            stop_ep = simulate_early_stop(val_acc)
            per_layer[str(layer)] = {
                "budgets_best_val_acc": budgets,
                "simulated_early_stop_epoch": stop_ep,
                "converged_within_max_epochs": stop_ep is not None,
                "first_epoch_at": {
                    f"{t:g}": first_epoch_at(val_acc, t) for t in THRESHOLDS
                },
                "best_val_acc": res["best_val_acc"],
                "best_epoch": res["best_epoch"],
                "final_val_acc": res["final_val_acc"],
                "test_acc_at_best": test_acc,
                "n_epochs_run": len(res["history"]),
            }
            histories[str(layer)] = val_acc
            print(f"  [{fam}] L{layer:2d}: stop={stop_ep} best={res['best_val_acc']:.4f}"
                  f"@{res['best_epoch']} test={test_acc:.4f}"
                  + ("" if stop_ep is not None else " [NOT CONVERGED in %d ep]" % args.max_epochs))

        np.savez(OUT_DIR / f"val_history_{fam}_seed{args.seed}.npz", **histories)
        summary["per_family"][fam] = per_layer

    # cross-reference EXP-003 1000ep results for the same seed/layer
    exp003_results = json.load(open(PROJECT_ROOT / "artifacts/EXP-20260810-003/results.json"))
    summary["exp003_reference"] = {}
    for fam in args.families:
        d = exp003_results[fam]["per_seed"][str(args.seed)]
        summary["exp003_reference"][fam] = {
            str(l): {
                "best_val_acc_1000ep": d["per_layer"][str(l)]["best_val_acc"],
                "best_epoch_1000ep": d["per_layer"][str(l)]["best_epoch"],
                "converged_1000ep": d["per_layer"][str(l)]["converged"],
            } for l in args.layers
        }

    with open(OUT_DIR / "convergence_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[conv-probe] saved {OUT_DIR / 'convergence_summary.json'}")


if __name__ == "__main__":
    main()
