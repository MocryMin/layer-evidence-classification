"""EXP-20260729-002 task 04: MLP probe - nonlinear-capacity control.

Question: is the mid-layer collapse a *linearity* limitation or an
optimisation problem? A single-hidden-layer MLP probe with ReLU, parameter
count matched to the plain linear head (919r; r=128 -> 117,632 vs plain
115,350, +2%), trained identically to the 03a plain/LN comparison.

Main run: r in {64, 128, 256} x layers 1-12, lr=1e-2 (the unified lr adopted
in the 6.1 matched control), seed 17, 100 epochs, batch 256, wd=0.01,
grad_clip=1.0, AdamW - identical to 03a plain except the head.
lr check: L6 x r x lr in {1e-3, 1e-2, 1e-1} to verify 1e-2 is reasonable for
the MLP (the "new lr" question).

Usage:
    python -u scripts/mlp_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.diag import _slice_layer  # noqa: E402
from src.heads import MLPHead, PlainHead  # noqa: E402
from src.probe import train_probe  # noqa: E402
from src.seeding import enable_determinism  # noqa: E402

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
HIDDEN = [64, 128, 256]
LAYERS = list(range(1, 13))
LR_CHECK_GRID = [1e-3, 1e-2, 1e-1]
LR_CHECK_LAYER = 6

PARAMS = {r: 768 * r + r + 150 * r for r in HIDDEN}  # 919r
PLAIN_PARAMS = 768 * 150 + 150


def _load(device):
    from src.cache import load_cache
    out = {}
    for split in ["train", "validation"]:
        out[split] = load_cache(CACHE_DIR / f"{split}_hidden.safetensors", device=device, dtype=torch.float32)
    return out


def run_one(train_cache, val_cache, r, layer, lr, device):
    tx = _slice_layer(train_cache["hidden"], layer)
    vx = _slice_layer(val_cache["hidden"], layer)
    ty = train_cache["labels"].to(device)
    vy = val_cache["labels"].to(device)
    head = MLPHead(IN_DIM, N_CLASSES, hidden=r).to(device)
    res = train_probe(
        head, tx, ty, vx, vy, optimizer="adamw", lr=lr, epochs=EPOCHS,
        batch_size=BATCH, weight_decay=WD, grad_clip=GRAD_CLIP, seed=SEED,
        device=device,
    )
    return {
        "r": r, "layer": layer, "lr": lr,
        "n_params": PARAMS[r],
        "best_val_acc": res["best_val_acc"], "best_val_nll": res["best_val_nll"],
        "best_epoch": res["best_epoch"],
        "final_val_acc": res["final_val_acc"], "final_val_nll": res["final_val_nll"],
    }


def _run_section(device, caches, centered=False):
    """Run MLPHead over all (r, layer); return {f'mlp_r{r}{_centered}_layer_{L}': res}."""
    out = {}
    for r in HIDDEN:
        for layer in LAYERS:
            head = MLPHead(IN_DIM, N_CLASSES, hidden=r).to(device)
            tag = f"mlp_r{r}" + ("_centered" if centered else "")
            tx = _slice_layer(caches["train"]["hidden"], layer)
            vx = _slice_layer(caches["validation"]["hidden"], layer)
            ty = caches["train"]["labels"].to(device)
            vy = caches["validation"]["labels"].to(device)
            if centered:
                tx = tx - caches["train_mean"][layer - 1]
                vx = vx - caches["train_mean"][layer - 1]
            res = train_probe(
                head, tx, ty, vx, vy, optimizer="adamw", lr=LR, epochs=EPOCHS,
                batch_size=BATCH, weight_decay=WD, grad_clip=GRAD_CLIP, seed=SEED,
                device=device,
            )
            out[f"{tag}_layer_{layer}"] = {
                "r": r, "layer": layer, "lr": LR, "centered": centered,
                "n_params": PARAMS[r],
                "best_val_acc": res["best_val_acc"], "best_val_nll": res["best_val_nll"],
                "best_epoch": res["best_epoch"],
                "final_val_acc": res["final_val_acc"], "final_val_nll": res["final_val_nll"],
            }
            print(f"  r={r:3d} layer={layer:2d}{' [centered]' if centered else ''}: "
                  f"best_val={res['best_val_acc']:.4f}@{res['best_epoch']} final={res['final_val_acc']:.4f}")
    return out


def _run_plain_centered(device, caches):
    """PlainHead on centered features (fixed cross-sample centering, no leakage)."""
    out = {}
    for layer in LAYERS:
        head = PlainHead(IN_DIM, N_CLASSES).to(device)
        tx = _slice_layer(caches["train"]["hidden"], layer) - caches["train_mean"][layer - 1]
        vx = _slice_layer(caches["validation"]["hidden"], layer) - caches["train_mean"][layer - 1]
        res = train_probe(
            head, tx, caches["train"]["labels"].to(device), vx, caches["validation"]["labels"].to(device),
            optimizer="adamw", lr=LR, epochs=EPOCHS, batch_size=BATCH,
            weight_decay=WD, grad_clip=GRAD_CLIP, seed=SEED, device=device,
        )
        out[f"plain_centered_layer_{layer}"] = {
            "layer": layer, "lr": LR, "centered": True, "n_params": PLAIN_PARAMS,
            "best_val_acc": res["best_val_acc"], "best_val_nll": res["best_val_nll"],
            "best_epoch": res["best_epoch"], "final_val_acc": res["final_val_acc"],
        }
        print(f"  plain centered layer={layer:2d}: best_val={res['best_val_acc']:.4f}@{res['best_epoch']}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--centered-only", action="store_true",
                    help="skip the raw-feature main/lr-check runs (already saved); run only the centered control")
    args = ap.parse_args()

    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mlp_probe] lr={LR} epochs={EPOCHS} batch={BATCH} wd={WD} grad_clip={GRAD_CLIP} "
          f"seed={SEED} device={device}")
    print(f"  params: plain={PLAIN_PARAMS}; MLP { {r: PARAMS[r] for r in HIDDEN} } "
          f"(relative to plain: { {r: f'{PARAMS[r]/PLAIN_PARAMS*100:.0f}%' for r in HIDDEN} })")
    c = _load(device)
    c["train_mean"] = c["train"]["hidden"].mean(dim=0)  # (12, 768) cross-sample mean per dim per layer

    out_path = OUT_DIR / "mlp_probe_lr1e-2.json"
    if args.centered_only and out_path.exists():
        results = json.loads(out_path.read_text())
        print(f"  loaded existing {out_path} (raw main/lr_check kept); updating 'centered' only")
    else:
        results = {
            "description": "MLP probe (nonlinear-capacity control). Head = Linear(768,r)+ReLU+Linear(r,150) "
                           "with 919r params; lr=1e-2 (unified lr, cf. 6.1 matched control), seed 17, 100ep, "
                           "batch 256, wd=0.01, grad_clip=1.0, AdamW - identical to 03a plain except head. "
                           "The centered control subtracts the train cross-sample mean from all splits (fixed "
                           "linear transform; no leakage) to test whether the near-constant CLS component is "
                           "what kills the ReLU MLP (uniform-prediction attractor).",
            "config": {"lr": LR, "epochs": EPOCHS, "batch_size": BATCH, "weight_decay": WD,
                       "grad_clip": GRAD_CLIP, "seed": SEED, "hidden": HIDDEN, "layers": LAYERS,
                       "plain_params": PLAIN_PARAMS, "mlp_params": PARAMS,
                       "cache": str(CACHE_DIR)},
            "main": {}, "lr_check": {}, "centered": {},
        }

    if not args.centered_only:
        print("\n=== main run (lr=1e-2, all 12 layers, r in {64,128,256}) ===")
        results["main"] = _run_section(device, c, centered=False)
        print("\n=== lr check (L6, r in {64,128,256}, lr in {1e-3,1e-2,1e-1}) ===")
        for r in HIDDEN:
            for lr in LR_CHECK_GRID:
                head = MLPHead(IN_DIM, N_CLASSES, hidden=r).to(device)
                tx = _slice_layer(c["train"]["hidden"], LR_CHECK_LAYER)
                vx = _slice_layer(c["validation"]["hidden"], LR_CHECK_LAYER)
                res = train_probe(
                    head, tx, c["train"]["labels"].to(device), vx, c["validation"]["labels"].to(device),
                    optimizer="adamw", lr=lr, epochs=EPOCHS, batch_size=BATCH,
                    weight_decay=WD, grad_clip=GRAD_CLIP, seed=SEED, device=device,
                )
                results["lr_check"][f"L6_r{r}_lr{lr:g}"] = {
                    "r": r, "lr": lr, "best_val_acc": res["best_val_acc"],
                    "best_epoch": res["best_epoch"], "final_val_acc": res["final_val_acc"],
                }
                print(f"  L6 r={r:3d} lr={lr:g}: best_val={res['best_val_acc']:.4f}@{res['best_epoch']}")

    print("\n=== centered control (x - train_mean; plain + MLP, lr=1e-2, all 12 layers) ===")
    results["centered"] = {}
    results["centered"].update(_run_plain_centered(device, c))
    results["centered"].update(_run_section(device, c, centered=True))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved {out_path}")


if __name__ == "__main__":
    main()
