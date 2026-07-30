"""EXP-20260729-002 task 03h: is AdamW's mid-layer collapse fundamental?

Ablation on layer 6, seed 17, frozen base + instruction-prompt cache (the
original collapse scenario: plain AdamW lr=1e-2 wd=0.01 batch=256 100ep ->
0.027 val acc). Tests whether the failure was a fundamental first-order
incapability or an artefact of (a) the 100-epoch cap, (b) weight_decay=0.01,
(c) mini-batch noise, (d) init.

Conditions (lr=1e-2, grad_clip=0 to remove it as a confound, CE loss):
  Group A - OLS init (Ridge alpha=1e-6), batch=256:
    A1  Adam   wd=0
    A2  AdamW  wd=1e-4
    A3  AdamW  wd=1e-2        (original wd)  -- does acc drop from 0.917?
  Group A-full - OLS init, full-batch (is the OLS point stable under CE itself?):
    A1f AdamW  wd=0
    A3f AdamW  wd=1e-2
  Group B - Xavier init:
    B1  AdamW  wd=0      batch=256
    B2  AdamW  wd=0      batch=full
    B3  AdamW  wd=0.01   batch=256           (original config, converged)
    B4  AdamW  wd=0.01   batch=full          -- mini-batch noise test

Training: CE, train to convergence - stop when relative train-loss change over
a window of k=10 epochs falls below r=1e-4, with min_epochs=20 and
max_epochs=20000. Records per-epoch train_loss / val_acc / val_nll; reports
epoch-0 (init) val acc, best val acc, final val acc, converged epoch.

Usage:
    python -u scripts/adam_ablation.py
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

from src.cache import load_cache  # noqa: E402
from src.heads import PlainHead  # noqa: E402
from src.seeding import enable_determinism, seed_all  # noqa: E402

CACHE_DIR = Path("artifacts/EXP-20260729-001/cache")
LAYER = 6
SEED = 17
LR = 1e-2
N_CLASSES = 150
IN_DIM = 768
K = 10            # convergence window (epochs)
R = 1e-4          # relative train-loss change threshold
MIN_EPOCHS = 20
MAX_EPOCHS = 20000
RIDGE_ALPHA = 1e-6


def _load(device):
    tr = load_cache(CACHE_DIR / "train_hidden.safetensors", device=device, dtype=torch.float32)
    va = load_cache(CACHE_DIR / "validation_hidden.safetensors", device=device, dtype=torch.float32)
    te = load_cache(CACHE_DIR / "test_hidden.safetensors", device=device, dtype=torch.float32)
    def sl(c, l): return c["hidden"][:, l - 1, :].contiguous()
    return (sl(tr, LAYER), tr["labels"].to(device),
            sl(va, LAYER), va["labels"].to(device),
            sl(te, LAYER), te["labels"].to(device))


def ols_init(xtr, ytr):
    """Ridge(alpha=1e-6, svd) -> (coef (C,D), intercept (C,))."""
    from sklearn.linear_model import RidgeClassifier
    clf = RidgeClassifier(alpha=RIDGE_ALPHA, fit_intercept=True, solver="svd")
    clf.fit(xtr.cpu().numpy().astype(np.float32), ytr.cpu().numpy().astype(np.int64))
    return clf


@torch.no_grad()
def evaluate(head, x, y):
    head.eval()
    logits = head(x)
    loss = F.cross_entropy(logits, y, reduction="mean").item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return acc, loss


def make_head(init: str, ols_coef, ols_intercept, device):
    head = PlainHead(IN_DIM, N_CLASSES).to(device)
    if init == "xavier":
        pass  # PlainHead already xavier-inits
    elif init == "ols":
        with torch.no_grad():
            head.fc.weight.copy_(torch.as_tensor(ols_coef, dtype=torch.float32, device=device))
            head.fc.bias.copy_(torch.as_tensor(ols_intercept, dtype=torch.float32, device=device))
    else:
        raise ValueError(init)
    return head


def make_opt(name: str, params, wd: float):
    if name == "adam":
        return torch.optim.Adam(params, lr=LR, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=LR, weight_decay=wd)
    raise ValueError(name)


def train_condition(name, init, opt_name, wd, batch, xtr, ytr, xva, yva, xte, yte,
                    ols_coef, ols_intercept, device):
    seed_all(SEED)
    head = make_head(init, ols_coef, ols_intercept, device)
    opt = make_opt(opt_name, head.parameters(), wd)
    full = batch >= xtr.shape[0]
    if not full:
        ds = TensorDataset(xtr, ytr)
        g = torch.Generator().manual_seed(SEED)
        loader = DataLoader(ds, batch_size=batch, shuffle=True, generator=g)
    else:
        loader = None

    acc0, nll0 = evaluate(head, xva, yva)
    history = []
    stopped = "max_epochs"
    for epoch in range(1, MAX_EPOCHS + 1):
        head.train()
        losses = []
        if loader is None:
            opt.zero_grad()
            loss = F.cross_entropy(head(xtr), ytr)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        else:
            for xb, yb in loader:
                opt.zero_grad()
                loss = F.cross_entropy(head(xb), yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        train_loss = float(np.mean(losses))
        acc, nll = evaluate(head, xva, yva)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": acc, "val_nll": nll})
        if epoch >= max(MIN_EPOCHS, K + 1):
            prev = history[epoch - 1 - K]["train_loss"]
            rel = abs(train_loss - prev) / max(abs(prev), 1e-12)
            if rel < R:
                stopped = f"converged@{epoch} (rel={rel:.2e})"
                break
    final_acc, final_nll = evaluate(head, xva, yva)
    test_acc, test_nll = evaluate(head, xte, yte)
    best = max(history, key=lambda h: (h["val_acc"], -h["val_nll"]))
    summary = {
        "condition": name, "init": init, "optimizer": opt_name, "weight_decay": wd,
        "batch": "full" if full else batch, "lr": LR,
        "init_val_acc": acc0, "init_val_nll": nll0,
        "best_val_acc": best["val_acc"], "best_val_nll": best["val_nll"], "best_epoch": best["epoch"],
        "final_val_acc": final_acc, "final_val_nll": final_nll,
        "final_test_acc": test_acc, "final_test_nll": test_nll,
        "stopped": stopped, "n_epochs": len(history),
    }
    flag = "DROP" if (init == "ols" and final_acc < acc0 - 0.01) else ("RESCUE" if final_acc > 0.5 else "fail")
    print(f"  {name:30s} init={init:6s} {opt_name:5s} wd={wd:<6g} batch={'full' if full else batch:>4} "
          f"| init={acc0:.4f} best={best['val_acc']:.4f}@{best['epoch']} final={final_acc:.4f} "
          f"test={test_acc:.4f} ep={len(history)} [{flag}]")
    return {"summary": summary, "history": history}


CONDITIONS = [
    ("A1_ols_adam_wd0_b256",      "ols",    "adam",  0.0,   256),
    ("A2_ols_adamw_wd1e-4_b256",  "ols",    "adamw", 1e-4,  256),
    ("A3_ols_adamw_wd1e-2_b256",  "ols",    "adamw", 1e-2,  256),
    ("A1f_ols_adamw_wd0_bfull",   "ols",    "adamw", 0.0,   15000),
    ("A3f_ols_adamw_wd1e-2_bfull","ols",    "adamw", 1e-2,  15000),
    ("B1_xav_adamw_wd0_b256",     "xavier", "adamw", 0.0,   256),
    ("B2_xav_adamw_wd0_bfull",    "xavier", "adamw", 0.0,   15000),
    ("B3_xav_adamw_wd1e-2_b256",  "xavier", "adamw", 1e-2,  256),
    ("B4_xav_adamw_wd1e-2_bfull", "xavier", "adamw", 1e-2,  15000),
]


def main():
    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[adam_ablation] layer={LAYER} seed={SEED} lr={LR} device={device}")
    print(f"  cache={CACHE_DIR} (frozen base + instruction prompt)")
    xtr, ytr, xva, yva, xte, yte = _load(device)
    print(f"  train={xtr.shape} val={xva.shape} test={xte.shape}")
    ols = ols_init(xtr, ytr)
    # sanity: OLS val acc should match the 0.917 reference
    ols_val = (ols.predict(xva.cpu().numpy()).astype(np.int64) == yva.cpu().numpy()).mean()
    ols_test = (ols.predict(xte.cpu().numpy()).astype(np.int64) == yte.cpu().numpy()).mean()
    print(f"  OLS(Ridge a={RIDGE_ALPHA}) reference: val={ols_val:.4f} test={ols_test:.4f}")

    results = {
        "layer": LAYER, "seed": SEED, "lr": LR, "k": K, "r": R,
        "min_epochs": MIN_EPOCHS, "max_epochs": MAX_EPOCHS, "grad_clip": 0.0,
        "cache": str(CACHE_DIR),
        "ols_reference": {"val_acc": float(ols_val), "test_acc": float(ols_test), "alpha": RIDGE_ALPHA},
        "conditions": {},
    }
    for name, init, opt_name, wd, batch in CONDITIONS:
        res = train_condition(name, init, opt_name, wd, batch,
                              xtr, ytr, xva, yva, xte, yte,
                              ols.coef_, ols.intercept_, device)
        results["conditions"][name] = res

    out = Path("artifacts/EXP-20260729-002/03h_adam_ablation/adam_ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
