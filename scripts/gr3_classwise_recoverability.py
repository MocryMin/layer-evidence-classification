"""gr3 step 1: layer + class-wise recoverability analysis on the two WOS-46985
baselines (DeBERTa-v3-base / modernBERT-base), ridge family, test split, per
user_exp_plans/fragmented_exp_gr3.md and the EXP-001 definitions:

    R_l     = P(yhat_l = y  | yhat_L != y)          (recoverability)
    H_l     = P(yhat_l != y  | yhat_L == y)          (harm rate)
    R_{l,c} = P(yhat_l = y  | y = c, yhat_L != c)
    H_{l,c} = P(yhat_l != y  | y = c, yhat_L == c)
    R_oracle, Acc_oracle, D_JS(e_c || r_c) per EXP-001 eq. (12)-(16).

The baseline results.json already stores the raw quantities; this script
recomputes them from ridge_test_pred.npy + test labels as a cross-check, then
adds the new analyses: per-class table (R_max, argmax layer), layer-agreement,
L1-domain aggregation, harm-side summary.

Usage: python scripts/gr3_classwise_recoverability.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ART = ROOT / "artifacts/fragmented-experiments"
MODELS = {
    "deberta": dict(art=ART / "DeBERTaV3BaseWOS46985Baseline_260812_04"),
    "modernbert": dict(art=ART / "ModernBERTBaseWOS46985Baseline_260812_05"),
}
OUT = ART / "class-wiseRecoverabilityWOS46985_260818_01"
FAMILY = "ridge"

L1_NAMES = ["CS", "ECE", "Psychology", "MAE", "Civil", "Medical", "biochemistry"]


def load_class_names() -> tuple[np.ndarray, np.ndarray]:
    """(l2_names (134,), l2 -> l1 index (134,)); from raw parquet row order."""
    df = pd.read_parquet(ROOT / "data/raw/wos46985/wos46895.parquet")
    z = np.load(ROOT / "data/processed/wos46985/wos46985_split.npz",
                allow_pickle=True)
    lab = z["labels"]
    raw_l1 = df["label_description"].str[0].values
    raw_l2 = df["label_description"].str[1].values
    l2_names = []
    l2_to_l1 = []
    for j in range(134):
        names = set(raw_l2[lab[:, 7 + j] == 1])
        assert len(names) == 1, f"L2 col {j} ambiguous: {names}"
        l2_names.append(names.pop())
        l1_cols = np.unique(lab[lab[:, 7 + j] == 1, :7].argmax(1))
        l2_to_l1.append(int(l1_cols[0]))  # primary domain (3 classes cross-listed)
    for j in range(7):  # each L1 column maps to exactly one raw L1 name
        assert len(set(raw_l1[lab[:, j] == 1])) == 1
    return np.array(l2_names, dtype=object), np.array(l2_to_l1)


def analyse(model: str, art: Path, y: np.ndarray, l2_names, l2_to_l1) -> dict:
    pred = np.load(art / f"{FAMILY}_test_pred.npy").astype(np.int64)  # (L, N)
    L = pred.shape[0]
    final = pred[-1]
    err_L = final != y
    cor_L = ~err_L
    n_err, n_cor = int(err_L.sum()), int(cor_L.sum())
    acc_L = n_cor / len(y)

    correct = pred == y                                    # (L, N)
    R_l = correct[:, err_L].mean(1)                        # (L,)
    H_l = (~correct)[:, cor_L].mean(1)
    oracle_rec = correct.any(0) & err_L                    # oracle-recoverable
    acc_or = (cor_L | (correct.any(0) & err_L)).mean()
    R_or = oracle_rec.sum() / n_err

    # class-wise fractions num/den (EXP-001: report unsimplified fractions)
    C = int(y.max()) + 1
    onehot = np.eye(C, dtype=np.float64)[y]                # (N, C)
    n_c = onehot.sum(0)                                    # (C,)
    err_c = (onehot.T @ err_L).astype(int)                 # final-layer errors per class
    cor_c = (onehot.T @ cor_L).astype(int)
    # R_{l,c}: correct at l AND final wrong, per class / err_c
    joint = (correct & err_L).astype(np.float64)           # (L, N)
    R_num = joint @ onehot                                  # (L, C)
    R_den = err_c                                            # (C,)
    H_num = ((~correct) & cor_L).astype(np.float64) @ onehot
    H_den = cor_c
    rec_num = (oracle_rec.astype(np.float64) @ onehot).astype(int)  # oracle per class

    with np.errstate(invalid="ignore", divide="ignore"):
        R_lc = R_num / np.where(R_den > 0, R_den, np.nan)
        H_lc = H_num / np.where(H_den > 0, H_den, np.nan)
    mid = R_lc[: L - 1]                                    # exclude final layer row
    with np.errstate(invalid="ignore"):
        R_max = np.nanmax(mid, 0)
        argmax_l = np.nanargmax(mid, 0) + 1                # layer id 1..L-1

    # D_JS between error-class dist e_c and recoverable-class dist r_c
    e_c = err_c / err_c.sum()
    r_c = rec_num / rec_num.sum()
    m = 0.5 * (e_c + r_c)
    def _kl(p, q):
        mask = p > 0
        return float((p[mask] * np.log2(p[mask] / q[mask])).sum())
    d_js = 0.5 * (_kl(e_c, m) + _kl(r_c, m))

    # L1-domain aggregation (weighted by per-class error counts)
    dom = {}
    for d, name in enumerate(L1_NAMES):
        cs = l2_to_l1 == d
        dom[name] = dict(
            n_classes=int(cs.sum()),
            n_err=int(err_c[cs].sum()),
            n_rec=int(rec_num[cs].sum()),
            R_oracle=float(rec_num[cs].sum() / max(err_c[cs].sum(), 1)),
            R_l_weighted=(R_num[:, cs].sum(1) /
                          np.maximum(err_c[cs].sum(), 1)).tolist(),
        )

    out = dict(
        model=model, n_layers=L, n_test=len(y), acc_L=float(acc_L),
        n_err=n_err, n_cor=n_cor,
        R_l=[dict(l=i + 1, ratio=float(R_l[i])) for i in range(L)],
        H_l=[dict(l=i + 1, ratio=float(H_l[i])) for i in range(L)],
        acc_oracle=float(acc_or), R_oracle=float(R_or),
        oracle_gain=float(acc_or - acc_L), d_js_class=float(d_js),
        classwise=dict(
            coverage=int((err_c > 0).sum()),
            R_lc=[[f"{int(R_num[l, c])}/{int(R_den[c])}" if R_den[c] > 0 else None
                   for c in range(C)] for l in range(L)],
            H_lc=[[f"{int(H_num[l, c])}/{int(H_den[c])}" if H_den[c] > 0 else None
                   for c in range(C)] for l in range(L)],
            n_c=n_c.astype(int).tolist(), n_err_c=err_c.tolist(),
            n_rec_c=rec_num.tolist(),
            R_max=R_max.tolist(), argmax_layer=argmax_l.tolist(),
        ),
        domains=dom,
    )
    return out


def crosscheck(model: str, art: Path, out: dict) -> list[str]:
    """Recomputed quantities must reproduce the baseline results.json."""
    ref = json.load(open(art / "results.json"))["families"][FAMILY]
    rec, probs = ref["recoverability"], []
    def close(a, b, tol=1e-9, name=""):
        if abs(a - b) > tol:
            probs.append(f"{model}.{name}: {a} != {b}")
    close(out["acc_L"], rec["acc_L"], name="acc_L")
    close(out["acc_oracle"], rec["acc_oracle"], name="acc_oracle")
    close(out["R_oracle"], rec["R_oracle"], name="R_oracle")
    close(out["d_js_class"], rec["d_js_class"], name="d_js_class")
    for l, ref in rec["R_l"].items():          # baseline stores mid layers only
        close(out["R_l"][int(l) - 1]["ratio"], ref["ratio"], name=f"R_l[{l}]")
        close(out["H_l"][int(l) - 1]["ratio"], rec["H_l"][l]["ratio"],
              name=f"H_l[{l}]")
    return probs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    l2_names, l2_to_l1 = load_class_names()
    results, all_probs = {}, []
    for model, cfg in MODELS.items():
        y = np.load(cfg["art"] / "cache/test_hidden.npz")["labels"].astype(np.int64)
        out = analyse(model, cfg["art"], y, l2_names, l2_to_l1)
        all_probs += crosscheck(model, cfg["art"], out)
        results[model] = out
        cw = out["classwise"]
        print(f"[{model}] acc_L {out['acc_L']:.4f} R_oracle {out['R_oracle']:.4f} "
              f"D_JS {out['d_js_class']:.4f} | R_max>=0.5: "
              f"{sum(v >= 0.5 for v in cw['R_max'])}/134")
    assert not all_probs, f"crosscheck failed: {all_probs}"
    json.dump(dict(l2_names=l2_names.tolist(), results=results),
              open(OUT / "analysis.json", "w"), indent=1)
    print(f"[done] {OUT / 'analysis.json'} (crosscheck vs results.json: OK)")


if __name__ == "__main__":
    main()
