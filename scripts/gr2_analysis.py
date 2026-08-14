"""Analyze gr2 mudularized_layer_probe data per user_exp_plans/gr2_data_analysis_plan.md.

1. Class-conditioned transition utility: for every pair (i,j),
   ``Gain_{(i,j)|c} = acc(i,j | y=c) - acc(i | y=c)`` and ``var_c(Gain)``.
2. Path feature regression (task-4 random paths, exact duplicates removed):
   per-path features (len; repeat_count = adjacent-repeat edges; backward-jump
   edges; canonical-adjacent edges p->p+1; longest canonical run; start with
   [1] / [1,2]; distinct layers; ratios; start/tail layer) -> regressions:
   a) linear: len/start_layer/tail_layer on acc;
   b) linear: all features on acc^res (length-residualized);
   c) random forest: all features on acc^res.
3. Weighted bigram/trigram vocabulary over paths ranked by acc:
   bigram weight = acc^res / sigma_len / (len-1) per occurrence,
   trigram weight = acc^res / sigma_len / (len-2).

Usage: python scripts/gr2_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (not needed; placeholder removed below)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fragmented import load_clinc_plus  # noqa: E402

ART = ROOT / "artifacts/fragmented-experiments/mudularized_layer_probe_260813_01"
OUT = ART / "analysis"
N_CLASSES = 150
SEED = 17


def load_nodes() -> tuple[dict, np.ndarray]:
    """path tuple -> (row index, val_acc); preds (N_nodes x 3000 int16)."""
    idx: dict[tuple, tuple[int, float]] = {}
    for i, line in enumerate((ART / "nodes.jsonl").read_text(encoding="utf-8").splitlines()):
        r = json.loads(line)
        idx[tuple(r["path"])] = (i, r["val_acc"])
    preds = np.load(ART / "nodes_pred.npy")
    return idx, preds


def per_class_acc(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """acc per class c: mean(pred == y | y == c), vectorized over 150 classes."""
    correct = (pred == y)
    onehot = np.eye(N_CLASSES, dtype=np.float64)[y]          # (N, C)
    num = onehot.T @ correct.astype(np.float64)              # (C,)
    den = onehot.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / np.where(den > 0, den, np.nan)


# --------------------------------------------------------------------------- #
# Task 1 — class-conditioned transition utility
# --------------------------------------------------------------------------- #
def task1(idx: dict, preds: np.ndarray, y: np.ndarray) -> dict:
    var_gain = np.full((12, 12), np.nan)
    mean_gain = np.full((12, 12), np.nan)
    acc_single = {}
    for i in range(1, 13):
        row, _ = idx[(i,)]
        acc_single[i] = per_class_acc(preds[row], y)
    for i in range(1, 13):
        for j in range(1, 13):
            row, _ = idx[(i, j)]
            acc_pair = per_class_acc(preds[row], y)
            gain = acc_pair - acc_single[i]
            with np.errstate(invalid="ignore"):
                var_gain[i - 1, j - 1] = np.nanvar(gain)
                mean_gain[i - 1, j - 1] = np.nanmean(gain)
    np.save(OUT / "task1_class_gain_var.npy", var_gain)
    (OUT / "task1_class_gain.json").write_text(json.dumps({
        "definition": "Gain_{(i,j)|c} = acc(i,j|y=c) - acc(i|y=c); matrix = var over "
                      "the 150 classes of Gain (population var, 20 val samples/class)",
        "var_gain": var_gain.tolist(), "mean_gain": mean_gain.tolist(),
    }, indent=2), encoding="utf-8")
    return {"var_gain": var_gain, "mean_gain": mean_gain}


# --------------------------------------------------------------------------- #
# Task 2 — path features + regression
# --------------------------------------------------------------------------- #
def path_features(p: tuple) -> dict:
    L = len(p)
    edges = list(zip(p[:-1], p[1:]))
    repeat_count = sum(1 for a, b in edges if a == b)
    backward = sum(1 for a, b in edges if b < a)
    canonical = sum(1 for a, b in edges if b == a + 1)
    run = cur = 1
    for a, b in edges:
        cur = cur + 1 if b == a + 1 else 1
        run = max(run, cur)
    distinct = len(set(p))
    f = {
        "len": L,
        "repeat_count": repeat_count,                    # adjacent repeats
        "repeat_count_all": sum(p.count(x) - 1 for x in set(p)),  # extra occurrences
        "backward_jump_count": backward,
        "canonical_adjacent_edge_count": canonical,
        "longest_canonical_run": run,
        "start_with_1": int(p[0] == 1),
        "start_with_12": int(p[0] == 1 and p[1] == 2),
        "distinct_layer_count": distinct,
        "distinct_ratio": distinct / L,
        "repeat_ratio": repeat_count / (L - 1),
        "backward_jump_ratio": backward / (L - 1),
        "canonical_edge_ratio": canonical / (L - 1),
        "longest_canonical_run_ratio": run / L,
        "start_layer": p[0],
        "tail_layer": p[-1],
    }
    return f


FEATURES_REGRESSION = [
    "len", "repeat_count", "backward_jump_count", "canonical_adjacent_edge_count",
    "longest_canonical_run", "start_with_1", "start_with_12", "distinct_layer_count",
    "distinct_ratio", "repeat_ratio", "backward_jump_ratio", "canonical_edge_ratio",
    "longest_canonical_run_ratio", "start_layer", "tail_layer",
]


def linear_report(X_tr, X_te, y_tr, y_te, names: list[str]) -> dict:
    """Linear regression with standardized features; in-sample + held-out R2."""
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Z_tr, Z_te = (X_tr - mu) / sd, (X_te - mu) / sd
    lr = LinearRegression().fit(Z_tr, y_tr)
    r2_tr = float(lr.score(Z_tr, y_tr))
    r2_te = float(lr.score(Z_te, y_te))
    return {
        "r2_in_sample": r2_tr, "r2_held_out": r2_te,
        "coef_standardized": {n: float(c) for n, c in zip(names, lr.coef_)},
        "intercept": float(lr.intercept_),
    }


def rf_report(X_tr, X_te, y_tr, y_te, names: list[str]) -> dict:
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1,
                               oob_score=True)
    rf.fit(X_tr, y_tr)
    return {
        "r2_oob": float(rf.oob_score_),
        "r2_held_out": float(rf.score(X_te, y_te)),
        "feature_importances": {n: float(v) for n, v in zip(names, rf.feature_importances_)},
    }


def task2(idx: dict, paths: list[tuple], accs: np.ndarray) -> dict:
    # dedupe exact duplicates
    seen: dict[tuple, int] = {}
    keep = []
    for p, a in zip(paths, accs):
        if p not in seen:
            seen[p] = len(keep)
            keep.append((p, a))
    paths, accs = zip(*keep)
    paths, accs = list(paths), np.array(accs)
    print(f"[task2] {len(paths)} unique paths (deduped from 4500)")

    feats = np.array([[path_features(p)[f] for f in FEATURES_REGRESSION] for p in paths])
    lens = feats[:, 0].astype(int)
    mu_k = np.array([accs[lens == k].mean() for k in range(3, 13)])
    sigma_k = np.array([accs[lens == k].std() for k in range(3, 13)])
    acc_res = accs - mu_k[lens - 3]

    # full feature rows incl. extras, for the CSV
    all_f = [path_features(p) for p in paths]
    header = (list(all_f[0]) + ["acc", "acc_res"])
    import csv
    with (OUT / "task2_path_features.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path"] + header)
        for p, a, ar, fr in zip(paths, accs, acc_res, all_f):
            w.writerow(["".join(map(str, p))] + [fr[h] for h in header[:-2]] + [a, ar])

    X = feats.astype(np.float64)
    # correlations of each feature with acc_res
    corr = {}
    for k, name in enumerate(FEATURES_REGRESSION):
        corr[name] = float(np.corrcoef(X[:, k], acc_res)[0, 1])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, acc_res, test_size=0.2, random_state=SEED)
    # a) linear: len, start_layer, tail_layer on raw acc
    cols_a = [FEATURES_REGRESSION.index(n) for n in ("len", "start_layer", "tail_layer")]
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        X, accs, test_size=0.2, random_state=SEED)
    lin_a = linear_report(Xa_tr[:, cols_a], Xa_te[:, cols_a], ya_tr, ya_te,
                          ["len", "start_layer", "tail_layer"])
    lin_b = linear_report(X_tr, X_te, y_tr, y_te, FEATURES_REGRESSION)
    rf_c = rf_report(X_tr, X_te, y_tr, y_te, FEATURES_REGRESSION)

    out = {
        "n_paths_deduped": len(paths),
        "acc_res_definition": "acc(P) - mean acc over same-length deduped paths",
        "mu_k": {str(k): float(v) for k, v in zip(range(3, 13), mu_k)},
        "sigma_k": {str(k): float(v) for k, v in zip(range(3, 13), sigma_k)},
        "corr_with_acc_res": corr,
        "regression_a_linear_len_start_tail_on_acc": lin_a,
        "regression_b_linear_all_on_acc_res": lin_b,
        "regression_c_rf_all_on_acc_res": rf_c,
    }
    (OUT / "task2_regressions.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Task 3 — weighted bigram/trigram vocabulary
# --------------------------------------------------------------------------- #
def task3(idx: dict, paths: list[tuple], accs: np.ndarray) -> dict:
    seen: dict[tuple, int] = {}
    keep = []
    for p, a in zip(paths, accs):
        if p not in seen:
            seen[p] = len(keep)
            keep.append((p, a))
    paths, accs = zip(*keep)
    paths, accs = list(paths), np.array(accs)
    lens = np.array([len(p) for p in paths])
    sigma_k = np.array([accs[lens == k].std() for k in range(3, 13)])
    acc_res = accs - np.array([accs[lens == k].mean() for k in range(3, 13)])[lens - 3]

    bigram_w: dict[tuple, float] = {}
    trigram_w: dict[tuple, float] = {}
    for p, ar, L in zip(paths, acc_res, lens):
        w2 = ar / sigma_k[L - 3] / (L - 1)
        w3 = ar / sigma_k[L - 3] / (L - 2)
        for a, b in zip(p[:-1], p[1:]):
            bigram_w[(a, b)] = bigram_w.get((a, b), 0.0) + w2
        for a, b, c in zip(p[:-2], p[1:-1], p[2:]):
            trigram_w[(a, b, c)] = trigram_w.get((a, b, c), 0.0) + w3
    top_b = sorted(bigram_w.items(), key=lambda kv: -kv[1])[:20]
    bot_b = sorted(bigram_w.items(), key=lambda kv: kv[1])[:10]
    top_t = sorted(trigram_w.items(), key=lambda kv: -kv[1])[:20]
    bot_t = sorted(trigram_w.items(), key=lambda kv: kv[1])[:10]
    out = {
        "definition": "weight per occurrence = acc^res / sigma_len / (len-1) for bigrams, "
                      "/(len-2) for trigrams; aggregated over deduped paths",
        "n_paths": len(paths),
        "bigram_top20": [[list(k), round(v, 4)] for k, v in top_b],
        "bigram_bottom10": [[list(k), round(v, 4)] for k, v in bot_b],
        "trigram_top20": [[list(k), round(v, 4)] for k, v in top_t],
        "trigram_bottom10": [[list(k), round(v, 4)] for k, v in bot_t],
        "n_distinct_bigrams": len(bigram_w),
        "n_distinct_trigrams": len(trigram_w),
    }
    (OUT / "task3_vocab.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx, preds = load_nodes()
    y = load_clinc_plus()["validation"][1]
    print(f"[data] nodes={len(idx)} preds={preds.shape} val_labels={len(y)}")

    r1 = task1(idx, preds, y)
    v = r1["var_gain"]
    print(f"[task1] var_c(Gain) mean {np.nanmean(v):.5f} max {np.nanmax(v):.5f} "
          f"at {np.unravel_index(np.nanargmax(v), v.shape)}")

    paths = [tuple(p) for p in json.loads((ART / "random_paths.json").read_text())]
    accs = np.array([idx[p][1] for p in paths])
    r2 = task2(idx, paths, accs)
    print("[task2] corr with acc_res (sorted):")
    for name, c in sorted(r2["corr_with_acc_res"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:>32}: {c:+.4f}")
    a = r2["regression_a_linear_len_start_tail_on_acc"]
    b = r2["regression_b_linear_all_on_acc_res"]
    c = r2["regression_c_rf_all_on_acc_res"]
    print(f"[task2a] linear len/start/tail on acc: R2 in {a['r2_in_sample']:.4f} "
          f"held-out {a['r2_held_out']:.4f}")
    print(f"[task2b] linear all on acc_res: R2 in {b['r2_in_sample']:.4f} "
          f"held-out {b['r2_held_out']:.4f}")
    print(f"[task2c] RF all on acc_res: R2 oob {c['r2_oob']:.4f} "
          f"held-out {c['r2_held_out']:.4f}")
    print("[task2c] RF importances (top8):")
    for name, v in sorted(c["feature_importances"].items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {name:>32}: {v:.4f}")

    r3 = task3(idx, paths, accs)
    print("[task3] bigram top10:", r3["bigram_top20"][:10])
    print("[task3] trigram top10:", r3["trigram_top20"][:10])
    print(f"[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
