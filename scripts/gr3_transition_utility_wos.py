"""gr3 step 2: class-conditioned transition utility (gr2 analysis plan item 1)
on the DeBERTa WOS-46985 modular layer probe, TEST split (primary; val side
check from the original run's nodes_pred.npy).

For every pair (i,j):  Gain_{(i,j)|c} = acc(i,j | y=c) - acc(i | y=c)
                       var_c(Gain), mean_c(Gain)  (c = 1..134 test classes)

Usage: python scripts/gr3_transition_utility_wos.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fragmented import load_wos_46985  # noqa: E402

ART = ROOT / "artifacts/fragmented-experiments/DeBERTaV3BaseWOS46985LayerProbe_260814_01"
N_CLASSES = 134
N_LAYERS = 12
OUT = ART / "analysis_gr3"


def per_class_acc(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    correct = pred == y
    onehot = np.eye(N_CLASSES, dtype=np.float64)[y]
    num = onehot.T @ correct.astype(np.float64)
    den = onehot.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / np.where(den > 0, den, np.nan)


def load_index(jsonl: Path, acc_key: str) -> tuple[dict, list]:
    idx = {}
    for i, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines()):
        r = json.loads(line)
        idx[tuple(r["path"])] = (i, r[acc_key])
    return idx, sorted(idx)  # sorted path order for stable output


def main() -> None:
    OUT.mkdir(exist_ok=True)
    y_test = load_wos_46985()["test"][1]
    y_val = load_wos_46985()["validation"][1]

    # ---- test (replay run) ----
    idx_t, order_t = load_index(ART / "nodes_test.jsonl", "test_acc")
    preds_t = np.load(ART / "nodes_test_pred.npy").astype(np.int64)
    assert preds_t.shape[1] == len(y_test), preds_t.shape

    single_pc = {}
    for i in range(1, N_LAYERS + 1):
        row = idx_t[(i,)][0]
        single_pc[i] = per_class_acc(preds_t[row], y_test)

    G = np.full((N_LAYERS, N_LAYERS, N_CLASSES), np.nan)   # per-class gain
    A_test = np.full((N_LAYERS, N_LAYERS), np.nan)
    with np.errstate(invalid="ignore"):
        for i in range(1, N_LAYERS + 1):
            for j in range(1, N_LAYERS + 1):
                row = idx_t[(i, j)][0]
                A_test[i - 1, j - 1] = idx_t[(i, j)][1]
                G[i - 1, j - 1] = per_class_acc(preds_t[row], y_test) - single_pc[i]

    ok = ~np.isnan(G).any(2)                                # classes with data
    var_c = np.full((N_LAYERS, N_LAYERS), np.nan)
    mean_c = np.full((N_LAYERS, N_LAYERS), np.nan)
    for i in range(N_LAYERS):
        for j in range(N_LAYERS):
            g = G[i, j][ok[i, j]]
            if g.size:
                var_c[i, j] = g.var()
                mean_c[i, j] = g.mean()

    # binomial noise reference: per-class acc diff of two probes sharing
    # nothing would have se ~ sqrt(2 p(1-p)/n_c); use observed p ~ 0.45
    n_c = np.bincount(y_test, minlength=N_CLASSES).astype(float)
    se0 = np.sqrt(2 * 0.45 * 0.55 / np.maximum(n_c, 1))     # (134,)
    var0 = np.nanmean((se0 ** 2)[np.isfinite(se0)])

    out = dict(
        split="test", n=len(y_test),
        mean_class_n=float(n_c.mean()), min_class_n=int(n_c.min()),
        var_c=var_c.tolist(), mean_c=mean_c.tolist(),
        A_test=A_test.tolist(),
        single_test_acc=[idx_t[(i,)][1] for i in range(1, N_LAYERS + 1)],
        gain_test=(A_test - np.array(
            [[idx_t[(i,)][1]] * N_LAYERS for i in range(1, N_LAYERS + 1)]))
        .tolist(),
        per_class_gain={f"{i},{j}": {c: float(G[i - 1, j - 1, c])
                                     for c in range(N_CLASSES)
                                     if np.isfinite(G[i - 1, j - 1, c])}
                        for i in range(1, N_LAYERS + 1)
                        for j in range(1, N_LAYERS + 1)},
        noise_ref=dict(
            definition="se of acc diff of two independent probes at p=0.45, "
                       "mean over classes; var0 = mean se^2",
            mean_se=float(np.mean(se0)), var0=float(var0),
            note="same-sample probes share variance; var0 is an upper bound "
                 "on pure-sampling variance of Gain"),
    )
    np.savez(OUT / "task1_class_gain.npz", G=G, var_c=var_c, mean_c=mean_c,
             A_test=A_test, n_c=n_c)

    # ---- val side-check (original run preds, all nodes) ----
    idx_v, _ = load_index(ART / "nodes.jsonl", "val_acc")
    preds_v = np.load(ART / "nodes_pred.npy").astype(np.int64)
    var_v = np.full((N_LAYERS, N_LAYERS), np.nan)
    for i in range(1, N_LAYERS + 1):
        sv = per_class_acc(preds_v[idx_v[(i,)][0]], y_val)
        for j in range(1, N_LAYERS + 1):
            pv = per_class_acc(preds_v[idx_v[(i, j)][0]], y_val)
            g = pv - sv
            var_v[i - 1, j - 1] = np.nanvar(g)
    out["var_c_val_sidecheck"] = var_v.tolist()
    n_cv = np.bincount(y_val, minlength=N_CLASSES).astype(float)
    out["noise_ref"]["mean_class_n_val"] = float(n_cv.mean())

    out["split_half"] = split_half(preds_t, idx_t, y_test)

    json.dump(out, open(OUT / "task1_class_gain.json", "w"), indent=1)
    r = np.unravel_index(np.nanargmax(var_c), var_c.shape)
    print(f"[test] var_c(Gain): mean {np.nanmean(var_c):.5f} "
          f"max {np.nanmax(var_c):.5f} at [{r[0]+1},{r[1]+1}] "
          f"(diag mean {np.nanmean(np.diag(var_c)):.5f})")
    print(f"[test] noise var0 {var0:.5f} -> mean var_c/var0 = "
          f"{np.nanmean(var_c)/var0:.2f}")
    print(f"[val ] var_c side-check mean {np.nanmean(var_v):.5f}")
    sh = out["split_half"]
    print(f"[test] split-half corr: cell {np.mean(sh['corr_cell']):.3f} "
          f"class-effect {np.mean(sh['corr_class_effect']):.3f}")
    print(f"[done] {OUT}")




def split_half(preds: np.ndarray, idx: dict, y: np.ndarray,
               n_rep: int = 3, min_n: int = 20) -> dict:
    """Split-half reliability of the centered per-class gains.

    Repeatedly split each class's test samples into halves; compute the
    class-centered gains on each half; report corr(half A, half B) both per
    (pair,class) cell and for the class effect (mean over pairs).  A positive
    correlation separates real class-conditioned heterogeneity from sampling
    noise (which is independent across halves only for the probe-error part;
    see report for the correlated-share caveat).
    """
    rows = [idx[(i,)][0] for i in range(1, N_LAYERS + 1)]          # singles
    pair_rows = np.array([[idx[(i, j)][0] if i != j else -1
                           for j in range(1, N_LAYERS + 1)]
                          for i in range(1, N_LAYERS + 1)])
    rng = np.random.default_rng(17)
    keep_cls = np.bincount(y, minlength=N_CLASSES) >= min_n
    corr_cell, corr_eff = [], []
    for _ in range(n_rep):
        half = np.zeros(len(y), bool)
        for c in range(N_CLASSES):
            ix = np.where(y == c)[0]
            rng.shuffle(ix)
            half[ix[:len(ix) // 2]] = True
        ga, gb = [], []
        for i in range(1, N_LAYERS + 1):
            pa = per_class_acc(preds[rows[i - 1]][half], y[half])
            pb = per_class_acc(preds[rows[i - 1]][~half], y[~half])
            for j in range(1, N_LAYERS + 1):
                if i == j:
                    continue
                ga.append(per_class_acc(preds[pair_rows[i - 1, j - 1]][half],
                                        y[half]) - pa)
                gb.append(per_class_acc(preds[pair_rows[i - 1, j - 1]][~half],
                                        y[~half]) - pb)
        ga = np.asarray(ga)                       # (132, 134)
        gb = np.asarray(gb)
        ca = ga - ga.mean(1, keepdims=True)
        cb = gb - gb.mean(1, keepdims=True)
        kc = np.tile(keep_cls, len(ga))
        corr_cell.append(np.corrcoef(ca.ravel()[kc], cb.ravel()[kc])[0, 1])
        corr_eff.append(np.corrcoef(ca.mean(0)[keep_cls],
                                    cb.mean(0)[keep_cls])[0, 1])
    return dict(corr_cell=[float(x) for x in corr_cell],
                corr_class_effect=[float(x) for x in corr_eff],
                min_class_n=min_n, n_rep=n_rep, seed=17)


if __name__ == "__main__":
    main()
