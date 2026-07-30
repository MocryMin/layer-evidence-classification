"""Metric definitions for EXP-20260729-001.

All metrics are computed from raw logits so the definitions live in one
testable place. Per-sample metrics feed the predictions parquet; aggregate
metrics feed ``layer_metrics.csv``; recoverability / oracle / divergence are
computed across layers per seed.

Conventions:
- class-wise ratios always carry (numerator, denominator); a zero denominator
  is reported as ``NA`` (numpy nan), never as 0.
- ``0 * log 0 = 0`` in all entropy / KL terms.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Per-sample metrics
# --------------------------------------------------------------------------- #
def per_sample_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Compute per-sample metrics from logits.

    Args:
        logits: (N, C) float.
        labels: (N,) int in [0, C).

    Returns dict of (N,) numpy arrays: ``probs`` (N,C), ``prediction``,
    ``nll``, ``probability_margin``, ``logit_margin``, ``gold_margin``,
    ``entropy``, ``correct``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    N, C = logits.shape
    assert labels.shape == (N,)

    # softmax (numerically stable)
    z = logits - logits.max(axis=1, keepdims=True)
    expz = np.exp(z)
    probs = expz / expz.sum(axis=1, keepdims=True)

    pred = logits.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)

    # NLL = -log p[y]
    nll = -np.log(probs[np.arange(N), labels] + EPS)

    # probability margin: top1 - top2 prob
    top2p = np.partition(probs, -2, axis=1)[:, -2:]
    prob_margin = top2p[:, 1] - top2p[:, 0]
    # logit margin: top1 - top2 logit
    top2z = np.partition(logits, -2, axis=1)[:, -2:]
    logit_margin = top2z[:, 1] - top2z[:, 0]

    # gold margin: z_y - max_{c != y} z_c
    masked = logits.copy()
    masked[np.arange(N), labels] = -np.inf
    max_other = masked.max(axis=1)
    gold_margin = logits[np.arange(N), labels] - max_other

    # predictive entropy
    entropy = -(probs * np.log(probs + EPS)).sum(axis=1)

    return {
        "probs": probs,
        "prediction": pred.astype(np.int64),
        "nll": nll,
        "probability_margin": prob_margin,
        "logit_margin": logit_margin,
        "gold_margin": gold_margin,
        "entropy": entropy,
        "correct": correct,
    }


# --------------------------------------------------------------------------- #
# Aggregate metrics (one layer, one seed)
# --------------------------------------------------------------------------- #
def confusion_matrix(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    idx = labels * n_classes + preds
    cm = np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    return cm


def macro_f1(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp  # predicted c, true != c
    fn = cm.sum(axis=1) - tp  # true c, predicted != c
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.where(denom == 0, 1, denom), 0.0)
    return float(f1.mean())


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width confidence bins."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    N = len(confidence)
    if N == 0:
        return float("nan")
    bin_idx = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    bin_idx = np.maximum(bin_idx, 0)
    ece_val = 0.0
    for m in range(n_bins):
        in_bin = bin_idx == m
        cnt = int(in_bin.sum())
        if cnt == 0:
            continue
        acc_m = correct[in_bin].mean()
        conf_m = confidence[in_bin].mean()
        ece_val += (cnt / N) * abs(acc_m - conf_m)
    return float(ece_val)


def aggregate_metrics(logits: np.ndarray, labels: np.ndarray, n_classes: int, n_bins: int = 10) -> dict:
    """Aggregate metrics for one (seed, layer) on a split."""
    ps = per_sample_metrics(logits, labels)
    cm = confusion_matrix(labels, ps["prediction"], n_classes)
    confidence = ps["probs"].max(axis=1)
    return {
        "accuracy": float(ps["correct"].mean()),
        "macro_f1": macro_f1(cm),
        "nll": float(ps["nll"].mean()),
        "probability_margin": float(ps["probability_margin"].mean()),
        "logit_margin": float(ps["logit_margin"].mean()),
        "gold_margin": float(ps["gold_margin"].mean()),
        "entropy": float(ps["entropy"].mean()),
        "ece": ece(confidence, ps["correct"], n_bins),
        "confusion_matrix": cm,
    }


# --------------------------------------------------------------------------- #
# Recoverability / oracle / class-wise (across layers, one seed)
# --------------------------------------------------------------------------- #
def _ratio(num: float, denom: float):
    """Return (num, denom, ratio or NA)."""
    if denom == 0:
        return float(num), float(denom), float("nan")
    return float(num), float(denom), float(num) / float(denom)


def recoverability(
    layer_correct: np.ndarray,
    layer_pred: np.ndarray,
    labels: np.ndarray,
    final_layer_idx: int,
    n_classes: int,
) -> dict:
    """Compute recoverability across layers for one seed.

    Args:
        layer_correct: (L, N) bool/float, row i == correctness of probe layer i+1.
        layer_pred: (L, N) int.
        labels: (N,) int.
        final_layer_idx: row index of the final layer (e.g. 11 for layer 12).
    """
    layer_correct = np.asarray(layer_correct, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64)
    L, N = layer_correct.shape

    final_correct = layer_correct[final_layer_idx]
    final_wrong = ~final_correct
    mid_mask = np.array([i != final_layer_idx for i in range(L)])
    any_mid_correct = layer_correct[mid_mask].any(axis=0)  # (N,)

    # oracle
    oracle_correct = final_correct | any_mid_correct
    acc_L = float(final_correct.mean())
    acc_oracle = float(oracle_correct.mean())
    num_R_oracle = int((any_mid_correct & final_wrong).sum())
    denom_R_oracle = int(final_wrong.sum())
    _, _, R_oracle = _ratio(num_R_oracle, denom_R_oracle)

    # identity check: Acc_oracle == Acc_L + (1 - Acc_L) * R_oracle
    if denom_R_oracle > 0:
        expected = acc_L + (1 - acc_L) * R_oracle
        assert abs(acc_oracle - expected) < 1e-6, (
            f"oracle identity failed: Acc_oracle={acc_oracle} != {expected} "
            f"(Acc_L={acc_L}, R_oracle={R_oracle})"
        )

    # per-layer R_l, H_l
    R_l, H_l = {}, {}
    for i in range(L):
        if i == final_layer_idx:
            continue
        layer = i + 1  # probe layer number (1-based)
        l_correct = layer_correct[i]
        num_R = int((l_correct & final_wrong).sum())
        denom_R = int(final_wrong.sum())
        num_H = int((~l_correct & final_correct).sum())
        denom_H = int(final_correct.sum())
        R_l[layer] = _ratio(num_R, denom_R)
        H_l[layer] = _ratio(num_H, denom_H)

    # class-wise R_{l,c}, H_{l,c}
    R_lc, H_lc = {}, {}
    for i in range(L):
        if i == final_layer_idx:
            continue
        layer = i + 1
        l_correct = layer_correct[i]
        for c in range(n_classes):
            y_c = labels == c
            # R_{l,c}: y=c & final wrong & l correct / (y=c & final wrong)
            num_Rc = int((y_c & final_wrong & l_correct).sum())
            denom_Rc = int((y_c & final_wrong).sum())
            # H_{l,c}: y=c & final correct & l wrong / (y=c & final correct)
            num_Hc = int((y_c & final_correct & ~l_correct).sum())
            denom_Hc = int((y_c & final_correct).sum())
            R_lc[(layer, c)] = _ratio(num_Rc, denom_Rc)
            H_lc[(layer, c)] = _ratio(num_Hc, denom_Hc)

    # class-wise divergence D_JS
    n_err_c = np.array([int((labels == c).sum() and (final_wrong & (labels == c)).sum())
                        if False else int((final_wrong & (labels == c)).sum())
                        for c in range(n_classes)], dtype=np.float64)
    n_rec_c = np.array([int((any_mid_correct & final_wrong & (labels == c)).sum())
                        for c in range(n_classes)], dtype=np.float64)
    d_js = classwise_js_divergence(n_err_c, n_rec_c)

    return {
        "acc_L": acc_L,
        "acc_oracle": acc_oracle,
        "R_oracle": R_oracle,
        "num_R_oracle": num_R_oracle,
        "denom_R_oracle": denom_R_oracle,
        "oracle_gain": acc_oracle - acc_L,
        "R_l": R_l,
        "H_l": H_l,
        "R_lc": R_lc,
        "H_lc": H_lc,
        "n_err_c": n_err_c,
        "n_rec_c": n_rec_c,
        "d_js_class": d_js,
    }


def classwise_js_divergence(n_err_c: np.ndarray, n_rec_c: np.ndarray) -> float:
    """Normalized class-wise JS divergence in [0, 1].

    Compares the class distribution of all final-layer errors (e) against the
    class distribution of oracle-recoverable errors (r). NA if either support
    is zero.
    """
    n_err_c = np.asarray(n_err_c, dtype=np.float64)
    n_rec_c = np.asarray(n_rec_c, dtype=np.float64)
    s_err = n_err_c.sum()
    s_rec = n_rec_c.sum()
    if s_err == 0 or s_rec == 0:
        return float("nan")
    e = n_err_c / s_err
    r = n_rec_c / s_rec
    m = 0.5 * (e + r)

    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    js = 0.5 * kl(e, m) + 0.5 * kl(r, m)
    return float(js / np.log(2.0))
