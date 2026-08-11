"""EXP-20260810-003: intermediate-layer recoverability under validated probes.

Restores the EXP-001 verification ($H_1, H_1', H_2$) using three probe families
that EXP-002 showed to be valid on frozen DeBERTa-v3-base mid-layer CLS:

- ``centered_plain`` (primary): PlainHead on train-mean-centered features.
  Same hypothesis class as EXP-001's plain probe; fixed centering changes only
  the optimisation geometry.
- ``ln_plain`` (robustness control): LNHead (per-sample LayerNorm + affine)
  on raw features.
- ``ridge`` (solver reference): RidgeClassifier alpha grid (alpha=0 realised
  as OLS via numpy.lstsq). Deterministic, single run, no cross-seed CI.

Gradient probes (centered_plain, ln_plain) use full-batch AdamW (lr=1e-2,
wd=0, Xavier) with early stopping (min_ep=100, max_ep=1000, patience=100,
min_delta=1e-4). All hyperparameters/checkpoints are selected on validation;
test is evaluated once. Every probe family compares intermediate layers
against its own layer-12 baseline.

Reuses the EXP-001 hidden-state cache, label maps, seeds, and the metric /
analysis functions in :mod:`src.metrics` and :mod:`src.analysis`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .config import PROJECT_ROOT
from .heads import build_head
from .metrics import aggregate_metrics, confusion_matrix, macro_f1, per_sample_metrics, recoverability
from .probe import eval_probe, train_probe_fullbatch_es
from .seeding import enable_determinism

CONFIG_PATH = PROJECT_ROOT / "configs" / "exp003_config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load the EXP-003 YAML config as a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Cache loading and centering
# --------------------------------------------------------------------------- #
def load_caches(cache_dir: Path, device: torch.device) -> dict:
    """Load the three EXP-001 splits as float32 tensors on ``device``."""
    from .cache import load_cache

    out = {}
    for split in ["train", "validation", "test"]:
        out[split] = load_cache(
            cache_dir / f"{split}_hidden.safetensors", device=device, dtype=torch.float32
        )
    return out


def compute_train_means(train_hidden: torch.Tensor) -> torch.Tensor:
    """Per-layer cross-sample mean of the train CLS features.

    Args:
        train_hidden: (Ntrain, 12, 768) tensor.

    Returns:
        (12, 768) tensor, the per-layer mean mu_l used for centering.
    """
    return train_hidden.mean(dim=0)


# --------------------------------------------------------------------------- #
# Ridge / OLS
# --------------------------------------------------------------------------- #
def _ols_fit(train_x: np.ndarray, train_y: np.ndarray, n_classes: int) -> np.ndarray:
    """Ordinary least squares with bias via numpy.lstsq.

    Solves ``[X | 1] @ W = Y_onehot`` where ``Y_onehot`` is the 0/1 one-hot
    encoding (matching sklearn RidgeClassifier's internal label binariser for
    multiclass). Returns ``W`` of shape ``(d+1, C)``; the last row is the bias.
    """
    n = train_x.shape[0]
    X_aug = np.hstack([train_x, np.ones((n, 1), dtype=np.float64)])
    Y_onehot = np.zeros((n, n_classes), dtype=np.float64)
    Y_onehot[np.arange(n), train_y] = 1.0
    W, _, _, _ = np.linalg.lstsq(X_aug, Y_onehot, rcond=None)
    return W


def _ols_decision(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Decision scores from an OLS weight matrix ``W`` of shape ``(d+1, C)``."""
    n = x.shape[0]
    X_aug = np.hstack([x, np.ones((n, 1), dtype=np.float64)])
    return X_aug @ W


def fit_ridge_grid(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    alphas: list[float],
    n_classes: int,
) -> dict:
    """Fit RidgeClassifier for each alpha (alpha=0 -> OLS); select best by val acc.

    For ``alpha > 0`` uses ``sklearn.RidgeClassifier(solver='svd',
    fit_intercept=True)`` (matching EXP-002 task 03g). For ``alpha == 0`` uses
    a standalone OLS via :func:`_ols_fit`. The best alpha is chosen by
    validation accuracy, ties broken by lower alpha (less regularisation).

    Test predictions are computed **only at the selected best alpha** (one-time
    test evaluation; no test information enters alpha selection).

    Returns ``{"best_alpha", "best_val_acc", "val_pred", "test_pred",
    "test_decision", "per_alpha"}`` where ``per_alpha`` contains only
    ``val_acc`` per alpha.
    """
    from sklearn.linear_model import RidgeClassifier

    # Phase 1: fit each alpha on train, score on validation (no test access).
    per_alpha: dict[float, dict] = {}
    ols_W = None
    for alpha in alphas:
        if alpha == 0:
            if ols_W is None:
                ols_W = _ols_fit(train_x, train_y, n_classes)
            val_dec = _ols_decision(ols_W, val_x)
        else:
            clf = RidgeClassifier(alpha=alpha, fit_intercept=True, solver="svd")
            clf.fit(train_x, train_y)
            val_dec = clf.decision_function(val_x)
        val_pred = val_dec.argmax(axis=1)
        val_acc = float((val_pred == val_y).mean())
        per_alpha[alpha] = {"val_acc": val_acc}

    best_alpha = max(per_alpha, key=lambda a: (per_alpha[a]["val_acc"], -a))

    # Phase 2: one-time test evaluation at the selected best alpha.
    if best_alpha == 0:
        if ols_W is None:
            ols_W = _ols_fit(train_x, train_y, n_classes)
        val_dec = _ols_decision(ols_W, val_x)
        test_dec = _ols_decision(ols_W, test_x)
    else:
        clf = RidgeClassifier(alpha=best_alpha, fit_intercept=True, solver="svd")
        clf.fit(train_x, train_y)
        val_dec = clf.decision_function(val_x)
        test_dec = clf.decision_function(test_x)

    return {
        "best_alpha": best_alpha,
        "best_val_acc": per_alpha[best_alpha]["val_acc"],
        "val_pred": val_dec.argmax(axis=1),
        "test_pred": test_dec.argmax(axis=1),
        "test_decision": test_dec,
        "per_alpha": per_alpha,
    }


# --------------------------------------------------------------------------- #
# Gradient probe family (centered_plain, ln_plain)
# --------------------------------------------------------------------------- #
def run_gradient_family(
    caches: dict,
    family_name: str,
    family_cfg: dict,
    training_cfg: dict,
    seeds: list[int],
    layers: list[int],
    n_classes: int,
    device: torch.device,
) -> dict:
    """Train one gradient probe family across all (seed, layer) pairs.

    Returns a dict with per-seed, per-layer results (scalars only; predictions
    and logits are returned as numpy arrays under ``"test_pred"`` and
    ``"test_logits"`` keys for the caller to persist).
    """
    head_type = family_cfg["head_type"]
    centering = family_cfg.get("centering", "none")
    tr_h = caches["train"]["hidden"]      # (Ntr, 12, 768)
    va_h = caches["validation"]["hidden"]
    te_h = caches["test"]["hidden"]
    tr_y = caches["train"]["labels"].to(device)
    va_y = caches["validation"]["labels"].to(device)
    te_y = caches["test"]["labels"].to(device)

    train_means = compute_train_means(tr_h) if centering == "train_mean" else None

    per_seed: dict[int, dict] = {}
    for seed in seeds:
        per_layer: dict[int, dict] = {}
        test_preds: list[np.ndarray] = []     # per layer, (Ntest,)
        test_logits_list: list[np.ndarray] = []  # per layer, (Ntest, C)
        for layer in layers:
            tx = tr_h[:, layer - 1, :].contiguous()
            vx = va_h[:, layer - 1, :].contiguous()
            tex = te_h[:, layer - 1, :].contiguous()
            if train_means is not None:
                mu = train_means[layer - 1]
                tx = tx - mu
                vx = vx - mu
                tex = tex - mu

            head = build_head(head_type, in_dim=768, n_classes=n_classes)
            res = train_probe_fullbatch_es(
                head=head,
                train_x=tx, train_y=tr_y,
                val_x=vx, val_y=va_y,
                lr=training_cfg["lr"],
                weight_decay=training_cfg["weight_decay"],
                grad_clip=training_cfg.get("grad_clip", 0.0),
                min_epochs=training_cfg["min_epochs"],
                max_epochs=training_cfg["max_epochs"],
                patience=training_cfg["patience"],
                min_delta=training_cfg["min_delta"],
                seed=seed,
                device=device,
            )

            # one-time test evaluation with the best-state head
            head.load_state_dict(res["best_state"])
            head.to(device).eval()
            with torch.no_grad():
                te_logits = []
                for s in range(0, tex.shape[0], 1024):
                    te_logits.append(head(tex[s:s + 1024]))
                te_logits = torch.cat(te_logits, dim=0).cpu().numpy().astype(np.float32)

            te_logits_np = te_logits
            te_pred = te_logits_np.argmax(axis=1)
            te_labels_np = te_y.cpu().numpy()
            agg = aggregate_metrics(te_logits_np, te_labels_np, n_classes, n_bins=10)

            per_layer[layer] = {
                "best_epoch": res["best_epoch"],
                "best_val_acc": res["best_val_acc"],
                "best_val_nll": res["best_val_nll"],
                "final_val_acc": res["final_val_acc"],
                "final_val_nll": res["final_val_nll"],
                "converged": res["converged"],
                "stop_reason": res["stop_reason"],
                "val_history": res["history"],
                "test_acc": agg["accuracy"],
                "test_macro_f1": agg["macro_f1"],
                "test_nll": agg["nll"],
                "test_probability_margin": agg["probability_margin"],
                "test_logit_margin": agg["logit_margin"],
                "test_gold_margin": agg["gold_margin"],
                "test_entropy": agg["entropy"],
                "test_ece": agg["ece"],
            }
            test_preds.append(te_pred.astype(np.int16))
            test_logits_list.append(te_logits_np.astype(np.float16))

            flag = "" if res["converged"] else " [NON-CONVERGED]"
            print(f"  [{family_name}] seed={seed:3d} L{layer:2d}: "
                  f"val={res['best_val_acc']:.4f}@{res['best_epoch']}"
                  f" test={agg['accuracy']:.4f}{flag}")

        # recoverability across all 12 layers for this seed
        labels_np = te_y.cpu().numpy()
        layer_pred = np.stack(test_preds)               # (L, Ntest) int16
        layer_correct = (layer_pred == labels_np)        # (L, Ntest) bool
        rec = recoverability(
            layer_correct=layer_correct,
            layer_pred=layer_pred,
            labels=labels_np,
            final_layer_idx=layers.index(max(layers)),  # index of layer 12
            n_classes=n_classes,
        )

        per_seed[seed] = {
            "per_layer": {str(l): per_layer[l] for l in layers},
            "test_pred": np.stack(test_preds),            # (L, Ntest) int16
            "test_logits": np.stack(test_logits_list),    # (L, Ntest, C) float16
            "recoverability": _recoverability_to_json(rec),
        }

    return {"family": family_name, "per_seed": per_seed}


# --------------------------------------------------------------------------- #
# Ridge probe family
# --------------------------------------------------------------------------- #
def run_ridge_family(
    caches: dict,
    family_cfg: dict,
    layers: list[int],
    n_classes: int,
) -> dict:
    """Fit the Ridge alpha grid on every layer (deterministic, single run)."""
    alphas = family_cfg["alphas"]
    tr_h = caches["train"]["hidden"].cpu().numpy()
    va_h = caches["validation"]["hidden"].cpu().numpy()
    te_h = caches["test"]["hidden"].cpu().numpy()
    tr_y = caches["train"]["labels"].cpu().numpy()
    va_y = caches["validation"]["labels"].cpu().numpy()
    te_y = caches["test"]["labels"].cpu().numpy()

    per_layer: dict[int, dict] = {}
    test_preds: list[np.ndarray] = []
    for layer in layers:
        tx = tr_h[:, layer - 1, :].astype(np.float64)
        vx = va_h[:, layer - 1, :].astype(np.float64)
        tex = te_h[:, layer - 1, :].astype(np.float64)
        res = fit_ridge_grid(tx, tr_y, vx, va_y, tex, alphas, n_classes)
        te_pred = res["test_pred"]
        cm = confusion_matrix(te_y, te_pred, n_classes)
        per_layer[layer] = {
            "best_alpha": res["best_alpha"],
            "best_val_acc": res["best_val_acc"],
            "test_acc": float((te_pred == te_y).mean()),
            "test_macro_f1": macro_f1(cm),
            "per_alpha": res["per_alpha"],
        }
        test_preds.append(te_pred.astype(np.int16))
        print(f"  [ridge] L{layer:2d}: alpha={res['best_alpha']:g} "
              f"val={res['best_val_acc']:.4f} test={per_layer[layer]['test_acc']:.4f}")

    layer_pred = np.stack(test_preds)  # (L, Ntest)
    layer_correct = (layer_pred == te_y)
    rec = recoverability(
        layer_correct=layer_correct,
        layer_pred=layer_pred,
        labels=te_y,
        final_layer_idx=layers.index(max(layers)),
        n_classes=n_classes,
    )
    return {
        "family": "ridge",
        "per_layer": {str(l): per_layer[l] for l in layers},
        "test_pred": layer_pred,
        "recoverability": _recoverability_to_json(rec),
    }


# --------------------------------------------------------------------------- #
# JSON serialisation helpers
# --------------------------------------------------------------------------- #
def _recoverability_to_json(rec: dict) -> dict:
    """Strip non-serialisable arrays from a recoverability result dict."""
    import math

    def clean_ratio(r):
        num, den, ratio = r
        return {"num": int(num), "den": int(den), "ratio": None if math.isnan(ratio) else float(ratio)}

    return {
        "acc_L": rec["acc_L"],
        "acc_oracle": rec["acc_oracle"],
        "R_oracle": rec["R_oracle"],
        "num_R_oracle": rec["num_R_oracle"],
        "denom_R_oracle": rec["denom_R_oracle"],
        "oracle_gain": rec["oracle_gain"],
        "R_l": {str(l): clean_ratio(r) for l, r in rec["R_l"].items()},
        "H_l": {str(l): clean_ratio(r) for l, r in rec["H_l"].items()},
        "d_js_class": rec["d_js_class"],
        "n_err_c": rec["n_err_c"].tolist(),
        "n_rec_c": rec["n_rec_c"].tolist(),
    }


def _ratio_lookup(rec: dict, key: str) -> dict:
    """Return the {layer: ratio-or-NA} view from a recoverability JSON dict."""
    out = {}
    for l, r in rec[key].items():
        ratio = r["ratio"]
        out[int(l)] = float("nan") if ratio is None else ratio
    return out


# --------------------------------------------------------------------------- #
# Per-family analysis and hypothesis judgement
# --------------------------------------------------------------------------- #
def analyze_gradient_family(
    family_results: dict,
    seeds: list[int],
    final_layer: int,
    epsilon_1: float,
    epsilon_2: float,
    n_resamples: int,
    ci: float,
) -> dict:
    """Bootstrap CI + H1/H1'/H2 judgement for a gradient probe family.

    Uses 10-seed paired bootstrap (10000 resamples, percentile 95% CI) from
    :mod:`src.analysis`. The candidate intermediate layer is selected by mean
    *validation* accuracy; test results enter only the CI.
    """
    from .analysis import (
        bootstrap_ci_mean,
        judge_h1,
        judge_h1prime,
        judge_h2,
        select_candidate_intermediate,
    )

    per_seed = family_results["per_seed"]
    val_acc = {(s, l): per_seed[s]["per_layer"][str(l)]["best_val_acc"]
               for s in seeds for l in [int(x) for x in per_seed[s]["per_layer"]]}
    test_acc = {(s, l): per_seed[s]["per_layer"][str(l)]["test_acc"]
                for s in seeds for l in [int(x) for x in per_seed[s]["per_layer"]]}

    candidate, mean_val = select_candidate_intermediate(val_acc, seeds, final_layer)

    h1 = judge_h1(test_acc, seeds, candidate, final_layer, epsilon_1, n_resamples, ci)
    h1p = judge_h1prime(test_acc, seeds, candidate, final_layer, n_resamples, ci)

    oracle_gain = {s: per_seed[s]["recoverability"]["oracle_gain"] for s in seeds}
    r_oracle = {s: per_seed[s]["recoverability"]["R_oracle"] for s in seeds}
    d_js = {s: per_seed[s]["recoverability"]["d_js_class"] for s in seeds}
    rec_count = {s: per_seed[s]["recoverability"]["num_R_oracle"] for s in seeds}
    h2 = judge_h2(oracle_gain, r_oracle, d_js, rec_count, seeds, epsilon_2, n_resamples, ci)

    # convergence tally (how many (seed, layer) runs hit max_epochs)
    n_runs = 0
    n_nonconverged = 0
    for s in seeds:
        for l in [int(x) for x in per_seed[s]["per_layer"]]:
            n_runs += 1
            if not per_seed[s]["per_layer"][str(l)]["converged"]:
                n_nonconverged += 1

    return {
        "family": family_results["family"],
        "candidate_layer": candidate,
        "mean_val_acc_per_layer": {str(l): v for l, v in mean_val.items()},
        "h1": h1,
        "h1_prime": h1p,
        "h2": h2,
        "convergence": {
            "n_runs": n_runs,
            "n_nonconverged": n_nonconverged,
            "nonconverged_fraction": n_nonconverged / n_runs if n_runs else 0.0,
        },
    }


def analyze_ridge_family(
    family_results: dict,
    final_layer: int,
    epsilon_1: float,
    epsilon_2: float,
) -> dict:
    """Point-estimate H1/H1'/H2 judgement for the Ridge family (no CI).

    Ridge is deterministic (single run, seed-agnostic), so there is no
    cross-seed bootstrap. Judgement is a direct comparison of point estimates.
    """
    per_layer = family_results["per_layer"]
    rec = family_results["recoverability"]
    layers = sorted(int(l) for l in per_layer)
    mid_layers = [l for l in layers if l != final_layer]

    # candidate = intermediate layer with highest validation accuracy
    candidate = max(mid_layers, key=lambda l: per_layer[str(l)]["best_val_acc"])
    acc_L = per_layer[str(final_layer)]["test_acc"]
    acc_cand = per_layer[str(candidate)]["test_acc"]
    oracle_gain = rec["oracle_gain"]

    return {
        "family": "ridge",
        "candidate_layer": candidate,
        "acc_L": acc_L,
        "acc_candidate": acc_cand,
        "d1": acc_L - acc_cand,
        "d2": acc_cand - acc_L,
        "oracle_gain": oracle_gain,
        "R_oracle": rec["R_oracle"],
        "d_js_class": rec["d_js_class"],
        "recoverable_count": rec["num_R_oracle"],
        "h1": {
            "supported": bool((acc_L - acc_cand) < epsilon_1),
            "criterion": "Acc_L - Acc_candidate < epsilon_1 (point estimate, no CI)",
            "epsilon_1": epsilon_1,
        },
        "h1_prime": {
            "supported": bool(acc_cand > acc_L),
            "criterion": "Acc_candidate > Acc_L (point estimate, no CI)",
        },
        "h2": {
            "supported": bool(oracle_gain > 0),
            "strong_continuation": bool(oracle_gain > epsilon_2),
            "criterion_statistical": "oracle_gain > 0 (point estimate, no CI)",
            "criterion_strong": f"oracle_gain > epsilon_2 ({epsilon_2})",
            "epsilon_2": epsilon_2,
        },
    }


# --------------------------------------------------------------------------- #
# Cross-family accept protocol (plan §3.2)
# --------------------------------------------------------------------------- #
def cross_family_accept(
    centered: dict,
    ln: dict,
    ridge: dict,
) -> dict:
    """Combine the three families into a per-hypothesis accept verdict.

    Rules (plan §3.2):
    - primary (centered_plain) accepts  -> "supported"
    - all three accept                  -> "very_strong"
    - primary refuses, control(s) accept -> "probe_sensitive"
    - none accept                       -> "not_supported"
    """
    def verdict(primary_sup, ln_sup, ridge_sup):
        if primary_sup and ln_sup and ridge_sup:
            return "very_strong"
        if primary_sup:
            return "supported"
        if ln_sup or ridge_sup:
            return "probe_sensitive"
        return "not_supported"

    hyps = ["h1", "h1_prime", "h2"]
    out = {}
    for h in hyps:
        p = centered[h]["supported"]
        l = ln[h]["supported"]
        r = ridge[h]["supported"]
        out[h] = {
            "primary_centered": p,
            "ln_control": l,
            "ridge_reference": r,
            "verdict": verdict(p, l, r),
        }
    return out

