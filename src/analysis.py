"""Bootstrap confidence intervals and hypothesis judgement for EXP-20260729-001.

Seed bootstrap (10 seeds, 10000 resamples, percentile 95% CI):
- paired for d1(l), d2(l), g (each seed contributes one paired difference);
- ordinary for D_JS^class.

Candidate intermediate layer for H1 / H1' is selected by mean *validation*
accuracy; all-layer test results stay exploratory (AgentProtocol §7).
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci_mean(
    samples,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean of ``samples``.

    Returns ``(mean, lower, upper)``.
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = samples[idx].mean(axis=1)
    alpha = 1.0 - ci
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(samples.mean()), lo, hi


def select_candidate_intermediate(
    val_acc: dict,
    seeds: list[int],
    final_layer: int,
) -> tuple[int, dict]:
    """Pick the intermediate layer with the highest mean validation accuracy.

    ``val_acc`` maps ``(seed, layer) -> val accuracy``. Returns
    ``(candidate_layer, {layer: mean_val_acc})``.
    """
    layers = sorted({l for (_, l) in val_acc if l != final_layer})
    means = {l: float(np.mean([val_acc[(s, l)] for s in seeds])) for l in layers}
    candidate = max(means, key=means.get)
    return candidate, means


def judge_h1(
    test_acc: dict,
    seeds: list[int],
    candidate: int,
    final_layer: int,
    epsilon_1: float,
    n_resamples: int,
    ci: float,
) -> dict:
    """H1: intermediate layer non-inferior to final. Supported if CI_upper(d1) < eps1."""
    d1 = np.array([test_acc[(s, final_layer)] - test_acc[(s, candidate)] for s in seeds])
    mean, lo, hi = bootstrap_ci_mean(d1, n_resamples, ci)
    return {
        "hypothesis": "H1",
        "candidate_layer": candidate,
        "d1_per_seed": d1.tolist(),
        "d1_mean": mean,
        "d1_ci_lower": lo,
        "d1_ci_upper": hi,
        "epsilon_1": epsilon_1,
        "supported": bool(hi < epsilon_1),
        "criterion": "CI_upper(d1) < epsilon_1",
    }


def judge_h1prime(
    test_acc: dict,
    seeds: list[int],
    candidate: int,
    final_layer: int,
    n_resamples: int,
    ci: float,
) -> dict:
    """H1': intermediate layer superior to final. Supported if CI_lower(d2) > 0."""
    d2 = np.array([test_acc[(s, candidate)] - test_acc[(s, final_layer)] for s in seeds])
    mean, lo, hi = bootstrap_ci_mean(d2, n_resamples, ci)
    return {
        "hypothesis": "H1'",
        "candidate_layer": candidate,
        "d2_per_seed": d2.tolist(),
        "d2_mean": mean,
        "d2_ci_lower": lo,
        "d2_ci_upper": hi,
        "supported": bool(lo > 0),
        "trend": bool(mean > 0 and lo <= 0),
        "criterion": "CI_lower(d2) > 0",
    }


def judge_h2(
    oracle_gain: dict,
    r_oracle: dict,
    d_js: dict,
    recoverable_count: dict,
    seeds: list[int],
    epsilon_2: float,
    n_resamples: int,
    ci: float,
) -> dict:
    """H2: corrective recoverability. Statistical support if CI_lower(g) > 0.

    Strong continuation criterion (not the statistical definition): mean(g) > eps2.
    """
    g = np.array([oracle_gain[s] for s in seeds])
    g_mean, g_lo, g_hi = bootstrap_ci_mean(g, n_resamples, ci)
    djs = np.array([d_js[s] for s in seeds])
    djs_mean, djs_lo, djs_hi = bootstrap_ci_mean(djs, n_resamples, ci)
    r_arr = np.array([r_oracle[s] for s in seeds])
    rec_arr = np.array([recoverable_count[s] for s in seeds])
    return {
        "hypothesis": "H2",
        "g_per_seed": g.tolist(),
        "g_mean": g_mean,
        "g_ci_lower": g_lo,
        "g_ci_upper": g_hi,
        "supported": bool(g_lo > 0),
        "epsilon_2": epsilon_2,
        "strong_continuation": bool(g_mean > epsilon_2),
        "criterion_statistical": "CI_lower(g) > 0",
        "criterion_strong": "mean(g) > epsilon_2",
        "R_oracle_mean": float(np.mean(r_arr)),
        "R_oracle_std": float(np.std(r_arr)),
        "recoverable_count_mean": float(np.mean(rec_arr)),
        "recoverable_count_std": float(np.std(rec_arr)),
        "d_js_mean": djs_mean,
        "d_js_std": float(np.std(djs)),
        "d_js_ci_lower": djs_lo,
        "d_js_ci_upper": djs_hi,
    }
