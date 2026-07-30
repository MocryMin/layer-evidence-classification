"""Tests for src/analysis.py: bootstrap CI, candidate selection, hypothesis logic."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import (  # noqa: E402
    bootstrap_ci_mean,
    judge_h1,
    judge_h1prime,
    judge_h2,
    select_candidate_intermediate,
)


class TestBootstrap(unittest.TestCase):
    def test_mean_is_exact(self):
        s = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lo, hi = bootstrap_ci_mean(s, n_resamples=5000, ci=0.95, seed=0)
        self.assertAlmostEqual(mean, 3.0)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)

    def test_ci_shrinks_with_low_variance(self):
        rng = np.random.default_rng(0)
        wide = rng.normal(0, 5.0, size=10)
        tight = rng.normal(0, 0.1, size=10)
        _, wlo, whi = bootstrap_ci_mean(wide, 5000, 0.95, seed=1)
        _, tlo, thi = bootstrap_ci_mean(tight, 5000, 0.95, seed=1)
        self.assertLess(thi - tlo, whi - wlo)

    def test_deterministic_with_seed(self):
        s = [0.1, 0.5, 0.9, -0.3, 0.2]
        r1 = bootstrap_ci_mean(s, 2000, 0.95, seed=42)
        r2 = bootstrap_ci_mean(s, 2000, 0.95, seed=42)
        self.assertEqual(r1, r2)


class TestCandidateSelection(unittest.TestCase):
    def test_picks_highest_mean_val_acc(self):
        seeds = [1, 2, 3]
        final = 12
        # layer 6 has highest mean val acc
        val_acc = {
            (1, 6): 0.9, (2, 6): 0.91, (3, 6): 0.92,   # mean 0.91
            (1, 3): 0.8, (2, 3): 0.81, (3, 3): 0.82,   # mean 0.81
            (1, 12): 0.95, (2, 12): 0.95, (3, 12): 0.95,
        }
        cand, means = select_candidate_intermediate(val_acc, seeds, final)
        self.assertEqual(cand, 6)
        self.assertNotIn(final, means)


class TestHypotheses(unittest.TestCase):
    def setUp(self):
        # 3 seeds; final layer acc 0.90 each; candidate layer acc 0.895 each
        self.seeds = [1, 2, 3]
        self.final = 12
        self.cand = 6
        self.test_acc = {
            (1, 12): 0.900, (2, 12): 0.900, (3, 12): 0.900,
            (1, 6): 0.895, (2, 6): 0.895, (3, 6): 0.895,
        }

    def test_h1_supported_when_ci_upper_below_eps(self):
        # d1 = 0.005 each -> CI_upper well below 0.02 -> supported
        res = judge_h1(self.test_acc, self.seeds, self.cand, self.final, 0.02, 2000, 0.95)
        self.assertTrue(res["supported"])
        self.assertLess(res["d1_ci_upper"], 0.02)

    def test_h1_not_supported_when_gap_large(self):
        ta = {k: (v - 0.05 if k[1] == 6 else v) for k, v in self.test_acc.items()}
        res = judge_h1(ta, self.seeds, self.cand, self.final, 0.02, 2000, 0.95)
        # d1 = 0.055 -> CI_upper > 0.02 -> not supported
        self.assertFalse(res["supported"])

    def test_h1prime_supported_when_candidate_beats_final(self):
        ta = {k: (v + 0.02 if k[1] == 6 else v) for k, v in self.test_acc.items()}
        res = judge_h1prime(ta, self.seeds, self.cand, self.final, 2000, 0.95)
        self.assertTrue(res["supported"])
        self.assertGreater(res["d2_ci_lower"], 0.0)

    def test_h1prime_trend_not_confirmation(self):
        # candidate slightly above on mean but CI crosses 0
        ta = {
            (1, 12): 0.900, (2, 12): 0.900, (3, 12): 0.900,
            (1, 6): 0.920, (2, 6): 0.880, (3, 6): 0.910,  # mean 0.9033 > 0.9 but crosses
        }
        res = judge_h1prime(ta, self.seeds, self.cand, self.final, 5000, 0.95)
        self.assertFalse(res["supported"])

    def test_h2_supported_and_strong(self):
        oracle_gain = {1: 0.07, 2: 0.08, 3: 0.09}
        r_oracle = {1: 0.7, 2: 0.8, 3: 0.9}
        d_js = {1: 0.1, 2: 0.12, 3: 0.11}
        rec = {1: 225, 2: 250, 3: 270}
        res = judge_h2(oracle_gain, r_oracle, d_js, rec, self.seeds, 0.05, 2000, 0.95)
        self.assertTrue(res["supported"])
        self.assertTrue(res["strong_continuation"])
        self.assertGreater(res["g_ci_lower"], 0.0)

    def test_h2_not_supported_when_gain_zero(self):
        oracle_gain = {1: 0.0, 2: 0.0, 3: 0.0}
        r_oracle = {1: 0.0, 2: 0.0, 3: 0.0}
        d_js = {1: float("nan"), 2: float("nan"), 3: float("nan")}
        rec = {1: 0, 2: 0, 3: 0}
        res = judge_h2(oracle_gain, r_oracle, d_js, rec, self.seeds, 0.05, 2000, 0.95)
        self.assertFalse(res["supported"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
