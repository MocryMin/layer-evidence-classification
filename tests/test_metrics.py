"""Tests for src/metrics.py: per-sample metrics, aggregate, recoverability, D_JS."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (  # noqa: E402
    aggregate_metrics,
    classwise_js_divergence,
    confusion_matrix,
    ece,
    macro_f1,
    per_sample_metrics,
    recoverability,
)


class TestPerSample(unittest.TestCase):
    def test_basic_two_samples(self):
        logits = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
        labels = np.array([0, 2])
        ps = per_sample_metrics(logits, labels)
        self.assertEqual(ps["prediction"].tolist(), [0, 2])
        self.assertTrue(np.allclose(ps["correct"], [1.0, 1.0]))
        # probs sum to 1
        self.assertTrue(np.allclose(ps["probs"].sum(axis=1), [1.0, 1.0]))
        # nll for sample 0 = -log(softmax(2,1,0)[0])
        p0 = math.exp(2) / (math.exp(2) + math.exp(1) + math.exp(0))
        self.assertAlmostEqual(ps["nll"][0], -math.log(p0), places=6)
        # logit margin = 1 for both
        self.assertTrue(np.allclose(ps["logit_margin"], [1.0, 1.0]))
        # gold margin = z_y - max other = 2-1 = 1
        self.assertTrue(np.allclose(ps["gold_margin"], [1.0, 1.0]))
        # prob margin = p_top1 - p_top2
        p1 = math.exp(1) / (math.exp(2) + math.exp(1) + math.exp(0))
        self.assertAlmostEqual(ps["probability_margin"][0], p0 - p1, places=6)
        # entropy non-negative
        self.assertTrue((ps["entropy"] >= 0).all())

    def test_gold_margin_when_wrong(self):
        # pred is class 1 but gold is 0 -> gold margin negative
        logits = np.array([[0.0, 5.0, 0.0]])
        labels = np.array([0])
        ps = per_sample_metrics(logits, labels)
        # z_0 = 0, max_other = 5 -> gold_margin = -5
        self.assertAlmostEqual(ps["gold_margin"][0], -5.0)
        self.assertEqual(ps["prediction"][0], 1)
        self.assertTrue(np.allclose(ps["correct"], [0.0]))


class TestAggregate(unittest.TestCase):
    def test_confusion_and_macro_f1(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        preds = np.array([0, 0, 1, 2, 2, 2])
        cm = confusion_matrix(labels, preds, 3)
        # class 0: 2/2 correct; class 1: 1 correct, 1 -> class 2; class 2: 2/2 correct
        self.assertEqual(cm[0, 0], 2)
        self.assertEqual(cm[1, 1], 1)
        self.assertEqual(cm[1, 2], 1)
        self.assertEqual(cm[2, 2], 2)
        # macro F1: class0=1.0, class1: tp=1,fp=0,fn=1 -> 2/(2+0+1)=2/3, class2: tp=2,fp=1,fn=0 -> 4/5
        f1 = macro_f1(cm)
        self.assertAlmostEqual(f1, (1.0 + 2 / 3 + 4 / 5) / 3, places=6)

    def test_ece_perfect_calibration(self):
        # all confidence 1.0 and all correct -> ECE 0
        conf = np.ones(100)
        correct = np.ones(100)
        self.assertAlmostEqual(ece(conf, correct, 10), 0.0, places=8)

    def test_ece_known(self):
        # 2 bins-ish: half confidence 0.5 all wrong, half 1.0 all correct
        conf = np.array([0.5] * 50 + [1.0] * 50)
        correct = np.array([0.0] * 50 + [1.0] * 50)
        # bin for 0.5 -> index 5, bin for 1.0 -> index 9 (clamped)
        # ECE = 0.5*|0-0.5| + 0.5*|1-1| = 0.25
        self.assertAlmostEqual(ece(conf, correct, 10), 0.25, places=6)

    def test_aggregate_runs(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((50, 5))
        labels = rng.integers(0, 5, size=50)
        agg = aggregate_metrics(logits, labels, n_classes=5)
        self.assertIn("accuracy", agg)
        self.assertEqual(agg["confusion_matrix"].shape, (5, 5))


class TestRecoverability(unittest.TestCase):
    """Construct a 12-layer, 10-sample, 3-class case with known R/H/oracle/D_JS."""

    def setUp(self):
        self.labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        L, N = 12, 10
        correct = np.zeros((L, N), dtype=bool)
        # final layer (row 11): correct on 0..5
        correct[11] = [True] * 6 + [False] * 4
        # layer 5 (row 4): correct on 1,2,3,4,5,6,7 (wrong on 0,8,9)
        correct[4] = [False, True, True, True, True, True, True, True, False, False]
        # other mid layers: all wrong (no recovery)
        self.layer_correct = correct
        self.final_idx = 11

    def test_oracle_identity(self):
        res = recoverability(self.layer_correct, None, self.labels, self.final_idx, 3)
        # Acc_L = 0.6, R_oracle = 2/4 = 0.5, Acc_oracle = 0.8
        self.assertAlmostEqual(res["acc_L"], 0.6)
        self.assertAlmostEqual(res["R_oracle"], 0.5)
        self.assertAlmostEqual(res["acc_oracle"], 0.8)
        self.assertAlmostEqual(res["oracle_gain"], 0.2)
        self.assertEqual(res["num_R_oracle"], 2)
        self.assertEqual(res["denom_R_oracle"], 4)
        # identity asserted inside; if we reach here it passed

    def test_R_l_H_l(self):
        res = recoverability(self.layer_correct, None, self.labels, self.final_idx, 3)
        # layer 5 (key=5): R_5 = 2/4 = 0.5, H_5 = 1/6
        r5_num, r5_den, r5_val = res["R_l"][5]
        self.assertEqual((r5_num, r5_den), (2, 4))
        self.assertAlmostEqual(r5_val, 0.5)
        h5_num, h5_den, h5_val = res["H_l"][5]
        self.assertEqual((h5_num, h5_den), (1, 6))
        self.assertAlmostEqual(h5_val, 1 / 6)

    def test_classwise_R_H(self):
        res = recoverability(self.layer_correct, None, self.labels, self.final_idx, 3)
        # R_{5,0}: y=0 & fw = {6,9}, layer5 correct on {6} -> 1/2
        self.assertEqual(res["R_lc"][(5, 0)][0:2], (1, 2))
        # R_{5,1}: y=1 & fw = {7}, correct -> 1/1
        self.assertEqual(res["R_lc"][(5, 1)][0:2], (1, 1))
        # R_{5,2}: y=2 & fw = {8}, wrong -> 0/1
        self.assertEqual(res["R_lc"][(5, 2)][0:2], (0, 1))
        # H_{5,0}: y=0 & fc = {0,1}, layer5 wrong on {0} -> 1/2
        self.assertEqual(res["H_lc"][(5, 0)][0:2], (1, 2))

    def test_d_js_known(self):
        # n_err = [2,1,1], n_rec = [1,1,0]
        res = recoverability(self.layer_correct, None, self.labels, self.final_idx, 3)
        self.assertTrue(np.allclose(res["n_err_c"], [2, 1, 1]))
        self.assertTrue(np.allclose(res["n_rec_c"], [1, 1, 0]))
        # manual D_JS
        e = np.array([2, 1, 1]) / 4
        r = np.array([1, 1, 0]) / 2
        m = 0.5 * (e + r)

        def kl(p, q):
            mask = p > 0
            return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

        expected = (0.5 * kl(e, m) + 0.5 * kl(r, m)) / math.log(2)
        self.assertAlmostEqual(res["d_js_class"], expected, places=6)
        self.assertGreater(res["d_js_class"], 0.0)
        self.assertLess(res["d_js_class"], 1.0)

    def test_d_js_identical_distributions(self):
        # if recoverable errors mirror final-error distribution, D_JS = 0
        n_err = np.array([2, 1, 1])
        n_rec = np.array([4, 2, 2])  # same proportions
        self.assertAlmostEqual(classwise_js_divergence(n_err, n_rec), 0.0, places=8)

    def test_d_js_na_when_no_errors(self):
        self.assertTrue(math.isnan(classwise_js_divergence(np.zeros(3), np.array([1, 0, 0]))))
        self.assertTrue(math.isnan(classwise_js_divergence(np.array([1, 0, 0]), np.zeros(3))))

    def test_zero_denominator_is_NA(self):
        # all samples correct at final layer -> no final-wrong -> R_l denominator 0 -> NA
        correct = np.zeros((12, 4), dtype=bool)
        correct[11] = True  # final correct on all
        res = recoverability(correct, None, np.array([0, 0, 1, 1]), 11, 2)
        for layer, (num, den, val) in res["R_l"].items():
            self.assertEqual(den, 0)
            self.assertTrue(math.isnan(val))
        self.assertTrue(math.isnan(res["R_oracle"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
