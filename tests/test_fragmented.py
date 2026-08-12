"""Tests for src/fragmented.py (gr1 shared probe infrastructure)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fragmented import (  # noqa: E402
    classwise_summary,
    judge_collapse,
    layer_variance_stats,
    load_clinc_plus,
    load_wos_46985,
    run_gradient_family_frag,
    run_ridge_family_frag,
)


def _tiny_data(n=200, layers=3, d=16, n_classes=5, seed=0):
    rng = np.random.default_rng(seed)
    hidden = {}
    labels = {}
    for k, n_k in [("train", n), ("validation", 60), ("test", 80)]:
        hidden[k] = rng.normal(size=(n_k, layers, d)).astype(np.float16)
        labels[k] = rng.integers(0, n_classes, size=n_k).astype(np.int16)
    return hidden, labels


class TestLoaders(unittest.TestCase):
    def test_load_clinc_plus_matches_exp003_cache(self):
        """Loader must reproduce EXP-001/003 cache population (rows, labels)."""
        data = load_clinc_plus()
        sizes = [len(v[0]) for v in data.values()]
        self.assertEqual(sizes, [15000, 3000, 4500])
        from src.cache import load_cache
        for split in ["train", "validation", "test"]:
            c = load_cache(
                f"artifacts/EXP-20260729-001/cache/{split}_hidden.safetensors",
                device="cpu", dtype=torch.float32)
            np.testing.assert_array_equal(
                data[split][1], c["labels"].numpy(),
                err_msg=f"labels mismatch on {split}")

    def test_load_wos_46985_split_counts(self):
        data = load_wos_46985()
        self.assertEqual([len(v[0]) for v in data.values()], [30070, 7518, 9397])
        for k, (_, y) in data.items():
            self.assertGreaterEqual(y.min(), 0)
            self.assertLess(y.max(), 134)
            self.assertEqual(len(set(y.tolist())), 134)  # all L2 classes present


class TestVariance(unittest.TestCase):
    def test_stats_and_collapse(self):
        rng = np.random.default_rng(0)
        feat = np.concatenate([rng.normal(size=(100, 3, 16)),
                               np.full((100, 1, 16), 1e-6)], axis=1)
        stats = layer_variance_stats(feat)
        self.assertEqual(len(stats), 4)
        self.assertLess(stats[3]["inter_std"], 1e-3)
        decision = judge_collapse(stats)
        self.assertTrue(decision["collapsed"])
        self.assertEqual(decision["gradient_family"], "centered_plain")
        self.assertEqual(decision["weight_decay"], 0.0)

    def test_healthy(self):
        rng = np.random.default_rng(1)
        stats = layer_variance_stats(rng.normal(size=(100, 3, 16)))
        d2 = judge_collapse(stats)
        self.assertFalse(d2["collapsed"])
        self.assertEqual(d2["gradient_family"], "plain")
        self.assertEqual(d2["weight_decay"], 1e-2)


class TestClasswise(unittest.TestCase):
    def test_parses_tuple_keys(self):
        rec = {"R_lc": {"(1, 0)": {"num": 5, "den": 10, "ratio": 0.5},
                        "(2, 0)": {"num": 8, "den": 10, "ratio": 0.8},
                        "(2, 3)": {"num": 2, "den": 0, "ratio": None}}}
        cs = classwise_summary(rec)
        # class 0 (max R 0.8) covered; class 3 has den 0 -> not counted
        self.assertEqual(cs["coverage"], 1)
        self.assertEqual(cs["n_R_ge_0_5"], 1)
        self.assertEqual(cs["n_R_ge_0_8"], 1)


class TestProbeSuites(unittest.TestCase):
    def test_gradient_family_tiny(self):
        hidden, labels = _tiny_data()
        fam = {"name": "plain", "head_type": "plain", "centering": "none"}
        training = dict(lr=1e-2, weight_decay=1e-2, grad_clip=0.0,
                        min_epochs=1, max_epochs=10, patience=2,
                        min_delta=1e-4, seed=17)
        res = run_gradient_family_frag(hidden, labels, fam, training,
                                       [1, 2, 3], 3, 5, torch.device("cpu"))
        self.assertEqual(set(res["per_layer"]), {"1", "2", "3"})
        for p in res["per_layer"].values():
            self.assertGreaterEqual(p["test_acc"], 0.0)
            self.assertLessEqual(p["test_acc"], 1.0)
            self.assertGreaterEqual(p["n_epochs_run"], 1)
        self.assertEqual(res["test_pred"].shape, (3, 80))
        self.assertIn("oracle_gain", res["recoverability"])
        self.assertIn("R_lc", res["recoverability"])

    def test_centered_and_ln_families_tiny(self):
        hidden, labels = _tiny_data()
        base = dict(lr=1e-2, weight_decay=0.0, grad_clip=0.0,
                    min_epochs=1, max_epochs=10, patience=2,
                    min_delta=1e-4, seed=17)
        for fam in [{"name": "centered_plain", "head_type": "plain",
                     "centering": "train_mean"},
                    {"name": "ln_plain", "head_type": "ln", "centering": "none"}]:
            res = run_gradient_family_frag(hidden, labels, fam, base,
                                           [1, 2, 3], 3, 5, torch.device("cpu"))
            self.assertEqual(set(res["per_layer"]), {"1", "2", "3"})

    def test_ridge_family_tiny(self):
        hidden, labels = _tiny_data()
        res = run_ridge_family_frag(hidden, labels, [0, 1e-4, 1.0],
                                    [1, 2, 3], 3, 5)
        self.assertEqual(set(res["per_layer"]), {"1", "2", "3"})
        for p in res["per_layer"].values():
            self.assertIn(p["best_alpha"], {0.0, 1e-4, 1.0})
            self.assertIn("per_alpha", p)
        self.assertEqual(res["test_pred"].shape, (3, 80))


if __name__ == "__main__":
    unittest.main(verbosity=2)
