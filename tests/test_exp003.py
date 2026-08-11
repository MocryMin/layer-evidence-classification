"""Tests for EXP-003: full-batch early-stopping training, Ridge/OLS, accept protocol."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp003 import (  # noqa: E402
    _ols_fit,
    analyze_ridge_family,
    compute_train_means,
    cross_family_accept,
    fit_ridge_grid,
)
from src.heads import PlainHead  # noqa: E402
from src.probe import train_probe_fullbatch_es  # noqa: E402


def make_separable(n_per_class=80, in_dim=16, n_classes=4, seed=0):
    """Linearly separable data: class signal concentrated on one dimension each."""
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in range(n_classes):
        x = torch.randn(n_per_class, in_dim, generator=g) * 0.3
        x[:, c] += 8.0
        xs.append(x)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


class TestTrainProbeFullbatchES(unittest.TestCase):
    def test_converges_and_returns_best_state(self):
        train_x, train_y = make_separable(80, 16, 4, seed=0)
        val_x, val_y = make_separable(20, 16, 4, seed=1)
        head = PlainHead(16, 4)
        res = train_probe_fullbatch_es(
            head=head, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
            lr=1e-2, weight_decay=0.0, grad_clip=0.0,
            min_epochs=5, max_epochs=200, patience=20, min_delta=1e-4,
            seed=42, device=torch.device("cpu"),
        )
        self.assertTrue(res["converged"], "should early-stop on separable data")
        self.assertEqual(res["stop_reason"], "early_stop")
        self.assertGreater(res["best_val_acc"], 0.95)
        self.assertIsNotNone(res["best_state"])
        self.assertGreater(res["best_epoch"], 0)
        self.assertEqual(len(res["history"]), res["best_epoch"] if res["converged"]
                         else len(res["history"]))

    def test_max_epochs_cap_when_not_converged(self):
        # near-random data: signal too weak to improve -> should hit max_epochs
        g = torch.Generator().manual_seed(0)
        train_x = torch.randn(100, 8, generator=g)
        train_y = torch.randint(0, 3, (100,), generator=g)
        val_x = torch.randn(30, 8, generator=g)
        val_y = torch.randint(0, 3, (30,), generator=g)
        head = PlainHead(8, 3)
        res = train_probe_fullbatch_es(
            head=head, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
            lr=1e-3, weight_decay=0.0, grad_clip=0.0,
            min_epochs=3, max_epochs=10, patience=100, min_delta=1e-4,
            seed=0, device=torch.device("cpu"),
        )
        self.assertFalse(res["converged"])
        self.assertEqual(res["stop_reason"], "max_epochs")
        self.assertEqual(len(res["history"]), 10)

    def test_best_state_loads_and_predicts(self):
        train_x, train_y = make_separable(60, 12, 3, seed=2)
        val_x, val_y = make_separable(15, 12, 3, seed=3)
        head = PlainHead(12, 3)
        res = train_probe_fullbatch_es(
            head=head, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
            lr=1e-2, weight_decay=0.0, grad_clip=0.0,
            min_epochs=3, max_epochs=100, patience=15, min_delta=1e-4,
            seed=7, device=torch.device("cpu"),
        )
        head2 = PlainHead(12, 3)
        head2.load_state_dict(res["best_state"])
        head2.eval()
        with torch.no_grad():
            acc = (head2(val_x).argmax(1) == val_y).float().mean().item()
        self.assertAlmostEqual(acc, res["best_val_acc"], places=4)


class TestOLSAndRidge(unittest.TestCase):
    def test_ols_matches_ridge_tiny_alpha(self):
        # OLS (alpha=0) should match Ridge(alpha=1e-8) predictions on separable data
        rng = np.random.default_rng(0)
        n, d, c = 200, 10, 4
        X = rng.standard_normal((n, d))
        W_true = rng.standard_normal((d, c))
        W_true[:c, :] = np.eye(c) * 5.0
        y = (X @ W_true).argmax(axis=1)
        Xval = rng.standard_normal((50, d))
        yval = (Xval @ W_true).argmax(axis=1)
        Xtest = rng.standard_normal((30, d))

        W_ols = _ols_fit(X, y, c)
        ols_pred = _ols_decision_wrap(W_ols, Xtest)

        from sklearn.linear_model import RidgeClassifier
        clf = RidgeClassifier(alpha=1e-8, fit_intercept=True, solver="svd")
        clf.fit(X, y)
        ridge_pred = clf.predict(Xtest)

        # predictions should agree on the vast majority of samples
        agree = (ols_pred == ridge_pred).mean()
        self.assertGreater(agree, 0.95, f"OLS vs Ridge(alpha=1e-8) agreement {agree:.3f}")

    def test_fit_ridge_grid_selects_best_alpha(self):
        rng = np.random.default_rng(1)
        n, d, c = 300, 12, 5
        X = rng.standard_normal((n, d))
        W_true = np.zeros((d, c))
        W_true[:c, :] = np.eye(c) * 10.0
        y = (X @ W_true).argmax(axis=1)
        Xv = rng.standard_normal((60, d))
        yv = (Xv @ W_true).argmax(axis=1)
        Xt = rng.standard_normal((40, d))
        yt = (Xt @ W_true).argmax(axis=1)

        alphas = [0, 1e-6, 1e-3, 1.0, 100.0]
        res = fit_ridge_grid(X, y, Xv, yv, Xt, alphas, c)
        self.assertIn("best_alpha", res)
        self.assertIn(res["best_alpha"], alphas)
        test_acc = float((res["test_pred"] == yt).mean())
        self.assertGreaterEqual(test_acc, 0.85)
        # alpha=0 should be in per_alpha
        self.assertIn(0, res["per_alpha"])
        # per_alpha should contain only val_acc (no test info, no raw arrays)
        for a, d in res["per_alpha"].items():
            self.assertEqual(set(d.keys()), {"val_acc"})

    def test_ols_fit_shape_and_bias(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 8))
        y = rng.integers(0, 3, 50)
        W = _ols_fit(X, y, 3)
        self.assertEqual(W.shape, (9, 3))  # d+1 rows (bias), C cols


class TestComputeTrainMeans(unittest.TestCase):
    def test_shape_and_centering(self):
        h = torch.randn(100, 12, 768)
        means = compute_train_means(h)
        self.assertEqual(means.shape, (12, 768))
        # centering with the train mean should give ~zero mean per layer
        centered = h - means
        self.assertTrue(torch.allclose(centered.mean(dim=0), torch.zeros(12, 768), atol=1e-5))


class TestCrossFamilyAccept(unittest.TestCase):
    def _mk(self, h1, h1p, h2):
        return {
            "h1": {"supported": h1},
            "h1_prime": {"supported": h1p},
            "h2": {"supported": h2},
        }

    def test_all_accept_is_very_strong(self):
        c = self._mk(True, True, True)
        l = self._mk(True, True, True)
        r = self._mk(True, True, True)
        accept = cross_family_accept(c, l, r)
        for h in ["h1", "h1_prime", "h2"]:
            self.assertEqual(accept[h]["verdict"], "very_strong")

    def test_primary_only_is_supported(self):
        c = self._mk(True, True, True)
        l = self._mk(False, False, False)
        r = self._mk(False, False, False)
        accept = cross_family_accept(c, l, r)
        for h in ["h1", "h1_prime", "h2"]:
            self.assertEqual(accept[h]["verdict"], "supported")

    def test_primary_refuses_controls_accept_is_probe_sensitive(self):
        c = self._mk(False, False, False)
        l = self._mk(True, False, True)
        r = self._mk(False, True, False)
        accept = cross_family_accept(c, l, r)
        for h in ["h1", "h1_prime", "h2"]:
            self.assertEqual(accept[h]["verdict"], "probe_sensitive")

    def test_none_accept_is_not_supported(self):
        c = self._mk(False, False, False)
        l = self._mk(False, False, False)
        r = self._mk(False, False, False)
        accept = cross_family_accept(c, l, r)
        for h in ["h1", "h1_prime", "h2"]:
            self.assertEqual(accept[h]["verdict"], "not_supported")


def _ols_decision_wrap(W, x):
    """Helper for test: decision scores from OLS weight matrix."""
    n = x.shape[0]
    X_aug = np.hstack([x, np.ones((n, 1))])
    return (X_aug @ W).argmax(axis=1)


if __name__ == "__main__":
    unittest.main()
