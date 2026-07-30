"""Tests for src/head.py: init, training, checkpoint selection, reload."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.head import LinearHead, evaluate, predict_with_checkpoint, train_head  # noqa: E402


def make_separable(n_per_class=50, in_dim=8, n_classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in range(n_classes):
        x = torch.randn(n_per_class, in_dim, generator=g) * 0.5
        x[:, c] += 5.0  # class signal
        xs.append(x)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


class TestHead(unittest.TestCase):
    def test_init_xavier_zero_bias(self):
        head = LinearHead(8, 3)
        torch.manual_seed(123)
        # bias must be exactly zero
        self.assertTrue(torch.all(head.linear.bias == 0.0))
        # weights must be non-trivial (Xavier-uniform, not all zero)
        self.assertGreater(head.linear.weight.abs().sum().item(), 0.0)

    def test_train_selects_best_and_saves_checkpoints(self):
        train_x, train_y = make_separable(50, 8, 3, seed=0)
        val_x, val_y = make_separable(10, 8, 3, seed=1)
        with tempfile.TemporaryDirectory() as td:
            res = train_head(
                train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
                seed=17, lr=1e-2, epochs=5, batch_size=32,
                weight_decay=0.01, grad_clip=1.0, in_dim=8, n_classes=3,
                device=torch.device("cpu"), ckpt_dir=Path(td),
            )
            self.assertEqual(len(res["val_history"]), 5)
            self.assertIn(res["best_epoch"], [1, 2, 3, 4, 5])
            # best val acc must beat random (1/3) on separable data
            self.assertGreater(res["best_val_acc"], 1 / 3)
            # every-epoch checkpoints saved
            for e in range(1, 6):
                self.assertTrue((Path(td) / f"epoch_{e:03d}.pt").exists())
            self.assertTrue((Path(td) / "best_checkpoint.json").exists())
            # best satisfies tie-break rule: no other epoch has (higher acc) or
            # (equal acc and lower nll)
            import json
            best = res
            for h in res["val_history"]:
                if h["epoch"] == best["best_epoch"]:
                    continue
                worse = (h["validation_accuracy"] < best["best_val_acc"]) or (
                    h["validation_accuracy"] == best["best_val_acc"]
                    and h["validation_nll"] >= best["best_val_nll"]
                )
                self.assertTrue(worse, f"epoch {h['epoch']} violates tie-break")

    def test_predict_with_checkpoint(self):
        train_x, train_y = make_separable(50, 8, 3, seed=0)
        val_x, val_y = make_separable(10, 8, 3, seed=1)
        with tempfile.TemporaryDirectory() as td:
            res = train_head(
                train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
                seed=17, lr=1e-2, epochs=3, batch_size=32,
                weight_decay=0.01, grad_clip=1.0, in_dim=8, n_classes=3,
                device=torch.device("cpu"), ckpt_dir=Path(td),
            )
            ckpt = Path(td) / f"epoch_{res['best_epoch']:03d}.pt"
            logits = predict_with_checkpoint(ckpt, val_x, 8, 3, torch.device("cpu"))
            self.assertEqual(logits.shape, (30, 3))
            # reloaded head matches the recorded best val acc
            from src.head import LinearHead
            head = LinearHead(8, 3)
            head.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            _, acc, _ = evaluate(head, val_x, val_y)
            self.assertAlmostEqual(acc, res["best_val_acc"], places=5)

    def test_seed_pairing_same_init_across_layers(self):
        """Same seed -> identical head init regardless of layer (paired design)."""
        torch.manual_seed(99)
        h1 = LinearHead(8, 3)
        w1 = h1.linear.weight.clone()
        # reset with same seed
        from src.seeding import seed_all
        seed_all(17)
        h2 = LinearHead(8, 3)
        seed_all(17)
        h3 = LinearHead(8, 3)
        self.assertTrue(torch.allclose(h2.linear.weight, h3.linear.weight))
        self.assertTrue(torch.allclose(h2.linear.bias, h3.linear.bias))


if __name__ == "__main__":
    unittest.main(verbosity=2)
