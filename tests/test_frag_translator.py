"""Focused algebra tests for the fragmented translator experiment."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from frag_translator import (  # noqa: E402
    apply_T,
    delta_head,
    eff_bias,
    translated_logits,
)


class TestTranslatorAlgebra(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.n, self.d, self.c, self.r = 5, 8, 4, 3
        self.x = torch.randn(self.n, self.d, dtype=torch.float64)
        self.W = torch.randn(self.d, self.c, dtype=torch.float64)
        self.b_c = torch.randn(self.c, dtype=torch.float64)
        self.state = {
            "A": torch.randn(self.r, self.d, dtype=torch.float64),
            "B": torch.randn(self.d, self.r, dtype=torch.float64),
            "b": torch.randn(self.d, dtype=torch.float64),
        }

    def test_feature_and_effective_head_forms_are_identical(self):
        """Translator bias must appear once in either equivalent form."""
        feature_form = translated_logits(
            self.state, self.x, self.W, self.b_c, full_rank=False)

        state_without_bias = {"A": self.state["A"], "B": self.state["B"]}
        W_eff = self.W + delta_head(
            state_without_bias, self.W, full_rank=False)
        b_eff = eff_bias(self.state, self.W, self.b_c, full_rank=False)
        effective_head_form = self.x @ W_eff + b_eff

        torch.testing.assert_close(feature_form, effective_head_form)

    def test_mixing_forms_double_counts_bias(self):
        """Regression guard for the original evaluation bug."""
        correct = translated_logits(
            self.state, self.x, self.W, self.b_c, full_rank=False)
        transformed = apply_T(self.state, self.x, full_rank=False)
        b_eff = eff_bias(self.state, self.W, self.b_c, full_rank=False)
        mixed = transformed @ self.W + b_eff

        expected_extra = (self.state["b"] @ self.W).expand_as(correct)
        torch.testing.assert_close(mixed - correct, expected_extra)

    def test_class_specific_bias_can_change_argmax(self):
        """A sample-independent class vector is not a common scalar shift."""
        x = torch.zeros(1, 2, dtype=torch.float64)
        W = torch.eye(2, dtype=torch.float64)
        b_c = torch.tensor([0.1, 0.0], dtype=torch.float64)
        state = {
            "A": torch.zeros(1, 2, dtype=torch.float64),
            "B": torch.zeros(2, 1, dtype=torch.float64),
            "b": torch.tensor([0.0, 1.0], dtype=torch.float64),
        }

        before = (x @ W + b_c).argmax(dim=1)
        after = translated_logits(state, x, W, b_c, full_rank=False).argmax(dim=1)

        self.assertEqual(before.item(), 0)
        self.assertEqual(after.item(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
