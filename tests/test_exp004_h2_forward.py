from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h2_forward import (  # noqa: E402
    canonical_cycle_path,
    fixed_head_logits,
    gold_rank_from_logits,
    h2_simulation_counts,
    longest_cached_prefix,
    runtime_hours,
)


class TestH2ForwardHelpers(unittest.TestCase):
    def test_canonical_cycle_extends_past_twelve(self):
        self.assertEqual(canonical_cycle_path(3), (1, 2, 3))
        self.assertEqual(canonical_cycle_path(14), tuple(range(1, 13)) + (1, 2))

    def test_longest_prefix_uses_path_identity(self):
        state2 = torch.tensor([2.0])
        state3 = torch.tensor([3.0])
        cache = {(1, 2): state2, (1, 2, 4): state3}
        depth, state = longest_cached_prefix((1, 2, 4, 5), cache)
        self.assertEqual(depth, 3)
        self.assertIs(state, state3)
        depth, state = longest_cached_prefix((2, 1), cache)
        self.assertEqual((depth, state), (0, None))

    def test_fixed_head_preserves_fp64_artifact_scoring(self):
        hidden = torch.tensor([[[1.0, 2.0], [99.0, 99.0]]], dtype=torch.float16)
        weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
        bias = torch.tensor([0.5, -0.5], dtype=torch.float64)
        logits = fixed_head_logits(hidden, weight, bias)
        self.assertEqual(logits.dtype, torch.float64)
        torch.testing.assert_close(logits, torch.tensor([[1.5, 3.5]], dtype=torch.float64))

    def test_gold_rank_tie_break_matches_argmax(self):
        logits = torch.tensor([3.0, 3.0, 2.0, 1.0], dtype=torch.float64)
        self.assertEqual(gold_rank_from_logits(logits, 0), 1)
        self.assertEqual(gold_rank_from_logits(logits, 1), 2)
        self.assertEqual(gold_rank_from_logits(logits, 2), 3)
        self.assertEqual(int(logits.argmax()), 0)

    def test_full_h2_simulation_count(self):
        counts = h2_simulation_counts()
        self.assertEqual(counts["primary_tuning"], 588_000)
        self.assertEqual(counts["binary_tuning"], 588_000)
        self.assertEqual(counts["primary_test"], 900_000)
        self.assertEqual(counts["binary_test"], 900_000)
        self.assertEqual(counts["random_test"], 900_000)
        self.assertEqual(counts["total"], 3_876_000)

    def test_runtime_conversion(self):
        self.assertAlmostEqual(runtime_hours(3_600_000, 1.0), 1.0)


if __name__ == "__main__":
    unittest.main()
