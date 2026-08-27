from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h1 import (  # noqa: E402
    ArcExample,
    DeadlineController,
    atomic_torch_save,
    atomic_write_json,
    chance_accuracy,
    format_arc_prompt,
    make_fit_discover_indices,
    masked_accuracy,
    stratified_fold_ids,
    valid_choice_mask,
)


class TestArcProtocol(unittest.TestCase):
    def test_split_is_complete_disjoint_and_deterministic(self):
        first = make_fit_discover_indices(2251, 1750, 17)
        second = make_fit_discover_indices(2251, 1750, 17)
        self.assertEqual(first, second)
        self.assertEqual(len(first["fit"]), 1750)
        self.assertEqual(len(first["discover"]), 501)
        self.assertFalse(set(first["fit"]) & set(first["discover"]))

    def test_prompt_relabels_numeric_options_by_position(self):
        example = ArcExample("x", "Two plus two?", ("3", "4", "5"), 1)
        text = format_arc_prompt(
            example,
            {
                "instruction": "Reply with a letter.",
                "question_prefix": "Question:",
                "answer_prefix": "Answer:",
            },
        )
        self.assertIn("A. 3", text)
        self.assertIn("B. 4", text)
        self.assertIn("C. 5", text)
        self.assertNotIn("D.", text)

    def test_invalid_choices_never_win(self):
        counts = torch.tensor([3, 4, 5])
        mask = valid_choice_mask(counts)
        logits = torch.tensor(
            [[0.0, 1.0, 2.0, 99.0, 100.0], [0.0, 1.0, 2.0, 3.0, 100.0], [0, 1, 2, 3, 4]],
            dtype=torch.float32,
        )
        labels = torch.tensor([2, 3, 4])
        self.assertEqual(masked_accuracy(logits, labels, mask), 1.0)

    def test_chance_uses_per_item_choice_count(self):
        self.assertAlmostEqual(chance_accuracy([3, 4, 5]), (1 / 3 + 1 / 4 + 1 / 5) / 3)

    def test_stratified_folds_are_deterministic_and_complete(self):
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        counts = [4] * len(labels)
        first = stratified_fold_ids(labels, counts, 3, 17)
        second = stratified_fold_ids(labels, counts, 3, 17)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {0, 1, 2})
        self.assertEqual([first.count(fold) for fold in range(3)], [4, 4, 4])


class TestAtomicArtifactsAndDeadline(unittest.TestCase):
    def test_atomic_writes_are_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_write_json(root / "value.json", {"b": 2, "a": 1})
            self.assertEqual(json.loads((root / "value.json").read_text()), {"a": 1, "b": 2})
            tensor = torch.arange(4)
            atomic_torch_save(root / "value.pt", {"tensor": tensor})
            torch.testing.assert_close(torch.load(root / "value.pt")["tensor"], tensor)
            self.assertFalse(list(root.glob(".*.tmp-*")))

    def test_deadline_requires_timezone(self):
        with self.assertRaises(ValueError):
            DeadlineController("2099-01-01T08:00:00", 10)

    def test_future_deadline_has_soft_reserve(self):
        stop = datetime.now(timezone.utc) + timedelta(hours=1)
        deadline = DeadlineController(stop.isoformat(), 10)
        self.assertAlmostEqual(
            (deadline.hard_stop - deadline.soft_stop).total_seconds(), 600, delta=0.1
        )
        self.assertGreater(deadline.seconds_to_soft_stop(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
