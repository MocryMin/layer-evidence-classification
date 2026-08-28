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
from src.exp004_h1_cache import GlobalPrefixCache, prefix_key  # noqa: E402
from src.exp004_h1_cache_policy import (  # noqa: E402
    CacheCostModel,
    GpuPrefixCache,
    select_cache_plan,
)
from scripts.exp004_h1_structured_pilot import structured_path_pool  # noqa: E402
from src.exp004_h1_search import (  # noqa: E402
    SOURCE_ORDER,
    keep_throttled_source_turn,
    parent_probabilities,
    path_key,
    propose_candidate,
    source_temperature,
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


class TestGlobalPrefixCache(unittest.TestCase):
    def test_cross_source_identity_is_only_the_prefix(self):
        self.assertEqual(prefix_key([1, 4, 28]), prefix_key([1, 4, 28]))
        self.assertNotEqual(prefix_key([1, 4, 28]), prefix_key([1, 5, 28]))

    def test_deepest_complete_prefix_and_leaf_lru_eviction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = GlobalPrefixCache(
                index_path=root / "index.sqlite3",
                ssd_root=root / "ssd",
                hdd_root=root / "hdd",
                ssd_cap_bytes=5,
                hdd_cap_bytes=20,
                config_hash="test-hash",
            )
            try:
                for path, payload in (([1], b"1234"), ([1, 2], b"5678")):
                    cache.prepare_write(path)
                    target = cache.shard_path(path, "fit", 0, 1, writing=True)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    cache.finalize_write(path)
                match = cache.deepest_complete_prefix([1, 2, 3])
                self.assertIsNotNone(match)
                self.assertEqual(match["path"], [1, 2])
                statuses = {cache.node([1])["cache_status"], cache.node([1, 2])["cache_status"]}
                self.assertEqual(statuses, {"ssd", "hdd"})
                self.assertGreaterEqual(cache.stats()["registered_nodes"], 3)
            finally:
                cache.close()


class TestCostAwareCacheHierarchy(unittest.TestCase):
    def test_measured_thirty_percent_depth_gates(self):
        model = CacheCostModel()
        self.assertEqual(model.minimum_depth(5, "ssd"), 4)
        self.assertIsNone(model.minimum_depth(5, "hdd"))
        self.assertEqual(model.minimum_depth(36, "hdd"), 34)
        self.assertEqual(model.minimum_depth(5, "gpu"), 2)

    def test_shallow_ssd_prefix_recomputes_but_deep_prefix_pages_into_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = GlobalPrefixCache(
                index_path=root / "index.sqlite3",
                ssd_root=root / "ssd",
                hdd_root=root / "hdd",
                ssd_cap_bytes=100,
                hdd_cap_bytes=100,
                config_hash="test-hash",
            )
            gpu = GpuPrefixCache(
                cap_bytes=100,
                device=torch.device("cpu"),
                config_hash="test-hash",
            )
            try:
                cache.prepare_write([1, 2])
                shallow = cache.shard_path([1, 2], "fit", 0, 1, writing=True)
                shallow.parent.mkdir(parents=True, exist_ok=True)
                shallow.write_bytes(b"x")
                cache.finalize_write([1, 2])
                plan = select_cache_plan(
                    [1, 2, 3, 4, 5],
                    disk_cache=cache,
                    gpu_cache=gpu,
                    cost_model=CacheCostModel(),
                )
                self.assertEqual(plan["action"], "recompute")

                cache.prepare_write([1, 2, 3, 4])
                deep = cache.shard_path([1, 2, 3, 4], "fit", 0, 1, writing=True)
                deep.parent.mkdir(parents=True, exist_ok=True)
                deep.write_bytes(b"y")
                cache.finalize_write([1, 2, 3, 4])
                plan = select_cache_plan(
                    [1, 2, 3, 4, 5],
                    disk_cache=cache,
                    gpu_cache=gpu,
                    cost_model=CacheCostModel(),
                )
                self.assertEqual(plan["action"], "cache")
                self.assertEqual(plan["tier"], "ssd")
                self.assertEqual(plan["path"], [1, 2, 3, 4])
                self.assertGreaterEqual(plan["predicted_fractional_saving"], 0.30)
            finally:
                cache.close()


class TestStructuredPilotPool(unittest.TestCase):
    def test_pool_has_expected_transparent_controls(self):
        pool = structured_path_pool(list(range(1, 29)))
        self.assertEqual(len(pool), 84)
        self.assertEqual(len({item["path_id"] for item in pool}), 84)
        by_id = {item["path_id"]: item for item in pool}
        self.assertEqual(by_id["skip_L01"]["path"], list(range(2, 29)))
        self.assertEqual(by_id["repeat_L28"]["path"][-2:], [28, 28])
        self.assertEqual(by_id["swap_L01_L02"]["path"][:3], [2, 1, 3])


class TestFrozenDiscoveryPolicy(unittest.TestCase):
    def setUp(self):
        canonical = list(range(1, 29))
        canonical_entry = {
            "path_id": "canonical",
            "path": canonical,
            "task_accuracy_discover": 0.9,
        }
        self.populations = {source: [] for source in SOURCE_ORDER}
        self.populations["S1"] = [canonical_entry]
        self.populations["S2"] = [canonical_entry]

    def proposal(self, source, known=None):
        import numpy as np

        return propose_candidate(
            source,
            self.populations,
            set() if known is None else known,
            np.random.default_rng(17),
            max_path_length=36,
            temperature=0.05,
            softmax_weight=0.75,
            max_attempts=100,
        )

    def test_parent_mixture_is_positive_and_normalized(self):
        entries = [
            {"task_accuracy_discover": 0.9},
            {"task_accuracy_discover": 0.1},
        ]
        probabilities = parent_probabilities(entries, temperature=0.05, softmax_weight=0.75)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(all(probabilities > 0))

    def test_source_shapes_match_frozen_generators(self):
        s1 = self.proposal("S1")
        self.assertEqual(len(s1["path"]), 29)
        self.assertEqual(s1["path"][-1], 28)
        s2 = self.proposal("S2")
        self.assertNotEqual(s2["path"], list(range(1, 29)))
        s3 = self.proposal("S3", {path_key([1]), path_key([28])})
        self.assertEqual(len(s3["path"]), 1)

    def test_sourcewise_s1_root_is_two_endpoint_nodes(self):
        self.populations["S1"] = [
            {"path_id": "S1_seed", "path": [1, 28], "task_accuracy_discover": 0.2}
        ]
        child = self.proposal("S1")
        self.assertEqual(len(child["path"]), 3)
        self.assertEqual(child["path"][0], 1)
        self.assertEqual(child["path"][-1], 28)

    def test_duplicate_retry_is_deterministic(self):
        first = self.proposal("S1")
        second = self.proposal("S1")
        self.assertEqual(first, second)

    def test_source_local_temperature_schedule(self):
        initials = {"S1": 0.3, "S2": 1.0, "S3": 0.3, "S4": 0.3, "S5": 0.3}
        self.assertEqual(source_temperature("S1", 0, initials), 0.3)
        self.assertEqual(source_temperature("S2", 10, initials), 1.0)
        self.assertAlmostEqual(source_temperature("S2", 11, initials), 0.52)
        self.assertAlmostEqual(source_temperature("S3", 100, initials), 2.3)

    def test_operational_source_throttle_is_source_local(self):
        import numpy as np

        rng = np.random.default_rng(17)
        self.assertTrue(
            keep_throttled_source_turn(99, rng, threshold=100, keep_probability=0.35)
        )
        draws = [
            keep_throttled_source_turn(100, rng, threshold=100, keep_probability=0.35)
            for _ in range(10_000)
        ]
        self.assertAlmostEqual(sum(draws) / len(draws), 0.35, delta=0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
