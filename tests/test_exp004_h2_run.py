from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exp004_h2_run import (  # noqa: E402
    aggregate_records,
    atomic_write_json,
    read_json,
    stable_search_seed,
)


def sample(canonical_correct: bool, shorter: bool, recovered: bool):
    return {
        "canonical": {"correct": canonical_correct},
        "shorter_correct": shorter,
        "recovered": recovered,
        "elapsed_seconds": 1.0,
        "cache": {
            "exact_result_cache_hits": 2,
            "unique_paths": 10,
            "transformer_blocks_executed": 50,
        },
    }


class TestH2FullRunnerHelpers(unittest.TestCase):
    def test_atomic_compact_json_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x.json"
            atomic_write_json(path, {"x": [1, 2], "unicode": "路径"})
            self.assertEqual(read_json(path), {"x": [1, 2], "unicode": "路径"})
            self.assertFalse(list(path.parent.glob(".x.json.tmp-*")))

    def test_tuning_aggregate_uses_canonical_strata(self):
        records = [
            sample(True, True, False),
            sample(True, False, False),
            sample(False, False, True),
            sample(False, False, False),
        ]
        metrics = aggregate_records(records)
        self.assertEqual((metrics["n_pos"], metrics["n_neg"]), (2, 2))
        self.assertEqual((metrics["R_short"], metrics["R_recov"]), (0.5, 0.5))
        self.assertEqual(metrics["exact_result_cache_hits"], 8)

    def test_common_random_seed_ignores_grid_parameters(self):
        first = stable_search_seed(17, "validation", "mcts", 123)
        second = stable_search_seed(17, "validation", "mcts", 123)
        random_control = stable_search_seed(17, "validation", "random", 123)
        self.assertEqual(first, second)
        self.assertNotEqual(first, random_control)

    def test_full_config_binds_frozen_semantics_and_gates_test(self):
        config = yaml.safe_load((ROOT / "configs/exp004_h2_full_v2.yaml").read_text())
        semantics = ROOT / config["semantics_config"]
        self.assertEqual(hashlib.sha256(semantics.read_bytes()).hexdigest(), config["semantics_config_sha256"])
        self.assertTrue(config["authorization"]["allow_validation"])
        self.assertTrue(config["authorization"]["allow_test_after_tuning_complete"])
        self.assertEqual(config["dataset"]["padding"], "fixed_to_each_split_observed_max_length")
        audit = config["dataset"]["canonical_validation_audit"]
        self.assertEqual(config["dataset"]["canonical_scan_batch_size"], 1)
        self.assertEqual(audit["expected_operational_correct"], 2704)
        self.assertEqual(audit["source_reference"]["expected_correct"], 2701)
        self.assertEqual(audit["source_reference"]["expected_operational_delta"], 3)
        numerical_audit = ROOT / audit["numerical_audit"]["artifact"]
        self.assertEqual(
            hashlib.sha256(numerical_audit.read_bytes()).hexdigest(),
            audit["numerical_audit"]["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
