from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h2_search import run_mcts_search, run_random_search  # noqa: E402


class FakeEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, path):
        self.calls += 1
        route = tuple(path)
        correct = len(route) <= 2 and route != (1, 2, 3)
        return {
            "path": list(route),
            "predicted_class": 1 if correct else 0,
            "gold_class": 1,
            "correct": correct,
            "gold_rank": 1 if correct else 2,
            "exact_result_cache_hit": False,
            "cached_prefix_length": 0,
            "executed_transformer_blocks": len(route),
            "new_prefixes": len(route),
        }

    def summary(self):
        return {"evaluation_calls": self.calls}


class TestCompleteSearchEngines(unittest.TestCase):
    def test_mcts_extends_exactly_one_node_per_simulation(self):
        evaluator = FakeEvaluator()
        result = run_mcts_search(
            evaluator.evaluate,
            evaluator.summary,
            canonical_path=(1, 2, 3),
            reward_kind="reciprocal_gold_rank",
            exploration_c=1.0,
            length_lambda=0.1,
            simulations=20,
            explore_probability=0.1,
            min_path_length=1,
            max_path_length=5,
            total_model_layers=3,
            search_seed=17,
        )
        self.assertEqual(len(result["trace"]), 20)
        self.assertEqual([row["simulation_round"] for row in result["trace"]], list(range(1, 21)))
        self.assertNotIn([1, 2, 3], result["alternatives"]["shortest_paths"])
        self.assertEqual(result["cache"]["evaluation_calls"], 21)

    def test_random_control_adds_one_tree_node_per_budget_unit(self):
        evaluator = FakeEvaluator()
        result = run_random_search(
            evaluator.evaluate,
            evaluator.summary,
            canonical_path=(1, 2, 3),
            simulations=20,
            min_path_length=1,
            max_path_length=5,
            search_seed=17,
        )
        self.assertEqual(len(result["trace"]), 20)
        self.assertTrue(all("action" in row for row in result["trace"]))

    def test_search_is_seed_deterministic(self):
        first = FakeEvaluator()
        second = FakeEvaluator()
        kwargs = dict(
            canonical_path=(1, 2, 3),
            reward_kind="binary_correctness",
            exploration_c=0.5,
            length_lambda=0.5,
            simulations=15,
            explore_probability=0.1,
            min_path_length=1,
            max_path_length=5,
            total_model_layers=3,
            search_seed=23,
        )
        left = run_mcts_search(first.evaluate, first.summary, **kwargs)
        right = run_mcts_search(second.evaluate, second.summary, **kwargs)
        self.assertEqual(left["trace"], right["trace"])


if __name__ == "__main__":
    unittest.main()
