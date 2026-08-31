from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h2_mcts import (  # noqa: E402
    PathAction,
    SearchNode,
    apply_action,
    choose_random_control_proposal,
    choose_tree_policy_mode,
    enumerate_legal_actions,
    expand_and_backpropagate,
    random_tuning_indices,
    select_tuning_grid,
    summarize_correct_alternatives,
    ucb_score,
    wilson_two_sided_lower,
)


class TestH2Actions(unittest.TestCase):
    def test_repeat_r_means_extra_copies(self):
        path = (1, 2, 3, 4)
        action = PathAction("repeat", start=1, width=2, extra_repetitions=3)
        self.assertEqual(apply_action(path, action), (1, 2, 3, 2, 3, 2, 3, 2, 3, 4))

    def test_skip_uses_zero_based_half_open_slice(self):
        self.assertEqual(apply_action((1, 2, 3, 4), PathAction("skip", 1, 2)), (1, 4))

    def test_enumeration_is_deterministic_and_respects_bounds(self):
        first = enumerate_legal_actions((1, 2, 3), min_path_length=1, max_path_length=5)
        second = enumerate_legal_actions((1, 2, 3), min_path_length=1, max_path_length=5)
        self.assertEqual(first, second)
        children = [apply_action((1, 2, 3), action) for action in first]
        self.assertTrue(children)
        self.assertTrue(all(1 <= len(child) <= 5 for child in children))
        self.assertNotIn(PathAction("skip", 0, 3), first)


class TestH2TreeStatistics(unittest.TestCase):
    def test_root_has_no_statistics_and_new_node_starts_at_one_visit(self):
        root = SearchNode.root((1, 2, 3))
        child = expand_and_backpropagate(
            root, PathAction("skip", 1, 1), 0.25, node_id="n1", max_path_length=5
        )
        self.assertIsNone(root.q)
        self.assertIsNone(root.visits)
        self.assertEqual(child.q, 0.25)
        self.assertEqual(child.visits, 1)

    def test_additional_leaf_visit_updates_only_non_root_ancestors(self):
        root = SearchNode.root((1, 2, 3))
        parent = expand_and_backpropagate(
            root, PathAction("repeat", 0, 1, 1), 0.5, node_id="n1", max_path_length=8
        )
        leaf = expand_and_backpropagate(
            parent, PathAction("skip", 2, 1), 0.25, node_id="n2", max_path_length=8
        )
        self.assertEqual((leaf.q, leaf.visits), (0.25, 1))
        self.assertEqual((parent.q, parent.visits), (0.75, 2))
        self.assertEqual((root.q, root.visits), (None, None))

    def test_ucb_uses_current_one_based_simulation_round(self):
        root = SearchNode.root((1, 2, 3))
        child = expand_and_backpropagate(
            root, PathAction("skip", 1, 1), 0.5, node_id="n1", max_path_length=5
        )
        score = ucb_score(
            child,
            current_simulation_round=4,
            exploration_c=2.0,
            length_lambda=0.3,
            total_model_layers=12,
        )
        expected = 0.5 + 2.0 * math.sqrt(math.log(4)) - 0.3 * 2 / 12
        self.assertAlmostEqual(score, expected)

    def test_first_possible_ucb_on_round_two_uses_V_equal_two(self):
        root = SearchNode.root((1, 2, 3))
        child = expand_and_backpropagate(
            root, PathAction("skip", 1, 1), 0.5, node_id="n1", max_path_length=5
        )
        score = ucb_score(
            child,
            current_simulation_round=2,
            exploration_c=1.0,
            length_lambda=0.0,
            total_model_layers=12,
        )
        self.assertAlmostEqual(score, 0.5 + math.sqrt(math.log(2)))

    def test_tree_policy_forces_empty_side_cases(self):
        rng = np.random.default_rng(17)
        self.assertEqual(
            choose_tree_policy_mode(
                explored_child_count=0,
                unexplored_action_count=2,
                explore_probability=0.0,
                rng=rng,
            ),
            "explore",
        )
        self.assertEqual(
            choose_tree_policy_mode(
                explored_child_count=2,
                unexplored_action_count=0,
                explore_probability=1.0,
                rng=rng,
            ),
            "ucb",
        )


class TestH2TuningAndControls(unittest.TestCase):
    def test_tuning_sampling_is_random_stratified_and_reproducible(self):
        flags = [True] * 50 + [False] * 50
        first = random_tuning_indices(flags, np.random.default_rng(17), count_per_group=30)
        second = random_tuning_indices(flags, np.random.default_rng(17), count_per_group=30)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first["canonical_correct"])), 30)
        self.assertEqual(len(set(first["canonical_wrong"])), 30)
        self.assertTrue(all(i < 50 for i in first["canonical_correct"]))
        self.assertTrue(all(i >= 50 for i in first["canonical_wrong"]))

    def test_grid_tie_breaks_on_recovery_then_random(self):
        records = [
            {"c": 0, "lambda": 0, "J": 0.4, "R_recov": 0.3},
            {"c": 1, "lambda": 0, "J": 0.5, "R_recov": 0.2},
            {"c": 5, "lambda": 0, "J": 0.5, "R_recov": 0.4},
            {"c": 10, "lambda": 0, "J": 0.5, "R_recov": 0.4},
        ]
        selected = select_tuning_grid(records, np.random.default_rng(17))
        self.assertIn(selected["c"], {5, 10})
        self.assertEqual(selected["R_recov"], 0.4)

    def test_random_control_samples_nodes_not_unique_paths(self):
        root = SearchNode.root((1, 2, 3))
        duplicate_a = expand_and_backpropagate(
            root, PathAction("skip", 1, 1), 1.0, node_id="a", max_path_length=6
        )
        duplicate_b = SearchNode(
            node_id="b",
            path=duplicate_a.path,
            parent=root,
            action=PathAction("skip", 1, 1),
            q=1.0,
            visits=1,
        )
        visited = [root, duplicate_a, duplicate_b]
        seen = set()
        rng = np.random.default_rng(17)
        for _ in range(100):
            parent, action = choose_random_control_proposal(
                visited, rng, min_path_length=1, max_path_length=6
            )
            seen.add(parent.node_id)
            self.assertIn(
                action,
                enumerate_legal_actions(parent.path, min_path_length=1, max_path_length=6),
            )
        self.assertEqual(seen, {"root", "a", "b"})


class TestH2ReportingRules(unittest.TestCase):
    def test_two_sided_wilson_lower_endpoint(self):
        self.assertEqual(wilson_two_sided_lower(0, 10), 0.0)
        self.assertAlmostEqual(wilson_two_sided_lower(5, 10), 0.236593090512564, places=12)
        self.assertAlmostEqual(wilson_two_sided_lower(10, 10), 0.7224672001371107, places=12)

    def test_all_tied_shortest_paths_returned_and_canonical_excluded(self):
        canonical = [1, 2, 3]
        result = summarize_correct_alternatives(
            [
                {"path": canonical, "correct": True},
                {"path": canonical, "correct": True},
                {"path": [2], "correct": True},
                {"path": [1], "correct": True},
                {"path": [1], "correct": True},
                {"path": [1, 2], "correct": True},
            ],
            canonical,
        )
        self.assertEqual(result["shortest_length"], 1)
        self.assertEqual(result["shortest_paths"], [[1], [2]])
        self.assertTrue(result["has_shorter_than_canonical"])

    def test_no_correct_alternative_is_explicit_absence(self):
        result = summarize_correct_alternatives(
            [{"path": [1, 2, 3], "correct": True}, {"path": [2], "correct": False}],
            [1, 2, 3],
        )
        self.assertEqual(
            result,
            {
                "has_correct_alternative": False,
                "shortest_length": None,
                "shortest_paths": [],
                "has_shorter_than_canonical": False,
            },
        )

    def test_machine_config_carries_user_frozen_semantics(self):
        config = yaml.safe_load((ROOT / "configs/exp004_h2_mcts_v1.yaml").read_text())
        self.assertFalse(config["search"]["ucb"]["root_has_q_or_v"])
        self.assertEqual(config["search"]["ucb"]["explored_non_root_initialization"]["v"], 1)
        self.assertEqual(config["tuning"]["sample_selection"], "stratified_uniform_random_without_replacement")
        self.assertEqual(config["evaluation"]["confidence_interval"]["sidedness"], "two_sided")
        self.assertEqual(config["random_control"]["parent_sampling"], "uniform_over_all_visited_tree_nodes")
        self.assertFalse(config["runtime"]["allow_validation"])
        self.assertFalse(config["runtime"]["allow_test"])


if __name__ == "__main__":
    unittest.main()
