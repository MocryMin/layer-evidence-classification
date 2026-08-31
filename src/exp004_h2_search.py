"""Complete in-memory sample-wise search engines for EXP-004 H2."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Sequence

import numpy as np

from .exp004_h2_mcts import (
    PathAction,
    SearchNode,
    apply_action,
    choose_random_control_proposal,
    choose_tree_policy_mode,
    choose_uniform_action,
    enumerate_legal_actions,
    expand_and_backpropagate,
    reciprocal_rank_reward,
    select_max_ucb,
    summarize_correct_alternatives,
)


RewardKind = Literal["reciprocal_gold_rank", "binary_correctness"]


class SearchInterrupted(RuntimeError):
    pass


def _reward(evaluation: dict[str, Any], reward_kind: RewardKind) -> float:
    if reward_kind == "reciprocal_gold_rank":
        return reciprocal_rank_reward(int(evaluation["gold_rank"]))
    if reward_kind == "binary_correctness":
        return float(bool(evaluation["correct"]))
    raise ValueError(f"unknown reward kind: {reward_kind}")


def _action_record(action: PathAction) -> dict[str, Any]:
    return {
        "kind": action.kind,
        "start": action.start,
        "width": action.width,
        "extra_repetitions": action.extra_repetitions,
    }


def _outcome(
    canonical: dict[str, Any],
    trace: list[dict[str, Any]],
    canonical_path: Sequence[int],
    evaluator_summary: dict[str, Any],
    *,
    search_kind: str,
    search_seed: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    alternatives = summarize_correct_alternatives(trace, canonical_path)
    return {
        "search_kind": search_kind,
        "search_seed": search_seed,
        "canonical": canonical,
        "alternatives": alternatives,
        "recovered": (not bool(canonical["correct"]))
        and alternatives["has_correct_alternative"],
        "shorter_correct": bool(canonical["correct"])
        and alternatives["has_shorter_than_canonical"],
        "trace": trace,
        "cache": evaluator_summary,
        "elapsed_seconds": elapsed_seconds,
    }


def run_mcts_search(
    evaluate: Callable[[Sequence[int]], dict[str, Any]],
    evaluator_summary: Callable[[], dict[str, Any]],
    *,
    canonical_path: Sequence[int],
    reward_kind: RewardKind,
    exploration_c: float,
    length_lambda: float,
    simulations: int,
    explore_probability: float,
    min_path_length: int,
    max_path_length: int,
    total_model_layers: int,
    search_seed: int,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    import time

    if simulations < 1:
        raise ValueError("simulations must be positive")
    started = time.perf_counter()
    rng = np.random.default_rng(search_seed)
    canonical_route = tuple(int(layer) for layer in canonical_path)
    root = SearchNode.root(canonical_route)
    canonical = evaluate(canonical_route)
    action_cache: dict[tuple[int, ...], list[PathAction]] = {}
    trace: list[dict[str, Any]] = []

    for simulation_round in range(1, simulations + 1):
        if should_stop is not None and should_stop():
            raise SearchInterrupted("stop requested")
        node = root
        tree_depth = 0
        while True:
            actions = action_cache.get(node.path)
            if actions is None:
                actions = enumerate_legal_actions(
                    node.path,
                    min_path_length=min_path_length,
                    max_path_length=max_path_length,
                )
                action_cache[node.path] = actions
            explored_actions = {child.action for child in node.children}
            unexplored = [action for action in actions if action not in explored_actions]
            mode = choose_tree_policy_mode(
                explored_child_count=len(node.children),
                unexplored_action_count=len(unexplored),
                explore_probability=explore_probability,
                rng=rng,
            )
            if mode == "ucb":
                node = select_max_ucb(
                    node.children,
                    rng,
                    current_simulation_round=simulation_round,
                    exploration_c=exploration_c,
                    length_lambda=length_lambda,
                    total_model_layers=total_model_layers,
                )
                tree_depth += 1
                continue

            action = choose_uniform_action(unexplored, rng)
            child_path = apply_action(node.path, action)
            evaluation = evaluate(child_path)
            reward = _reward(evaluation, reward_kind)
            child = expand_and_backpropagate(
                node,
                action,
                reward,
                node_id=f"n{simulation_round}",
                min_path_length=min_path_length,
                max_path_length=max_path_length,
            )
            trace.append(
                {
                    "simulation_round": simulation_round,
                    "tree_depth": tree_depth + 1,
                    "parent_node_id": node.node_id,
                    "node_id": child.node_id,
                    "action": _action_record(action),
                    "reward": reward,
                    **evaluation,
                }
            )
            break

    return _outcome(
        canonical,
        trace,
        canonical_route,
        evaluator_summary(),
        search_kind=f"mcts_{reward_kind}",
        search_seed=search_seed,
        elapsed_seconds=time.perf_counter() - started,
    )


def run_random_search(
    evaluate: Callable[[Sequence[int]], dict[str, Any]],
    evaluator_summary: Callable[[], dict[str, Any]],
    *,
    canonical_path: Sequence[int],
    simulations: int,
    min_path_length: int,
    max_path_length: int,
    search_seed: int,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    import time

    started = time.perf_counter()
    rng = np.random.default_rng(search_seed)
    canonical_route = tuple(int(layer) for layer in canonical_path)
    root = SearchNode.root(canonical_route)
    canonical = evaluate(canonical_route)
    visited: list[SearchNode] = [root]
    trace: list[dict[str, Any]] = []
    for simulation_round in range(1, simulations + 1):
        if should_stop is not None and should_stop():
            raise SearchInterrupted("stop requested")
        parent, action = choose_random_control_proposal(
            visited,
            rng,
            min_path_length=min_path_length,
            max_path_length=max_path_length,
        )
        child_path = apply_action(parent.path, action)
        evaluation = evaluate(child_path)
        reward = reciprocal_rank_reward(int(evaluation["gold_rank"]))
        child = SearchNode(
            node_id=f"n{simulation_round}",
            path=child_path,
            parent=parent,
            action=action,
            q=reward,
            visits=1,
        )
        parent.children.append(child)
        visited.append(child)
        trace.append(
            {
                "simulation_round": simulation_round,
                "tree_depth": None,
                "parent_node_id": parent.node_id,
                "node_id": child.node_id,
                "action": _action_record(action),
                "reward": reward,
                **evaluation,
            }
        )
    return _outcome(
        canonical,
        trace,
        canonical_route,
        evaluator_summary(),
        search_kind="uniform_node_then_action_random",
        search_seed=search_seed,
        elapsed_seconds=time.perf_counter() - started,
    )
