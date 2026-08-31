"""Pure operational semantics for the EXP-004 H2 MCTS protocol.

This module deliberately contains no model or dataset access.  It freezes the
tree, action, tuning, interval, and sample-summary rules before any H2
validation/test execution.  Layer identifiers in paths are one-based model
layer IDs; action ``start`` indices are zero-based positions in a path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Literal, Sequence

import numpy as np


ActionKind = Literal["skip", "repeat"]


@dataclass(frozen=True, order=True)
class PathAction:
    """A contiguous skip or repeat edit.

    ``extra_repetitions`` is zero for skip and is the number of *additional*
    copies for repeat.  Thus repeating a width-two block with
    ``extra_repetitions=3`` leaves four copies of the block in the child path.
    """

    kind: ActionKind
    start: int
    width: int
    extra_repetitions: int = 0

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("action start must be non-negative")
        if self.width < 1:
            raise ValueError("action width must be positive")
        if self.kind == "skip" and self.extra_repetitions != 0:
            raise ValueError("skip action cannot have extra repetitions")
        if self.kind == "repeat" and self.extra_repetitions < 1:
            raise ValueError("repeat action requires at least one extra copy")


@dataclass(eq=False)
class SearchNode:
    """One MCTS tree node; identical paths may occur in distinct nodes."""

    node_id: str
    path: tuple[int, ...]
    parent: SearchNode | None = None
    action: PathAction | None = None
    q: float | None = None
    visits: int | None = None
    children: list[SearchNode] = field(default_factory=list)

    @classmethod
    def root(cls, path: Sequence[int], node_id: str = "root") -> SearchNode:
        if not path:
            raise ValueError("root path cannot be empty")
        return cls(node_id=node_id, path=tuple(path))

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path cannot be empty")
        if self.is_root:
            if self.action is not None or self.q is not None or self.visits is not None:
                raise ValueError("root has no action, Q, or visit count")
        elif self.action is None or self.q is None or self.visits is None:
            raise ValueError("non-root node requires action, Q, and visits")


def apply_action(path: Sequence[int], action: PathAction) -> tuple[int, ...]:
    """Apply an action using zero-based, half-open slice semantics."""
    source = tuple(int(layer) for layer in path)
    end = action.start + action.width
    if end > len(source):
        raise ValueError("action block exceeds path")
    block = source[action.start:end]
    if action.kind == "skip":
        return source[:action.start] + source[end:]
    return source[:action.start] + block * (1 + action.extra_repetitions) + source[end:]


def enumerate_legal_actions(
    path: Sequence[int],
    *,
    min_path_length: int = 1,
    max_path_length: int = 18,
    widths: Sequence[int] = (1, 2, 3, 4),
    extra_repetitions: Sequence[int] = (1, 2, 3, 4),
) -> list[PathAction]:
    """Deterministically enumerate every path- and length-legal action."""
    n = len(path)
    if not 1 <= min_path_length <= max_path_length:
        raise ValueError("invalid path-length bounds")
    if not min_path_length <= n <= max_path_length:
        raise ValueError("input path is outside configured length bounds")
    actions: list[PathAction] = []
    for start in range(n):
        for width in widths:
            width = int(width)
            if width < 1 or start + width > n:
                continue
            if n - width >= min_path_length:
                actions.append(PathAction("skip", start, width))
            for repeats in extra_repetitions:
                repeats = int(repeats)
                if repeats >= 1 and n + width * repeats <= max_path_length:
                    actions.append(PathAction("repeat", start, width, repeats))
    return actions


def reciprocal_rank_reward(gold_rank: int) -> float:
    if gold_rank < 1:
        raise ValueError("gold rank is one-based and must be positive")
    return 1.0 / gold_rank


def binary_correctness_reward(is_correct: bool) -> float:
    return float(bool(is_correct))


def expand_and_backpropagate(
    parent: SearchNode,
    action: PathAction,
    reward: float,
    *,
    node_id: str,
    min_path_length: int = 1,
    max_path_length: int = 18,
) -> SearchNode:
    """Add one leaf and update non-root statistics exactly once.

    The newly explored node begins at ``Q=reward, v=1``.  The same reward and
    one additional visit are then propagated through its non-root ancestors.
    The root is an always-selected sentinel and never receives Q/v fields.
    """
    if not math.isfinite(reward):
        raise ValueError("reward must be finite")
    child_path = apply_action(parent.path, action)
    if not min_path_length <= len(child_path) <= max_path_length:
        raise ValueError("action violates path-length bounds")
    child = SearchNode(
        node_id=node_id,
        path=child_path,
        parent=parent,
        action=action,
        q=float(reward),
        visits=1,
    )
    parent.children.append(child)
    ancestor = parent
    while not ancestor.is_root:
        assert ancestor.q is not None and ancestor.visits is not None
        ancestor.q += float(reward)
        ancestor.visits += 1
        ancestor = ancestor.parent  # type: ignore[assignment]
    return child


def ucb_score(
    node: SearchNode,
    *,
    current_simulation_round: int,
    exploration_c: float,
    length_lambda: float,
    total_model_layers: int,
) -> float:
    """Compute equation (10), with V the current one-based sample round."""
    if node.is_root or node.q is None or node.visits is None:
        raise ValueError("UCB is defined only for visited non-root nodes")
    if node.visits < 1 or current_simulation_round < 1:
        raise ValueError("UCB requires v >= 1 and V >= 1")
    if node.visits > current_simulation_round:
        raise ValueError("node visits cannot exceed the current simulation round")
    if exploration_c < 0 or length_lambda < 0 or total_model_layers < 1:
        raise ValueError("invalid UCB parameters")
    exploitation = node.q / node.visits
    exploration = exploration_c * math.sqrt(math.log(current_simulation_round) / node.visits)
    length_penalty = length_lambda * len(node.path) / total_model_layers
    return exploitation + exploration - length_penalty


def select_max_ucb(
    children: Sequence[SearchNode],
    rng: np.random.Generator,
    **ucb_kwargs: Any,
) -> SearchNode:
    """Select maximum UCB, uniformly breaking exact numerical ties."""
    if not children:
        raise ValueError("cannot select UCB from an empty child set")
    scores = np.asarray([ucb_score(child, **ucb_kwargs) for child in children])
    tied = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0.0, atol=1e-12))
    return children[int(rng.choice(tied))]


def choose_tree_policy_mode(
    *,
    explored_child_count: int,
    unexplored_action_count: int,
    explore_probability: float,
    rng: np.random.Generator,
) -> Literal["explore", "ucb"]:
    """Apply the forced-edge cases and then the configured p_exp draw."""
    if explored_child_count < 0 or unexplored_action_count < 0:
        raise ValueError("child/action counts cannot be negative")
    if not 0 <= explore_probability <= 1:
        raise ValueError("explore_probability must lie in [0, 1]")
    if explored_child_count == 0 and unexplored_action_count == 0:
        raise ValueError("node has neither an explored nor unexplored child")
    if explored_child_count == 0:
        return "explore"
    if unexplored_action_count == 0:
        return "ucb"
    return "explore" if rng.random() < explore_probability else "ucb"


def choose_uniform_action(
    actions: Sequence[PathAction], rng: np.random.Generator
) -> PathAction:
    if not actions:
        raise ValueError("cannot sample from an empty action set")
    return actions[int(rng.integers(len(actions)))]


def choose_random_control_proposal(
    visited_nodes: Sequence[SearchNode],
    rng: np.random.Generator,
    *,
    min_path_length: int = 1,
    max_path_length: int = 18,
) -> tuple[SearchNode, PathAction]:
    """Uniform node, then uniform legal-action sampling for the random control.

    Previously chosen actions are not removed.  Each repeated draw is a valid
    budgeted simulation and may create another tree node or cached-path hit.
    """
    if not visited_nodes:
        raise ValueError("random control requires at least the root")
    parent = visited_nodes[int(rng.integers(len(visited_nodes)))]
    actions = enumerate_legal_actions(
        parent.path,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
    )
    if not actions:
        raise ValueError("selected random-control parent has no legal action")
    return parent, choose_uniform_action(actions, rng)


def random_tuning_indices(
    canonical_correct: Sequence[bool],
    rng: np.random.Generator,
    *,
    count_per_group: int = 30,
) -> dict[str, list[int]]:
    """Uniformly sample fixed canonical-correct/wrong validation strata."""
    flags = np.asarray(canonical_correct, dtype=bool)
    correct = np.flatnonzero(flags)
    wrong = np.flatnonzero(~flags)
    if count_per_group < 1 or len(correct) < count_per_group or len(wrong) < count_per_group:
        raise ValueError("insufficient samples in a tuning stratum")
    return {
        "canonical_correct": rng.choice(correct, count_per_group, replace=False).tolist(),
        "canonical_wrong": rng.choice(wrong, count_per_group, replace=False).tolist(),
    }


def select_tuning_grid(
    records: Sequence[dict[str, Any]], rng: np.random.Generator
) -> dict[str, Any]:
    """Maximise J, then R_recov, then uniformly choose an exact tie."""
    if not records:
        raise ValueError("grid records are empty")
    objective = np.asarray([float(record["J"]) for record in records])
    best_j = np.flatnonzero(np.isclose(objective, objective.max(), rtol=0.0, atol=1e-12))
    recoverability = np.asarray([float(records[i]["R_recov"]) for i in best_j])
    best_recovery_local = np.flatnonzero(
        np.isclose(recoverability, recoverability.max(), rtol=0.0, atol=1e-12)
    )
    finalists = best_j[best_recovery_local]
    return dict(records[int(rng.choice(finalists))])


def wilson_two_sided_lower(successes: int, total: int, confidence: float = 0.95) -> float:
    """Lower endpoint of a two-sided Wilson score interval.

    EXP-004 freezes confidence at 0.95.  The explicit argument is retained so
    callers cannot silently mistake this for a one-sided bound.
    """
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("Wilson counts require 0 <= successes <= total and total > 0")
    if confidence != 0.95:
        raise ValueError("EXP-004 H2 freezes a two-sided 95% interval")
    z = 1.959963984540054
    p = successes / total
    z2_over_n = z * z / total
    center = (p + z * z / (2 * total)) / (1 + z2_over_n)
    half_width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half_width /= 1 + z2_over_n
    return max(0.0, center - half_width)


def summarize_correct_alternatives(
    evaluated_paths: Iterable[dict[str, Any]], canonical_path: Sequence[int]
) -> dict[str, Any]:
    """Return all distinct tied-shortest correct *alternative* paths.

    Any tree node whose generated path equals the canonical sequence retains
    canonical identity and is excluded.  Absence is represented explicitly by
    an empty path list and ``None`` length.
    """
    canonical = tuple(int(layer) for layer in canonical_path)
    correct_alternatives = {
        tuple(int(layer) for layer in record["path"])
        for record in evaluated_paths
        if bool(record["correct"]) and tuple(int(layer) for layer in record["path"]) != canonical
    }
    if not correct_alternatives:
        return {
            "has_correct_alternative": False,
            "shortest_length": None,
            "shortest_paths": [],
            "has_shorter_than_canonical": False,
        }
    shortest_length = min(map(len, correct_alternatives))
    shortest = sorted(path for path in correct_alternatives if len(path) == shortest_length)
    return {
        "has_correct_alternative": True,
        "shortest_length": shortest_length,
        "shortest_paths": [list(path) for path in shortest],
        "has_shorter_than_canonical": shortest_length < len(canonical),
    }
