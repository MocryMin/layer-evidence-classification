"""Pure, deterministic search policy for the frozen EXP-004 H1 discovery.

The policy deliberately sees only path-specific task-head accuracy.  Native
readout accuracy and readability-gap labels are never accepted as inputs to
parent selection or proposal generation.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import numpy as np


SOURCE_ORDER = ("S1", "S2", "S3", "S4", "S5")


def path_key(path: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(layer) for layer in path)


def stable_path_id(path: Iterable[int]) -> str:
    encoded = ",".join(str(int(layer)) for layer in path).encode("ascii")
    return f"p_{hashlib.sha256(encoded).hexdigest()[:16]}"


def parent_probabilities(
    entries: list[dict[str, Any]],
    *,
    temperature: float,
    softmax_weight: float,
) -> np.ndarray:
    """Return the preregistered softmax/uniform mixture over parents."""
    if not entries:
        raise ValueError("parent population is empty")
    if temperature <= 0 or not 0 <= softmax_weight <= 1:
        raise ValueError("invalid parent-mixture parameters")
    scores = np.asarray([entry["task_accuracy_discover"] for entry in entries], dtype=np.float64)
    scaled = (scores - scores.max()) / temperature
    softmax = np.exp(scaled)
    softmax /= softmax.sum()
    uniform = np.full(len(entries), 1.0 / len(entries), dtype=np.float64)
    return softmax_weight * softmax + (1.0 - softmax_weight) * uniform


def _choose_parent(
    entries: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    temperature: float,
    softmax_weight: float,
) -> dict[str, Any]:
    probabilities = parent_probabilities(
        entries, temperature=temperature, softmax_weight=softmax_weight
    )
    return entries[int(rng.choice(len(entries), p=probabilities))]


def _s2_mutation(path: list[int], rng: np.random.Generator) -> tuple[list[int], dict[str, Any]]:
    operations = ["replace"]
    if len(path) > 1:
        operations.extend(["remove", "swap"])
    operation = operations[int(rng.integers(len(operations)))]
    child = path.copy()
    if operation == "remove":
        position = int(rng.integers(len(child)))
        removed = child.pop(position)
        return child, {"operation": operation, "position": position, "old_layer": removed}
    if operation == "replace":
        position = int(rng.integers(len(child)))
        old_layer = child[position]
        choices = [layer for layer in range(1, 29) if layer != old_layer]
        new_layer = choices[int(rng.integers(len(choices)))]
        child[position] = new_layer
        return child, {
            "operation": operation,
            "position": position,
            "old_layer": old_layer,
            "new_layer": new_layer,
        }
    positions = rng.choice(len(child), size=2, replace=False)
    left, right = int(positions[0]), int(positions[1])
    child[left], child[right] = child[right], child[left]
    return child, {"operation": operation, "positions": [left, right]}


def propose_candidate(
    source: str,
    populations: dict[str, list[dict[str, Any]]],
    known_paths: set[tuple[int, ...]],
    rng: np.random.Generator,
    *,
    max_path_length: int,
    temperature: float,
    softmax_weight: float,
    max_attempts: int,
) -> dict[str, Any] | None:
    """Propose one globally novel child; return ``None`` after fixed retries."""
    if source not in SOURCE_ORDER:
        raise ValueError(f"unknown source: {source}")
    for attempt in range(1, max_attempts + 1):
        entries = populations[source]
        if source == "S3" and not entries:
            child = [int(rng.integers(1, 29))]
            parent = None
            mutation = {"operation": "append_from_empty", "new_layer": child[0]}
        else:
            eligible = entries
            if source in {"S1", "S3", "S4", "S5"}:
                eligible = [entry for entry in entries if len(entry["path"]) < max_path_length]
            if not eligible:
                return None
            parent = _choose_parent(
                eligible,
                rng,
                temperature=temperature,
                softmax_weight=softmax_weight,
            )
            base = list(parent["path"])
            if source in {"S1", "S5"}:
                # S1/S5 preserve a terminal layer 28 and insert immediately before it.
                new_layer = int(rng.integers(1, 29))
                child = base[:-1] + [new_layer, 28]
                mutation = {
                    "operation": "insert_before_terminal_28",
                    "position": len(base) - 1,
                    "new_layer": new_layer,
                }
            elif source in {"S3", "S4"}:
                new_layer = int(rng.integers(1, 29))
                child = base + [new_layer]
                mutation = {
                    "operation": "append",
                    "position": len(base),
                    "new_layer": new_layer,
                }
            else:
                child, mutation = _s2_mutation(base, rng)
        key = path_key(child)
        if not 1 <= len(child) <= max_path_length:
            continue
        if any(layer < 1 or layer > 28 for layer in child):
            raise AssertionError("proposal left the 28-layer path alphabet")
        if key in known_paths:
            continue
        return {
            "path_id": stable_path_id(child),
            "source": source,
            "path": child,
            "parent_path_id": None if parent is None else parent["path_id"],
            "mutation": mutation,
            "proposal_attempt": attempt,
        }
    return None


def edit_distance(left: list[int], right: list[int]) -> int:
    """Exact Levenshtein distance for short layer-index paths."""
    previous = list(range(len(right) + 1))
    for i, lval in enumerate(left, start=1):
        current = [i]
        for j, rval in enumerate(right, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (lval != rval))
            )
        previous = current
    return previous[-1]
