#!/usr/bin/env python3
"""Derive a reproducible H2 cost envelope from the train-only benchmark."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h1 import atomic_write_json  # noqa: E402
from src.exp004_h2_forward import canonical_cycle_path, runtime_hours  # noqa: E402
from src.exp004_h2_mcts import (  # noqa: E402
    SearchNode,
    apply_action,
    enumerate_legal_actions,
    select_max_ucb,
)


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def common_prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    depth = 0
    for l_value, r_value in zip(left, right):
        if l_value != r_value:
            break
        depth += 1
    return depth


def action_profile(parent_length: int) -> dict[str, Any]:
    parent = canonical_cycle_path(parent_length)
    actions = enumerate_legal_actions(parent)
    suffixes: list[float] = []
    children: list[float] = []
    for action in actions:
        child = apply_action(parent, action)
        suffixes.append(float(len(child) - common_prefix_length(parent, child)))
        children.append(float(len(child)))
    return {
        "parent_length": parent_length,
        "n_legal_actions": len(actions),
        "child_length": distribution(children),
        "recomputed_suffix_length": distribution(suffixes),
    }


def cpu_policy_benchmark(repetitions: int = 10_000) -> dict[str, Any]:
    path = canonical_cycle_path(12)
    actions = enumerate_legal_actions(path)
    rng = np.random.default_rng(17)
    root = SearchNode.root(path)
    children = [
        SearchNode(
            node_id=f"n{index}",
            path=apply_action(path, action),
            parent=root,
            action=action,
            q=1.0 / (index % 150 + 1),
            visits=1,
        )
        for index, action in enumerate(actions)
    ]
    started = time.perf_counter()
    for _ in range(repetitions):
        enumerate_legal_actions(path)
    enumeration_ms = 1000.0 * (time.perf_counter() - started) / repetitions
    started = time.perf_counter()
    for _ in range(repetitions):
        select_max_ucb(
            children,
            rng,
            current_simulation_round=200,
            exploration_c=1.0,
            length_lambda=1.0,
            total_model_layers=12,
        )
    ucb_ms = 1000.0 * (time.perf_counter() - started) / repetitions
    return {
        "platform": platform.processor() or platform.machine(),
        "repetitions": repetitions,
        "children": len(children),
        "enumerate_legal_actions_ms": enumeration_ms,
        "ucb_select_152_children_ms": ucb_ms,
        "conservative_combined_ms_per_simulation": enumeration_ms + ucb_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "artifacts/EXP-20260831-004-h2-throughput"),
    )
    args = parser.parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    raw_path = artifact_root / "results.json"
    output_path = artifact_root / "complexity_analysis.json"
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite existing analysis: {output_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in raw["scenarios"]:
        if row["kind"] in {"sample_wise_full_miss", "sample_wise_prefix_hit"}:
            grouped[(row["kind"], int(row["evaluated_suffix_length"]))].extend(
                map(float, row["durations_ms"])
            )
    per_kind = {
        f"{kind}:L{length}": distribution(values)
        for (kind, length), values in sorted(grouped.items())
    }

    pooled: dict[int, list[float]] = defaultdict(list)
    for (_, length), values in grouped.items():
        pooled[length].extend(values)
    pooled_by_evaluated_layers = {
        str(length): distribution(values) for length, values in sorted(pooled.items())
    }
    measured_layers = np.asarray(sorted(pooled), dtype=np.float64)
    measured_means = np.asarray(
        [pooled_by_evaluated_layers[str(int(length))]["mean"] for length in measured_layers]
    )

    action_profiles = [action_profile(length) for length in range(1, 19)]
    canonical12_suffix = action_profiles[11]["recomputed_suffix_length"]["mean"]
    long18_suffix = action_profiles[17]["recomputed_suffix_length"]["mean"]
    canonical12_gpu_ms = float(np.interp(canonical12_suffix, measured_layers, measured_means))
    long18_gpu_ms = float(np.interp(long18_suffix, measured_layers, measured_means))

    cpu = cpu_policy_benchmark()
    cpu_ms = cpu["conservative_combined_ms_per_simulation"]
    simulations = int(raw["simulation_counts"]["total"])
    startup_samples = 4560
    preparation_mean = float(np.mean([row["timing_ms"]["mean"] for row in raw["preparation"]]))
    full12_mean = per_kind["sample_wise_full_miss:L12"]["mean"]
    startup_hours = runtime_hours(startup_samples, preparation_mean + full12_mean)

    recommended_ms = canonical12_gpu_ms + cpu_ms
    recommended_hours = runtime_hours(simulations, recommended_ms) + startup_hours
    reserve_multiplier = 1.25

    component_counts = raw["simulation_counts"]
    component_hours = {
        name: runtime_hours(int(count), recommended_ms)
        for name, count in component_counts.items()
        if name != "total"
    }

    cache_sensitivity = []
    for hit_rate in (0.0, 0.10, 0.25, 0.50):
        effective_ms = (1.0 - hit_rate) * canonical12_gpu_ms + cpu_ms
        cache_sensitivity.append({
            "exact_result_cache_hit_rate": hit_rate,
            "effective_mean_ms_per_simulation": effective_ms,
            "total_hours_including_startup": runtime_hours(simulations, effective_ms)
            + startup_hours,
        })

    batch_bounds = []
    for row in raw["scenarios"]:
        if row["kind"] != "same_path_batch_upper_bound":
            continue
        mean_per_sample = float(row["timing_ms"]["mean"]) / int(row["batch_size"])
        batch_bounds.append({
            "batch_size": int(row["batch_size"]),
            "mean_ms_per_sample": mean_per_sample,
            "perfect_grouping_total_gpu_hours": runtime_hours(simulations, mean_per_sample),
        })

    envelope = {
        "perfect_same_path_batch64_lower_bound_hours": next(
            row["perfect_grouping_total_gpu_hours"]
            for row in batch_bounds if row["batch_size"] == 64
        ),
        "prefix_cache_action_weighted_canonical12_hours": recommended_hours,
        "prefix_cache_action_weighted_long18_hours": runtime_hours(
            simulations, long18_gpu_ms + cpu_ms
        )
        + startup_hours,
        "no_prefix_full_L12_mean_hours": runtime_hours(
            simulations, full12_mean + cpu_ms
        )
        + startup_hours,
        "no_prefix_full_L18_mean_hours": runtime_hours(
            simulations, per_kind["sample_wise_full_miss:L18"]["mean"] + cpu_ms
        )
        + startup_hours,
        "recommended_reserved_hours": recommended_hours * reserve_multiplier,
        "rounded_operational_allocation_gpu_hours": 10.0,
    }

    analysis = {
        "experiment_id": raw["experiment_id"],
        "source_results": str(raw_path.relative_to(ROOT)),
        "source_git": raw["git"],
        "analysis_git": {
            "commit": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                capture_output=True, text=True, check=True,
            ).stdout.strip(),
            "dirty": bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT,
                capture_output=True, text=True, check=True,
            ).stdout.strip()),
        },
        "method": {
            "total_runtime_uses_arithmetic_mean_not_median": True,
            "prefix_estimate": (
                "uniform legal action at a length-12 parent with every exact FP16 parent prefix cached"
            ),
            "interpolation": "linear interpolation of pooled measured mean wall-time by evaluated layers",
            "cpu_overhead": "legal-action enumeration plus worst-case UCB over all 152 canonical children",
            "reserve_multiplier": reserve_multiplier,
        },
        "aggregate_timing_ms": {
            "per_kind": per_kind,
            "pooled_by_evaluated_layers": pooled_by_evaluated_layers,
        },
        "action_profiles": action_profiles,
        "cpu_policy_benchmark": cpu,
        "startup": {
            "samples": startup_samples,
            "mean_preparation_ms": preparation_mean,
            "mean_canonical_L12_ms": full12_mean,
            "hours": startup_hours,
        },
        "recommended_model": {
            "expected_suffix_layers_at_parent_L12": canonical12_suffix,
            "interpolated_gpu_mean_ms": canonical12_gpu_ms,
            "conservative_cpu_ms": cpu_ms,
            "mean_ms_per_simulation": recommended_ms,
            "total_hours_including_startup": recommended_hours,
            "component_hours": component_hours,
        },
        "cache_sensitivity": cache_sensitivity,
        "same_path_batching_bounds": batch_bounds,
        "runtime_envelope": envelope,
        "interpretive_limits": [
            "the real parent path-length distribution is unknown until a validation-only pilot",
            "exact-result cache hit rate is unknown and is set to zero in the recommended estimate",
            "the 25% reserve covers observed host synchronization spikes and ordinary orchestration overhead",
            "perfect same-path batching is not assumed in the recommended estimate",
        ],
    }
    atomic_write_json(output_path, analysis)

    mlflow_record = json.loads((artifact_root / "mlflow_run.json").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(mlflow_record["tracking_uri"])
    with mlflow.start_run(run_id=mlflow_record["run_id"]):
        mlflow.log_metrics({
            "recommended_mean_ms_per_simulation": recommended_ms,
            "recommended_total_gpu_hours": recommended_hours,
            "recommended_reserved_gpu_hours": envelope["recommended_reserved_hours"],
            "no_prefix_full_L12_mean_hours": envelope["no_prefix_full_L12_mean_hours"],
            "no_prefix_full_L18_mean_hours": envelope["no_prefix_full_L18_mean_hours"],
        })
        mlflow.log_artifact(str(output_path), artifact_path="evidence")
    print(json.dumps({
        "analysis": str(output_path),
        "recommended_model": analysis["recommended_model"],
        "runtime_envelope": envelope,
    }, indent=2))


if __name__ == "__main__":
    main()
