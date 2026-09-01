#!/usr/bin/env python3
"""Independent post-run integrity and descriptive audit for EXP-004 H2."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exp004_h2_run import atomic_write_json, git_state, read_json, sha256  # noqa: E402
from src.exp004_h2_mcts import wilson_two_sided_lower  # noqa: E402

VARIANTS = ("primary_rank", "binary_control", "random_control")
CANONICAL = tuple(range(1, 13))


def describe(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "distribution": {str(key): value for key, value in sorted(Counter(values).items())},
    }


def paired_table(left: list[bool], right: list[bool]) -> dict[str, Any]:
    if len(left) != len(right):
        raise AssertionError("paired arrays have different lengths")
    both = sum(a and b for a, b in zip(left, right, strict=True))
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(not a and b for a, b in zip(left, right, strict=True))
    neither = len(left) - both - left_only - right_only
    return {
        "n": len(left),
        "both": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither": neither,
        "left_rate": sum(left) / len(left),
        "right_rate": sum(right) / len(right),
        "paired_rate_difference_left_minus_right": (sum(left) - sum(right)) / len(left),
    }


def load_variant(root: Path, variant: str, summary: dict[str, Any]) -> dict[str, Any]:
    directory = root / "test" / variant
    canonical_correct: list[bool] = []
    shorter: list[bool] = []
    recovered: list[bool] = []
    has_alternative: list[bool] = []
    shortest_lengths: list[int | None] = []
    tied_shortest_counts: list[int] = []
    elapsed: list[float] = []
    source_rows: list[int] = []
    trace_records = 0
    invalid: list[str] = []

    for start in range(0, 4500, 20):
        end = min(start + 20, 4500)
        path = directory / f"shard_{start:05d}_{end:05d}.json"
        if not path.exists():
            invalid.append(f"missing {path.name}")
            continue
        shard = read_json(path)
        if (shard["start"], shard["end"], len(shard["records"])) != (
            start,
            end,
            end - start,
        ):
            invalid.append(f"invalid shard envelope {path.name}")
        for expected_index, record in enumerate(shard["records"], start=start):
            if int(record["split_index"]) != expected_index:
                invalid.append(f"index mismatch {variant}:{expected_index}")
            canonical = record["canonical"]
            alternative = record["alternatives"]
            trace = record["trace"]
            if tuple(canonical["path"]) != CANONICAL:
                invalid.append(f"canonical path mismatch {variant}:{expected_index}")
            if len(trace) != 200:
                invalid.append(f"trace length {variant}:{expected_index}={len(trace)}")
            elif [entry["simulation_round"] for entry in trace] != list(range(1, 201)):
                invalid.append(f"simulation rounds {variant}:{expected_index}")
            for entry in trace:
                route = entry["path"]
                if not 1 <= len(route) <= 18 or any(not 1 <= int(layer) <= 12 for layer in route):
                    invalid.append(f"illegal route {variant}:{expected_index}")
                    break
                expected_reward = (
                    float(bool(entry["correct"]))
                    if variant == "binary_control"
                    else 1.0 / int(entry["gold_rank"])
                )
                if not math.isclose(float(entry["reward"]), expected_reward, abs_tol=1e-15):
                    invalid.append(f"reward mismatch {variant}:{expected_index}")
                    break
            paths = alternative["shortest_paths"]
            length = alternative["shortest_length"]
            if bool(alternative["has_correct_alternative"]) != bool(paths):
                invalid.append(f"alternative presence {variant}:{expected_index}")
            if paths and (
                any(len(path_value) != length for path_value in paths)
                or paths != sorted(paths)
                or len({tuple(path_value) for path_value in paths}) != len(paths)
                or any(tuple(path_value) == CANONICAL for path_value in paths)
            ):
                invalid.append(f"shortest paths {variant}:{expected_index}")
            expected_recovered = (not bool(canonical["correct"])) and bool(
                alternative["has_correct_alternative"]
            )
            expected_shorter = bool(canonical["correct"]) and bool(
                alternative["has_shorter_than_canonical"]
            )
            if bool(record["recovered"]) != expected_recovered:
                invalid.append(f"recovered flag {variant}:{expected_index}")
            if bool(record["shorter_correct"]) != expected_shorter:
                invalid.append(f"shorter flag {variant}:{expected_index}")

            canonical_correct.append(bool(canonical["correct"]))
            shorter.append(bool(record["shorter_correct"]))
            recovered.append(bool(record["recovered"]))
            has_alternative.append(bool(alternative["has_correct_alternative"]))
            shortest_lengths.append(None if length is None else int(length))
            tied_shortest_counts.append(len(paths))
            elapsed.append(float(record["elapsed_seconds"]))
            source_rows.append(int(record["source_row_index"]))
            trace_records += len(trace)

    if invalid:
        raise AssertionError("; ".join(invalid[:20]))
    if (
        len(canonical_correct) != 4500
        or len(set(source_rows)) != 4500
        or source_rows != sorted(source_rows)
    ):
        raise AssertionError(f"incomplete or reordered {variant} records")
    n_pos = sum(canonical_correct)
    n_neg = len(canonical_correct) - n_pos
    n_short = sum(shorter)
    n_recov = sum(recovered)
    recomputed = {
        "n": 4500,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_short": n_short,
        "n_recov": n_recov,
        "R_short": n_short / n_pos,
        "R_recov": n_recov / n_neg,
        "R_short_wilson_two_sided_95_lower": wilson_two_sided_lower(n_short, n_pos),
        "R_recov_wilson_two_sided_95_lower": wilson_two_sided_lower(n_recov, n_neg),
    }
    for key, value in recomputed.items():
        if isinstance(value, float):
            if not math.isclose(float(summary[key]), value, abs_tol=1e-15):
                raise AssertionError(f"summary mismatch {variant}:{key}")
        elif summary[key] != value:
            raise AssertionError(f"summary mismatch {variant}:{key}")

    positive_lengths = [
        int(length)
        for is_positive, success, length in zip(
            canonical_correct, shorter, shortest_lengths, strict=True
        )
        if is_positive and success and length is not None
    ]
    recovered_lengths = [
        int(length)
        for is_positive, success, length in zip(
            canonical_correct, recovered, shortest_lengths, strict=True
        )
        if not is_positive and success and length is not None
    ]
    return {
        "canonical_correct": canonical_correct,
        "shorter": shorter,
        "recovered": recovered,
        "has_alternative": has_alternative,
        "shortest_lengths": shortest_lengths,
        "elapsed": elapsed,
        "source_rows": source_rows,
        "integrity": {
            "shards": 225,
            "records": 4500,
            "trace_records": trace_records,
            "simulations_per_record": 200,
            "all_checks_passed": True,
        },
        "recomputed_summary": recomputed,
        "all_correct_alternative_rate": sum(has_alternative) / 4500,
        "successful_positive_shortest_length": describe(positive_lengths),
        "successful_positive_layers_saved": describe([12 - value for value in positive_lengths]),
        "successful_recovery_shortest_length": describe(recovered_lengths),
        "shortest_tie_count": describe([value for value in tied_shortest_counts if value]),
        "raw_runtime": {
            "mean_seconds_per_search": statistics.fmean(elapsed),
            "total_seconds": sum(elapsed),
            "max_seconds": max(elapsed),
            "max_elapsed_split_index": elapsed.index(max(elapsed)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(ROOT / "artifacts/EXP-20260831-004-h2-mcts-v2"),
    )
    args = parser.parse_args()
    root = Path(args.root)
    manifest = read_json(root / "run_manifest.json")
    final = read_json(root / "final_summary.json")
    config = yaml.safe_load((root / "resolved_full_config.yaml").read_text(encoding="utf-8"))
    semantics = yaml.safe_load(
        (root / "resolved_semantics_config.yaml").read_text(encoding="utf-8")
    )

    if sha256(ROOT / config["semantics_config"]) != manifest["semantics_config_sha256"]:
        raise AssertionError("semantics hash mismatch")
    if sha256(ROOT / semantics["canonical_head"]["artifact"]) != manifest[
        "canonical_head_sha256"
    ]:
        raise AssertionError("canonical head hash mismatch")

    loaded = {
        variant: load_variant(root, variant, final["test"][variant])
        for variant in VARIANTS
    }
    reference = loaded["primary_rank"]["canonical_correct"]
    if any(loaded[variant]["canonical_correct"] != reference for variant in VARIANTS[1:]):
        raise AssertionError("canonical strata differ across variants")
    reference_source_rows = loaded["primary_rank"]["source_rows"]
    if any(loaded[variant]["source_rows"] != reference_source_rows for variant in VARIANTS[1:]):
        raise AssertionError("source-row order differs across variants")
    if sum(reference) != 3986:
        raise AssertionError("unexpected canonical test count")

    positive_indices = [index for index, correct in enumerate(reference) if correct]
    negative_indices = [index for index, correct in enumerate(reference) if not correct]
    comparisons = {}
    for comparator in ("binary_control", "random_control"):
        comparisons[f"primary_vs_{comparator}"] = {
            "shorter_on_canonical_correct": paired_table(
                [loaded["primary_rank"]["shorter"][index] for index in positive_indices],
                [loaded[comparator]["shorter"][index] for index in positive_indices],
            ),
            "recovery_on_canonical_wrong": paired_table(
                [loaded["primary_rank"]["recovered"][index] for index in negative_indices],
                [loaded[comparator]["recovered"][index] for index in negative_indices],
            ),
        }

    grid_files = sorted((root / "tuning").glob("*/grid_*.json"))
    if len(grid_files) != 98:
        raise AssertionError("tuning grid file count mismatch")
    tuning_seconds = sum(read_json(path)["metrics"]["grid_elapsed_seconds"] for path in grid_files)
    pause_audit = read_json(root / "pause_resume_audit.json")
    affected = int(pause_audit["runtime_telemetry_exception"]["affected_split_index"])
    random_elapsed = loaded["random_control"]["elapsed"]
    unaffected_random = [value for index, value in enumerate(random_elapsed) if index != affected]
    replacement = float(
        pause_audit["runtime_telemetry_exception"][
            "median_other_search_elapsed_seconds_in_same_shard"
        ]
    )
    corrected_random_total = sum(unaffected_random) + replacement
    started = datetime.fromisoformat(manifest["started_at"])
    completed = datetime.fromisoformat(final["completed_at"])
    estimated_pause = random_elapsed[affected] - replacement
    test_corrected_seconds = (
        sum(loaded["primary_rank"]["elapsed"])
        + sum(loaded["binary_control"]["elapsed"])
        + corrected_random_total
    )

    compact_variants = {}
    for variant, data in loaded.items():
        compact_variants[variant] = {
            key: value
            for key, value in data.items()
            if key not in {
                "canonical_correct",
                "shorter",
                "recovered",
                "has_alternative",
                "shortest_lengths",
                "elapsed",
                "source_rows",
            }
        }
    payload = {
        "experiment_id": final["experiment_id"],
        "execution_git": manifest["git"],
        "audit_git": git_state(),
        "completed_at": final["completed_at"],
        "integrity": {
            "tuning_grid_files": 98,
            "test_shards": 675,
            "test_records": 13500,
            "test_simulation_trace_records": 2_700_000,
            "canonical_strata_identical_across_variants": True,
            "canonical_correct": 3986,
            "canonical_wrong": 514,
            "canonical_accuracy": 3986 / 4500,
            "all_checks_passed": True,
        },
        "tuning_selection": {
            "primary_rank": {
                "c": final["test"]["primary_rank"]["c"],
                "lambda": final["test"]["primary_rank"]["lambda"],
            },
            "binary_control": {
                "c": final["test"]["binary_control"]["c"],
                "lambda": final["test"]["binary_control"]["lambda"],
            },
        },
        "variants": compact_variants,
        "paired_comparisons": comparisons,
        "acceptance": final["acceptance"],
        "runtime": {
            "wall_hours_including_pause": (completed - started).total_seconds() / 3600,
            "estimated_pause_seconds": estimated_pause,
            "wall_hours_excluding_estimated_pause": (
                (completed - started).total_seconds() - estimated_pause
            )
            / 3600,
            "tuning_grid_seconds": tuning_seconds,
            "test_search_seconds_pause_adjusted": test_corrected_seconds,
            "search_seconds_total_pause_adjusted": tuning_seconds + test_corrected_seconds,
            "random_mean_seconds_per_search_raw": statistics.fmean(random_elapsed),
            "random_mean_seconds_per_search_pause_adjusted": corrected_random_total / 4500,
            "pause_policy": pause_audit["runtime_telemetry_exception"]["policy"],
        },
        "artifact_hashes": {
            "final_summary_sha256": sha256(root / "final_summary.json"),
            "tuning_selection_sha256": sha256(root / "tuning_selection.json"),
            "test_access_gate_sha256": sha256(root / "test_access_gate.json"),
        },
    }
    output = root / "postrun_audit.json"
    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(f"[audit] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
