#!/usr/bin/env python3
"""Register completed EXP-004 H1 qualification and discovery artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h1 import atomic_write_json  # noqa: E402


TRACKING_URI = "sqlite:///" + str((ROOT / "mlruns.db").resolve())
EXPERIMENT = "EXP-004-readability-qualification"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def register(
    run_name: str,
    artifact_root: Path,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: list[Path],
    tags: dict[str, str] | None = None,
) -> str:
    record_path = artifact_root / "mlflow_run.json"
    if record_path.exists():
        return read_json(record_path)["run_id"]
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags(
            tags
            or {
                "experiment_phase": "engineering_qualification",
                "official_hypothesis_evidence": "false",
                "validation_accessed": "false",
                "test_accessed": "false",
            }
        )
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(metrics)
        for path in artifact_paths:
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path="evidence")
        run_id = active.info.run_id
    atomic_write_json(
        record_path,
        {
            "run_id": run_id,
            "run_name": run_name,
            "tracking_uri": TRACKING_URI,
            "experiment": EXPERIMENT,
        },
    )
    return run_id


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    qualification = ROOT / "artifacts/EXP-20260827-004-h1-qualification"
    q_manifest = read_json(qualification / "run_manifest.json")
    q_metrics = read_json(qualification / "qualification_metrics.json")
    q_run = register(
        "EXP-20260827-004-h1-qualification",
        qualification,
        {
            "git_commit": q_manifest["git"]["commit"],
            "config_hash": q_manifest["config_hash"],
            "model": "Llama-3.2-3B-Instruct",
            "dataset": "ARC-Easy official-train only",
            "fit_size": q_metrics["fit_size"],
            "discover_size": q_metrics["discover_size"],
        },
        {
            "canonical_native_accuracy_fit": q_metrics["canonical_native_accuracy_fit"],
            "canonical_native_accuracy_discover": q_metrics[
                "canonical_native_accuracy_discover"
            ],
            "provisional_task_accuracy_discover": q_metrics["canonical_task_head"][
                "eval_accuracy"
            ],
            "equivalence_feature_max_abs": q_metrics["equivalence"][
                "feature_max_abs"
            ],
            "equivalence_label_logit_max_abs": q_metrics["equivalence"][
                "label_logit_max_abs"
            ],
            "forward_batch_seconds_p95": q_metrics["forward_batch_seconds_p95"],
            "peak_cuda_gib": q_metrics["peak_cuda_bytes"] / 2**30,
        },
        [
            qualification / "run_manifest.json",
            qualification / "resolved_config.yaml",
            qualification / "split_indices.json",
            qualification / "canonical_equivalence.json",
            qualification / "qualification_metrics.json",
        ],
    )

    head = ROOT / "artifacts/EXP-20260827-004-h1-head-qualification-v2"
    h_manifest = read_json(head / "run_manifest.json")
    h_summary = read_json(head / "head_qualification.json")
    h_final = h_summary["refit_full_D_fit_evaluate_D_discover"]
    h_run = register(
        "EXP-20260827-004-h1-head-qualification-v2",
        head,
        {
            "git_commit": h_manifest["git"]["commit"],
            "config_hash": h_manifest["config_hash"],
            "selection_data": h_summary["selection_data"],
            "n_folds": 5,
            "selected_l2": h_summary["selected_l2"],
        },
        {
            "selected_cv_mean_accuracy": h_summary["selected_cv_mean_accuracy"],
            "selected_cv_std_accuracy": h_summary["selected_cv_std_accuracy"],
            "selected_cv_mean_cross_entropy": h_summary[
                "selected_cv_mean_cross_entropy"
            ],
            "refit_train_accuracy": h_final["train_accuracy"],
            "refit_discover_accuracy": h_final["eval_accuracy"],
            "refit_discover_cross_entropy": h_final["eval_cross_entropy"],
        },
        [
            head / "run_manifest.json",
            head / "head_qualification.json",
            head / "folds.json",
            ROOT / "configs/exp004_h1_head_qualification_v2.yaml",
        ],
    )

    pilot = ROOT / "artifacts/EXP-20260827-004-h1-structured-pilot"
    p_manifest = read_json(pilot / "run_manifest.json")
    p_summary = read_json(pilot / "pilot_summary.json")
    witness = read_json(pilot / "results/repeat_L28.json")
    p_run = register(
        "EXP-20260827-004-h1-structured-pilot",
        pilot,
        {
            "git_commit": p_manifest["git"]["commit"],
            "config_hash": p_manifest["config_hash"],
            "path_pool_hash": p_manifest["path_pool_hash"],
            "path_sources": "canonical,single_skip,single_repeat,adjacent_swap",
            "task_head_l2": 0.3,
        },
        {
            "n_paths": p_summary["n_paths"],
            "n_good": p_summary["n_good"],
            "n_readability_collapse": p_summary["n_readability_collapse"],
            "p_gap_among_good": p_summary["p_gap_among_good"],
            "canonical_task_accuracy_discover": p_summary[
                "canonical_task_accuracy_discover"
            ],
            "canonical_native_accuracy_discover": p_summary[
                "canonical_native_accuracy_discover"
            ],
            "witness_task_accuracy_discover": witness["task_accuracy_discover"],
            "witness_native_accuracy_discover": witness["native_accuracy_discover"],
            "witness_native_gap": witness["native_gap_from_canonical"],
            "witness_relative_native_gap": witness["relative_native_gap"],
            "peak_cuda_gib": p_summary["peak_cuda_bytes"] / 2**30,
        },
        [
            pilot / "run_manifest.json",
            pilot / "path_pool.json",
            pilot / "pilot_summary.json",
            pilot / "results/repeat_L28.json",
            pilot / "audits/repeat_L28.json",
            pilot / "audits/repeat_L28_batch8.json",
            ROOT / "configs/exp004_h1_structured_pilot.yaml",
        ],
    )

    legacy = ROOT / "artifacts/EXP-20260827-004-h1-discovery"
    legacy_manifest = read_json(legacy / "run_manifest.json")
    legacy_summary = read_json(legacy / "discovery_summary.json")
    legacy_metrics: dict[str, float] = {
        "n_discovered_candidates": legacy_summary["n_discovered_candidates"],
        "n_good": legacy_summary["n_good"],
        "n_readability_collapse_among_good": legacy_summary[
            "n_readability_collapse_among_good"
        ],
        "p_gap_among_good": legacy_summary["p_gap_among_good"],
        "canonical_task_accuracy_discover": legacy_summary[
            "canonical_task_accuracy_discover"
        ],
        "canonical_native_accuracy_discover": legacy_summary[
            "canonical_native_accuracy_discover"
        ],
        "gpu_hours_used": legacy_summary["gpu_hours_used"],
    }
    for source, values in legacy_summary["by_source"].items():
        legacy_metrics[f"{source}_n"] = values["n"]
        legacy_metrics[f"{source}_n_good"] = values["n_good"]
        legacy_metrics[f"{source}_n_gap"] = values["n_gap_among_good"]
        if values["p_gap_among_good"] is not None:
            legacy_metrics[f"{source}_p_gap"] = values["p_gap_among_good"]
    legacy_run = register(
        "EXP-20260827-004-h1-discovery",
        legacy,
        {
            "git_commit_at_freeze": legacy_manifest["git_at_freeze"]["commit"],
            "config_hash": legacy_manifest["config_hash"],
            "protocol_status": legacy_manifest["protocol_status"],
            "search_semantics": "legacy_fixed_temperature_mixture",
            "terminal_status": legacy_manifest["status"],
        },
        legacy_metrics,
        [
            legacy / "run_manifest.json",
            legacy / "resolved_config.yaml.json",
            legacy / "discovery_summary.json",
            legacy / "search_state.json",
            legacy / "results/repeat_L28.json",
            legacy / "results/p_9386832e3dab5b0b.json",
            ROOT / "user_exp_plans/EXP-20260827-004-h1-frozen-protocol.md",
            ROOT
            / "agent-BuildReports/experiments/"
            "EXP-20260828-004-H1-agent-report.md",
        ],
        tags={
            "experiment_phase": "train_discovery",
            "official_hypothesis_evidence": "train_discovery_only",
            "evidence_maturity": "discovery_preliminary",
            "search_semantics": "legacy_fixed_temperature_mixture",
            "validation_accessed": "false",
            "test_accessed": "false",
        },
    )

    sourcewise = ROOT / "artifacts/EXP-20260828-004-h1-sourcewise-rerun"
    sourcewise_manifest = read_json(sourcewise / "run_manifest.json")
    sourcewise_summary = read_json(sourcewise / "discovery_summary.json")
    sourcewise_metrics: dict[str, float] = {
        "n_discovered_candidates": sourcewise_summary["n_discovered_candidates"],
        "n_good": sourcewise_summary["n_good"],
        "n_readability_collapse_among_good": sourcewise_summary[
            "n_readability_collapse_among_good"
        ],
        "p_gap_among_good": sourcewise_summary["p_gap_among_good"],
        "canonical_task_accuracy_discover": sourcewise_summary[
            "canonical_task_accuracy_discover"
        ],
        "canonical_native_accuracy_discover": sourcewise_summary[
            "canonical_native_accuracy_discover"
        ],
        "gpu_hours_used": sourcewise_summary["gpu_hours_used"],
        "gpu_cache_hits": sourcewise_summary["gpu_prefix_cache"]["hits"],
        "gpu_cache_loads": sourcewise_summary["gpu_prefix_cache"]["loads"],
    }
    for source, values in sourcewise_summary["by_source"].items():
        sourcewise_metrics[f"{source}_n"] = values["n"]
        sourcewise_metrics[f"{source}_n_good"] = values["n_good"]
        sourcewise_metrics[f"{source}_n_gap"] = values["n_gap_among_good"]
        if values["p_gap_among_good"] is not None:
            sourcewise_metrics[f"{source}_p_gap"] = values["p_gap_among_good"]
    sourcewise_run = register(
        "EXP-20260828-004-h1-sourcewise-rerun",
        sourcewise,
        {
            "git_commit_at_freeze": sourcewise_manifest["git_at_freeze"]["commit"],
            "config_hash": sourcewise_manifest["config_hash"],
            "protocol_status": sourcewise_manifest["protocol_status"],
            "search_semantics": "corrected_sourcewise_temperature",
            "terminal_status": sourcewise_manifest["status"],
        },
        sourcewise_metrics,
        [
            sourcewise / "run_manifest.json",
            sourcewise / "resolved_config.yaml.json",
            sourcewise / "discovery_summary.json",
            sourcewise / "search_state.json",
            sourcewise / "operational_cache_policy.json",
            sourcewise / "results/repeat_L28.json",
            sourcewise / "results/p_84ec26e17f1a4b06.json",
            ROOT / "user_exp_plans/EXP-20260828-004-h1-sourcewise-rerun-protocol.md",
            ROOT
            / "agent-BuildReports/experiments/"
            "EXP-20260828-004-H1-agent-report.md",
        ],
        tags={
            "experiment_phase": "train_discovery",
            "official_hypothesis_evidence": "train_discovery_only",
            "evidence_maturity": "discovery_preliminary",
            "search_semantics": "corrected_sourcewise_temperature",
            "validation_accessed": "false",
            "test_accessed": "false",
        },
    )
    print(
        json.dumps(
            {
                "qualification": q_run,
                "head": h_run,
                "pilot": p_run,
                "legacy_discovery": legacy_run,
                "sourcewise_discovery": sourcewise_run,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
