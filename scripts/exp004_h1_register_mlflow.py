#!/usr/bin/env python3
"""Register completed EXP-004 H1 qualification artifacts in local MLflow."""
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
) -> str:
    record_path = artifact_root / "mlflow_run.json"
    if record_path.exists():
        return read_json(record_path)["run_id"]
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags(
            {
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
    print(json.dumps({"qualification": q_run, "head": h_run, "pilot": p_run}, indent=2))


if __name__ == "__main__":
    main()
