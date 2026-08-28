#!/usr/bin/env python3
"""Register the completed fragmented early-readability diagnostic in MLflow."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h1 import atomic_write_json  # noqa: E402


ARTIFACT_ROOT = ROOT / "artifacts/fragmented_llama_early_readability_260828_01"
RUN_NAME = "FRAG-20260828-004-early-readability"
EXPERIMENT = "EXP-004-readability-qualification"
TRACKING_URI = "sqlite:///" + str((ROOT / "mlruns.db").resolve())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    record_path = ARTIFACT_ROOT / "mlflow_run.json"
    if record_path.is_file():
        print(record_path.read_text(encoding="utf-8"))
        return
    manifest = read_json(ARTIFACT_ROOT / "run_manifest.json")
    if manifest["status"] != "completed":
        raise RuntimeError("refusing to register an incomplete diagnostic")
    if manifest["validation_accessed"] or manifest["test_accessed"]:
        raise RuntimeError("validation/test access flags are not false")
    summary = read_json(ARTIFACT_ROOT / "summary.json")
    variance = summary["variance"]
    selection = summary["smoke_selection"]
    baselines = summary["D_discover_baselines"]
    single = summary["single_block_summary"]
    curves = summary["curves"]
    equivalence = read_json(ARTIFACT_ROOT / "within_process_path_equivalence.json")
    config = read_json(ARTIFACT_ROOT / "resolved_config.json")
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=RUN_NAME) as active:
        mlflow.set_tags(
            {
                "experiment_phase": "fragmented_diagnostic",
                "official_hypothesis_evidence": "false",
                "validation_accessed": "false",
                "test_accessed": "false",
            }
        )
        mlflow.log_params(
            {
                "git_commit": git_commit,
                "config_hash": manifest["config_hash"],
                "split_hash": config["dataset"]["split_sha256"],
                "model": "Llama-3.2-3B-Instruct",
                "dataset": "ARC-Easy official-train D_fit/D_discover only",
                "fit_size": config["dataset"]["fit_size"],
                "discover_size": config["dataset"]["discover_size"],
                "n_paths": config["paths"]["n_paths"],
                "plain_lr": selection["adamw"]["plain"]["selected_learning_rate"],
                "ln_plain_lr": selection["adamw"]["ln_plain"][
                    "selected_learning_rate"
                ],
                "adamw_epochs": selection["adamw"]["plain"]["final_epochs"],
                "ridge_alpha": selection["ridge"]["selected_alpha"],
                "h1_head_l2": config["probes"]["h1_rms_lbfgs"]["l2"],
            }
        )
        metrics = {
            "variance_n_collapsed": variance["n_collapsed"],
            "variance_min_inter_sample_std": variance[
                "minimum_inter_sample_std_mean"
            ],
            "chance_accuracy_discover": baselines["choice_count_uniform_chance"],
            "posthoc_majority_accuracy_discover": baselines[
                "posthoc_majority_label_accuracy"
            ],
            "single_plain_max_accuracy": single["plain"]["maximum_accuracy"],
            "single_ln_plain_max_accuracy": single["ln_plain"]["maximum_accuracy"],
            "single_ridge_max_accuracy": single["ridge"]["maximum_accuracy"],
            "single_h1_max_accuracy": single["h1_rms_lbfgs"]["maximum_accuracy"],
            "same_process_feature_max_abs": equivalence["feature_max_abs"],
            "same_process_logit_max_abs": equivalence["logit_max_abs"],
        }
        for method in ("native", "plain", "ln_plain", "ridge", "h1_rms_lbfgs"):
            prefix = curves[method]["canonical_prefix"]
            metrics[f"prefix_{method}_L12_accuracy"] = prefix[11]["accuracy"]
            metrics[f"prefix_{method}_L13_accuracy"] = prefix[12]["accuracy"]
            metrics[f"prefix_{method}_L15_accuracy"] = prefix[14]["accuracy"]
            metrics[f"prefix_{method}_L16_accuracy"] = prefix[15]["accuracy"]
            metrics[f"prefix_{method}_L28_accuracy"] = prefix[27]["accuracy"]
        mlflow.log_metrics(metrics)
        evidence = [
            ARTIFACT_ROOT / "run_manifest.json",
            ARTIFACT_ROOT / "resolved_config.json",
            ARTIFACT_ROOT / "feature_integrity.json",
            ARTIFACT_ROOT / "within_process_path_equivalence.json",
            ARTIFACT_ROOT / "variance_summary.json",
            ARTIFACT_ROOT / "smoke/selection.json",
            ARTIFACT_ROOT / "summary.json",
            ARTIFACT_ROOT / "report.md",
            ROOT
            / "agent-BuildReports/experiments/FRAG-20260828-004-early-readability.md",
            ROOT / "configs/frag_llama_early_readability_260828_01.yaml",
        ]
        for path in evidence:
            mlflow.log_artifact(str(path), artifact_path="evidence")
        run_id = active.info.run_id
    record = {
        "run_id": run_id,
        "run_name": RUN_NAME,
        "experiment": EXPERIMENT,
        "tracking_uri": TRACKING_URI,
        "registered_git_commit": git_commit,
    }
    atomic_write_json(record_path, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
