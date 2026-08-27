#!/usr/bin/env python3
"""Choose provisional task-head regularisation using D_fit-only CV."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h1 import (  # noqa: E402
    DeadlineController,
    DeadlineReached,
    EventJournal,
    atomic_torch_save,
    atomic_write_json,
    canonical_json_hash,
    environment_summary,
    fit_masked_linear_head,
    git_state,
    load_yaml,
    stratified_fold_ids,
    valid_choice_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/exp004_h1_head_qualification.yaml"
    )
    parser.add_argument("--stop-at", required=True)
    return parser.parse_args()


def resolved(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_feature_split(source_run: Path, split: str) -> dict[str, Any]:
    paths = sorted((source_run / "features" / "canonical" / split).glob("shard_*.pt"))
    if not paths:
        raise RuntimeError(f"missing source features: {split}")
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    return {
        "features": torch.cat([item["features"] for item in shards]),
        "labels": torch.cat([item["labels"] for item in shards]),
        "choice_counts": torch.cat([item["choice_counts"] for item in shards]).long(),
    }


def run() -> int:
    args = parse_args()
    config_path = resolved(args.config).resolve()
    config = load_yaml(config_path)
    if config["status"] != "engineering_qualification_not_hypothesis_evidence":
        raise RuntimeError("head qualification must be explicitly non-official")
    if config["runtime"].get("allow_validation") or config["runtime"].get("allow_test"):
        raise RuntimeError("validation/test access is forbidden")
    deadline = DeadlineController(
        args.stop_at, int(config["runtime"]["reserve_minutes"])
    )
    deadline.install_signal_handlers()
    config_hash = canonical_json_hash(config)
    source_run = resolved(config["source_run"]).resolve()
    source_manifest = json.loads(
        (source_run / "run_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest["config_hash"] != config["source_config_hash"]:
        raise RuntimeError("source qualification config hash changed")

    output_root = resolved(config["runtime"]["artifact_root"]).resolve()
    if output_root.exists():
        raise RuntimeError(f"head qualification output already exists: {output_root}")
    output_root.mkdir(parents=True)
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "config_hash": config_hash,
        "source_run": str(source_run),
        "source_run_config_hash": source_manifest["config_hash"],
        "selection_data": config["selection_data"],
        "git": git_state(),
        "environment": environment_summary(),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hard_stop": deadline.hard_stop.isoformat(),
        "soft_stop": deadline.soft_stop.isoformat(),
        "official_hypothesis_evidence": False,
        "validation_accessed": False,
        "test_accessed": False,
    }
    atomic_write_json(output_root / "run_manifest.json", manifest)
    journal = EventJournal(output_root / "events.jsonl")
    journal.append("head_qualification_started", **manifest)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device = torch.device("cuda")
        fit = load_feature_split(source_run, "fit")
        discover = load_feature_split(source_run, "discover")
        fold_ids = torch.tensor(
            stratified_fold_ids(
                fit["labels"].tolist(),
                fit["choice_counts"].tolist(),
                int(config["n_folds"]),
                int(config["seed"]),
            ),
            dtype=torch.long,
        )
        atomic_write_json(
            output_root / "folds.json",
            {
                "n_folds": int(config["n_folds"]),
                "seed": int(config["seed"]),
                "fold_ids": fold_ids.tolist(),
                "fold_hash": canonical_json_hash(fold_ids.tolist()),
            },
        )
        x = fit["features"].to(device)
        y = fit["labels"].to(device)
        mask = valid_choice_mask(fit["choice_counts"], 5).to(device)
        solver = config["solver"]
        per_l2: list[dict[str, Any]] = []
        for l2 in [float(item) for item in config["l2_grid"]]:
            fold_records = []
            for fold in range(int(config["n_folds"])):
                deadline.checkpoint(next_unit_seconds=120.0)
                train_rows = (fold_ids != fold).to(device)
                eval_rows = (fold_ids == fold).to(device)
                result = fit_masked_linear_head(
                    x[train_rows],
                    y[train_rows],
                    mask[train_rows],
                    x[eval_rows],
                    y[eval_rows],
                    mask[eval_rows],
                    l2=l2,
                    max_iter=int(solver["max_iter"]),
                    tolerance_grad=float(solver["tolerance_grad"]),
                )
                result.pop("weight")
                result["fold"] = fold
                fold_records.append(result)
                journal.append("fold_completed", l2=l2, fold=fold, **result)
            record = {
                "l2": l2,
                "mean_eval_accuracy": statistics.fmean(
                    item["eval_accuracy"] for item in fold_records
                ),
                "std_eval_accuracy": statistics.stdev(
                    item["eval_accuracy"] for item in fold_records
                ),
                "mean_eval_cross_entropy": statistics.fmean(
                    item["eval_cross_entropy"] for item in fold_records
                ),
                "folds": fold_records,
            }
            per_l2.append(record)
            atomic_write_json(output_root / "partial_grid.json", {"per_l2": per_l2})

        # Accuracy first; CE breaks exact ties; stronger L2 breaks any remaining tie.
        best = min(
            per_l2,
            key=lambda item: (
                -item["mean_eval_accuracy"],
                item["mean_eval_cross_entropy"],
                -item["l2"],
            ),
        )
        deadline.checkpoint(next_unit_seconds=180.0)
        discover_mask = valid_choice_mask(discover["choice_counts"], 5)
        final = fit_masked_linear_head(
            fit["features"].to(device),
            fit["labels"].to(device),
            mask,
            discover["features"].to(device),
            discover["labels"].to(device),
            discover_mask.to(device),
            l2=float(best["l2"]),
            max_iter=int(solver["max_iter"]),
            tolerance_grad=float(solver["tolerance_grad"]),
        )
        weight = final.pop("weight")
        atomic_torch_save(
            output_root / "selected_head.pt",
            {
                "weight": weight,
                "selected_l2": best["l2"],
                "source_config_hash": source_manifest["config_hash"],
                "head_qualification_config_hash": config_hash,
            },
        )
        summary = {
            "status": config["status"],
            "official_hypothesis_evidence": False,
            "selection_data": config["selection_data"],
            "selected_l2": best["l2"],
            "selected_cv_mean_accuracy": best["mean_eval_accuracy"],
            "selected_cv_std_accuracy": best["std_eval_accuracy"],
            "selected_cv_mean_cross_entropy": best["mean_eval_cross_entropy"],
            "refit_full_D_fit_evaluate_D_discover": final,
            "per_l2": per_l2,
            "validation_accessed": False,
            "test_accessed": False,
        }
        atomic_write_json(output_root / "head_qualification.json", summary)
        manifest.update(
            {
                "status": "completed",
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "selected_l2": best["l2"],
            }
        )
        atomic_write_json(output_root / "run_manifest.json", manifest)
        journal.append("head_qualification_completed", **summary)
        print(json.dumps(summary, indent=2))
        return 0
    except DeadlineReached as exc:
        manifest.update(
            {
                "status": "soft_stopped",
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "reason": str(exc),
            }
        )
        atomic_write_json(output_root / "run_manifest.json", manifest)
        journal.append("head_qualification_soft_stopped", reason=str(exc))
        return 75
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        atomic_write_json(output_root / "run_manifest.json", manifest)
        journal.append(
            "head_qualification_failed", error_type=type(exc).__name__, error=str(exc)
        )
        raise


if __name__ == "__main__":
    raise SystemExit(run())
