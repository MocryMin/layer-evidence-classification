#!/usr/bin/env python3
"""Resumable near-canonical engineering pilot for EXP-004 H1.

The path pool is a fixed collection of transparent controls, not the official
multi-source adaptive search.  Its outputs qualify runtime and reveal pipeline
failures; they are explicitly not hypothesis-bearing EXP-004 results.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h1 import (  # noqa: E402
    DeadlineController,
    DeadlineReached,
    EventJournal,
    ModularLlamaExecutor,
    atomic_torch_save,
    atomic_write_json,
    canonical_json_hash,
    chance_accuracy,
    environment_summary,
    extract_path_feature_split,
    fit_masked_linear_head,
    git_state,
    load_arc_easy_split,
    load_path_feature_split,
    load_yaml,
    masked_accuracy,
    valid_choice_mask,
    verify_answer_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/exp004_h1_structured_pilot.yaml"
    )
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolved(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def structured_path_pool(canonical: list[int]) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = [
        {"path_id": "canonical", "source": "canonical", "path": canonical}
    ]
    for layer in canonical:
        paths.append(
            {
                "path_id": f"skip_L{layer:02d}",
                "source": "single_skip",
                "edit": {"operation": "remove", "layer": layer},
                "path": [item for item in canonical if item != layer],
            }
        )
    for position, layer in enumerate(canonical):
        repeated = canonical[: position + 1] + [layer] + canonical[position + 1 :]
        paths.append(
            {
                "path_id": f"repeat_L{layer:02d}",
                "source": "single_repeat",
                "edit": {"operation": "repeat", "layer": layer},
                "path": repeated,
            }
        )
    for position in range(len(canonical) - 1):
        left, right = canonical[position], canonical[position + 1]
        swapped = canonical.copy()
        swapped[position], swapped[position + 1] = right, left
        paths.append(
            {
                "path_id": f"swap_L{left:02d}_L{right:02d}",
                "source": "adjacent_swap",
                "edit": {"operation": "swap", "left": left, "right": right},
                "path": swapped,
            }
        )
    if len({item["path_id"] for item in paths}) != len(paths):
        raise AssertionError("duplicate structured path IDs")
    return paths


def load_source(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_root = resolved(config["source_qualification"]["path"])
    source_manifest = json.loads(
        (source_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest["config_hash"] != config["source_qualification"]["config_hash"]:
        raise RuntimeError("source qualification hash changed")
    source_config = load_yaml(source_root / "resolved_config.yaml")
    source_metrics = json.loads(
        (source_root / "qualification_metrics.json").read_text(encoding="utf-8")
    )
    head_root = resolved(config["head_qualification"]["path"])
    head_manifest = json.loads(
        (head_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    if head_manifest["config_hash"] != config["head_qualification"]["config_hash"]:
        raise RuntimeError("head qualification hash changed")
    head_summary = json.loads(
        (head_root / "head_qualification.json").read_text(encoding="utf-8")
    )
    selected_l2 = float(config["head_qualification"]["selected_l2"])
    if float(head_summary["selected_l2"]) != selected_l2:
        raise RuntimeError("selected task-head L2 changed")
    return source_config, source_metrics, head_summary


def prepare_run(
    output_root: Path,
    config: dict[str, Any],
    config_hash: str,
    path_pool: list[dict[str, Any]],
    deadline: DeadlineController,
    resume: bool,
) -> dict[str, Any]:
    manifest_path = output_root / "run_manifest.json"
    if output_root.exists():
        if not resume:
            raise RuntimeError(f"pilot exists; pass --resume: {output_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_hash"] != config_hash:
            raise RuntimeError("refusing resume: pilot config changed")
        manifest.setdefault("resume_history", []).append(
            {
                "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                "previous_status": manifest.get("status"),
                "git": git_state(),
                "hard_stop": deadline.hard_stop.isoformat(),
            }
        )
        manifest["status"] = "running"
    else:
        if resume:
            raise RuntimeError(f"cannot resume missing pilot: {output_root}")
        output_root.mkdir(parents=True)
        manifest = {
            "experiment_id": config["experiment_id"],
            "qualification_status": config["status"],
            "status": "running",
            "config_hash": config_hash,
            "path_pool_hash": canonical_json_hash(path_pool),
            "git": git_state(),
            "environment": environment_summary(),
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "official_hypothesis_evidence": False,
            "validation_accessed": False,
            "test_accessed": False,
        }
        atomic_write_json(output_root / "path_pool.json", path_pool)
    manifest["hard_stop"] = deadline.hard_stop.isoformat()
    manifest["soft_stop"] = deadline.soft_stop.isoformat()
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_completed_results(output_root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "results").glob("*.json"))
    ] if (output_root / "results").exists() else []


def run() -> int:
    args = parse_args()
    config_path = resolved(args.config).resolve()
    config = load_yaml(config_path)
    if config["status"] != "engineering_pilot_not_hypothesis_evidence":
        raise RuntimeError("structured pilot must be explicitly non-official")
    if config["runtime"].get("allow_validation") or config["runtime"].get("allow_test"):
        raise RuntimeError("validation/test access is forbidden")
    deadline = DeadlineController(
        args.stop_at, int(config["runtime"]["reserve_minutes"])
    )
    deadline.install_signal_handlers()
    source_config, source_metrics, head_summary = load_source(config)
    canonical = list(source_config["path"]["canonical"])
    path_pool = structured_path_pool(canonical)
    if len(path_pool) != int(config["path_pool"]["expected_total_paths"]):
        raise RuntimeError(f"unexpected path-pool size: {len(path_pool)}")
    config_hash = canonical_json_hash(config)
    output_root = resolved(config["runtime"]["artifact_root"]).resolve()
    manifest = prepare_run(
        output_root, config, config_hash, path_pool, deadline, args.resume
    )
    journal = EventJournal(output_root / "events.jsonl")
    journal.append(
        "pilot_session_started",
        pid=os.getpid(),
        hard_stop=deadline.hard_stop.isoformat(),
        soft_stop=deadline.soft_stop.isoformat(),
    )
    model = None
    try:
        source_root = resolved(config["source_qualification"]["path"])
        split_document = json.loads(
            (source_root / "split_indices.json").read_text(encoding="utf-8")
        )
        dataset_root = resolved(source_config["dataset"]["path"])
        all_train = load_arc_easy_split(dataset_root, "train")
        split_examples = {
            split: [all_train[index] for index in split_document[split]]
            for split in ("fit", "discover")
        }
        model_path = resolved(source_config["model"]["path"])
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer.padding_side = source_config["tokenization"]["padding_side"]
        answer_token_ids = verify_answer_tokens(tokenizer, source_config)
        deadline.checkpoint(next_unit_seconds=180.0)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.manual_seed(int(config["runtime"]["seed"]))
        torch.cuda.manual_seed_all(int(config["runtime"]["seed"]))
        if config["runtime"].get("deterministic", True):
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[source_config["model"]["dtype"]]
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=source_config["model"]["attention_implementation"],
        ).to("cuda")
        model.eval()
        model.config.use_cache = False
        executor = ModularLlamaExecutor(model, answer_token_ids)
        device = torch.device("cuda")
        feature_root = output_root / "features"
        results_root = output_root / "results"
        heads_root = output_root / "heads"
        results_root.mkdir(exist_ok=True)
        heads_root.mkdir(exist_ok=True)

        canonical_task = float(
            head_summary["refit_full_D_fit_evaluate_D_discover"]["eval_accuracy"]
        )
        canonical_native = float(source_metrics["canonical_native_accuracy_discover"])
        chance = float(source_metrics["chance_accuracy_discover"])
        existing = {item["path_id"] for item in load_completed_results(output_root)}

        for path_record in path_pool:
            deadline.checkpoint(next_unit_seconds=180.0)
            path_id = path_record["path_id"]
            result_path = results_root / f"{path_id}.json"
            if path_id in existing:
                continue
            if path_id == "canonical":
                result = {
                    **path_record,
                    "task_accuracy_fit": float(
                        head_summary["refit_full_D_fit_evaluate_D_discover"]["train_accuracy"]
                    ),
                    "task_accuracy_discover": canonical_task,
                    "native_accuracy_fit": float(source_metrics["canonical_native_accuracy_fit"]),
                    "native_accuracy_discover": canonical_native,
                    "delta_task_from_canonical": 0.0,
                    "native_gap_from_canonical": 0.0,
                    "relative_native_gap": 0.0,
                    "good_path": True,
                    "readability_collapse": False,
                    "source_reused": True,
                }
                atomic_write_json(result_path, result)
                existing.add(path_id)
                continue

            journal.append("path_started", path_id=path_id, path=path_record["path"])
            extraction = []
            for split_name in ("fit", "discover"):
                extraction.append(
                    extract_path_feature_split(
                        split_name,
                        split_examples[split_name],
                        split_document[split_name],
                        executor=executor,
                        tokenizer=tokenizer,
                        prompt_cfg=source_config["prompt"],
                        max_length_guard=int(source_config["tokenization"]["max_length_guard"]),
                        path=path_record["path"],
                        path_id=path_id,
                        batch_size=int(config["runtime"]["batch_size"]),
                        shard_size=int(config["runtime"]["shard_size"]),
                        config_hash=config_hash,
                        feature_root=feature_root,
                        deadline=deadline,
                        journal=journal,
                    )
                )
            fit = load_path_feature_split(feature_root, path_id, "fit")
            discover = load_path_feature_split(feature_root, path_id, "discover")
            fit_mask = valid_choice_mask(fit["choice_counts"], 5)
            discover_mask = valid_choice_mask(discover["choice_counts"], 5)
            head = fit_masked_linear_head(
                fit["features"].to(device),
                fit["labels"].to(device),
                fit_mask.to(device),
                discover["features"].to(device),
                discover["labels"].to(device),
                discover_mask.to(device),
                l2=float(config["head_qualification"]["selected_l2"]),
                max_iter=int(config["head"]["max_iter"]),
                tolerance_grad=float(config["head"]["tolerance_grad"]),
            )
            weight = head.pop("weight")
            atomic_torch_save(
                heads_root / f"{path_id}.pt",
                {"path_id": path_id, "path": path_record["path"], "weight": weight},
            )
            task_acc = float(head["eval_accuracy"])
            native_acc_fit = masked_accuracy(
                fit["native_label_logits"], fit["labels"], fit_mask
            )
            native_acc = masked_accuracy(
                discover["native_label_logits"], discover["labels"], discover_mask
            )
            delta_task = canonical_task - task_acc
            native_gap = canonical_native - native_acc
            denominator = canonical_native - chance
            relative_gap = native_gap / denominator if denominator > 0 else None
            good = delta_task <= float(config["criteria"]["epsilon_task"])
            collapse = bool(
                good
                and (
                    native_gap >= float(config["criteria"]["gap_absolute"])
                    or (
                        relative_gap is not None
                        and relative_gap >= float(config["criteria"]["gap_relative"])
                    )
                )
            )
            batch_times = sum(
                (item["batch_durations_seconds"] for item in extraction), []
            )
            result = {
                **path_record,
                "task_accuracy_fit": float(head["train_accuracy"]),
                "task_accuracy_discover": task_acc,
                "task_head_cross_entropy_fit": float(head["train_cross_entropy"]),
                "task_head_cross_entropy_discover": float(head["eval_cross_entropy"]),
                "task_head_closure_calls": int(head["closure_calls"]),
                "task_head_elapsed_seconds": float(head["elapsed_seconds"]),
                "native_accuracy_fit": native_acc_fit,
                "native_accuracy_discover": native_acc,
                "delta_task_from_canonical": delta_task,
                "native_gap_from_canonical": native_gap,
                "relative_native_gap": relative_gap,
                "good_path": good,
                "readability_collapse": collapse,
                "forward_batch_seconds_mean": (
                    sum(batch_times) / len(batch_times) if batch_times else None
                ),
                "forward_batch_seconds_max": max(batch_times) if batch_times else None,
            }
            atomic_write_json(result_path, result)
            existing.add(path_id)
            journal.append("path_completed", **result)
            print(
                f"[{len(existing):02d}/{len(path_pool)}] {path_id}: "
                f"task={task_acc:.4f} native={native_acc:.4f} good={good} collapse={collapse}",
                flush=True,
            )

        results = load_completed_results(output_root)
        good = [item for item in results if item["good_path"]]
        collapsed = [item for item in good if item["readability_collapse"]]
        noncanonical = [item for item in results if item["path_id"] != "canonical"]
        summary = {
            "status": config["status"],
            "official_hypothesis_evidence": False,
            "n_paths": len(results),
            "n_noncanonical": len(noncanonical),
            "n_good": len(good),
            "n_readability_collapse": len(collapsed),
            "p_gap_among_good": len(collapsed) / len(good) if good else None,
            "canonical_task_accuracy_discover": canonical_task,
            "canonical_native_accuracy_discover": canonical_native,
            "chance_accuracy_discover": chance_accuracy(
                [item.n_choices for item in split_examples["discover"]]
            ),
            "best_noncanonical_task": max(
                noncanonical, key=lambda item: item["task_accuracy_discover"]
            ) if noncanonical else None,
            "largest_native_gap_among_good": max(
                good, key=lambda item: item["native_gap_from_canonical"]
            ) if good else None,
            "source_counts": {
                source: sum(item["source"] == source for item in results)
                for source in sorted({item["source"] for item in results})
            },
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "validation_accessed": False,
            "test_accessed": False,
        }
        atomic_write_json(output_root / "pilot_summary.json", summary)
        manifest.update(
            {
                "status": "completed",
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "n_paths": len(results),
            }
        )
        atomic_write_json(output_root / "run_manifest.json", manifest)
        journal.append("pilot_completed", **summary)
        print(json.dumps(summary, indent=2))
        return 0
    except DeadlineReached as exc:
        manifest.update(
            {
                "status": "soft_stopped",
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "reason": str(exc),
                "n_completed_paths": len(load_completed_results(output_root)),
            }
        )
        atomic_write_json(output_root / "run_manifest.json", manifest)
        journal.append("pilot_soft_stopped", reason=str(exc))
        return 75
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "n_completed_paths": len(load_completed_results(output_root)),
            }
        )
        atomic_write_json(output_root / "run_manifest.json", manifest)
        journal.append(
            "pilot_failed", error_type=type(exc).__name__, error=str(exc)
        )
        raise
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(run())
