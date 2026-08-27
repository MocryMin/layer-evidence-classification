#!/usr/bin/env python3
"""Independently audit one structured-pilot readability-gap witness."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h1 import (  # noqa: E402
    DeadlineController,
    ModularLlamaExecutor,
    atomic_write_json,
    encode_arc_examples,
    fit_masked_linear_head,
    git_state,
    load_arc_easy_split,
    load_path_feature_split,
    load_yaml,
    mask_invalid_logits,
    masked_accuracy,
    valid_choice_mask,
    verify_answer_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-config", default="configs/exp004_h1_structured_pilot.yaml"
    )
    parser.add_argument("--path-id", default="repeat_L28")
    parser.add_argument("--recompute-samples", type=int, default=16)
    parser.add_argument("--stop-at", required=True)
    return parser.parse_args()


def resolved(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_canonical_split(source_root: Path, split: str) -> dict[str, Any]:
    paths = sorted((source_root / "features" / "canonical" / split).glob("shard_*.pt"))
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    return {
        "features": torch.cat([item["features"] for item in shards]),
        "native_label_logits": torch.cat([item["native_label_logits"] for item in shards]),
        "labels": torch.cat([item["labels"] for item in shards]),
        "choice_counts": torch.cat([item["choice_counts"] for item in shards]).long(),
    }


def prediction_counts(logits: torch.Tensor, mask: torch.Tensor) -> dict[str, int]:
    labels = "ABCDE"
    counts = Counter(mask_invalid_logits(logits, mask).argmax(dim=1).tolist())
    return {labels[index]: int(counts.get(index, 0)) for index in range(5)}


def run() -> int:
    args = parse_args()
    config = load_yaml(resolved(args.pilot_config))
    deadline = DeadlineController(
        args.stop_at, int(config["runtime"]["reserve_minutes"])
    )
    deadline.install_signal_handlers()
    pilot_root = resolved(config["runtime"]["artifact_root"])
    pilot_manifest = json.loads(
        (pilot_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    if pilot_manifest["status"] != "completed":
        raise RuntimeError("structured pilot is not complete")
    result = json.loads(
        (pilot_root / "results" / f"{args.path_id}.json").read_text(encoding="utf-8")
    )
    if not result["readability_collapse"]:
        raise RuntimeError(f"selected path is not a recorded collapse: {args.path_id}")

    source_root = resolved(config["source_qualification"]["path"])
    source_config = load_yaml(source_root / "resolved_config.yaml")
    split_document = json.loads(
        (source_root / "split_indices.json").read_text(encoding="utf-8")
    )
    all_train = load_arc_easy_split(resolved(source_config["dataset"]["path"]), "train")
    pilot_features = {
        split: load_path_feature_split(pilot_root / "features", args.path_id, split)
        for split in ("fit", "discover")
    }
    canonical_discover = load_canonical_split(source_root, "discover")
    saved_head = torch.load(
        pilot_root / "heads" / f"{args.path_id}.pt",
        map_location="cpu",
        weights_only=False,
    )["weight"].float()

    audit_path = pilot_root / "audits" / f"{args.path_id}.json"
    if audit_path.exists():
        raise RuntimeError(f"audit already exists: {audit_path}")
    deadline.checkpoint(next_unit_seconds=180.0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model_path = resolved(source_config["model"]["path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = source_config["tokenization"]["padding_side"]
    answer_token_ids = verify_answer_tokens(tokenizer, source_config)
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

    try:
        # Recompute a raw subset from original examples, independent of saved tensors.
        first_shard_path = sorted(
            (pilot_root / "features" / args.path_id / "fit").glob("shard_*.pt")
        )[0]
        first_shard = torch.load(first_shard_path, map_location="cpu", weights_only=False)
        n = min(args.recompute_samples, len(first_shard["original_indices"]))
        examples = [all_train[index] for index in first_shard["original_indices"][:n]]
        encoded = encode_arc_examples(tokenizer, examples, source_config["prompt"])
        recomputed_features, recomputed_logits = executor.forward_path(
            encoded["input_ids"].to("cuda"),
            encoded["attention_mask"].to("cuda"),
            result["path"],
        )
        saved_subset_features = first_shard["features"][:n]
        saved_subset_logits = first_shard["native_label_logits"][:n]
        feature_diff = (
            recomputed_features.to(dtype=torch.float16, device="cpu")
            - saved_subset_features
        ).abs()
        logit_diff = (recomputed_logits.float().cpu() - saved_subset_logits).abs()

        fit = pilot_features["fit"]
        discover = pilot_features["discover"]
        fit_mask = valid_choice_mask(fit["choice_counts"], 5)
        discover_mask = valid_choice_mask(discover["choice_counts"], 5)
        saved_task_fit_logits = F.linear(fit["features"].float(), saved_head)
        saved_task_discover_logits = F.linear(discover["features"].float(), saved_head)
        saved_metrics = {
            "task_accuracy_fit": masked_accuracy(
                saved_task_fit_logits, fit["labels"], fit_mask
            ),
            "task_accuracy_discover": masked_accuracy(
                saved_task_discover_logits, discover["labels"], discover_mask
            ),
            "native_accuracy_fit": masked_accuracy(
                fit["native_label_logits"], fit["labels"], fit_mask
            ),
            "native_accuracy_discover": masked_accuracy(
                discover["native_label_logits"], discover["labels"], discover_mask
            ),
        }

        deadline.checkpoint(next_unit_seconds=180.0)
        refit = fit_masked_linear_head(
            fit["features"].to("cuda"),
            fit["labels"].to("cuda"),
            fit_mask.to("cuda"),
            discover["features"].to("cuda"),
            discover["labels"].to("cuda"),
            discover_mask.to("cuda"),
            l2=float(config["head_qualification"]["selected_l2"]),
            max_iter=int(config["head"]["max_iter"]),
            tolerance_grad=float(config["head"]["tolerance_grad"]),
        )
        refit_weight = refit.pop("weight").float()
        head_diff = (refit_weight - saved_head).abs()

        canonical_x = canonical_discover["features"].float()
        witness_x = discover["features"].float()
        cosine = F.cosine_similarity(canonical_x, witness_x, dim=1)
        canonical_mask = valid_choice_mask(canonical_discover["choice_counts"], 5)
        canonical_native_logits = canonical_discover["native_label_logits"]
        task_pred = mask_invalid_logits(saved_task_discover_logits, discover_mask).argmax(1)
        native_pred = mask_invalid_logits(
            discover["native_label_logits"], discover_mask
        ).argmax(1)
        labels = discover["labels"]
        task_correct = task_pred == labels
        native_correct = native_pred == labels
        correctness = {
            "both_correct": int((task_correct & native_correct).sum().item()),
            "task_only_correct": int((task_correct & ~native_correct).sum().item()),
            "native_only_correct": int((~task_correct & native_correct).sum().item()),
            "both_wrong": int((~task_correct & ~native_correct).sum().item()),
        }
        audit = {
            "status": "engineering_witness_audit_not_hypothesis_evidence",
            "path_id": args.path_id,
            "path": result["path"],
            "git": git_state(),
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "raw_recompute": {
                "n_samples": n,
                "feature_bit_equal_after_float16_storage": bool(
                    torch.equal(
                        recomputed_features.to(dtype=torch.float16, device="cpu"),
                        saved_subset_features,
                    )
                ),
                "feature_max_abs": float(feature_diff.max().item()),
                "native_label_logit_bit_equal": bool(
                    torch.equal(recomputed_logits.float().cpu(), saved_subset_logits)
                ),
                "native_label_logit_max_abs": float(logit_diff.max().item()),
            },
            "saved_metric_recalculation": saved_metrics,
            "recorded_metrics": {
                key: result[key]
                for key in (
                    "task_accuracy_fit",
                    "task_accuracy_discover",
                    "native_accuracy_fit",
                    "native_accuracy_discover",
                )
            },
            "head_refit": {
                **refit,
                "weight_bit_equal": bool(torch.equal(refit_weight, saved_head)),
                "weight_max_abs": float(head_diff.max().item()),
                "weight_mean_abs": float(head_diff.mean().item()),
            },
            "representation_comparison_discover": {
                "canonical_witness_cosine_mean": float(cosine.mean().item()),
                "canonical_witness_cosine_min": float(cosine.min().item()),
                "canonical_witness_l2_mean": float(
                    (canonical_x - witness_x).norm(dim=1).mean().item()
                ),
                "canonical_feature_norm_mean": float(
                    canonical_x.norm(dim=1).mean().item()
                ),
                "witness_feature_norm_mean": float(witness_x.norm(dim=1).mean().item()),
            },
            "prediction_distribution_discover": {
                "gold": {
                    label: int(Counter(labels.tolist()).get(index, 0))
                    for index, label in enumerate("ABCDE")
                },
                "canonical_native": prediction_counts(
                    canonical_native_logits, canonical_mask
                ),
                "witness_native": prediction_counts(
                    discover["native_label_logits"], discover_mask
                ),
                "witness_task_head": prediction_counts(
                    saved_task_discover_logits, discover_mask
                ),
            },
            "correctness_cross_table_discover": correctness,
            "all_saved_features_finite": bool(
                torch.isfinite(fit["features"]).all()
                and torch.isfinite(discover["features"]).all()
            ),
            "validation_accessed": False,
            "test_accessed": False,
        }
        # Fail the audit if any directly recorded metric cannot be reproduced.
        for key, recalculated in saved_metrics.items():
            if abs(recalculated - float(result[key])) > 1e-7:
                raise RuntimeError(
                    f"saved metric mismatch for {key}: {recalculated} vs {result[key]}"
                )
        atomic_write_json(audit_path, audit)
        print(json.dumps(audit, indent=2))
        return 0
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(run())
