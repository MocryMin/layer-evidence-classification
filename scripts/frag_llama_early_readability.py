#!/usr/bin/env python3
"""Run the crash-safe fragmented Llama early-readability diagnostic."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h1 import (  # noqa: E402
    EventJournal,
    ModularLlamaExecutor,
    atomic_torch_save,
    atomic_write_json,
    canonical_json_hash,
    encode_arc_examples,
    environment_summary,
    fit_masked_linear_head,
    git_state,
    load_arc_easy_split,
    load_yaml,
    masked_accuracy,
    stratified_fold_ids,
    valid_choice_mask,
    verify_answer_tokens,
)
from src.frag_early_readability import (  # noqa: E402
    FAMILIES,
    fit_ridge,
    masked_numpy_accuracy,
    path_specs,
    select_smoke_value,
    train_adamw_fixed,
    train_adamw_smoke,
    variance_stats,
)


PHASES = ("features", "variance", "smoke", "probes", "report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/frag_llama_early_readability_260828_01.yaml"
    )
    parser.add_argument("--phase", choices=(*PHASES, "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolved(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def assert_scope(config: dict[str, Any]) -> None:
    if config["status"] != "fragmented_diagnostic_not_hypothesis_confirmation":
        raise RuntimeError("run must remain explicitly diagnostic")
    if config["scope"]["official_validation_access"] != "forbidden":
        raise RuntimeError("official validation access must remain forbidden")
    if config["scope"]["official_test_access"] != "forbidden":
        raise RuntimeError("official test access must remain forbidden")
    if config["runtime"].get("validation_accessed") or config["runtime"].get(
        "test_accessed"
    ):
        raise RuntimeError("validation/test access flags must be false")


def prepare_run(
    config: dict[str, Any], config_path: Path, *, resume: bool
) -> tuple[Path, dict[str, Any], EventJournal, str]:
    config_hash = canonical_json_hash(config)
    output_root = resolved(config["runtime"]["artifact_root"]).resolve()
    manifest_path = output_root / "run_manifest.json"
    if output_root.exists():
        if not manifest_path.is_file():
            raise RuntimeError(f"artifact directory lacks manifest: {output_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_hash"] != config_hash:
            raise RuntimeError("refusing resume because config changed")
        if not resume:
            raise RuntimeError(f"output exists; pass --resume: {output_root}")
        manifest.setdefault("resume_history", []).append(
            {"time": now(), "previous_status": manifest.get("status"), "git": git_state()}
        )
    else:
        output_root.mkdir(parents=True)
        manifest = {
            "experiment_id": config["experiment_id"],
            "status": "created",
            "diagnostic_only": True,
            "official_hypothesis_evidence": False,
            "config_path": str(config_path),
            "config_hash": config_hash,
            "started_at": now(),
            "git": git_state(),
            "environment": environment_summary(),
            "validation_accessed": False,
            "test_accessed": False,
            "completed_phases": [],
        }
    manifest["status"] = "running"
    manifest["last_started_at"] = now()
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(output_root / "resolved_config.json", config)
    journal = EventJournal(output_root / "events.jsonl")
    return output_root, manifest, journal, config_hash


def finish_phase(
    output_root: Path,
    manifest: dict[str, Any],
    journal: EventJournal,
    phase: str,
) -> None:
    if phase not in manifest["completed_phases"]:
        manifest["completed_phases"].append(phase)
    manifest["last_completed_phase"] = phase
    manifest["last_updated_at"] = now()
    atomic_write_json(output_root / "run_manifest.json", manifest)
    journal.append("phase_completed", phase=phase)


def load_split_definition(config: dict[str, Any]) -> tuple[list[Any], dict[str, list[int]]]:
    train = load_arc_easy_split(resolved(config["dataset"]["path"]), "train")
    split_path = resolved(config["dataset"]["split_indices"])
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    indices = {name: [int(v) for v in split_payload[name]] for name in ("fit", "discover")}
    observed_hash = canonical_json_hash(indices)
    expected_hash = config["dataset"]["split_sha256"]
    if observed_hash != expected_hash or split_payload.get("hash") != expected_hash:
        raise RuntimeError(
            f"split hash mismatch: observed={observed_hash}, expected={expected_hash}"
        )
    if len(indices["fit"]) != int(config["dataset"]["fit_size"]):
        raise RuntimeError("D_fit size mismatch")
    if len(indices["discover"]) != int(config["dataset"]["discover_size"]):
        raise RuntimeError("D_discover size mismatch")
    if set(indices["fit"]) & set(indices["discover"]):
        raise RuntimeError("fit/discover overlap")
    return train, indices


def shard_paths(output_root: Path, split: str) -> list[Path]:
    return sorted((output_root / "features" / split).glob("shard_*.pt"))


def load_features(output_root: Path, split: str) -> dict[str, Any]:
    paths = shard_paths(output_root, split)
    if not paths:
        raise RuntimeError(f"no extracted feature shards for {split}")
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    expected_start = 0
    for path, shard in zip(paths, shards):
        if int(shard["start"]) != expected_start:
            raise RuntimeError(f"non-contiguous shard sequence at {path}")
        expected_start = int(shard["end"])
    return {
        "raw": torch.cat([item["raw"] for item in shards], dim=0),
        "rms": torch.cat([item["rms"] for item in shards], dim=0),
        "native_logits": torch.cat([item["native_logits"] for item in shards], dim=0),
        "labels": torch.cat([item["labels"] for item in shards], dim=0).long(),
        "choice_counts": torch.cat(
            [item["choice_counts"] for item in shards], dim=0
        ).long(),
        "original_indices": sum([item["original_indices"] for item in shards], []),
        "sample_ids": sum([item["sample_ids"] for item in shards], []),
    }


def audit_extracted_features(config: dict[str, Any], output_root: Path) -> None:
    """Validate shape/finiteness and reproduce the prior canonical path."""
    expected_sizes = {
        "fit": int(config["dataset"]["fit_size"]),
        "discover": int(config["dataset"]["discover_size"]),
    }
    reference_root = resolved(config["audit"]["canonical_reference_feature_root"])
    audit: dict[str, Any] = {"splits": {}, "passed": True}
    for split, expected_size in expected_sizes.items():
        current = load_features(output_root, split)
        expected_shape = (
            expected_size,
            2,
            int(config["model"]["layers"]),
            int(config["model"]["hidden_size"]),
        )
        if tuple(current["raw"].shape) != expected_shape:
            raise RuntimeError(
                f"{split} raw shape {tuple(current['raw'].shape)} != {expected_shape}"
            )
        if not torch.isfinite(current["raw"]).all() or not torch.isfinite(
            current["rms"]
        ).all():
            raise RuntimeError(f"non-finite extracted feature in {split}")
        reference_paths = sorted((reference_root / split).glob("shard_*.pt"))
        if not reference_paths:
            raise RuntimeError(f"missing canonical reference features for {split}")
        reference_shards = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in reference_paths
        ]
        reference_features = torch.cat(
            [item["features"] for item in reference_shards], dim=0
        )
        reference_logits = torch.cat(
            [item["native_label_logits"] for item in reference_shards], dim=0
        )
        reference_indices = sum(
            [item["original_indices"] for item in reference_shards], []
        )
        if current["original_indices"] != reference_indices:
            raise RuntimeError(f"canonical reference sample order mismatch in {split}")
        difference = (
            current["rms"][:, FAMILIES.index("canonical_prefix"), -1].float()
            - reference_features.float()
        ).abs()
        current_logits = current["native_logits"][
            :, FAMILIES.index("canonical_prefix"), -1
        ]
        choice_mask = valid_choice_mask(current["choice_counts"], 5)
        current_predictions = current_logits.masked_fill(
            ~choice_mask, torch.finfo(current_logits.dtype).min
        ).argmax(dim=1)
        reference_predictions = reference_logits.masked_fill(
            ~choice_mask, torch.finfo(reference_logits.dtype).min
        ).argmax(dim=1)
        current_accuracy = masked_accuracy(
            current_logits, current["labels"], choice_mask
        )
        reference_accuracy = masked_accuracy(
            reference_logits, current["labels"], choice_mask
        )
        split_audit = {
            "n_samples": expected_size,
            "raw_shape": list(current["raw"].shape),
            "all_finite": True,
            "prefix_L28_reference_max_abs": float(difference.max().item()),
            "prefix_L28_reference_mean_abs": float(difference.mean().item()),
            "prefix_L28_reference_exact_fraction": float(
                (difference == 0).float().mean().item()
            ),
            "prefix_L28_reference_prediction_agreement": float(
                (current_predictions == reference_predictions).float().mean().item()
            ),
            "prefix_L28_current_native_accuracy": current_accuracy,
            "prefix_L28_reference_native_accuracy": reference_accuracy,
            "prefix_L28_native_accuracy_abs_difference": abs(
                current_accuracy - reference_accuracy
            ),
        }
        # The historical numeric thresholds were designed for two forwards in
        # one process.  We retain that stricter cross-run comparison as an
        # audit field, but gate cross-run reproducibility on predictions and
        # accuracy because BF16 SDPA reduction order can differ across runs.
        split_audit["within_historical_same_process_numeric_tolerance"] = (
            split_audit["prefix_L28_reference_max_abs"]
            <= float(config["audit"]["prefix_L28_max_abs_tolerance"])
            and split_audit["prefix_L28_reference_mean_abs"]
            <= float(config["audit"]["prefix_L28_mean_abs_tolerance"])
        )
        split_audit["semantic_within_tolerance"] = (
            split_audit["prefix_L28_reference_prediction_agreement"] >= 0.99
            and split_audit["prefix_L28_native_accuracy_abs_difference"] <= 0.01
        )
        audit["splits"][split] = split_audit
        audit["passed"] = audit["passed"] and split_audit["semantic_within_tolerance"]
    atomic_write_json(output_root / "feature_integrity.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"prefix_L28 canonical semantic audit failed: {audit}")


@torch.inference_mode()
def extract_features(
    config: dict[str, Any],
    output_root: Path,
    journal: EventJournal,
    config_hash: str,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feature extraction")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    train, indices = load_split_definition(config)
    model_path = resolved(config["model"]["path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = config["prompt"]["padding_side"]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    answer_config = {
        "tokenization": {
            "answer_labels": config["prompt"]["answer_labels"],
            "expected_answer_token_ids": config["prompt"]["expected_answer_token_ids"],
        }
    }
    answer_token_ids = verify_answer_tokens(tokenizer, answer_config)
    torch.manual_seed(int(config["runtime"]["seed"]))
    torch.cuda.manual_seed_all(int(config["runtime"]["seed"]))
    if config["runtime"].get("deterministic", True):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=config["model"]["attention_implementation"],
    ).eval().to("cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    executor = ModularLlamaExecutor(model, answer_token_ids)
    n_layers = int(config["model"]["layers"])
    if len(executor.backbone.layers) != n_layers:
        raise RuntimeError("model/config layer count mismatch")
    batch_size = int(config["runtime"]["batch_size"])
    shard_size = int(config["runtime"]["shard_size"])
    max_length = int(config["prompt"]["max_length_guard"])
    prompt_cfg = config["prompt"]

    for split in ("fit", "discover"):
        split_indices = indices[split]
        examples = [train[index] for index in split_indices]
        split_root = output_root / "features" / split
        split_root.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(examples), shard_size):
            end = min(start + shard_size, len(examples))
            path = split_root / f"shard_{start:05d}_{end:05d}.pt"
            expected_indices = split_indices[start:end]
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if (
                    saved["config_hash"] != config_hash
                    or saved["start"] != start
                    or saved["end"] != end
                    or saved["original_indices"] != expected_indices
                ):
                    raise RuntimeError(f"resume validation failed for {path}")
                print(f"reuse {split} {start}:{end}", flush=True)
                continue
            raw_batches = []
            rms_batches = []
            native_batches = []
            shard_t0 = time.monotonic()
            for batch_start in range(start, end, batch_size):
                batch_end = min(batch_start + batch_size, end)
                batch_examples = examples[batch_start:batch_end]
                encoded = encode_arc_examples(tokenizer, batch_examples, prompt_cfg)
                if encoded["input_ids"].shape[1] > max_length:
                    raise RuntimeError(
                        f"token length {encoded['input_ids'].shape[1]} exceeds {max_length}"
                    )
                input_ids = encoded["input_ids"].to("cuda", non_blocking=True)
                attention_mask = encoded["attention_mask"].to("cuda", non_blocking=True)
                embeddings, position_ids, causal_mask, position_embeddings = executor._inputs(
                    input_ids, attention_mask
                )
                single_terminals = []
                for layer in executor.backbone.layers:
                    hidden = layer(
                        embeddings,
                        attention_mask=causal_mask,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                    )
                    single_terminals.append(executor._terminal(hidden, attention_mask))
                prefix_terminals = []
                hidden = embeddings
                for layer in executor.backbone.layers:
                    hidden = layer(
                        hidden,
                        attention_mask=causal_mask,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                    )
                    prefix_terminals.append(executor._terminal(hidden, attention_mask))
                raw = torch.stack(
                    [torch.stack(single_terminals, dim=1), torch.stack(prefix_terminals, dim=1)],
                    dim=1,
                )
                rms = executor.backbone.norm(raw)
                native = executor.label_logits(rms)
                raw_batches.append(raw.to(dtype=torch.float16, device="cpu"))
                rms_batches.append(rms.to(dtype=torch.float16, device="cpu"))
                native_batches.append(native.float().cpu())
                print(
                    f"extract {split} {batch_start}:{batch_end}/{len(examples)}",
                    flush=True,
                )
            payload = {
                "config_hash": config_hash,
                "start": start,
                "end": end,
                "families": list(FAMILIES),
                "layers": list(range(1, n_layers + 1)),
                "raw": torch.cat(raw_batches, dim=0),
                "rms": torch.cat(rms_batches, dim=0),
                "native_logits": torch.cat(native_batches, dim=0),
                "labels": torch.tensor(
                    [item.answer_position for item in examples[start:end]], dtype=torch.long
                ),
                "choice_counts": torch.tensor(
                    [item.n_choices for item in examples[start:end]], dtype=torch.long
                ),
                "original_indices": expected_indices,
                "sample_ids": [item.sample_id for item in examples[start:end]],
                "elapsed_seconds": time.monotonic() - shard_t0,
            }
            atomic_torch_save(path, payload)
            journal.append(
                "feature_shard_completed",
                split=split,
                start=start,
                end=end,
                elapsed_seconds=payload["elapsed_seconds"],
            )
    del model
    torch.cuda.empty_cache()
    audit_extracted_features(config, output_root)


def analyse_variance(
    config: dict[str, Any], output_root: Path, journal: EventJournal
) -> None:
    threshold = float(
        config["variance"]["exp002_collapse_threshold_inter_sample_std_mean"]
    )
    rows = []
    for split in config["variance"]["report_splits"]:
        features = load_features(output_root, split)
        for transform in config["variance"]["report_transforms"]:
            tensor = features["raw" if transform == "raw" else "rms"]
            for family_index, family in enumerate(FAMILIES):
                for layer_index in range(tensor.shape[2]):
                    stats = variance_stats(tensor[:, family_index, layer_index])
                    stats["exp002_collapsed"] = (
                        stats["inter_sample_std_mean"] < threshold
                    )
                    rows.append(
                        {
                            "split": split,
                            "transform": transform,
                            "family": family,
                            "layer": layer_index + 1,
                            "path_id": (
                                f"single_L{layer_index + 1:02d}"
                                if family == "single_block"
                                else f"prefix_L{layer_index + 1:02d}"
                            ),
                            **stats,
                        }
                    )
    primary = [
        row
        for row in rows
        if row["split"] == config["variance"]["primary_split"]
        and row["transform"] == "raw"
    ]
    summary = {
        "primary_split": config["variance"]["primary_split"],
        "primary_transform": "raw",
        "threshold": threshold,
        "n_paths": len(primary),
        "n_collapsed": sum(bool(row["exp002_collapsed"]) for row in primary),
        "collapsed_path_ids": [row["path_id"] for row in primary if row["exp002_collapsed"]],
        "minimum_inter_sample_std_mean": min(
            row["inter_sample_std_mean"] for row in primary
        ),
        "minimum_path_id": min(primary, key=lambda row: row["inter_sample_std_mean"])[
            "path_id"
        ],
        "maximum_inter_sample_std_mean": max(
            row["inter_sample_std_mean"] for row in primary
        ),
    }
    atomic_write_json(output_root / "variance_stats.json", {"rows": rows})
    atomic_write_json(output_root / "variance_summary.json", summary)
    journal.append("variance_completed", **summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def smoke_paths(config: dict[str, Any]) -> list[dict[str, Any]]:
    reps = config["smoke"]["representative_paths"]
    return [
        spec
        for spec in path_specs(int(config["model"]["layers"]))
        if spec["layer"] in [int(value) for value in reps[spec["family"]]]
    ]


def feature_at(data: dict[str, Any], spec: dict[str, Any], key: str = "raw") -> torch.Tensor:
    family_index = FAMILIES.index(spec["family"])
    return data[key][:, family_index, int(spec["layer"]) - 1]


def run_smoke(config: dict[str, Any], output_root: Path, journal: EventJournal) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AdamW smoke")
    fit = load_features(output_root, "fit")
    seed = int(config["smoke"]["seed"])
    folds = torch.tensor(
        stratified_fold_ids(
            fit["labels"].tolist(),
            fit["choice_counts"].tolist(),
            int(config["smoke"]["internal_folds"]),
            seed,
        ),
        dtype=torch.long,
    )
    val_rows = folds == int(config["smoke"]["selection_fold"])
    train_rows = ~val_rows
    atomic_write_json(
        output_root / "smoke" / "folds.json",
        {
            "fold_ids": folds.tolist(),
            "fold_hash": canonical_json_hash(folds.tolist()),
            "selection_fold": int(config["smoke"]["selection_fold"]),
            "selection_is_D_fit_only": True,
        },
    )
    y = fit["labels"]
    counts = fit["choice_counts"]
    masks = valid_choice_mask(counts, 5)
    adam_cfg = config["smoke"]["adamw"]
    adam_path = output_root / "smoke" / "adamw_smoke.json"
    adam_rows = (
        json.loads(adam_path.read_text(encoding="utf-8"))["rows"]
        if adam_path.is_file()
        else []
    )
    done = {
        (row["probe"], row["path_id"], float(row["learning_rate"]))
        for row in adam_rows
    }
    for family in ("plain", "ln_plain"):
        for spec in smoke_paths(config):
            x = feature_at(fit, spec).float().to("cuda")
            for lr in [float(value) for value in adam_cfg["learning_rates"]]:
                key = (family, spec["path_id"], lr)
                if key in done:
                    continue
                result = train_adamw_smoke(
                    family=family,
                    train_x=x[train_rows].to("cuda"),
                    train_y=y[train_rows].to("cuda"),
                    train_mask=masks[train_rows].to("cuda"),
                    val_x=x[val_rows].to("cuda"),
                    val_y=y[val_rows].to("cuda"),
                    val_mask=masks[val_rows].to("cuda"),
                    lr=lr,
                    weight_decay=float(adam_cfg["weight_decay"]),
                    max_epochs=int(adam_cfg["max_epochs"]),
                    min_epochs=int(adam_cfg["min_epochs"]),
                    patience=int(adam_cfg["patience"]),
                    min_delta=float(adam_cfg["min_delta"]),
                    seed=seed,
                )
                result.pop("best_state", None)
                history = result.pop("history", [])
                row = {
                    "probe": family,
                    "path_id": spec["path_id"],
                    "path": spec["path"],
                    "learning_rate": lr,
                    "history_last": history[-1] if history else None,
                    **result,
                }
                adam_rows.append(row)
                atomic_write_json(adam_path, {"rows": adam_rows})
                journal.append("adamw_smoke_cell_completed", **row)
                print(
                    f"smoke {family} {spec['path_id']} lr={lr:g} acc={row['best_accuracy']:.4f}",
                    flush=True,
                )
            del x
            torch.cuda.empty_cache()

    selection: dict[str, Any] = {"D_fit_only": True, "adamw": {}, "ridge": {}}
    for family in ("plain", "ln_plain"):
        rows = [row for row in adam_rows if row["probe"] == family]
        selected_lr, summaries = select_smoke_value(rows, "learning_rate")
        epochs = [
            int(row["best_epoch"])
            for row in rows
            if float(row["learning_rate"]) == selected_lr and int(row["best_epoch"]) > 0
        ]
        if not epochs:
            raise RuntimeError(f"all AdamW smoke cells failed for {family}")
        selection["adamw"][family] = {
            "selected_learning_rate": selected_lr,
            "final_epochs": max(50, int(round(statistics.median(epochs)))),
            "summaries": summaries,
        }

    ridge_cfg = config["smoke"]["ridge"]
    ridge_path = output_root / "smoke" / "ridge_smoke.json"
    ridge_rows = (
        json.loads(ridge_path.read_text(encoding="utf-8"))["rows"]
        if ridge_path.is_file()
        else []
    )
    ridge_done = {(row["path_id"], float(row["alpha"])) for row in ridge_rows}
    train_numpy = train_rows.numpy()
    val_numpy = val_rows.numpy()
    for spec in smoke_paths(config):
        x = feature_at(fit, spec).float().numpy()
        for alpha in [float(value) for value in ridge_cfg["alphas"]]:
            if (spec["path_id"], alpha) in ridge_done:
                continue
            start = time.monotonic()
            _, decision = fit_ridge(
                x[train_numpy], y.numpy()[train_numpy], x[val_numpy], alpha
            )
            accuracy = masked_numpy_accuracy(
                decision, y.numpy()[val_numpy], counts.numpy()[val_numpy]
            )
            row = {
                "path_id": spec["path_id"],
                "path": spec["path"],
                "alpha": alpha,
                "best_accuracy": accuracy,
                "elapsed_seconds": time.monotonic() - start,
            }
            ridge_rows.append(row)
            atomic_write_json(ridge_path, {"rows": ridge_rows})
            journal.append("ridge_smoke_cell_completed", **row)
            print(
                f"smoke ridge {spec['path_id']} alpha={alpha:g} acc={accuracy:.4f}",
                flush=True,
            )
    values = sorted({float(row["alpha"]) for row in ridge_rows})
    ridge_summaries = {
        str(value): {
            "mean_accuracy": statistics.fmean(
                row["best_accuracy"]
                for row in ridge_rows
                if float(row["alpha"]) == value
            ),
            "n": sum(float(row["alpha"]) == value for row in ridge_rows),
        }
        for value in values
    }
    selected_alpha = max(
        values, key=lambda value: (ridge_summaries[str(value)]["mean_accuracy"], -value)
    )
    selection["ridge"] = {
        "selected_alpha": selected_alpha,
        "summaries": ridge_summaries,
    }
    atomic_write_json(output_root / "smoke" / "selection.json", selection)
    journal.append("smoke_selection_completed", selection=selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


def native_result(
    data: dict[str, Any], spec: dict[str, Any], split: str
) -> dict[str, Any]:
    logits = feature_at(data, spec, "native_logits")
    mask = valid_choice_mask(data["choice_counts"], 5)
    return {
        f"{split}_accuracy": masked_accuracy(logits, data["labels"], mask),
        f"{split}_predictions": logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        .argmax(dim=1)
        .tolist(),
    }


def run_probes(config: dict[str, Any], output_root: Path, journal: EventJournal) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full probes")
    fit = load_features(output_root, "fit")
    discover = load_features(output_root, "discover")
    selection_path = output_root / "smoke" / "selection.json"
    if not selection_path.is_file():
        raise RuntimeError("smoke selection is required before full probes")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result_root = output_root / "probe_paths"
    result_root.mkdir(parents=True, exist_ok=True)
    seed = int(config["runtime"]["seed"])
    fit_y_gpu = fit["labels"].to("cuda")
    discover_y_gpu = discover["labels"].to("cuda")
    fit_mask_gpu = valid_choice_mask(fit["choice_counts"], 5).to("cuda")
    discover_mask_gpu = valid_choice_mask(discover["choice_counts"], 5).to("cuda")
    for spec in path_specs(int(config["model"]["layers"])):
        output_path = result_root / f"{spec['path_id']}.json"
        prediction_path = result_root / f"{spec['path_id']}_predictions.pt"
        if output_path.is_file() and prediction_path.is_file():
            print(f"reuse probes {spec['path_id']}", flush=True)
            continue
        record: dict[str, Any] = {**spec, "methods": {}}
        predictions: dict[str, Any] = {
            "path_id": spec["path_id"],
            "labels": discover["labels"],
            "choice_counts": discover["choice_counts"],
        }
        native = native_result(fit, spec, "fit")
        native.update(native_result(discover, spec, "discover"))
        predictions["native"] = torch.tensor(native.pop("discover_predictions"))
        native.pop("fit_predictions")
        record["methods"]["native"] = native

        raw_fit = feature_at(fit, spec).float()
        raw_discover = feature_at(discover, spec).float()
        for probe in ("plain", "ln_plain"):
            chosen = selection["adamw"][probe]
            result = train_adamw_fixed(
                family=probe,
                train_x=raw_fit.to("cuda"),
                train_y=fit_y_gpu,
                train_mask=fit_mask_gpu,
                eval_x=raw_discover.to("cuda"),
                eval_y=discover_y_gpu,
                eval_mask=discover_mask_gpu,
                lr=float(chosen["selected_learning_rate"]),
                weight_decay=float(config["smoke"]["adamw"]["weight_decay"]),
                epochs=int(chosen["final_epochs"]),
                seed=seed,
            )
            logits = result.pop("eval_logits")
            predictions[probe] = logits.argmax(dim=1)
            record["methods"][probe] = {
                **result,
                "learning_rate": float(chosen["selected_learning_rate"]),
                "epochs": int(chosen["final_epochs"]),
            }

        alpha = float(selection["ridge"]["selected_alpha"])
        ridge, ridge_decision = fit_ridge(
            raw_fit.numpy(), fit["labels"].numpy(), raw_discover.numpy(), alpha
        )
        fit_decision = ridge.decision_function(raw_fit.numpy())
        from src.frag_early_readability import expand_ridge_decision

        fit_decision = expand_ridge_decision(ridge, raw_fit.numpy(), 5)
        record["methods"]["ridge"] = {
            "alpha": alpha,
            "train_accuracy": masked_numpy_accuracy(
                fit_decision, fit["labels"].numpy(), fit["choice_counts"].numpy()
            ),
            "eval_accuracy": masked_numpy_accuracy(
                ridge_decision,
                discover["labels"].numpy(),
                discover["choice_counts"].numpy(),
            ),
        }
        masked_ridge = ridge_decision.copy()
        for index, count in enumerate(discover["choice_counts"].tolist()):
            masked_ridge[index, int(count) :] = -1e30
        predictions["ridge"] = torch.from_numpy(masked_ridge.argmax(axis=1))

        rms_result = fit_masked_linear_head(
            feature_at(fit, spec, "rms").to("cuda"),
            fit_y_gpu,
            fit_mask_gpu,
            feature_at(discover, spec, "rms").to("cuda"),
            discover_y_gpu,
            discover_mask_gpu,
            l2=float(config["probes"]["h1_rms_lbfgs"]["l2"]),
            max_iter=int(config["probes"]["h1_rms_lbfgs"]["max_iter"]),
            tolerance_grad=float(config["probes"]["h1_rms_lbfgs"]["tolerance_grad"]),
        )
        weight = rms_result.pop("weight")
        rms_logits = torch.nn.functional.linear(
            feature_at(discover, spec, "rms").float(), weight.float()
        )
        rms_mask = valid_choice_mask(discover["choice_counts"], 5)
        predictions["h1_rms_lbfgs"] = rms_logits.masked_fill(
            ~rms_mask, torch.finfo(rms_logits.dtype).min
        ).argmax(dim=1)
        record["methods"]["h1_rms_lbfgs"] = rms_result

        atomic_torch_save(prediction_path, predictions)
        atomic_write_json(output_path, record)
        journal.append(
            "path_probes_completed",
            path_id=spec["path_id"],
            discover_accuracies={
                name: values.get("eval_accuracy", values.get("discover_accuracy"))
                for name, values in record["methods"].items()
            },
        )
        print(f"probes completed {spec['path_id']}", flush=True)
        torch.cuda.empty_cache()


def stable_onset(curve: list[dict[str, Any]], threshold: float) -> int | None:
    for index in range(len(curve) - 2):
        if all(row["accuracy"] >= threshold for row in curve[index : index + 3]):
            return int(curve[index]["layer"])
    return None


def build_report(config: dict[str, Any], output_root: Path, journal: EventJournal) -> None:
    result_paths = sorted((output_root / "probe_paths").glob("*_L??.json"))
    if len(result_paths) != int(config["paths"]["n_paths"]):
        raise RuntimeError(f"expected 56 probe results, found {len(result_paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    methods = ["native", "plain", "ln_plain", "ridge", "h1_rms_lbfgs"]
    curves: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {family: [] for family in FAMILIES} for method in methods
    }
    for record in records:
        for method in methods:
            values = record["methods"][method]
            accuracy = values.get("eval_accuracy", values.get("discover_accuracy"))
            curves[method][record["family"]].append(
                {"layer": record["layer"], "accuracy": accuracy, "path_id": record["path_id"]}
            )
    for method in methods:
        for family in FAMILIES:
            curves[method][family].sort(key=lambda row: row["layer"])
    onsets = {}
    for method in methods:
        canonical = curves[method]["canonical_prefix"]
        baseline = canonical[-1]["accuracy"]
        threshold = baseline - 0.05
        onsets[method] = {
            "prefix_L28_accuracy": baseline,
            "within_5pp_threshold": threshold,
            "first_prefix_within_5pp": next(
                (row["layer"] for row in canonical if row["accuracy"] >= threshold), None
            ),
            "first_three_consecutive_prefixes_within_5pp": stable_onset(
                canonical, threshold
            ),
        }
    variance = json.loads((output_root / "variance_summary.json").read_text(encoding="utf-8"))
    selection = json.loads((output_root / "smoke" / "selection.json").read_text(encoding="utf-8"))
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "completed_fragmented_diagnostic",
        "diagnostic_only": True,
        "official_hypothesis_evidence": False,
        "data": "ARC-Easy official train D_fit/D_discover only",
        "validation_accessed": False,
        "test_accessed": False,
        "variance": variance,
        "smoke_selection": selection,
        "early_exit_onsets": onsets,
        "curves": curves,
    }
    atomic_write_json(output_root / "summary.json", summary)

    lines = [
        f"# {config['experiment_id']}",
        "",
        "## Scope",
        "",
        "Fragmented diagnostic only. It does not modify or confirm EXP-004 H1. "
        "Only ARC-Easy official-train D_fit and D_discover were used; official validation and test were not accessed.",
        "",
        "## Pre-registered execution order",
        "",
        "1. Extract raw pre-final-RMSNorm last-token features for 28 single-block and 28 canonical-prefix paths.",
        "2. Audit inter-sample variance before fitting any readout.",
        "3. Select AdamW learning rates and RidgeClassifier alpha on one deterministic D_fit-only internal fold.",
        "4. Freeze those values, refit on all D_fit, and evaluate once on D_discover.",
        "",
        "## Variance audit",
        "",
        f"EXP-002 collapse threshold: `{variance['threshold']}`. Collapsed raw D_fit paths: "
        f"**{variance['n_collapsed']}/{variance['n_paths']}**. Minimum inter-sample std mean: "
        f"`{variance['minimum_inter_sample_std_mean']:.6g}` at `{variance['minimum_path_id']}`.",
        "",
        "## D_fit-only smoke selection",
        "",
        f"- Plain AdamW: lr `{selection['adamw']['plain']['selected_learning_rate']}`, "
        f"{selection['adamw']['plain']['final_epochs']} fixed full-data epochs.",
        f"- LN-Plain AdamW: lr `{selection['adamw']['ln_plain']['selected_learning_rate']}`, "
        f"{selection['adamw']['ln_plain']['final_epochs']} fixed full-data epochs.",
        f"- RidgeClassifier: alpha `{selection['ridge']['selected_alpha']}`.",
        "",
        "## Canonical-prefix early-exit summary on D_discover",
        "",
        "| Readout | Prefix L28 | First within 5 pp | First 3 consecutive within 5 pp |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        item = onsets[method]
        lines.append(
            f"| {method} | {item['prefix_L28_accuracy']:.4f} | "
            f"{item['first_prefix_within_5pp']} | "
            f"{item['first_three_consecutive_prefixes_within_5pp']} |"
        )
    lines.extend(
        [
            "",
            "## Complete D_discover curves",
            "",
            "| Family | Layer | Native | Plain | LN-Plain | Ridge | H1 RMS+LBFGS |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    record_by_key = {(record["family"], record["layer"]): record for record in records}
    for family in FAMILIES:
        for layer in range(1, int(config["model"]["layers"]) + 1):
            record = record_by_key[(family, layer)]
            values = []
            for method in methods:
                item = record["methods"][method]
                values.append(item.get("eval_accuracy", item.get("discover_accuracy")))
            lines.append(
                f"| {family} | {layer} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A non-collapsed variance result rules out EXP-001-style gross feature collapse, but does not establish task information. "
            "Ridge/LN-Plain success with Plain failure supports an optimisation/conditioning explanation. Failure of all linear readouts "
            "supports absence of linearly decodable task signal, not absence of all information. This diagnostic has no official-test claim.",
            "",
        ]
    )
    report_path = output_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    journal.append("report_completed", report=str(report_path))


def run() -> int:
    args = parse_args()
    config_path = resolved(args.config).resolve()
    config = load_yaml(config_path)
    assert_scope(config)
    output_root, manifest, journal, config_hash = prepare_run(
        config, config_path, resume=args.resume
    )
    selected = list(PHASES) if args.phase == "all" else [args.phase]
    try:
        for phase in selected:
            journal.append("phase_started", phase=phase)
            if phase == "features":
                extract_features(config, output_root, journal, config_hash)
            elif phase == "variance":
                analyse_variance(config, output_root, journal)
            elif phase == "smoke":
                run_smoke(config, output_root, journal)
            elif phase == "probes":
                run_probes(config, output_root, journal)
            elif phase == "report":
                build_report(config, output_root, journal)
            finish_phase(output_root, manifest, journal, phase)
        manifest["status"] = "completed" if args.phase == "all" else "phase_completed"
        manifest["finished_at"] = now()
        atomic_write_json(output_root / "run_manifest.json", manifest)
        return 0
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["last_error"] = {"type": type(exc).__name__, "message": str(exc), "time": now()}
        atomic_write_json(output_root / "run_manifest.json", manifest)
        journal.append("run_stopped", **manifest["last_error"])
        raise


if __name__ == "__main__":
    raise SystemExit(run())
