#!/usr/bin/env python3
"""Run the resumable EXP-004 H1 engineering qualification.

This entrypoint intentionally supports only canonical-path qualification.  The
adaptive H1 search must not be added until the final user-authored protocol has
frozen its remaining search choices.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
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
    fit_masked_linear_head,
    format_arc_prompt,
    git_state,
    load_arc_easy_split,
    load_yaml,
    make_fit_discover_indices,
    masked_accuracy,
    valid_choice_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/exp004_h1_qualification.yaml"
    )
    parser.add_argument("--stage", choices=["equivalence", "canonical"], required=True)
    parser.add_argument("--stop-at", required=True, help="ISO datetime with UTC offset")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reserve-minutes",
        type=int,
        default=None,
        help="override config soft-stop reserve",
    )
    return parser.parse_args()


def _resolved(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "input_ids"):
        ids = value.input_ids
    elif isinstance(value, dict):
        ids = value["input_ids"]
    else:
        ids = value
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(item) for item in ids]


def verify_answer_tokens(tokenizer: Any, config: dict[str, Any]) -> list[int]:
    """Verify the actual token immediately following the assistant header."""
    prompt_cfg = config["prompt"]
    example_text = "Question: 1+1?\nA. 1\nB. 2\n\nAnswer:"
    messages = [{"role": "user", "content": example_text}]
    base = _token_ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True
        )
    )
    observed: list[int] = []
    for label in config["tokenization"]["answer_labels"]:
        full = _token_ids(
            tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": label}],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
            )
        )
        if full[: len(base)] != base or len(full) <= len(base):
            raise RuntimeError(f"chat-template boundary is not prefix-stable for {label}")
        observed.append(full[len(base)])
    expected = [int(item) for item in config["tokenization"]["expected_answer_token_ids"]]
    if observed != expected:
        raise RuntimeError(f"answer-token mismatch: observed={observed}, expected={expected}")
    return observed


def encode_examples(tokenizer: Any, examples: Sequence[Any], prompt_cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    encoded = []
    for example in examples:
        prompt = format_arc_prompt(example, prompt_cfg)
        messages = [{"role": "user", "content": prompt}]
        ids = _token_ids(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True
            )
        )
        encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return tokenizer.pad(encoded, padding=True, return_tensors="pt")


def ensure_run_manifest(
    run_root: Path,
    config: dict[str, Any],
    config_path: Path,
    config_hash: str,
    resume: bool,
) -> dict[str, Any]:
    manifest_path = run_root / "run_manifest.json"
    current_git = git_state()
    if manifest_path.exists():
        if not resume:
            raise RuntimeError(f"run already exists; pass --resume: {run_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_hash"] != config_hash:
            raise RuntimeError("refusing resume: resolved configuration hash changed")
        if manifest["git"]["commit"] != current_git["commit"]:
            raise RuntimeError("refusing resume: Git commit changed")
        return manifest
    if resume:
        raise RuntimeError(f"cannot resume a missing run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    with (run_root / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config_source": str(config_path),
        "config_hash": config_hash,
        "git": current_git,
        "environment": environment_summary(),
        "official_hypothesis_evidence": False,
        "test_access_allowed": False,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def start_session(
    run_root: Path,
    stage: str,
    deadline: DeadlineController,
    config_hash: str,
) -> tuple[Path, EventJournal, dict[str, Any]]:
    session_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    session_path = run_root / "sessions" / f"{session_id}.json"
    session = {
        "session_id": session_id,
        "stage": stage,
        "status": "running",
        "pid": os.getpid(),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hard_stop": deadline.hard_stop.isoformat(),
        "soft_stop": deadline.soft_stop.isoformat(),
        "reserve_minutes": deadline.reserve_minutes,
        "config_hash": config_hash,
    }
    atomic_write_json(session_path, session)
    journal = EventJournal(run_root / "events.jsonl")
    journal.append("session_started", **session)
    return session_path, journal, session


def finish_session(
    session_path: Path,
    journal: EventJournal,
    session: dict[str, Any],
    status: str,
    **details: Any,
) -> None:
    session.update(
        {
            "status": status,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            **details,
        }
    )
    atomic_write_json(session_path, session)
    journal.append("session_finished", **session)


def equivalence_check(
    executor: ModularLlamaExecutor,
    tokenizer: Any,
    examples: Sequence[Any],
    config: dict[str, Any],
) -> dict[str, float]:
    batch = encode_examples(tokenizer, examples, config["prompt"])
    device = next(executor.causal_lm.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    canonical = config["path"]["canonical"]
    native_features, native_logits = executor.forward_native(input_ids, attention_mask)
    modular_features, modular_logits = executor.forward_path(
        input_ids, attention_mask, canonical
    )
    feature_abs = (native_features.float() - modular_features.float()).abs()
    logit_abs = (native_logits.float() - modular_logits.float()).abs()
    result = {
        "feature_max_abs": float(feature_abs.max().item()),
        "feature_mean_abs": float(feature_abs.mean().item()),
        "label_logit_max_abs": float(logit_abs.max().item()),
        "label_logit_mean_abs": float(logit_abs.mean().item()),
    }
    qual = config["qualification"]
    if result["feature_max_abs"] > float(qual["equivalence_max_abs_tolerance"]):
        raise RuntimeError(f"canonical feature equivalence failed: {result}")
    if result["feature_mean_abs"] > float(qual["equivalence_mean_abs_tolerance"]):
        raise RuntimeError(f"canonical feature mean error failed: {result}")
    if result["label_logit_max_abs"] > float(qual["equivalence_max_abs_tolerance"]):
        raise RuntimeError(f"canonical label-logit equivalence failed: {result}")
    return result


def extract_split(
    name: str,
    examples: Sequence[Any],
    original_indices: Sequence[int],
    *,
    executor: ModularLlamaExecutor,
    tokenizer: Any,
    config: dict[str, Any],
    config_hash: str,
    run_root: Path,
    deadline: DeadlineController,
    journal: EventJournal,
) -> dict[str, Any]:
    qual = config["qualification"]
    batch_size = int(qual["batch_size"])
    shard_size = int(qual["shard_size"])
    path = config["path"]["canonical"]
    feature_root = run_root / "features" / "canonical" / name
    feature_root.mkdir(parents=True, exist_ok=True)
    device = next(executor.causal_lm.parameters()).device
    batch_durations: list[float] = []
    n_reused = 0

    for shard_start in range(0, len(examples), shard_size):
        next_estimate = max(batch_durations[-5:] or [60.0])
        deadline.checkpoint(next_unit_seconds=next_estimate)
        shard_end = min(shard_start + shard_size, len(examples))
        shard_path = feature_root / f"shard_{shard_start:05d}_{shard_end:05d}.pt"
        expected_indices = list(original_indices[shard_start:shard_end])
        if shard_path.exists():
            saved = torch.load(shard_path, map_location="cpu", weights_only=False)
            if saved["config_hash"] != config_hash or saved["original_indices"] != expected_indices:
                raise RuntimeError(f"resume validation failed for {shard_path}")
            n_reused += shard_end - shard_start
            continue

        shard_examples = examples[shard_start:shard_end]
        features: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        journal.append(
            "shard_started",
            split=name,
            start=shard_start,
            end=shard_end,
            original_indices=expected_indices,
        )
        shard_t0 = time.monotonic()
        for batch_start in range(0, len(shard_examples), batch_size):
            next_estimate = max(batch_durations[-5:] or [60.0])
            deadline.checkpoint(next_unit_seconds=next_estimate)
            batch_examples = shard_examples[batch_start : batch_start + batch_size]
            encoded = encode_examples(tokenizer, batch_examples, config["prompt"])
            if encoded["input_ids"].shape[1] > int(config["tokenization"]["max_length_guard"]):
                raise RuntimeError(
                    f"token length {encoded['input_ids'].shape[1]} exceeds max_length_guard"
                )
            input_ids = encoded["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
            batch_t0 = time.monotonic()
            terminal, label_logits = executor.forward_path(input_ids, attention_mask, path)
            torch.cuda.synchronize(device)
            batch_durations.append(time.monotonic() - batch_t0)
            features.append(terminal.to(dtype=torch.float16, device="cpu"))
            logits.append(label_logits.float().cpu())

        payload = {
            "config_hash": config_hash,
            "split": name,
            "path": list(path),
            "start": shard_start,
            "end": shard_end,
            "original_indices": expected_indices,
            "sample_ids": [item.sample_id for item in shard_examples],
            "labels": torch.tensor(
                [item.answer_position for item in shard_examples], dtype=torch.long
            ),
            "choice_counts": torch.tensor(
                [item.n_choices for item in shard_examples], dtype=torch.int8
            ),
            "features": torch.cat(features, dim=0),
            "native_label_logits": torch.cat(logits, dim=0),
            "elapsed_seconds": time.monotonic() - shard_t0,
            "batch_durations_seconds": batch_durations[-len(features) :],
        }
        atomic_torch_save(shard_path, payload)
        journal.append(
            "shard_completed",
            split=name,
            start=shard_start,
            end=shard_end,
            path=str(shard_path.relative_to(run_root)),
            elapsed_seconds=payload["elapsed_seconds"],
        )

    return {
        "split": name,
        "n_samples": len(examples),
        "n_reused": n_reused,
        "batch_durations_seconds": batch_durations,
    }


def load_feature_split(run_root: Path, name: str) -> dict[str, Any]:
    shard_paths = sorted((run_root / "features" / "canonical" / name).glob("shard_*.pt"))
    if not shard_paths:
        raise RuntimeError(f"no completed shards for {name}")
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]
    expected = 0
    for shard in shards:
        if shard["start"] != expected:
            raise RuntimeError(f"non-contiguous shards in {name}: expected {expected}")
        expected = shard["end"]
    return {
        "features": torch.cat([item["features"] for item in shards]),
        "native_label_logits": torch.cat([item["native_label_logits"] for item in shards]),
        "labels": torch.cat([item["labels"] for item in shards]),
        "choice_counts": torch.cat([item["choice_counts"] for item in shards]).long(),
        "sample_ids": sum((item["sample_ids"] for item in shards), []),
    }


def run() -> int:
    args = parse_args()
    config_path = _resolved(ROOT, args.config).resolve()
    config = load_yaml(config_path)
    if config["status"] != "engineering_qualification_not_hypothesis_evidence":
        raise RuntimeError("this runner only accepts explicitly non-official qualification configs")
    if config["runtime"].get("allow_test", False):
        raise RuntimeError("test access is forbidden in H1 qualification")
    config_hash = canonical_json_hash(config)
    reserve = args.reserve_minutes or int(config["runtime"]["reserve_minutes"])
    deadline = DeadlineController(args.stop_at, reserve)
    deadline.install_signal_handlers()

    run_root = _resolved(ROOT, config["runtime"]["artifact_root"]).resolve()
    ensure_run_manifest(run_root, config, config_path, config_hash, args.resume)
    session_path, journal, session = start_session(
        run_root, args.stage, deadline, config_hash
    )
    print(
        f"[{config['experiment_id']}] stage={args.stage} hard_stop={deadline.hard_stop.isoformat()} "
        f"soft_stop={deadline.soft_stop.isoformat()}"
    )

    model = None
    try:
        deadline.checkpoint(next_unit_seconds=180.0)
        seed = int(config["runtime"]["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if config["runtime"].get("deterministic", True):
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False

        dataset_root = _resolved(ROOT, config["dataset"]["path"])
        train_examples = load_arc_easy_split(dataset_root, "train")
        expected_train = int(config["dataset"]["official_train_size"])
        if len(train_examples) != expected_train:
            raise RuntimeError(f"official train size changed: {len(train_examples)}")
        split_indices = make_fit_discover_indices(
            len(train_examples), int(config["split"]["fit_size"]), int(config["split"]["seed"])
        )
        if len(split_indices["discover"]) != int(config["split"]["discover_size"]):
            raise RuntimeError("configured discover size does not match train-fit")
        split_document = {
            **split_indices,
            "seed": int(config["split"]["seed"]),
            "source": "official_train_only",
            "hash": canonical_json_hash(split_indices),
        }
        split_path = run_root / "split_indices.json"
        if split_path.exists():
            existing = json.loads(split_path.read_text(encoding="utf-8"))
            if existing != split_document:
                raise RuntimeError("persisted fit/discover split changed")
        else:
            atomic_write_json(split_path, split_document)

        model_path = _resolved(ROOT, config["model"]["path"])
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer.padding_side = config["tokenization"]["padding_side"]
        answer_token_ids = verify_answer_tokens(tokenizer, config)
        journal.append("answer_tokens_verified", answer_token_ids=answer_token_ids)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this qualification")
        deadline.checkpoint(next_unit_seconds=180.0)
        torch.cuda.reset_peak_memory_stats()
        dtype_name = config["model"]["dtype"]
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
        load_t0 = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=config["model"]["attention_implementation"],
        ).to("cuda")
        model.eval()
        model.config.use_cache = False
        load_seconds = time.monotonic() - load_t0
        if len(model.model.layers) != int(config["model"]["expected_layers"]):
            raise RuntimeError("unexpected Llama layer count")
        executor = ModularLlamaExecutor(model, answer_token_ids)
        journal.append("model_loaded", elapsed_seconds=load_seconds)

        equivalence_n = int(config["qualification"]["equivalence_batch_size"])
        equivalence_examples = [train_examples[i] for i in split_indices["discover"][:equivalence_n]]
        equivalence = equivalence_check(executor, tokenizer, equivalence_examples, config)
        equivalence.update(
            {
                "n_samples": equivalence_n,
                "answer_token_ids": answer_token_ids,
                "model_load_seconds": load_seconds,
            }
        )
        atomic_write_json(run_root / "canonical_equivalence.json", equivalence)
        journal.append("canonical_equivalence_passed", **equivalence)
        print(json.dumps({"canonical_equivalence": equivalence}, indent=2))

        if args.stage == "equivalence":
            finish_session(
                session_path,
                journal,
                session,
                "completed",
                equivalence=equivalence,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            )
            return 0

        extraction_results = []
        for split_name in ("fit", "discover"):
            indices = split_indices[split_name]
            examples = [train_examples[index] for index in indices]
            extraction_results.append(
                extract_split(
                    split_name,
                    examples,
                    indices,
                    executor=executor,
                    tokenizer=tokenizer,
                    config=config,
                    config_hash=config_hash,
                    run_root=run_root,
                    deadline=deadline,
                    journal=journal,
                )
            )

        deadline.checkpoint(next_unit_seconds=300.0)
        fit = load_feature_split(run_root, "fit")
        discover = load_feature_split(run_root, "discover")
        device = torch.device("cuda")
        fit_mask = valid_choice_mask(fit["choice_counts"], 5)
        discover_mask = valid_choice_mask(discover["choice_counts"], 5)
        native_fit_acc = masked_accuracy(
            fit["native_label_logits"], fit["labels"], fit_mask
        )
        native_discover_acc = masked_accuracy(
            discover["native_label_logits"], discover["labels"], discover_mask
        )
        head_cfg = config["head"]
        head_result = fit_masked_linear_head(
            fit["features"].to(device),
            fit["labels"].to(device),
            fit_mask.to(device),
            discover["features"].to(device),
            discover["labels"].to(device),
            discover_mask.to(device),
            l2=float(head_cfg["l2"]),
            max_iter=int(head_cfg["max_iter"]),
            tolerance_grad=float(head_cfg["tolerance_grad"]),
        )
        atomic_torch_save(
            run_root / "canonical_task_head.pt",
            {
                "config_hash": config_hash,
                "weight": head_result.pop("weight"),
                "head_config": head_cfg,
            },
        )
        batch_times = sum(
            (item["batch_durations_seconds"] for item in extraction_results), []
        )
        batch_times_sorted = sorted(batch_times)
        p95 = (
            batch_times_sorted[round(0.95 * (len(batch_times_sorted) - 1))]
            if batch_times_sorted
            else None
        )
        metrics = {
            "status": config["status"],
            "official_hypothesis_evidence": False,
            "split_hash": split_document["hash"],
            "fit_size": len(fit["labels"]),
            "discover_size": len(discover["labels"]),
            "chance_accuracy_fit": chance_accuracy(fit["choice_counts"].tolist()),
            "chance_accuracy_discover": chance_accuracy(discover["choice_counts"].tolist()),
            "canonical_native_accuracy_fit": native_fit_acc,
            "canonical_native_accuracy_discover": native_discover_acc,
            "canonical_task_head": head_result,
            "equivalence": equivalence,
            "model_load_seconds": load_seconds,
            "forward_batch_seconds_p95": p95,
            "forward_batch_seconds_max": max(batch_times) if batch_times else None,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "test_accessed": False,
        }
        atomic_write_json(run_root / "qualification_metrics.json", metrics)
        journal.append("canonical_qualification_completed", **metrics)
        print(json.dumps(metrics, indent=2))
        finish_session(
            session_path, journal, session, "completed", metrics_path="qualification_metrics.json"
        )
        return 0
    except DeadlineReached as exc:
        finish_session(
            session_path,
            journal,
            session,
            "soft_stopped",
            reason=str(exc),
        )
        print(f"[soft-stop] {exc}")
        return 75
    except BaseException as exc:
        finish_session(
            session_path,
            journal,
            session,
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(run())
