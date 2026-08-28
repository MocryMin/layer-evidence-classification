#!/usr/bin/env python3
"""Frozen, resumable five-source train-only discovery for EXP-004 H1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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
    extract_cached_path_feature_split,
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
from src.exp004_h1_cache import GlobalPrefixCache  # noqa: E402
from src.exp004_h1_search import (  # noqa: E402
    SOURCE_ORDER,
    edit_distance,
    keep_throttled_source_turn,
    path_key,
    propose_candidate,
    source_temperature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp004_h1_frozen.yaml")
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def resolved(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[int]]]:
    root = resolved(config["source_qualification"]["path"])
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["config_hash"] != config["source_qualification"]["config_hash"]:
        raise RuntimeError("source qualification config changed")
    source_config = load_yaml(root / "resolved_config.yaml")
    if canonical_json_hash(source_config) != config["source_qualification"]["config_hash"]:
        raise RuntimeError("source qualification resolved config no longer matches its manifest")
    split_document = json.loads((root / "split_indices.json").read_text(encoding="utf-8"))
    split_hash = canonical_json_hash(
        {"fit": split_document.get("fit"), "discover": split_document.get("discover")}
    )
    if split_hash != config["split"]["indices_sha256"]:
        raise RuntimeError(f"split hash changed: {split_hash}")
    head_root = resolved(config["head_qualification"]["path"])
    head_manifest = json.loads((head_root / "run_manifest.json").read_text(encoding="utf-8"))
    head_summary = json.loads((head_root / "head_qualification.json").read_text(encoding="utf-8"))
    if head_manifest["config_hash"] != config["head_qualification"]["config_hash"]:
        raise RuntimeError("head qualification config changed")
    if float(head_summary["selected_l2"]) != float(config["head_qualification"]["selected_l2"]):
        raise RuntimeError("frozen head L2 no longer matches qualification")
    return source_config, split_document


def initial_state(config_hash: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "schema_version": 1,
        "config_hash": config_hash,
        "rng_state": rng.bit_generator.state,
        "source_cursor": 0,
        "gpu_seconds_used": 0.0,
        "completed_search_candidates": 0,
        "candidate_ordinal": 0,
        "skipped_source_turns": {source: 0 for source in SOURCE_ORDER},
        "throttle_skips": {source: 0 for source in SOURCE_ORDER},
        "source_good_counts": {source: 0 for source in SOURCE_ORDER},
        "populations": {source: [] for source in SOURCE_ORDER},
        "bootstrap": {"canonical": False, "repeat_L28": False, "S4_seed": False, "S5_seed": False},
        "inflight": None,
        "campaign_status": "running",
    }


def prepare_run(
    output_root: Path,
    config: dict[str, Any],
    config_hash: str,
    deadline: DeadlineController,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = output_root / "run_manifest.json"
    state_path = output_root / "search_state.json"
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    if output_root.exists():
        if not resume:
            raise RuntimeError(f"discovery exists; pass --resume: {output_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if manifest["config_hash"] != config_hash or state["config_hash"] != config_hash:
            raise RuntimeError("refusing resume: frozen configuration changed")
        state.setdefault("throttle_skips", {source: 0 for source in SOURCE_ORDER})
        state.setdefault("source_good_counts", {source: 0 for source in SOURCE_ORDER})
        manifest.setdefault("sessions", []).append(
            {"started_at": now, "hard_stop": deadline.hard_stop.isoformat(), "git": git_state()}
        )
        state["campaign_status"] = "running"
    else:
        if resume:
            raise RuntimeError(f"cannot resume missing discovery: {output_root}")
        output_root.mkdir(parents=True)
        manifest = {
            "experiment_id": config["experiment_id"],
            "protocol_status": config["status"],
            "config_hash": config_hash,
            "git_at_freeze": git_state(),
            "environment": environment_summary(),
            "started_at": now,
            "official_hypothesis_evidence": "train_discovery_only",
            "validation_accessed": False,
            "test_accessed": False,
            "sessions": [{"started_at": now, "hard_stop": deadline.hard_stop.isoformat(), "git": git_state()}],
        }
        state = initial_state(config_hash, int(config["search"]["seed"]))
        atomic_write_json(output_root / "resolved_config.yaml.json", config)
    manifest["status"] = "running"
    manifest["hard_stop"] = deadline.hard_stop.isoformat()
    manifest["soft_stop"] = deadline.soft_stop.isoformat()
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(state_path, state)
    return manifest, state


def load_results(output_root: Path) -> list[dict[str, Any]]:
    root = output_root / "results"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))] if root.exists() else []


def artifact_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def non_cache_artifact_bytes(output_root: Path) -> int:
    """Measure hypothesis artifacts without walking the separately capped cache."""
    total = 0
    for name in ("features", "heads", "results"):
        target = output_root / name
        if target.exists():
            total += artifact_bytes(target)
    for item in output_root.iterdir():
        if item.is_file() and not item.name.startswith("prefix_cache_index.sqlite3"):
            total += item.stat().st_size
    return total


def add_population_entry(state: dict[str, Any], source: str, result: dict[str, Any]) -> None:
    entries = state["populations"][source]
    if not any(entry["path_id"] == result["path_id"] for entry in entries):
        entries.append(
            {
                "path_id": result["path_id"],
                "path": result["path"],
                "task_accuracy_discover": result["task_accuracy_discover"],
            }
        )


def summary_document(
    output_root: Path,
    state: dict[str, Any],
    canonical_task: float | None,
    canonical_native: float | None,
    chance: float | None,
    prefix_cache: GlobalPrefixCache | None = None,
) -> dict[str, Any]:
    results = load_results(output_root)
    discovered = [item for item in results if item.get("count_in_prevalence", False)]
    good = [item for item in discovered if item["good_path"]]
    collapsed = [item for item in good if item["readability_collapse"]]
    by_source = {}
    for source in SOURCE_ORDER:
        source_items = [item for item in discovered if item["source"] == source]
        source_good = [item for item in source_items if item["good_path"]]
        source_gap = [item for item in source_good if item["readability_collapse"]]
        by_source[source] = {
            "n": len(source_items),
            "n_good": len(source_good),
            "n_gap_among_good": len(source_gap),
            "p_gap_among_good": len(source_gap) / len(source_good) if source_good else None,
        }
    control = next((item for item in results if item["path_id"] == "repeat_L28"), None)
    return {
        "status": state["campaign_status"],
        "n_results_total": len(results),
        "n_discovered_candidates": len(discovered),
        "n_good": len(good),
        "n_readability_collapse_among_good": len(collapsed),
        "p_gap_among_good": len(collapsed) / len(good) if good else None,
        "by_source": by_source,
        "repeat_L28_control": control,
        "canonical_task_accuracy_discover": canonical_task,
        "canonical_native_accuracy_discover": canonical_native,
        "chance_accuracy_discover": chance,
        "gpu_seconds_used": state["gpu_seconds_used"],
        "gpu_hours_used": state["gpu_seconds_used"] / 3600.0,
        "source_good_counts_for_temperature": state.get("source_good_counts"),
        "operational_throttle_skips": state.get("throttle_skips"),
        "prefix_cache": None if prefix_cache is None else prefix_cache.stats(),
        "validation_accessed": False,
        "test_accessed": False,
    }


def run() -> int:
    args = parse_args()
    config_path = resolved(args.config).resolve()
    config = load_yaml(config_path)
    accepted_statuses = {
        "frozen_official_train_discovery",
        "frozen_sourcewise_rerun_train_discovery",
    }
    if config["status"] not in accepted_statuses:
        raise RuntimeError("official discovery requires the frozen status marker")
    if config["runtime"].get("allow_validation") or config["runtime"].get("allow_test"):
        raise RuntimeError("validation/test access is forbidden")
    if tuple(config["search"]["source_order"]) != SOURCE_ORDER:
        raise RuntimeError("source order changed")
    protocol_path = Path(config["protocol_document"]["wsl_path"])
    # The host document may later receive H2 edits.  Its full-file hash is an
    # initial-freeze witness; resumptions are governed by the immutable config
    # hash already stored in the run state.
    if not args.resume and file_sha256(protocol_path) != config["protocol_document"]["sha256"]:
        raise RuntimeError("host EXP-004 protocol document changed after freeze")
    supplement = config.get("protocol_supplement")
    if supplement and file_sha256(resolved(supplement["path"])) != supplement["sha256"]:
        raise RuntimeError("H1 protocol supplement changed after freeze")

    deadline = DeadlineController(args.stop_at, int(config["runtime"]["reserve_minutes"]))
    deadline.install_signal_handlers()
    source_config, split_document = load_source(config)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "config_hash": canonical_json_hash(config),
                    "protocol_sha256": file_sha256(protocol_path),
                    "split_hash": config["split"]["indices_sha256"],
                    "validation_accessed": False,
                    "test_accessed": False,
                },
                indent=2,
            )
        )
        return 0
    canonical_path = list(source_config["path"]["canonical"])
    repeat_path = canonical_path + [28]
    config_hash = canonical_json_hash(config)
    output_root = resolved(config["runtime"]["artifact_root"]).resolve()
    manifest, state = prepare_run(output_root, config, config_hash, deadline, args.resume)
    state_path = output_root / "search_state.json"
    journal = EventJournal(output_root / "events.jsonl")
    journal.append("discovery_session_started", pid=os.getpid(), hard_stop=deadline.hard_stop.isoformat())

    model = None
    prefix_cache: GlobalPrefixCache | None = None
    resident_anchor: float | None = None
    canonical_task: float | None = None
    canonical_native: float | None = None
    chance: float | None = None

    def persist_state() -> None:
        atomic_write_json(state_path, state)

    def account_gpu_time() -> None:
        nonlocal resident_anchor
        if resident_anchor is not None:
            now = time.monotonic()
            state["gpu_seconds_used"] += now - resident_anchor
            resident_anchor = now
            persist_state()

    try:
        dataset_root = resolved(source_config["dataset"]["path"])
        train = load_arc_easy_split(dataset_root, "train")
        split_examples = {
            split: [train[index] for index in split_document[split]] for split in ("fit", "discover")
        }
        chance = chance_accuracy([item.n_choices for item in split_examples["discover"]])
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
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[source_config["model"]["dtype"]]
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=source_config["model"]["attention_implementation"],
        ).to("cuda")
        model.eval()
        model.config.use_cache = False
        resident_anchor = time.monotonic()
        executor = ModularLlamaExecutor(model, answer_token_ids)
        device = torch.device("cuda")
        feature_root = output_root / "features"
        results_root = output_root / "results"
        heads_root = output_root / "heads"
        results_root.mkdir(exist_ok=True)
        heads_root.mkdir(exist_ok=True)

        cache_config = config.get("prefix_cache", {})
        if cache_config.get("enabled", False):
            prefix_cache = GlobalPrefixCache(
                index_path=output_root / "prefix_cache_index.sqlite3",
                ssd_root=resolved(cache_config["ssd_root"]),
                hdd_root=Path(cache_config["hdd_root"]),
                ssd_cap_bytes=int(float(cache_config["ssd_cap_gib"]) * 1024**3),
                hdd_cap_bytes=int(float(cache_config["hdd_cap_gib"]) * 1024**3),
                config_hash=config_hash,
            )

        def cache_target_for(record: dict[str, Any]) -> list[int] | None:
            if prefix_cache is None:
                return None
            explicit = record.get("cache_target")
            if explicit is not None:
                return list(explicit) or None
            path = list(record["path"])
            if record["source"] in {"S1", "S5"}:
                return path[:-1] or None
            if record["source"] in {"S3", "S4"}:
                return path or None
            return None

        def process(record: dict[str, Any]) -> dict[str, Any]:
            nonlocal canonical_task, canonical_native
            result_path = results_root / f"{record['path_id']}.json"
            if result_path.exists():
                return json.loads(result_path.read_text(encoding="utf-8"))
            deadline.checkpoint(next_unit_seconds=180.0)
            journal.append("path_started", path_id=record["path_id"], source=record["source"], path=record["path"])
            cache_target = cache_target_for(record)
            cached_prefix = None
            if prefix_cache is not None:
                prefix_cache.register_path(record["path"])
                if cache_target is not None:
                    prefix_cache.prepare_write(cache_target)
                cached_prefix = prefix_cache.deepest_complete_prefix(record["path"])
            extraction = []
            for split_name in ("fit", "discover"):
                extractor = (
                    extract_cached_path_feature_split
                    if prefix_cache is not None
                    else extract_path_feature_split
                )
                kwargs = {
                    "executor": executor,
                    "tokenizer": tokenizer,
                    "prompt_cfg": source_config["prompt"],
                    "max_length_guard": int(source_config["tokenization"]["max_length_guard"]),
                    "path": record["path"],
                    "path_id": record["path_id"],
                    "batch_size": int(config["runtime"]["batch_size"]),
                    "shard_size": int(config["runtime"]["shard_size"]),
                    "config_hash": config_hash,
                    "feature_root": feature_root,
                    "deadline": deadline,
                    "journal": journal,
                }
                if prefix_cache is not None:
                    kwargs.update(
                        prefix_cache=prefix_cache,
                        cached_prefix=cached_prefix,
                        cache_target=cache_target,
                    )
                extraction.append(
                    extractor(
                        split_name,
                        split_examples[split_name],
                        split_document[split_name],
                        **kwargs,
                    )
                )
            if prefix_cache is not None and cache_target is not None:
                target_node = prefix_cache.node(cache_target)
                if target_node is not None and target_node["cache_status"] == "partial_ssd":
                    prefix_cache.finalize_write(cache_target)
            fit = load_path_feature_split(feature_root, record["path_id"], "fit")
            discover = load_path_feature_split(feature_root, record["path_id"], "discover")
            fit_mask = valid_choice_mask(fit["choice_counts"], 5)
            discover_mask = valid_choice_mask(discover["choice_counts"], 5)
            head = fit_masked_linear_head(
                fit["features"].to(device), fit["labels"].to(device), fit_mask.to(device),
                discover["features"].to(device), discover["labels"].to(device), discover_mask.to(device),
                l2=float(config["head_qualification"]["selected_l2"]),
                max_iter=int(config["head"]["max_iter"]),
                tolerance_grad=float(config["head"]["tolerance_grad"]),
            )
            weight = head.pop("weight")
            atomic_torch_save(heads_root / f"{record['path_id']}.pt", {"path_id": record["path_id"], "path": record["path"], "weight": weight})
            task_acc = float(head["eval_accuracy"])
            native_fit = masked_accuracy(fit["native_label_logits"], fit["labels"], fit_mask)
            native_acc = masked_accuracy(discover["native_label_logits"], discover["labels"], discover_mask)
            if record["path_id"] == "canonical":
                base_task, base_native = task_acc, native_acc
            else:
                base_task, base_native = canonical_task, canonical_native
                if base_task is None or base_native is None:
                    raise RuntimeError("canonical must complete before other paths")
            delta_task = float(base_task - task_acc)
            native_gap = float(base_native - native_acc)
            denominator = float(base_native - chance)
            relative_gap = native_gap / denominator if denominator > 0 else None
            good = delta_task <= float(config["criteria"]["epsilon_task"])
            collapse = bool(good and (
                native_gap >= float(config["criteria"]["gap_absolute"])
                or (relative_gap is not None and relative_gap >= float(config["criteria"]["gap_relative"]))
            ))
            result = {
                **record,
                "task_accuracy_fit": float(head["train_accuracy"]),
                "task_accuracy_discover": task_acc,
                "task_head_cross_entropy_fit": float(head["train_cross_entropy"]),
                "task_head_cross_entropy_discover": float(head["eval_cross_entropy"]),
                "task_head_closure_calls": int(head["closure_calls"]),
                "task_head_elapsed_seconds": float(head["elapsed_seconds"]),
                "native_accuracy_fit": native_fit,
                "native_accuracy_discover": native_acc,
                "delta_task_from_canonical": delta_task,
                "native_gap_from_canonical": native_gap,
                "relative_native_gap": relative_gap,
                "good_path": good,
                "readability_collapse": collapse,
                "edit_distance_from_canonical": edit_distance(record["path"], canonical_path),
                "count_in_prevalence": bool(record.get("count_in_prevalence", False)),
                "forward_batch_seconds": sum((item["batch_durations_seconds"] for item in extraction), []),
                "cached_prefix": [] if cached_prefix is None else cached_prefix["path"],
                "cache_tier": None if cached_prefix is None else cached_prefix["cache_status"],
                "cache_target": cache_target,
            }
            atomic_write_json(result_path, result)
            journal.append("path_completed", path_id=record["path_id"], source=record["source"], task_accuracy=task_acc, native_accuracy=native_acc, good=good, collapse=collapse)
            return result

        policy_version = config["search"].get("policy_version", "legacy_fixed_mixture")
        # Bootstrap controls and source roots. Each completed item is durable and idempotent.
        canonical_record = {"path_id": "canonical", "source": "baseline", "path": canonical_path, "count_in_prevalence": False}
        canonical_result = process(canonical_record)
        canonical_task = float(canonical_result["task_accuracy_discover"])
        canonical_native = float(canonical_result["native_accuracy_discover"])
        if policy_version == "sourcewise_temperature_v2":
            add_population_entry(state, "S2", canonical_result)
        else:
            for source in ("S1", "S2"):
                add_population_entry(state, source, canonical_result)
        state["bootstrap"]["canonical"] = True
        persist_state()

        repeat_result = process({"path_id": "repeat_L28", "source": "structured_control", "path": repeat_path, "count_in_prevalence": False})
        state["bootstrap"]["repeat_L28"] = True
        persist_state()

        seed_specs = [("S4", [1], "S4_seed"), ("S5", [28], "S5_seed")]
        if policy_version == "sourcewise_temperature_v2":
            seed_specs.insert(0, ("S1", [1, 28], "S1_seed"))
            state["bootstrap"].setdefault("S1_seed", False)
        for source, path, marker in seed_specs:
            seed_result = process({"path_id": f"{source}_seed", "source": source, "path": path, "parent_path_id": None, "mutation": {"operation": "fixed_seed"}, "count_in_prevalence": policy_version != "sourcewise_temperature_v2"})
            add_population_entry(state, source, seed_result)
            if policy_version != "sourcewise_temperature_v2" and not state["bootstrap"][marker]:
                state["completed_search_candidates"] += 1
            state["bootstrap"][marker] = True
            persist_state()
        account_gpu_time()

        campaign_limit = float(config["campaign"]["cumulative_model_resident_gpu_hours"]) * 3600.0
        artifact_cap = config["campaign"].get(
            "non_cache_artifact_cap_gib", config["campaign"].get("artifact_cap_gib")
        )
        if artifact_cap is None:
            raise RuntimeError("campaign artifact cap is missing")
        artifact_limit = int(artifact_cap) * 1024**3
        candidate_cap = int(config["search"]["candidate_cap"])
        while True:
            if state["gpu_seconds_used"] >= campaign_limit:
                state["campaign_status"] = "completed_gpu_budget"
                break
            if state["completed_search_candidates"] >= candidate_cap:
                state["campaign_status"] = "completed_candidate_safety_cap"
                break
            deadline.checkpoint(next_unit_seconds=180.0)
            if state["completed_search_candidates"] % 25 == 0 and non_cache_artifact_bytes(output_root) >= artifact_limit:
                state["campaign_status"] = "completed_artifact_safety_cap"
                break

            if state["inflight"] is None:
                source = SOURCE_ORDER[int(state["source_cursor"])]
                rng = np.random.default_rng()
                rng.bit_generator.state = state["rng_state"]
                if policy_version == "sourcewise_temperature_v2":
                    keep_turn = keep_throttled_source_turn(
                        int(state["source_good_counts"][source]),
                        rng,
                        threshold=int(config["search"]["operational_throttle"]["good_threshold"]),
                        keep_probability=float(config["search"]["operational_throttle"]["keep_probability"]),
                    )
                    state["rng_state"] = rng.bit_generator.state
                    if not keep_turn:
                        state["throttle_skips"][source] += 1
                        state["source_cursor"] = (int(state["source_cursor"]) + 1) % len(SOURCE_ORDER)
                        persist_state()
                        journal.append("source_turn_throttle_skipped", source=source, source_good_count=state["source_good_counts"][source])
                        continue
                known = {path_key(item["path"]) for item in load_results(output_root)}
                if policy_version == "sourcewise_temperature_v2":
                    temperature = source_temperature(
                        source,
                        int(state["source_good_counts"][source]),
                        {key: float(value) for key, value in config["search"]["initial_temperatures"].items()},
                    )
                    softmax_weight = 1.0
                else:
                    temperature = float(config["search"]["temperature"])
                    softmax_weight = float(config["search"]["parent_softmax_weight"])
                candidate = propose_candidate(
                    source, state["populations"], known, rng,
                    max_path_length=int(config["search"]["max_path_length"]),
                    temperature=temperature,
                    softmax_weight=softmax_weight,
                    max_attempts=int(config["search"]["proposal_attempts"]),
                )
                state["rng_state"] = rng.bit_generator.state
                if candidate is None:
                    state["skipped_source_turns"][source] += 1
                    state["source_cursor"] = (int(state["source_cursor"]) + 1) % len(SOURCE_ORDER)
                    persist_state()
                    journal.append("source_turn_skipped", source=source)
                    continue
                state["candidate_ordinal"] += 1
                candidate["candidate_ordinal"] = state["candidate_ordinal"]
                candidate["count_in_prevalence"] = True
                state["inflight"] = candidate
                persist_state()  # Commits RNG and identity before expensive work.
                journal.append("candidate_committed", **candidate)

            candidate = state["inflight"]
            result = process(candidate)
            add_population_entry(state, candidate["source"], result)
            state["completed_search_candidates"] += 1
            if policy_version == "sourcewise_temperature_v2" and result["good_path"]:
                state["source_good_counts"][candidate["source"]] += 1
            state["source_cursor"] = (int(state["source_cursor"]) + 1) % len(SOURCE_ORDER)
            state["inflight"] = None
            account_gpu_time()
            summary = summary_document(output_root, state, canonical_task, canonical_native, chance, prefix_cache)
            atomic_write_json(output_root / "discovery_summary.json", summary)
            print(
                f"[{state['completed_search_candidates']:04d}] {result['source']} {result['path_id']} "
                f"task={result['task_accuracy_discover']:.4f} native={result['native_accuracy_discover']:.4f} "
                f"good={result['good_path']} collapse={result['readability_collapse']} "
                f"gpu_h={state['gpu_seconds_used']/3600:.3f}",
                flush=True,
            )

        account_gpu_time()
        persist_state()
        summary = summary_document(output_root, state, canonical_task, canonical_native, chance, prefix_cache)
        atomic_write_json(output_root / "discovery_summary.json", summary)
        manifest.update({"status": state["campaign_status"], "finished_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        atomic_write_json(output_root / "run_manifest.json", manifest)
        journal.append("discovery_campaign_completed", **summary)
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    except DeadlineReached as exc:
        account_gpu_time()
        state["campaign_status"] = "session_soft_stopped"
        persist_state()
        summary = summary_document(output_root, state, canonical_task, canonical_native, chance, prefix_cache)
        atomic_write_json(output_root / "discovery_summary.json", summary)
        manifest.update({"status": "session_soft_stopped", "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"), "reason": str(exc)})
        atomic_write_json(output_root / "run_manifest.json", manifest)
        journal.append("discovery_session_soft_stopped", reason=str(exc), gpu_seconds_used=state["gpu_seconds_used"])
        return 75
    except BaseException as exc:
        account_gpu_time()
        state["campaign_status"] = "failed_resumable"
        persist_state()
        manifest.update({"status": "failed_resumable", "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"), "error_type": type(exc).__name__, "error": str(exc)})
        atomic_write_json(output_root / "run_manifest.json", manifest)
        journal.append("discovery_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if prefix_cache is not None:
            prefix_cache.close()
        if model is not None:
            del model
        if torch.cuda.is_available():
            # A poisoned CUDA context can make cleanup itself raise and obscure
            # the already-persisted root failure.  A fresh process owns the
            # recovery; cleanup is best-effort only.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(run())
