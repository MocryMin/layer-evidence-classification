#!/usr/bin/env python3
"""Resumable full EXP-004 H2 tuning and test runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h2_forward import (  # noqa: E402
    H2ModularDeberta,
    SamplePathEvaluator,
    fixed_head_logits,
)
from src.exp004_h2_mcts import (  # noqa: E402
    random_tuning_indices,
    select_tuning_grid,
    wilson_two_sided_lower,
)
from src.exp004_h2_search import (  # noqa: E402
    SearchInterrupted,
    run_mcts_search,
    run_random_search,
)
from src.seeding import enable_determinism, seed_all  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip())
    return {"commit": commit, "dirty": dirty}


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class StopController:
    def __init__(self, stop_at: str | None):
        self.requested = False
        self.stop_time: datetime | None = None
        if stop_at:
            parsed = datetime.fromisoformat(stop_at)
            if parsed.tzinfo is None:
                raise ValueError("--stop-at must include a timezone offset")
            self.stop_time = parsed
        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)

    def _signal(self, signum, _frame) -> None:
        self.requested = True
        print(f"[stop] signal={signum}; stopping at next simulation checkpoint", flush=True)

    def should_stop(self) -> bool:
        if self.requested:
            return True
        if self.stop_time is not None:
            return datetime.now(self.stop_time.tzinfo) >= self.stop_time
        return False


def load_split(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in {"validation", "test"}:
        raise ValueError("H2 runner loads only validation or test")
    filename = config["dataset"][f"{split}_file"]
    frame = pd.read_parquet(ROOT / config["dataset"]["root"] / filename)
    drop_id = int(config["dataset"]["drop_intent_id"])
    frame = frame[frame["intent"] != drop_id].copy()
    records = []
    for filtered_index, (source_index, row) in enumerate(frame.iterrows()):
        original_label = int(row["intent"])
        records.append({
            "split_index": filtered_index,
            "source_row_index": int(source_index),
            "text": str(row["text"]),
            "gold_class": original_label - 1 if original_label > drop_id else original_label,
        })
    expected = 3000 if split == "validation" else 4500
    if len(records) != expected:
        raise RuntimeError(f"unexpected {split} size {len(records)} != {expected}")
    return records


def tokenize_records(tokenizer, records: list[dict[str, Any]], prompt: str) -> list[list[int]]:
    texts = [prompt.format(utterance=record["text"]) for record in records]
    return tokenizer(texts, padding=False, truncation=False)["input_ids"]


def padded_batch(
    tokenizer,
    token_ids: Sequence[Sequence[int]],
    device: torch.device,
    *,
    fixed_length: int,
):
    items = [
        {"input_ids": list(ids), "attention_mask": [1] * len(ids)} for ids in token_ids
    ]
    if any(len(ids) > fixed_length for ids in token_ids):
        raise ValueError("token sequence exceeds fixed split padding length")
    encoded = tokenizer.pad(
        items,
        padding="max_length",
        max_length=fixed_length,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def deterministic_rank(values: np.ndarray, gold: int) -> int:
    return 1 + int((values > values[gold]).sum()) + int(
        (values[:gold] == values[gold]).sum()
    )


@torch.inference_mode()
def canonical_scan(
    records: list[dict[str, Any]],
    token_ids: list[list[int]],
    *,
    tokenizer,
    executor: H2ModularDeberta,
    weight: torch.Tensor,
    bias: torch.Tensor,
    canonical_path: Sequence[int],
    batch_size: int,
    fixed_length: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(records), batch_size):
        end = min(start + batch_size, len(records))
        ids, mask = padded_batch(
            tokenizer, token_ids[start:end], device, fixed_length=fixed_length
        )
        prepared = executor.prepare(ids, mask)
        hidden, _ = executor.forward_from_prefix(prepared, canonical_path)
        logits = fixed_head_logits(hidden, weight, bias).cpu().numpy()
        for offset, values in enumerate(logits):
            record = records[start + offset]
            predicted = int(values.argmax())
            rows.append({
                "split_index": record["split_index"],
                "source_row_index": record["source_row_index"],
                "gold_class": record["gold_class"],
                "predicted_class": predicted,
                "correct": predicted == record["gold_class"],
                "gold_rank": deterministic_rank(values, record["gold_class"]),
            })
        if end % 640 == 0 or end == len(records):
            print(
                f"[canonical-scan] {end}/{len(records)} elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    return rows


def stable_search_seed(base_seed: int, split: str, family: str, sample_index: int) -> int:
    encoded = f"{base_seed}|{split}|{family}|{sample_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


def make_evaluator(
    record: dict[str, Any],
    ids: list[int],
    *,
    tokenizer,
    executor: H2ModularDeberta,
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: torch.device,
    pad_length: int,
) -> SamplePathEvaluator:
    input_ids, attention_mask = padded_batch(
        tokenizer, [ids], device, fixed_length=pad_length
    )
    prepared = executor.prepare(input_ids, attention_mask)
    return SamplePathEvaluator(
        executor, prepared, weight, bias, int(record["gold_class"])
    )


def run_one_search(
    record: dict[str, Any],
    ids: list[int],
    *,
    split: str,
    family: str,
    reward_kind: str | None,
    c_value: float | None,
    lambda_value: float | None,
    simulations: int,
    semantics: dict[str, Any],
    config: dict[str, Any],
    tokenizer,
    executor: H2ModularDeberta,
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: torch.device,
    pad_length: int,
    stop: StopController,
) -> dict[str, Any]:
    evaluator = make_evaluator(
        record,
        ids,
        tokenizer=tokenizer,
        executor=executor,
        weight=weight,
        bias=bias,
        device=device,
        pad_length=pad_length,
    )
    seed = stable_search_seed(
        int(config["search_runtime"]["seed"]),
        split,
        "random" if family == "random" else "mcts",
        int(record["split_index"]),
    )
    canonical = semantics["model_and_task"]["canonical_path"]
    search = semantics["search"]
    if family == "random":
        outcome = run_random_search(
            evaluator.evaluate,
            evaluator.summary,
            canonical_path=canonical,
            simulations=simulations,
            min_path_length=int(search["min_path_length"]),
            max_path_length=int(search["max_path_length"]),
            search_seed=seed,
            should_stop=stop.should_stop,
        )
    else:
        if reward_kind is None or c_value is None or lambda_value is None:
            raise ValueError("MCTS search requires reward, c, and lambda")
        outcome = run_mcts_search(
            evaluator.evaluate,
            evaluator.summary,
            canonical_path=canonical,
            reward_kind=reward_kind,
            exploration_c=float(c_value),
            length_lambda=float(lambda_value),
            simulations=simulations,
            explore_probability=float(search["explore_probability"]),
            min_path_length=int(search["min_path_length"]),
            max_path_length=int(search["max_path_length"]),
            total_model_layers=int(semantics["model_and_task"]["n_transformer_layers"]),
            search_seed=seed,
            should_stop=stop.should_stop,
        )
    return {
        "split_index": record["split_index"],
        "source_row_index": record["source_row_index"],
        "gold_class": record["gold_class"],
        **outcome,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [record for record in records if record["canonical"]["correct"]]
    negative = [record for record in records if not record["canonical"]["correct"]]
    n_short = sum(bool(record["shorter_correct"]) for record in positive)
    n_recov = sum(bool(record["recovered"]) for record in negative)
    r_short = n_short / len(positive) if positive else None
    r_recov = n_recov / len(negative) if negative else None
    result = {
        "n": len(records),
        "n_pos": len(positive),
        "n_neg": len(negative),
        "n_short": n_short,
        "n_recov": n_recov,
        "R_short": r_short,
        "R_recov": r_recov,
        "mean_elapsed_seconds_per_search": float(
            np.mean([record["elapsed_seconds"] for record in records])
        ),
        "exact_result_cache_hits": sum(
            int(record["cache"]["exact_result_cache_hits"]) for record in records
        ),
        "unique_paths": sum(int(record["cache"]["unique_paths"]) for record in records),
        "transformer_blocks_executed": sum(
            int(record["cache"]["transformer_blocks_executed"]) for record in records
        ),
    }
    if positive:
        result["R_short_wilson_two_sided_95_lower"] = wilson_two_sided_lower(
            n_short, len(positive)
        )
    if negative:
        result["R_recov_wilson_two_sided_95_lower"] = wilson_two_sided_lower(
            n_recov, len(negative)
        )
    return result


def init_artifact(
    config_path: Path,
    semantics_path: Path,
    config: dict[str, Any],
    semantics: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    artifact_root = ROOT / config["artifacts"]["root"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_root / config["artifacts"]["manifest"]
    state = git_state()
    if state["dirty"]:
        raise RuntimeError("official H2 run requires a clean Git worktree")
    head_path = ROOT / semantics["canonical_head"]["artifact"]
    expected = {
        "experiment_id": config["experiment_id"],
        "git": state,
        "full_config_sha256": sha256(config_path),
        "semantics_config_sha256": sha256(semantics_path),
        "canonical_head_sha256": sha256(head_path),
        "authorization": config["authorization"],
        "validation_access_authorized": True,
        "test_access_authorized_after_tuning": True,
    }
    if manifest_path.exists():
        existing = read_json(manifest_path)
        for key in (
            "experiment_id", "git", "full_config_sha256",
            "semantics_config_sha256", "canonical_head_sha256",
        ):
            if existing[key] != expected[key]:
                raise RuntimeError(f"resume manifest mismatch for {key}")
        manifest = existing
    else:
        manifest = {**expected, "started_at": datetime.now().astimezone().isoformat()}
        atomic_write_json(manifest_path, manifest)
        (artifact_root / "resolved_full_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        (artifact_root / "resolved_semantics_config.yaml").write_text(
            yaml.safe_dump(semantics, sort_keys=False), encoding="utf-8"
        )
    return artifact_root, manifest


def ensure_mlflow(config: dict[str, Any], artifact_root: Path, manifest: dict[str, Any]) -> str:
    tracking = config["mlflow"]["tracking_uri"]
    if tracking == "sqlite:///mlruns.db":
        tracking = "sqlite:///" + str((ROOT / "mlruns.db").resolve())
    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment(config["mlflow"]["experiment"])
    record_path = artifact_root / "mlflow_run.json"
    if record_path.exists():
        return read_json(record_path)["run_id"]
    with mlflow.start_run(run_name=config["experiment_id"]) as active:
        mlflow.set_tags({
            "experiment_phase": "official_h2",
            "validation_accessed": "true",
            "test_access_requires_completed_tuning": "true",
        })
        mlflow.log_params({
            "git_commit": manifest["git"]["commit"],
            "full_config_sha256": manifest["full_config_sha256"],
            "semantics_config_sha256": manifest["semantics_config_sha256"],
            "canonical_head_sha256": manifest["canonical_head_sha256"],
            "simulations_per_search": 200,
        })
        run_id = active.info.run_id
    atomic_write_json(record_path, {
        "run_id": run_id,
        "tracking_uri": tracking,
        "experiment": config["mlflow"]["experiment"],
    })
    return run_id


def log_metrics(run_id: str, tracking_uri: str, metrics: dict[str, float], step: int = 0):
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics, step=step)


def ensure_tuning_indices(
    validation_records,
    validation_tokens,
    *,
    config,
    semantics,
    artifact_root,
    tokenizer,
    executor,
    weight,
    bias,
    pad_length,
    device,
    stop=None,
) -> dict[str, Any]:
    indices_path = artifact_root / config["artifacts"]["tuning_indices"]
    if indices_path.exists():
        existing = read_json(indices_path)
        if int(existing["fixed_padding_length"]) != int(pad_length):
            raise RuntimeError("validation padding length changed on resume")
        return existing
    scan = canonical_scan(
        validation_records,
        validation_tokens,
        tokenizer=tokenizer,
        executor=executor,
        weight=weight,
        bias=bias,
        canonical_path=semantics["model_and_task"]["canonical_path"],
        batch_size=int(config["dataset"]["canonical_scan_batch_size"]),
        fixed_length=pad_length,
        device=device,
    )
    scan_path = artifact_root / "validation_canonical_scan.json"
    atomic_write_json(scan_path, scan)
    audit = config["dataset"]["canonical_validation_audit"]
    observed_correct = sum(row["correct"] for row in scan)
    if observed_correct != int(audit["expected_correct"]) or len(scan) != int(
        audit["expected_total"]
    ):
        raise RuntimeError(
            "canonical validation audit failed: "
            f"{observed_correct}/{len(scan)} != "
            f"{audit['expected_correct']}/{audit['expected_total']}"
        )
    selected = random_tuning_indices(
        [row["correct"] for row in scan],
        np.random.default_rng(int(semantics["tuning"]["selection_seed"])),
        count_per_group=int(semantics["tuning"]["canonical_correct_count"]),
    )
    payload = {
        **selected,
        "selection_seed": int(semantics["tuning"]["selection_seed"]),
        "canonical_correct_scan_count": observed_correct,
        "canonical_wrong_scan_count": sum(not row["correct"] for row in scan),
        "scan_path": str(scan_path.relative_to(ROOT)),
        "fixed_padding_length": pad_length,
    }
    atomic_write_json(indices_path, payload)
    return payload


def run_preflight(
    tuning_indices,
    validation_records,
    validation_tokens,
    **kwargs,
) -> None:
    config = kwargs["config"]
    artifact_root = kwargs["artifact_root"]
    path = artifact_root / "preflight.json"
    if path.exists():
        print("[preflight] already complete", flush=True)
        return
    pool = tuning_indices["canonical_correct"] + tuning_indices["canonical_wrong"]
    selected = pool[: int(config["preflight"]["validation_samples"])]
    rows = []
    scan = read_json(artifact_root / "validation_canonical_scan.json")
    started = time.perf_counter()
    for index in selected:
        record = validation_records[index]
        common = dict(
            record=record,
            ids=validation_tokens[index],
            split="validation",
            simulations=int(config["preflight"]["simulations_per_search"]),
            **{key: value for key, value in kwargs.items() if key not in {"artifact_root"}},
        )
        primary = run_one_search(
            family="mcts",
            reward_kind="reciprocal_gold_rank",
            c_value=float(config["preflight"]["c"]),
            lambda_value=float(config["preflight"]["lambda"]),
            **common,
        )
        random = run_one_search(
            family="random",
            reward_kind=None,
            c_value=None,
            lambda_value=None,
            **common,
        )
        expected = scan[index]
        for observed in (primary["canonical"], random["canonical"]):
            for key in ("predicted_class", "correct", "gold_rank"):
                if observed[key] != expected[key]:
                    raise RuntimeError(
                        f"canonical scan/search mismatch sample={index} key={key}: "
                        f"{observed[key]} != {expected[key]}"
                    )
        rows.append({"primary": primary, "random": random})
    payload = {
        "validation_only": True,
        "test_accessed": False,
        "n_samples": len(selected),
        "simulations_per_search": int(config["preflight"]["simulations_per_search"]),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    atomic_write_json(path, payload)
    per_search = payload["elapsed_seconds"] / (2 * len(selected))
    print(f"[preflight] complete mean_search={per_search:.3f}s path={path}", flush=True)


def tuning_grid_filename(index: int, c_value: float, lambda_value: float) -> str:
    c_text = str(c_value).replace(".", "p")
    l_text = str(lambda_value).replace(".", "p")
    return f"grid_{index:02d}_c{c_text}_lambda{l_text}.json"


def run_tuning_family(
    reward_label: str,
    reward_kind: str,
    tuning_indices: dict[str, Any],
    validation_records,
    validation_tokens,
    *,
    run_id: str,
    tracking_uri: str,
    **kwargs,
) -> dict[str, Any]:
    config = kwargs["config"]
    semantics = kwargs["semantics"]
    artifact_root = kwargs["artifact_root"]
    stop = kwargs["stop"]
    directory = artifact_root / "tuning" / reward_label
    directory.mkdir(parents=True, exist_ok=True)
    sample_indices = tuning_indices["canonical_correct"] + tuning_indices["canonical_wrong"]
    grids = list(product(semantics["search"]["c_grid"], semantics["search"]["lambda_grid"]))
    metric_records: list[dict[str, Any]] = []
    family_started = time.perf_counter()
    for grid_index, (c_value, lambda_value) in enumerate(grids):
        output = directory / tuning_grid_filename(grid_index, c_value, lambda_value)
        if output.exists():
            payload = read_json(output)
            metric_records.append(payload["metrics"])
            continue
        if stop.should_stop():
            raise SearchInterrupted("stop requested before tuning grid")
        records = []
        started = time.perf_counter()
        for sample_index in sample_indices:
            records.append(run_one_search(
                validation_records[sample_index],
                validation_tokens[sample_index],
                split="validation",
                family="mcts",
                reward_kind=reward_kind,
                c_value=float(c_value),
                lambda_value=float(lambda_value),
                simulations=int(semantics["search"]["simulations_per_sample"]),
                **{key: value for key, value in kwargs.items() if key != "artifact_root"},
            ))
        metrics = aggregate_records(records)
        metrics.update({
            "grid_index": grid_index,
            "c": float(c_value),
            "lambda": float(lambda_value),
            "J": 0.5 * metrics["R_short"] + 0.5 * metrics["R_recov"],
            "grid_elapsed_seconds": time.perf_counter() - started,
            "artifact": str(output.relative_to(ROOT)),
        })
        atomic_write_json(output, {"metrics": metrics, "records": records})
        metric_records.append(metrics)
        log_metrics(
            run_id,
            tracking_uri,
            {
                f"tuning_{reward_label}_J": metrics["J"],
                f"tuning_{reward_label}_R_short": metrics["R_short"],
                f"tuning_{reward_label}_R_recov": metrics["R_recov"],
                f"tuning_{reward_label}_seconds": metrics["grid_elapsed_seconds"],
            },
            step=grid_index,
        )
        if config["search_runtime"]["gpu_empty_cache_between_grids_or_shards"]:
            torch.cuda.empty_cache()
        complete = grid_index + 1
        elapsed = time.perf_counter() - family_started
        eta = elapsed / complete * (len(grids) - complete)
        print(
            f"[tuning:{reward_label}] {complete}/{len(grids)} c={c_value} "
            f"lambda={lambda_value} J={metrics['J']:.4f} "
            f"short={metrics['R_short']:.4f} recov={metrics['R_recov']:.4f} "
            f"grid={metrics['grid_elapsed_seconds']:.1f}s eta={eta/60:.1f}m",
            flush=True,
        )

    selected = select_tuning_grid(
        metric_records,
        np.random.default_rng(int(semantics["tuning"]["selection_seed"])),
    )
    return {"reward_label": reward_label, "selected": selected, "all_metrics": metric_records}


def ensure_tuning_selection(
    tuning_indices,
    validation_records,
    validation_tokens,
    *,
    run_id,
    tracking_uri,
    **kwargs,
) -> dict[str, Any]:
    config = kwargs["config"]
    artifact_root = kwargs["artifact_root"]
    selection_path = artifact_root / config["artifacts"]["tuning_selection"]
    if selection_path.exists():
        return read_json(selection_path)
    primary = run_tuning_family(
        "primary_rank",
        "reciprocal_gold_rank",
        tuning_indices,
        validation_records,
        validation_tokens,
        run_id=run_id,
        tracking_uri=tracking_uri,
        **kwargs,
    )
    binary = run_tuning_family(
        "binary_control",
        "binary_correctness",
        tuning_indices,
        validation_records,
        validation_tokens,
        run_id=run_id,
        tracking_uri=tracking_uri,
        **kwargs,
    )
    payload = {
        "selection_rule": "max_J_then_max_R_recov_then_seeded_uniform_random",
        "selection_seed": int(kwargs["semantics"]["tuning"]["selection_seed"]),
        "primary_rank": primary,
        "binary_control": binary,
        "test_access_now_permitted": True,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    atomic_write_json(selection_path, payload)
    return payload


def test_variant_spec(variant: str, selection: dict[str, Any]):
    if variant == "primary_rank":
        chosen = selection["primary_rank"]["selected"]
        return "mcts", "reciprocal_gold_rank", chosen["c"], chosen["lambda"]
    if variant == "binary_control":
        chosen = selection["binary_control"]["selected"]
        return "mcts", "binary_correctness", chosen["c"], chosen["lambda"]
    if variant == "random_control":
        return "random", None, None, None
    raise ValueError(variant)


def run_test_variant(
    variant,
    selection,
    test_records,
    test_tokens,
    *,
    run_id,
    tracking_uri,
    **kwargs,
) -> dict[str, Any]:
    config = kwargs["config"]
    semantics = kwargs["semantics"]
    artifact_root = kwargs["artifact_root"]
    stop = kwargs["stop"]
    directory = artifact_root / "test" / variant
    directory.mkdir(parents=True, exist_ok=True)
    shard_size = int(config["search_runtime"]["test_shard_size"])
    family, reward_kind, c_value, lambda_value = test_variant_spec(variant, selection)
    stage_started = time.perf_counter()
    completed_samples = 0
    for start in range(0, len(test_records), shard_size):
        end = min(start + shard_size, len(test_records))
        output = directory / f"shard_{start:05d}_{end:05d}.json"
        if output.exists():
            completed_samples += end - start
            continue
        if stop.should_stop():
            raise SearchInterrupted("stop requested before test shard")
        records = []
        shard_started = time.perf_counter()
        for index in range(start, end):
            records.append(run_one_search(
                test_records[index],
                test_tokens[index],
                split="test",
                family=family,
                reward_kind=reward_kind,
                c_value=c_value,
                lambda_value=lambda_value,
                simulations=int(semantics["search"]["simulations_per_sample"]),
                **{key: value for key, value in kwargs.items() if key != "artifact_root"},
            ))
        payload = {
            "variant": variant,
            "start": start,
            "end": end,
            "c": c_value,
            "lambda": lambda_value,
            "reward_kind": reward_kind,
            "elapsed_seconds": time.perf_counter() - shard_started,
            "records": records,
        }
        atomic_write_json(output, payload)
        completed_samples += end - start
        if config["search_runtime"]["gpu_empty_cache_between_grids_or_shards"]:
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - stage_started
        eta = elapsed / completed_samples * (len(test_records) - completed_samples)
        print(
            f"[test:{variant}] {completed_samples}/{len(test_records)} "
            f"shard={payload['elapsed_seconds']:.1f}s eta={eta/60:.1f}m",
            flush=True,
        )

    all_records = []
    for start in range(0, len(test_records), shard_size):
        end = min(start + shard_size, len(test_records))
        payload = read_json(directory / f"shard_{start:05d}_{end:05d}.json")
        all_records.extend(payload["records"])
    if len(all_records) != len(test_records):
        raise RuntimeError(f"incomplete {variant} test records")
    all_records.sort(key=lambda row: row["split_index"])
    if [row["split_index"] for row in all_records] != list(range(len(test_records))):
        raise RuntimeError(f"duplicate or missing {variant} test indices")
    summary = aggregate_records(all_records)
    summary.update({
        "variant": variant,
        "c": c_value,
        "lambda": lambda_value,
        "reward_kind": reward_kind,
    })
    atomic_write_json(directory / "summary.json", summary)
    log_metrics(run_id, tracking_uri, {
        f"test_{variant}_R_short": summary["R_short"],
        f"test_{variant}_R_recov": summary["R_recov"],
        f"test_{variant}_R_short_wilson_lower": summary["R_short_wilson_two_sided_95_lower"],
        f"test_{variant}_R_recov_wilson_lower": summary["R_recov_wilson_two_sided_95_lower"],
    })
    return summary


def run_test(selection, *, config, artifact_root, run_id, tracking_uri, **kwargs):
    if not selection.get("test_access_now_permitted"):
        raise RuntimeError("test gate closed: tuning selection is incomplete")
    gate_path = artifact_root / config["artifacts"]["test_access_gate"]
    if not gate_path.exists():
        atomic_write_json(gate_path, {
            "opened_at": datetime.now().astimezone().isoformat(),
            "reason": "both primary and binary tuning selections are complete",
            "tuning_selection_sha256": sha256(
                artifact_root / config["artifacts"]["tuning_selection"]
            ),
            "authorization": config["authorization"],
        })
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("test_accessed", "true")
        mlflow.log_artifact(str(gate_path), artifact_path="evidence")
    print("[gate] tuning complete; loading official test for the first time", flush=True)
    test_records = load_split(config, "test")
    test_tokens = tokenize_records(kwargs["tokenizer"], test_records, config["dataset"]["prompt"])
    test_pad_length = max(map(len, test_tokens))
    summaries = {}
    for variant in ("primary_rank", "binary_control", "random_control"):
        summaries[variant] = run_test_variant(
            variant,
            selection,
            test_records,
            test_tokens,
            config=config,
            artifact_root=artifact_root,
            run_id=run_id,
            tracking_uri=tracking_uri,
            pad_length=test_pad_length,
            **kwargs,
        )
    epsilon_short = float(kwargs["semantics"]["evaluation"]["acceptance_lower_endpoint"]["R_short"])
    epsilon_recov = float(kwargs["semantics"]["evaluation"]["acceptance_lower_endpoint"]["R_recov"])
    primary = summaries["primary_rank"]
    final = {
        "experiment_id": config["experiment_id"],
        "completed_at": datetime.now().astimezone().isoformat(),
        "tuning_selection": selection,
        "test": summaries,
        "acceptance": {
            "epsilon_R_short": epsilon_short,
            "epsilon_R_recov": epsilon_recov,
            "efficiency": primary["R_short_wilson_two_sided_95_lower"] >= epsilon_short,
            "recoverability": primary["R_recov_wilson_two_sided_95_lower"] >= epsilon_recov,
        },
    }
    final["acceptance"]["H2"] = (
        final["acceptance"]["efficiency"] and final["acceptance"]["recoverability"]
    )
    final_path = artifact_root / config["artifacts"]["final_summary"]
    atomic_write_json(final_path, final)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(final_path), artifact_path="evidence")
        mlflow.log_artifact(
            str(artifact_root / config["artifacts"]["tuning_selection"]),
            artifact_path="evidence",
        )
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/exp004_h2_full_v1.yaml"))
    parser.add_argument("--stage", choices=["preflight", "tuning", "test", "all"], default="all")
    parser.add_argument("--stop-at", default=None, help="timezone-aware ISO timestamp")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    semantics_path = ROOT / config["semantics_config"]
    semantics = yaml.safe_load(semantics_path.read_text(encoding="utf-8"))
    if sha256(semantics_path) != config["semantics_config_sha256"]:
        raise RuntimeError("semantics config hash changed")
    if not config["authorization"]["allow_validation"]:
        raise RuntimeError("validation gate is closed")

    stop = StopController(args.stop_at or config["search_runtime"]["stop_at"])
    seed_all(int(config["search_runtime"]["seed"]))
    enable_determinism()
    torch.set_float32_matmul_precision(config["model"]["float32_matmul_precision"])
    if not torch.cuda.is_available():
        raise RuntimeError("official H2 requires CUDA")
    device = torch.device("cuda")
    artifact_root, manifest = init_artifact(
        config_path, semantics_path, config, semantics
    )
    run_id = ensure_mlflow(config, artifact_root, manifest)
    tracking_uri = read_json(artifact_root / "mlflow_run.json")["tracking_uri"]

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / config["model"]["path"]))
    model = AutoModel.from_pretrained(
        str(ROOT / config["model"]["path"]), dtype=torch.float32
    ).to(device).eval()
    executor = H2ModularDeberta(model)
    head_npz = np.load(ROOT / semantics["canonical_head"]["artifact"])
    weight = torch.from_numpy(head_npz[semantics["canonical_head"]["weight_key"]]).to(device)
    bias = torch.from_numpy(head_npz[semantics["canonical_head"]["bias_key"]]).to(device)

    validation_records = load_split(config, "validation")
    validation_tokens = tokenize_records(
        tokenizer, validation_records, config["dataset"]["prompt"]
    )
    validation_pad_length = max(map(len, validation_tokens))
    shared = dict(
        config=config,
        semantics=semantics,
        artifact_root=artifact_root,
        tokenizer=tokenizer,
        executor=executor,
        weight=weight,
        bias=bias,
        device=device,
        pad_length=validation_pad_length,
        stop=stop,
    )
    try:
        tuning_indices = ensure_tuning_indices(
            validation_records,
            validation_tokens,
            **shared,
        )
        if args.stage in {"preflight", "all"}:
            run_preflight(
                tuning_indices,
                validation_records,
                validation_tokens,
                **shared,
            )
            if args.stage == "preflight":
                return
        selection = None
        if args.stage in {"tuning", "all"}:
            selection = ensure_tuning_selection(
                tuning_indices,
                validation_records,
                validation_tokens,
                run_id=run_id,
                tracking_uri=tracking_uri,
                **shared,
            )
            if args.stage == "tuning":
                print(json.dumps({
                    "tuning_selection": str(
                        artifact_root / config["artifacts"]["tuning_selection"]
                    )
                }), flush=True)
                return
        if args.stage == "test":
            selection_path = artifact_root / config["artifacts"]["tuning_selection"]
            if not selection_path.exists():
                raise RuntimeError("cannot run test before complete tuning selection")
            selection = read_json(selection_path)
        if args.stage in {"test", "all"}:
            if not config["authorization"]["allow_test_after_tuning_complete"]:
                raise RuntimeError("test authorization is absent")
            final = run_test(
                selection,
                config=config,
                artifact_root=artifact_root,
                run_id=run_id,
                tracking_uri=tracking_uri,
                semantics=semantics,
                tokenizer=tokenizer,
                executor=executor,
                weight=weight,
                bias=bias,
                device=device,
                stop=stop,
            )
            print(json.dumps({"final_summary": final["acceptance"]}), flush=True)
    except SearchInterrupted as error:
        print(f"[stopped] {error}; completed checkpoints are resumable", flush=True)
        raise SystemExit(130) from error


if __name__ == "__main__":
    main()
