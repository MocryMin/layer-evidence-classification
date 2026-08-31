#!/usr/bin/env python3
"""Train-only GPU throughput preflight for EXP-004 H2.

This script never calls the shared three-split CLINC loader.  It reads only the
official train parquet, so running it cannot accidentally expose validation or
test examples before the H2 protocol is executable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from frag_modular_probe import ModularStack, forward_path  # noqa: E402
from src.exp004_h1 import atomic_write_json  # noqa: E402
from src.exp004_h2_forward import (  # noqa: E402
    H2ModularDeberta,
    canonical_cycle_path,
    fixed_head_logits,
    h2_simulation_counts,
    runtime_hours,
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
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return {"commit": commit, "dirty": dirty}


def nvidia_snapshot() -> dict[str, str]:
    fields = [
        "name", "driver_version", "memory.total", "memory.used",
        "utilization.gpu", "power.draw", "power.limit", "temperature.gpu",
        "clocks.current.sm", "clocks.current.memory",
    ]
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(", ")
    return dict(zip(fields, result, strict=True))


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def timed_cuda_calls(
    call: Callable[[], Any], *, warmup: int, repetitions: int, device: torch.device
) -> tuple[dict[str, float], list[float]]:
    for _ in range(warmup):
        value = call()
        if isinstance(value, torch.Tensor):
            _ = value.reshape(-1)[0].item()
    torch.cuda.synchronize(device)
    durations: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        value = call()
        if isinstance(value, torch.Tensor):
            _ = value.reshape(-1)[0].item()  # search needs the reward on the host
        torch.cuda.synchronize(device)
        durations.append((time.perf_counter() - started) * 1000.0)
    return distribution(durations), durations


def select_quantile_indices(lengths: np.ndarray, quantiles: list[float]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    used: set[int] = set()
    for quantile in quantiles:
        target = float(np.quantile(lengths, quantile))
        order = np.argsort(np.abs(lengths - target), kind="stable")
        index = next(int(item) for item in order if int(item) not in used)
        used.add(index)
        records.append({"quantile": quantile, "index": index, "tokens": int(lengths[index])})
    return records


def encode(tokenizer, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    batch = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)


def legacy_fp16_branch(
    legacy: ModularStack,
    x0: torch.Tensor,
    mask: torch.Tensor,
    path: tuple[int, ...],
) -> torch.Tensor:
    hidden = x0
    for layer_id in path:
        hidden = forward_path(
            legacy, hidden, mask, [layer_id - 1], batch=len(x0), out_dtype=torch.float16
        )
    return hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs/exp004_h2_throughput_v1.yaml")
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["runtime"]["artifact_root"]
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing throughput artifact: {output}")
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / config["runtime"]["resolved_config_name"])

    if not torch.cuda.is_available():
        raise RuntimeError("H2 throughput preflight requires CUDA")
    device = torch.device("cuda")
    seed_all(int(config["benchmark"]["seed"]))
    enable_determinism()
    torch.set_float32_matmul_precision(config["model"]["float32_matmul_precision"])
    before_gpu = nvidia_snapshot()
    torch.cuda.reset_peak_memory_stats(device)

    # Train parquet only: do not call load_clinc_plus(), which eagerly reads all splits.
    frame = pd.read_parquet(ROOT / config["data"]["train_parquet"])
    frame = frame[frame["intent"] != int(config["data"]["drop_intent_id"])]
    prompt = config["data"]["prompt"]
    texts = [prompt.format(utterance=text) for text in frame["text"].tolist()]

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / config["model"]["path"]))
    tokenized = tokenizer(texts, padding=False, truncation=False)["input_ids"]
    lengths = np.asarray([len(ids) for ids in tokenized], dtype=np.int64)
    representatives = select_quantile_indices(lengths, list(config["data"]["quantiles"]))

    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(ROOT / config["model"]["path"]), dtype=torch.float32
    ).to(device).eval()
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    executor = H2ModularDeberta(model)
    legacy = ModularStack(model)

    head_path = ROOT / config["head"]["artifact"]
    head_npz = np.load(head_path)
    weight = torch.from_numpy(head_npz[config["head"]["weight_key"]]).to(device)
    bias = torch.from_numpy(head_npz[config["head"]["bias_key"]]).to(device)

    prepared_samples: list[tuple[dict[str, Any], Any, torch.Tensor]] = []
    preparation_rows: list[dict[str, Any]] = []
    for representative in representatives:
        sample_text = texts[representative["index"]]
        ids, mask = encode(tokenizer, [sample_text], device)
        prep_summary, prep_durations = timed_cuda_calls(
            lambda ids=ids, mask=mask: executor.prepare(ids, mask).x0,
            warmup=int(config["benchmark"]["warmup_repetitions"]),
            repetitions=int(config["benchmark"]["measured_repetitions"]),
            device=device,
        )
        prepared = executor.prepare(ids, mask)
        prepared_samples.append((representative, prepared, mask))
        preparation_rows.append({
            **representative,
            "kind": "embedding_and_constants",
            "timing_ms": prep_summary,
            "durations_ms": prep_durations,
        })

    # Bit-exact compatibility against the source implementation on train examples.
    parity: list[dict[str, Any]] = []
    representative, prepared, mask = prepared_samples[1]
    for path in (
        tuple(range(1, 13)),
        (1, 2, 6, 4, 9),
        (1, 1, 2),
    ):
        current, _ = executor.forward_from_prefix(prepared, path)
        reference = legacy_fp16_branch(legacy, prepared.x0, mask, path)
        record = {
            "path": list(path),
            "torch_equal": bool(torch.equal(current, reference)),
            "max_abs_diff": float((current.float() - reference.float()).abs().max().item()),
        }
        parity.append(record)
        if not record["torch_equal"]:
            raise RuntimeError(f"H2 forward is not bit-exact with branch stack: {record}")

    warmup = int(config["benchmark"]["warmup_repetitions"])
    repetitions = int(config["benchmark"]["measured_repetitions"])
    scenario_rows: list[dict[str, Any]] = []

    # Sample-wise full misses at representative token lengths.
    for representative, prepared, _ in prepared_samples:
        for path_length in config["benchmark"]["full_path_lengths"]:
            path = canonical_cycle_path(int(path_length))

            def full_call(prepared=prepared, path=path):
                hidden, _ = executor.forward_from_prefix(prepared, path)
                return fixed_head_logits(hidden, weight, bias).argmax(dim=1)

            torch.cuda.reset_peak_memory_stats(device)
            summary, durations = timed_cuda_calls(
                full_call, warmup=warmup, repetitions=repetitions, device=device
            )
            scenario_rows.append({
                "kind": "sample_wise_full_miss",
                **representative,
                "batch_size": 1,
                "path_length": len(path),
                "cached_prefix_length": 0,
                "evaluated_suffix_length": len(path),
                "timing_ms": summary,
                "durations_ms": durations,
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            })

    # Exact prefix hits: build an 18-layer route once and time only the suffix.
    route18 = canonical_cycle_path(18)
    for representative, prepared, _ in prepared_samples:
        _, prefixes = executor.forward_from_prefix(prepared, route18, capture_prefixes=True)
        for suffix_length in config["benchmark"]["suffix_lengths"]:
            suffix_length = int(suffix_length)
            prefix_length = len(route18) - suffix_length
            cached = prefixes[route18[:prefix_length]] if prefix_length else None

            def suffix_call(
                prepared=prepared,
                path=route18,
                prefix_length=prefix_length,
                cached=cached,
            ):
                hidden, _ = executor.forward_from_prefix(
                    prepared,
                    path,
                    cached_prefix_length=prefix_length,
                    cached_hidden=cached,
                )
                return fixed_head_logits(hidden, weight, bias).argmax(dim=1)

            torch.cuda.reset_peak_memory_stats(device)
            summary, durations = timed_cuda_calls(
                suffix_call, warmup=warmup, repetitions=repetitions, device=device
            )
            scenario_rows.append({
                "kind": "sample_wise_prefix_hit",
                **representative,
                "batch_size": 1,
                "path_length": len(route18),
                "cached_prefix_length": prefix_length,
                "evaluated_suffix_length": suffix_length,
                "timing_ms": summary,
                "durations_ms": durations,
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            })

    # Same-path batching is an optimistic scheduler bound, not the current runner.
    rng = np.random.default_rng(int(config["data"]["batch_sample_seed"]))
    maximum_batch = max(map(int, config["benchmark"]["same_path_batch_sizes"]))
    batch_indices = rng.choice(len(texts), size=maximum_batch, replace=False)
    batch_path = canonical_cycle_path(int(config["benchmark"]["batch_path_length"]))
    for batch_size in config["benchmark"]["same_path_batch_sizes"]:
        batch_size = int(batch_size)
        ids, mask = encode(tokenizer, [texts[i] for i in batch_indices[:batch_size]], device)
        prepared = executor.prepare(ids, mask)

        def batch_call(prepared=prepared, path=batch_path):
            hidden, _ = executor.forward_from_prefix(prepared, path)
            return fixed_head_logits(hidden, weight, bias).argmax(dim=1)

        torch.cuda.reset_peak_memory_stats(device)
        summary, durations = timed_cuda_calls(
            batch_call, warmup=warmup, repetitions=repetitions, device=device
        )
        scenario_rows.append({
            "kind": "same_path_batch_upper_bound",
            "batch_size": batch_size,
            "tokens_padded": int(ids.shape[1]),
            "path_length": len(batch_path),
            "cached_prefix_length": 0,
            "evaluated_suffix_length": len(batch_path),
            "timing_ms": summary,
            "durations_ms": durations,
            "samples_per_second_median": 1000.0 * batch_size / summary["median"],
            "milliseconds_per_sample_median": summary["median"] / batch_size,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        })

    # Host-only exact-result cache lookup, including argmax retrieval.
    result_cache = {tuple(range(1, 13)): 17}
    key = tuple(range(1, 13))
    cache_durations: list[float] = []
    for _ in range(100_000):
        started = time.perf_counter_ns()
        _ = result_cache[key]
        cache_durations.append((time.perf_counter_ns() - started) / 1_000_000.0)

    counts = h2_simulation_counts()
    estimates: dict[str, dict[str, float]] = {}
    median_token = min(representatives, key=lambda item: abs(item["quantile"] - 0.5))
    for row in scenario_rows:
        if row["kind"] not in {"sample_wise_full_miss", "sample_wise_prefix_hit"}:
            continue
        if row["index"] != median_token["index"]:
            continue
        label = (
            f"full_L{row['path_length']}" if row["kind"] == "sample_wise_full_miss"
            else f"prefix_suffix_L{row['evaluated_suffix_length']}"
        )
        milliseconds = row["timing_ms"]["median"]
        estimates[label] = {
            "median_ms_per_simulation": milliseconds,
            "total_gpu_hours_for_3876000_simulations": runtime_hours(counts["total"], milliseconds),
        }

    after_gpu = nvidia_snapshot()
    results = {
        "experiment_id": config["experiment_id"],
        "scope": config["scope"],
        "git": git_state(),
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "head_artifact": {
            "path": config["head"]["artifact"],
            "sha256": sha256(head_path),
            "weight_shape": list(weight.shape),
            "bias_shape": list(bias.shape),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "model_load_seconds": model_load_seconds,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "gpu_before": before_gpu,
            "gpu_after": after_gpu,
            "peak_cuda_bytes_observed": max(
                [int(torch.cuda.max_memory_allocated(device))]
                + [int(row["peak_cuda_bytes"]) for row in scenario_rows]
            ),
        },
        "train_data": {
            "n": len(texts),
            "token_lengths": distribution(lengths.astype(float).tolist()),
            "representatives": representatives,
        },
        "parity": parity,
        "preparation": preparation_rows,
        "scenarios": scenario_rows,
        "cache_hit_timing_ms": distribution(cache_durations),
        "simulation_counts": counts,
        "complexity_estimates": estimates,
        "limitations": [
            "short burst benchmark; sustained thermal throttling is not measured",
            "same-path batching is an optimistic scheduler bound, not an implemented H2 scheduler",
            "prefix-hit estimates assume exact FP16 prefix states are retained in GPU memory",
            "full-run cache hit rate and searched path-length distribution are unknown before pilot",
        ],
    }
    results_path = output / config["runtime"]["results_name"]
    atomic_write_json(results_path, results)

    tracking_uri = "sqlite:///" + str((ROOT / "mlruns.db").resolve())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("EXP-004-readability-qualification")
    with mlflow.start_run(run_name=config["experiment_id"]) as active:
        mlflow.set_tags({
            "experiment_phase": "engineering_throughput_preflight",
            "official_hypothesis_evidence": "false",
            "validation_accessed": "false",
            "test_accessed": "false",
        })
        mlflow.log_params({
            "git_commit": results["git"]["commit"],
            "git_dirty": results["git"]["dirty"],
            "model": "deberta-v3-base",
            "dataset": "CLINC150 official-train only",
            "total_simulations_estimated": counts["total"],
            "branch_boundary_dtype": "float16",
            "compute_dtype": "float32",
        })
        for label, estimate in estimates.items():
            mlflow.log_metric(f"{label}_median_ms", estimate["median_ms_per_simulation"])
            mlflow.log_metric(
                f"{label}_total_gpu_hours",
                estimate["total_gpu_hours_for_3876000_simulations"],
            )
        mlflow.log_artifact(str(results_path), artifact_path="evidence")
        mlflow.log_artifact(str(output / config["runtime"]["resolved_config_name"]), artifact_path="evidence")
        run_id = active.info.run_id
    atomic_write_json(output / "mlflow_run.json", {
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "experiment": "EXP-004-readability-qualification",
    })
    print(json.dumps({
        "results": str(results_path),
        "mlflow_run_id": run_id,
        "complexity_estimates": estimates,
    }, indent=2))


if __name__ == "__main__":
    main()
