#!/usr/bin/env python3
"""Validation-only numerical audit for the EXP-004 H2 canonical baseline.

The source head was evaluated by the fragmented pipeline in batches of 512,
whereas H2 evaluates one sample at a time.  This audit verifies tokenization
identity and quantifies any prediction changes caused solely by inference
batch shape.  It never loads the test split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exp004_h2_run import (  # noqa: E402
    atomic_write_json,
    canonical_scan,
    load_split,
    padded_batch,
    tokenize_records,
)
from scripts.frag_modular_probe import tokenize_data  # noqa: E402
from src.exp004_h2_forward import H2ModularDeberta  # noqa: E402
from src.seeding import enable_determinism, seed_all  # noqa: E402


def compare_tokenization(tokenizer, records, prompt: str) -> tuple[list[list[int]], dict]:
    texts = [prompt.format(utterance=record["text"]) for record in records]
    legacy_ids, legacy_mask, legacy_max = tokenize_data(tokenizer, texts)
    current = tokenize_records(tokenizer, records, prompt)
    mismatches = []
    for index, ids in enumerate(current):
        length = int(legacy_mask[index].sum().item())
        legacy = legacy_ids[index, :length].tolist()
        if list(ids) != legacy:
            mismatches.append({
                "split_index": index,
                "legacy": legacy,
                "current": list(ids),
            })
    return current, {
        "legacy_max_length": int(legacy_max),
        "current_max_length": int(max(map(len, current))),
        "n_sequence_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
    }


def compare_scans(scans: dict[str, list[dict]]) -> dict:
    comparisons = {}
    keys = list(scans)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            disagreements = []
            correctness_flips = []
            rank_changes = []
            for left_row, right_row in zip(scans[left], scans[right], strict=True):
                index = int(left_row["split_index"])
                if left_row["predicted_class"] != right_row["predicted_class"]:
                    disagreements.append(index)
                if left_row["correct"] != right_row["correct"]:
                    correctness_flips.append({
                        "split_index": index,
                        left: {
                            "predicted_class": left_row["predicted_class"],
                            "correct": left_row["correct"],
                            "gold_rank": left_row["gold_rank"],
                        },
                        right: {
                            "predicted_class": right_row["predicted_class"],
                            "correct": right_row["correct"],
                            "gold_rank": right_row["gold_rank"],
                        },
                    })
                if left_row["gold_rank"] != right_row["gold_rank"]:
                    rank_changes.append(index)
            comparisons[f"{left}_vs_{right}"] = {
                "prediction_disagreements": len(disagreements),
                "prediction_disagreement_indices": disagreements,
                "correctness_flips": len(correctness_flips),
                "correctness_flip_details": correctness_flips,
                "gold_rank_changes": len(rank_changes),
                "gold_rank_change_indices": rank_changes,
            }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs/exp004_h2_full_v1.yaml")
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64, 512])
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "artifacts/EXP-20260831-004-h2-numerical-audit/results.json"
        ),
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    semantics = yaml.safe_load(
        (ROOT / config["semantics_config"]).read_text(encoding="utf-8")
    )
    seed_all(int(config["search_runtime"]["seed"]))
    enable_determinism()
    torch.set_float32_matmul_precision(config["model"]["float32_matmul_precision"])
    if not torch.cuda.is_available():
        raise RuntimeError("numerical audit requires CUDA")
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / config["model"]["path"]))
    records = load_split(config, "validation")
    token_ids, token_audit = compare_tokenization(
        tokenizer, records, config["dataset"]["prompt"]
    )
    fixed_length = max(map(len, token_ids))
    # Also require the batch-one padding helper to reproduce the corresponding
    # globally padded legacy row exactly before any model comparison.
    legacy_texts = [
        config["dataset"]["prompt"].format(utterance=record["text"])
        for record in records
    ]
    legacy_ids, legacy_mask, _ = tokenize_data(tokenizer, legacy_texts)
    padded_mismatches = []
    for index, ids in enumerate(token_ids):
        current_ids, current_mask = padded_batch(
            tokenizer, [ids], torch.device("cpu"), fixed_length=fixed_length
        )
        if not torch.equal(current_ids[0], legacy_ids[index]) or not torch.equal(
            current_mask[0], legacy_mask[index]
        ):
            padded_mismatches.append(index)
    token_audit["n_globally_padded_row_mismatches"] = len(padded_mismatches)
    token_audit["globally_padded_row_mismatch_indices"] = padded_mismatches[:20]

    model = AutoModel.from_pretrained(
        str(ROOT / config["model"]["path"]), dtype=torch.float32
    ).to(device).eval()
    executor = H2ModularDeberta(model)
    head_npz = np.load(ROOT / semantics["canonical_head"]["artifact"])
    weight = torch.from_numpy(
        head_npz[semantics["canonical_head"]["weight_key"]]
    ).to(device)
    bias = torch.from_numpy(
        head_npz[semantics["canonical_head"]["bias_key"]]
    ).to(device)

    scans = {}
    summaries = {}
    for batch_size in args.batch_sizes:
        label = f"batch_{batch_size}"
        print(f"[audit] canonical scan {label}", flush=True)
        rows = canonical_scan(
            records,
            token_ids,
            tokenizer=tokenizer,
            executor=executor,
            weight=weight,
            bias=bias,
            canonical_path=semantics["model_and_task"]["canonical_path"],
            batch_size=batch_size,
            fixed_length=fixed_length,
            device=device,
        )
        scans[label] = rows
        summaries[label] = {
            "correct": sum(bool(row["correct"]) for row in rows),
            "wrong": sum(not bool(row["correct"]) for row in rows),
            "total": len(rows),
        }

    payload = {
        "scope": "validation_only",
        "test_accessed": False,
        "source_expected": config["dataset"]["canonical_validation_audit"],
        "tokenization": token_audit,
        "fixed_padding_length": fixed_length,
        "canonical_summaries": summaries,
        "batch_shape_comparisons": compare_scans(scans),
    }
    output = Path(args.output)
    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(f"[audit] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
