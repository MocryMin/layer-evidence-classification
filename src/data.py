"""CLINC150 data preparation for EXP-20260729-001.

Loads the ``plus`` config of ``clinc/clinc_oos`` from local parquet, drops the
OOS class (label id 42), remaps the 150 in-scope intents to contiguous ids
``0..149`` (original names preserved), wraps each utterance in the fixed prompt,
and tokenises with left truncation and no padding (dynamic padding happens in
the backbone forward pass).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Offline by default: all data is local parquet.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from datasets import Dataset, load_dataset  # noqa: E402

OOS_NAME = "oos"


def _parquet_file(dataset_root: Path, config: str, split: str) -> Path:
    return Path(dataset_root) / config / f"{split}-00000-of-00001.parquet"


def build_label_maps(dataset_root: Path, config: str, drop_oos_label: int):
    """Build ``label2id`` (name -> 0..149) and ``id2label`` (str(id) -> name).

    In-scope original ids are sorted ascending and mapped to ``0..149`` so the
    mapping is deterministic. Returns ``(label2id, id2label, in_scope_ids)``.
    """
    train_file = _parquet_file(dataset_root, config, "train")
    ds = load_dataset("parquet", data_files={"train": str(train_file)}, split="train")
    names = ds.features["intent"].names
    # sanity: the OOS id really is "oos"
    assert names[drop_oos_label] == OOS_NAME, (
        f"label {drop_oos_label} is {names[drop_oos_label]!r}, expected 'oos'"
    )
    in_scope_ids = sorted(i for i in range(len(names)) if i != drop_oos_label)
    assert len(in_scope_ids) == 150, f"expected 150 in-scope ids, got {len(in_scope_ids)}"
    label2id = {names[i]: new_id for new_id, i in enumerate(in_scope_ids)}
    id2label = {str(new_id): names[i] for new_id, i in enumerate(in_scope_ids)}
    return label2id, id2label, in_scope_ids


def load_split(
    dataset_root: Path,
    config: str,
    split: str,
    drop_oos_label: int,
    in_scope_ids: list[int],
) -> Dataset:
    """Load one split, drop OOS, remap labels to 0..149.

    Returns a ``Dataset`` with columns ``text``, ``label`` (remapped 0..149) and
    ``sample_id`` (0..N-1 in the filtered split).
    """
    file = _parquet_file(dataset_root, config, split)
    ds = load_dataset("parquet", data_files={split: str(file)}, split=split)
    ds = ds.filter(lambda x: x["intent"] != drop_oos_label)
    remap = {orig: new for new, orig in enumerate(in_scope_ids)}

    def _map_fn(batch):
        labels = [remap[i] for i in batch["intent"]]
        return {"text": batch["text"], "label": labels}

    ds = ds.map(_map_fn, batched=True, remove_columns=ds.column_names)
    ds = ds.add_column("sample_id", list(range(len(ds))))
    return ds


def make_prompt(utterance: str, template: str) -> str:
    return template.format(utterance=utterance)


def tokenise_split(
    ds: Dataset,
    tokenizer,
    prompt: str,
    max_length: int,
    truncation_side: str = "left",
) -> dict:
    """Tokenise with the prompt and left truncation, no padding.

    Returns ``{"input_ids": List[List[int]], "attention_mask": List[List[int]],
    "labels": np.ndarray, "sample_ids": np.ndarray}``.
    """
    original_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side
    try:
        texts = [make_prompt(t, prompt) for t in ds["text"]]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )
    finally:
        tokenizer.truncation_side = original_side
    import numpy as np

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": np.asarray(ds["label"], dtype=np.int64),
        "sample_ids": np.asarray(ds["sample_id"], dtype=np.int64),
        "texts": ds["text"],
    }


def save_label_maps(label2id: dict, id2label: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(out_dir / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
