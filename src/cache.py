"""One-pass frozen-backbone hidden-state caching.

Runs ``DeBERTa-v3-base`` once per split with ``output_hidden_states=True``,
extracts the CLS token (index 0) of Transformer layers 1..12, and stores them
as float16 safetensors together with remapped labels and sample ids.

CLS representations are batch-independent because padding tokens are masked in
self-attention, so dynamic padding (longest-in-batch) is safe and reproducible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pad_batch(input_ids_list, pad_id: int):
    maxlen = max(len(ids) for ids in input_ids_list)
    b = len(input_ids_list)
    input_ids = torch.full((b, maxlen), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((b, maxlen), dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        input_ids[i, : len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    return input_ids, attention_mask


def cache_split(
    model,
    tokenizer,
    tokenised: dict,
    device: torch.device,
    batch_size: int = 128,
    n_layers: int = 12,
    hidden_state_offset: int = 1,
    cls_index: int = 0,
) -> dict:
    """Forward the frozen backbone over one split; return CLS hidden states.

    Returns ``{"hidden": torch.Tensor (N, n_layers, 768) float32,
    "labels": torch.Tensor (N,) int64, "sample_ids": torch.Tensor (N,) int64}``.
    """
    model.eval()
    pad_id = tokenizer.pad_token_id
    input_ids_list = tokenised["input_ids"]
    labels = torch.as_tensor(tokenised["labels"], dtype=torch.long)
    sample_ids = torch.as_tensor(tokenised["sample_ids"], dtype=torch.long)

    all_cls = []
    order = []  # sample_ids in the order they were processed
    for start in range(0, len(input_ids_list), batch_size):
        batch_ids = input_ids_list[start : start + batch_size]
        input_ids, attention_mask = _pad_batch(batch_ids, pad_id)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hs = outputs.hidden_states
        assert len(hs) == n_layers + 1, (
            f"expected {n_layers + 1} hidden states (embedding + {n_layers} layers), "
            f"got {len(hs)}"
        )
        # layers 1..n_layers -> hidden_states[offset : offset+n_layers], CLS at cls_index
        layers = [hs[hidden_state_offset + l - 1][:, cls_index, :] for l in range(1, n_layers + 1)]
        cls = torch.stack(layers, dim=1)  # (B, n_layers, 768)
        all_cls.append(cls.cpu())
        order.extend(sample_ids[start : start + batch_size].tolist())

    hidden = torch.cat(all_cls, dim=0)  # (N, n_layers, 768)
    # restore original sample order (batching preserves order here, but be explicit)
    order = torch.tensor(order, dtype=torch.long)
    assert torch.equal(order, sample_ids), "sample order changed during caching"
    return {"hidden": hidden, "labels": labels, "sample_ids": sample_ids}


def save_cache(cache: dict, out_path: Path, cache_dtype: str = "float16") -> dict:
    """Save a split cache as safetensors; return manifest entry."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hidden = cache["hidden"]
    if cache_dtype == "float16":
        hidden_store = hidden.to(torch.float16)
    elif cache_dtype == "float32":
        hidden_store = hidden.to(torch.float32)
    else:
        raise ValueError(f"unsupported cache_dtype {cache_dtype}")
    hidden_store = hidden_store.contiguous()
    labels = cache["labels"].to(torch.int64).contiguous()
    sample_ids = cache["sample_ids"].to(torch.int64).contiguous()
    save_file(
        {"hidden": hidden_store, "labels": labels, "sample_ids": sample_ids},
        str(out_path),
    )
    return {
        "file": str(out_path),
        "shapes": {
            "hidden": list(hidden_store.shape),
            "labels": list(labels.shape),
            "sample_ids": list(sample_ids.shape),
        },
        "dtypes": {
            "hidden": str(hidden_store.dtype).replace("torch.", ""),
            "labels": str(labels.dtype).replace("torch.", ""),
            "sample_ids": str(sample_ids.dtype).replace("torch.", ""),
        },
        "sha256": _sha256(out_path),
    }


def load_cache(path: Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> dict:
    """Load a cached split. ``hidden`` is moved to ``device`` and cast to ``dtype``."""
    from safetensors.torch import load_file

    data = load_file(str(path))
    return {
        "hidden": data["hidden"].to(device=device, dtype=dtype),
        "labels": data["labels"].to(torch.int64),
        "sample_ids": data["sample_ids"].to(torch.int64),
    }


def write_manifest(manifest: dict, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
