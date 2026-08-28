"""Infrastructure for the EXP-004 H1 engineering qualification.

The official H1 search protocol is still being frozen.  This module therefore
contains only reusable, protocol-neutral pieces: ARC-Easy formatting, masked
multiple-choice metrics, a modular Llama forward, atomic artifacts, and a
required wall-clock deadline.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def canonical_json_hash(value: Any) -> str:
    """Return a stable SHA256 over a JSON-serialisable object."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace *path* with JSON, never exposing a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    """Durably replace *path* with a torch artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class EventJournal:
    """Append-only, fsync-on-commit JSONL event journal."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def append(self, event: str, **payload: Any) -> None:
        record = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": event,
            **payload,
        }
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class DeadlineReached(RuntimeError):
    """Raised at a safe boundary when the session must stop."""


class DeadlineController:
    """Required absolute deadline with a soft-stop reserve and signal support."""

    def __init__(self, stop_at: str, reserve_minutes: int):
        hard = datetime.fromisoformat(stop_at)
        if hard.tzinfo is None or hard.utcoffset() is None:
            raise ValueError("--stop-at must include an explicit UTC offset")
        if reserve_minutes < 1:
            raise ValueError("reserve_minutes must be at least 1")
        now = datetime.now().astimezone()
        if hard <= now:
            raise ValueError(f"--stop-at is not in the future: {hard.isoformat()}")
        self.hard_stop = hard
        self.soft_stop = hard - timedelta(minutes=reserve_minutes)
        if self.soft_stop <= now:
            raise ValueError("deadline is too close to preserve the requested reserve")
        self.reserve_minutes = reserve_minutes
        self.signal_received: int | None = None

    def install_signal_handlers(self) -> None:
        def _mark(signum: int, _frame: Any) -> None:
            self.signal_received = signum

        signal.signal(signal.SIGINT, _mark)
        signal.signal(signal.SIGTERM, _mark)

    def seconds_to_hard_stop(self) -> float:
        return (self.hard_stop - datetime.now().astimezone()).total_seconds()

    def seconds_to_soft_stop(self) -> float:
        return (self.soft_stop - datetime.now().astimezone()).total_seconds()

    def stop_requested(self) -> bool:
        return self.signal_received is not None or self.seconds_to_soft_stop() <= 0

    def checkpoint(self, next_unit_seconds: float = 0.0) -> None:
        if self.signal_received is not None:
            raise DeadlineReached(f"received signal {self.signal_received}")
        if self.seconds_to_soft_stop() <= next_unit_seconds:
            raise DeadlineReached(
                "soft deadline reached or insufficient time for the next atomic unit"
            )


@dataclass(frozen=True)
class ArcExample:
    sample_id: str
    question: str
    choices: tuple[str, ...]
    answer_position: int

    @property
    def n_choices(self) -> int:
        return len(self.choices)


def _answer_position(answer_key: Any, labels: Sequence[Any]) -> int:
    key = str(answer_key)
    str_labels = [str(item) for item in labels]
    if key not in str_labels:
        raise ValueError(f"answer key {key!r} is absent from choices.label={str_labels}")
    return str_labels.index(key)


def load_arc_easy_split(dataset_root: Path, split: str) -> list[ArcExample]:
    """Load a local ARC-Easy parquet split without using the network."""
    path = dataset_root / f"{split}-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    examples: list[ArcExample] = []
    for row_number, row in frame.iterrows():
        raw_choices = row["choices"]
        labels = list(raw_choices["label"])
        texts = tuple(str(item) for item in raw_choices["text"])
        answer_position = _answer_position(row["answerKey"], labels)
        if not 2 <= len(texts) <= 5:
            raise ValueError(f"unsupported number of choices at row {row_number}: {len(texts)}")
        examples.append(
            ArcExample(
                sample_id=str(row.get("id", row_number)),
                question=str(row["question"]),
                choices=texts,
                answer_position=answer_position,
            )
        )
    return examples


def make_fit_discover_indices(
    n_samples: int, n_fit: int, seed: int
) -> dict[str, list[int]]:
    """Create a fixed disjoint fit/discover split over official train."""
    if not 0 < n_fit < n_samples:
        raise ValueError("n_fit must be between 0 and n_samples")
    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)
    split = {"fit": sorted(indices[:n_fit]), "discover": sorted(indices[n_fit:])}
    assert set(split["fit"]).isdisjoint(split["discover"])
    assert sorted(split["fit"] + split["discover"]) == list(range(n_samples))
    return split


def format_arc_prompt(example: ArcExample, prompt_cfg: dict[str, Any]) -> str:
    """Format one zero-shot ARC item, relabelling options by position A--E."""
    alphabet = "ABCDE"
    if example.n_choices > len(alphabet):
        raise ValueError("more than five choices are unsupported")
    lines = [str(prompt_cfg["instruction"]).strip(), ""]
    lines.append(f"{prompt_cfg.get('question_prefix', 'Question:')} {example.question}")
    for position, choice in enumerate(example.choices):
        lines.append(f"{alphabet[position]}. {choice}")
    answer_prefix = str(prompt_cfg.get("answer_prefix", "Answer:"))
    if answer_prefix:
        lines.extend(["", answer_prefix])
    return "\n".join(lines)


def valid_choice_mask(choice_counts: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
    """Return [N, C] bool mask where columns below each choice count are valid."""
    if choice_counts.ndim != 1:
        raise ValueError("choice_counts must be one-dimensional")
    if torch.any(choice_counts < 1) or torch.any(choice_counts > n_classes):
        raise ValueError("choice count outside supported class range")
    columns = torch.arange(n_classes, device=choice_counts.device)
    return columns.unsqueeze(0) < choice_counts.unsqueeze(1)


def mask_invalid_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"logits/mask shape mismatch: {logits.shape} vs {mask.shape}")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    masked = mask_invalid_logits(logits, mask)
    if torch.any(~mask.gather(1, labels[:, None]).squeeze(1)):
        raise ValueError("a gold label is masked as invalid")
    return float((masked.argmax(dim=1) == labels).float().mean().item())


def chance_accuracy(choice_counts: Iterable[int]) -> float:
    counts = list(choice_counts)
    if not counts:
        raise ValueError("empty choice-count collection")
    return sum(1.0 / int(count) for count in counts) / len(counts)


class ModularLlamaExecutor:
    """Compose pretrained Llama decoder blocks in an arbitrary 1-based path."""

    def __init__(self, causal_lm: torch.nn.Module, answer_token_ids: Sequence[int]):
        if len(answer_token_ids) != 5:
            raise ValueError("exactly five A--E answer token ids are required")
        self.causal_lm = causal_lm
        self.backbone = causal_lm.model
        self.answer_token_ids = tuple(int(item) for item in answer_token_ids)

    def _inputs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        from transformers.models.llama.modeling_llama import create_causal_mask

        embeddings = self.backbone.embed_tokens(input_ids)
        position_ids = torch.arange(
            embeddings.shape[1], device=embeddings.device, dtype=torch.long
        ).unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.backbone.config,
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = self.backbone.rotary_emb(
            embeddings, position_ids=position_ids
        )
        return embeddings, position_ids, causal_mask, position_embeddings

    @staticmethod
    def _terminal(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        terminal_positions = attention_mask.long().sum(dim=1) - 1
        if torch.any(terminal_positions < 0):
            raise ValueError("empty token sequence")
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, terminal_positions]

    def label_logits(self, terminal: torch.Tensor) -> torch.Tensor:
        rows = self.causal_lm.lm_head.weight[
            torch.tensor(self.answer_token_ids, device=terminal.device)
        ]
        return F.linear(terminal, rows)

    @torch.inference_mode()
    def forward_path(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        path: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, position_ids, causal_mask, position_embeddings = self._inputs(
            input_ids, attention_mask
        )
        n_layers = len(self.backbone.layers)
        for layer_id in path:
            if not 1 <= int(layer_id) <= n_layers:
                raise ValueError(f"layer id {layer_id} outside 1..{n_layers}")
            hidden = self.backbone.layers[int(layer_id) - 1](
                hidden,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
        hidden = self.backbone.norm(hidden)
        terminal = self._terminal(hidden, attention_mask)
        return terminal, self.label_logits(terminal)

    @torch.inference_mode()
    def forward_path_from_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        path: Sequence[int],
        *,
        cached_prefix_length: int = 0,
        cached_hidden: torch.Tensor | None = None,
        capture_prefix_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Continue a path from a globally cached raw hidden state.

        Cached and captured tensors are before the final RMSNorm and contain
        every token.  This is necessary because another decoder block needs the
        whole sequence, not only the terminal token.
        """
        embeddings, position_ids, causal_mask, position_embeddings = self._inputs(
            input_ids, attention_mask
        )
        if not 0 <= cached_prefix_length <= len(path):
            raise ValueError("cached prefix length is outside the path")
        if cached_prefix_length:
            if cached_hidden is None:
                raise ValueError("a non-empty cached prefix requires hidden state")
            if tuple(cached_hidden.shape) != tuple(embeddings.shape):
                raise ValueError(
                    f"cached hidden shape {tuple(cached_hidden.shape)} != {tuple(embeddings.shape)}"
                )
            hidden = cached_hidden.to(device=embeddings.device, dtype=embeddings.dtype)
        else:
            hidden = embeddings
        if capture_prefix_length is not None and not (
            cached_prefix_length <= capture_prefix_length <= len(path)
        ):
            raise ValueError("capture prefix must be reachable from the cached prefix")
        captured = hidden if capture_prefix_length == cached_prefix_length else None
        n_layers = len(self.backbone.layers)
        for offset, layer_id in enumerate(path[cached_prefix_length:], start=cached_prefix_length + 1):
            if not 1 <= int(layer_id) <= n_layers:
                raise ValueError(f"layer id {layer_id} outside 1..{n_layers}")
            hidden = self.backbone.layers[int(layer_id) - 1](
                hidden,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
            if offset == capture_prefix_length:
                captured = hidden
        normalised = self.backbone.norm(hidden)
        terminal = self._terminal(normalised, attention_mask)
        return terminal, self.label_logits(terminal), captured

    @torch.inference_mode()
    def forward_native(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        terminal = self._terminal(output.last_hidden_state, attention_mask)
        return terminal, self.label_logits(terminal)


def token_ids(value: Any) -> list[int]:
    """Normalise tokenizer/BatchEncoding outputs to one flat ID list."""
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
    """Verify the exact A--E token immediately after the assistant header."""
    example_text = "Question: 1+1?\nA. 1\nB. 2\n\nAnswer:"
    messages = [{"role": "user", "content": example_text}]
    base = token_ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True
        )
    )
    observed: list[int] = []
    for label in config["tokenization"]["answer_labels"]:
        full = token_ids(
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


def encode_arc_examples(
    tokenizer: Any, examples: Sequence[ArcExample], prompt_cfg: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Apply the frozen chat template and dynamically right-pad one batch."""
    encoded = []
    for example in examples:
        prompt = format_arc_prompt(example, prompt_cfg)
        messages = [{"role": "user", "content": prompt}]
        ids = token_ids(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True
            )
        )
        encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return tokenizer.pad(encoded, padding=True, return_tensors="pt")


def extract_path_feature_split(
    split_name: str,
    examples: Sequence[ArcExample],
    original_indices: Sequence[int],
    *,
    executor: ModularLlamaExecutor,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    max_length_guard: int,
    path: Sequence[int],
    path_id: str,
    batch_size: int,
    shard_size: int,
    config_hash: str,
    feature_root: Path,
    deadline: DeadlineController,
    journal: EventJournal,
) -> dict[str, Any]:
    """Extract one path/split in crash-safe shards, reusing completed shards."""
    split_root = feature_root / path_id / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    device = next(executor.causal_lm.parameters()).device
    batch_durations: list[float] = []
    n_reused = 0
    for shard_start in range(0, len(examples), shard_size):
        next_estimate = max(batch_durations[-5:] or [60.0])
        deadline.checkpoint(next_unit_seconds=next_estimate)
        shard_end = min(shard_start + shard_size, len(examples))
        shard_path = split_root / f"shard_{shard_start:05d}_{shard_end:05d}.pt"
        expected_indices = list(original_indices[shard_start:shard_end])
        if shard_path.exists():
            saved = torch.load(shard_path, map_location="cpu", weights_only=False)
            if (
                saved["config_hash"] != config_hash
                or saved["original_indices"] != expected_indices
                or saved["path"] != list(path)
            ):
                raise RuntimeError(f"resume validation failed for {shard_path}")
            n_reused += shard_end - shard_start
            continue

        shard_examples = examples[shard_start:shard_end]
        features: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        shard_batch_durations: list[float] = []
        journal.append(
            "path_shard_started",
            path_id=path_id,
            split=split_name,
            start=shard_start,
            end=shard_end,
        )
        shard_t0 = time.monotonic()
        for batch_start in range(0, len(shard_examples), batch_size):
            next_estimate = max(batch_durations[-5:] or [60.0])
            deadline.checkpoint(next_unit_seconds=next_estimate)
            batch_examples = shard_examples[batch_start : batch_start + batch_size]
            encoded = encode_arc_examples(tokenizer, batch_examples, prompt_cfg)
            if encoded["input_ids"].shape[1] > max_length_guard:
                raise RuntimeError(
                    f"token length {encoded['input_ids'].shape[1]} exceeds max_length_guard"
                )
            input_ids = encoded["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
            batch_t0 = time.monotonic()
            terminal, label_logits = executor.forward_path(input_ids, attention_mask, path)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.monotonic() - batch_t0
            batch_durations.append(duration)
            shard_batch_durations.append(duration)
            features.append(terminal.to(dtype=torch.float16, device="cpu"))
            logits.append(label_logits.float().cpu())

        payload = {
            "config_hash": config_hash,
            "path_id": path_id,
            "path": list(path),
            "split": split_name,
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
            "batch_durations_seconds": shard_batch_durations,
        }
        atomic_torch_save(shard_path, payload)
        journal.append(
            "path_shard_completed",
            path_id=path_id,
            split=split_name,
            start=shard_start,
            end=shard_end,
            elapsed_seconds=payload["elapsed_seconds"],
        )
    return {
        "path_id": path_id,
        "split": split_name,
        "n_samples": len(examples),
        "n_reused": n_reused,
        "batch_durations_seconds": batch_durations,
    }


def extract_cached_path_feature_split(
    split_name: str,
    examples: Sequence[ArcExample],
    original_indices: Sequence[int],
    *,
    executor: ModularLlamaExecutor,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    max_length_guard: int,
    path: Sequence[int],
    path_id: str,
    batch_size: int,
    shard_size: int,
    config_hash: str,
    feature_root: Path,
    deadline: DeadlineController,
    journal: EventJournal,
    prefix_cache: Any,
    cached_prefix: dict[str, Any] | None,
    cache_target: Sequence[int] | None,
) -> dict[str, Any]:
    """Extract features while reading/writing one global prefix-cache payload."""
    split_root = feature_root / path_id / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    device = next(executor.causal_lm.parameters()).device
    batch_durations: list[float] = []
    n_reused = 0
    cached_prefix_path = [] if cached_prefix is None else list(cached_prefix["path"])
    cached_status = None if cached_prefix is None else str(cached_prefix["cache_status"])
    target_path = None if cache_target is None else list(cache_target)
    target_node = None if target_path is None else prefix_cache.node(target_path)
    write_target = bool(target_node and target_node["cache_status"] == "partial_ssd")

    for shard_start in range(0, len(examples), shard_size):
        next_estimate = max(batch_durations[-5:] or [60.0])
        deadline.checkpoint(next_unit_seconds=next_estimate)
        shard_end = min(shard_start + shard_size, len(examples))
        shard_path = split_root / f"shard_{shard_start:05d}_{shard_end:05d}.pt"
        expected_indices = list(original_indices[shard_start:shard_end])
        target_shard = None
        if write_target and target_path is not None:
            target_shard = prefix_cache.shard_path(
                target_path, split_name, shard_start, shard_end, writing=True
            )
        reusable_feature = shard_path.exists() and (target_shard is None or target_shard.exists())
        if reusable_feature:
            saved = torch.load(shard_path, map_location="cpu", weights_only=False)
            if (
                saved["config_hash"] != config_hash
                or saved["original_indices"] != expected_indices
                or saved["path"] != list(path)
            ):
                raise RuntimeError(f"resume validation failed for {shard_path}")
            n_reused += shard_end - shard_start
            continue

        cached_batches = None
        if cached_prefix is not None:
            cache_shard_path = prefix_cache.shard_path(
                cached_prefix_path, split_name, shard_start, shard_end
            )
            cache_payload = torch.load(cache_shard_path, map_location="cpu", weights_only=False)
            if (
                cache_payload["config_hash"] != config_hash
                or cache_payload["prefix"] != cached_prefix_path
                or cache_payload["original_indices"] != expected_indices
            ):
                raise RuntimeError(f"prefix-cache validation failed for {cache_shard_path}")
            cached_batches = cache_payload["hidden_batches"]

        shard_examples = examples[shard_start:shard_end]
        features: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        captured_batches: list[torch.Tensor] = []
        shard_batch_durations: list[float] = []
        journal.append(
            "path_shard_started",
            path_id=path_id,
            split=split_name,
            start=shard_start,
            end=shard_end,
            cached_prefix=cached_prefix_path,
            cache_tier=cached_status,
            cache_target=target_path,
        )
        shard_t0 = time.monotonic()
        for batch_number, batch_start in enumerate(range(0, len(shard_examples), batch_size)):
            next_estimate = max(batch_durations[-5:] or [60.0])
            deadline.checkpoint(next_unit_seconds=next_estimate)
            batch_examples = shard_examples[batch_start : batch_start + batch_size]
            encoded = encode_arc_examples(tokenizer, batch_examples, prompt_cfg)
            if encoded["input_ids"].shape[1] > max_length_guard:
                raise RuntimeError(
                    f"token length {encoded['input_ids'].shape[1]} exceeds max_length_guard"
                )
            input_ids = encoded["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
            hidden = None
            if cached_batches is not None:
                hidden = cached_batches[batch_number]
            batch_t0 = time.monotonic()
            terminal, label_logits, captured = executor.forward_path_from_prefix(
                input_ids,
                attention_mask,
                path,
                cached_prefix_length=len(cached_prefix_path),
                cached_hidden=hidden,
                capture_prefix_length=None if not write_target else len(target_path or []),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.monotonic() - batch_t0
            batch_durations.append(duration)
            shard_batch_durations.append(duration)
            features.append(terminal.to(dtype=torch.float16, device="cpu"))
            logits.append(label_logits.float().cpu())
            if write_target:
                if captured is None:
                    raise RuntimeError("requested cache target was not captured")
                captured_batches.append(captured.to(dtype=torch.bfloat16, device="cpu"))

        if target_shard is not None:
            atomic_torch_save(
                target_shard,
                {
                    "config_hash": config_hash,
                    "prefix": target_path,
                    "split": split_name,
                    "start": shard_start,
                    "end": shard_end,
                    "original_indices": expected_indices,
                    "hidden_batches": captured_batches,
                    "batch_size": batch_size,
                },
            )
        payload = {
            "config_hash": config_hash,
            "path_id": path_id,
            "path": list(path),
            "split": split_name,
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
            "batch_durations_seconds": shard_batch_durations,
            "cached_prefix": cached_prefix_path,
            "cache_tier": cached_status,
            "cache_target": target_path,
        }
        atomic_torch_save(shard_path, payload)
        journal.append(
            "path_shard_completed",
            path_id=path_id,
            split=split_name,
            start=shard_start,
            end=shard_end,
            elapsed_seconds=payload["elapsed_seconds"],
            cached_prefix=cached_prefix_path,
            cache_target=target_path,
        )
    return {
        "path_id": path_id,
        "split": split_name,
        "n_samples": len(examples),
        "n_reused": n_reused,
        "batch_durations_seconds": batch_durations,
        "cached_prefix": cached_prefix_path,
        "cache_tier": cached_status,
        "cache_target": target_path,
    }


def load_path_feature_split(
    feature_root: Path, path_id: str, split_name: str
) -> dict[str, Any]:
    paths = sorted((feature_root / path_id / split_name).glob("shard_*.pt"))
    if not paths:
        raise RuntimeError(f"no completed shards for {path_id}/{split_name}")
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    expected = 0
    for shard in shards:
        if shard["start"] != expected:
            raise RuntimeError(
                f"non-contiguous shards for {path_id}/{split_name}: expected {expected}"
            )
        expected = shard["end"]
    return {
        "features": torch.cat([item["features"] for item in shards]),
        "native_label_logits": torch.cat([item["native_label_logits"] for item in shards]),
        "labels": torch.cat([item["labels"] for item in shards]),
        "choice_counts": torch.cat([item["choice_counts"] for item in shards]).long(),
        "sample_ids": sum((item["sample_ids"] for item in shards), []),
    }


def git_state() -> dict[str, Any]:
    """Return current commit and dirty state without modifying Git config."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        )
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration is not a mapping: {path}")
    return value


def environment_summary() -> dict[str, Any]:
    import transformers

    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_name,
    }


def fit_masked_linear_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_mask: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    eval_mask: torch.Tensor,
    *,
    l2: float,
    max_iter: int,
    tolerance_grad: float,
) -> dict[str, Any]:
    """Fit a deterministic bias-free 5-way head with full-batch L-BFGS."""
    if train_features.ndim != 2 or train_features.shape[1] <= 0:
        raise ValueError("train_features must have shape [N, D]")
    device = train_features.device
    head = torch.nn.Linear(train_features.shape[1], 5, bias=False, device=device)
    torch.nn.init.zeros_(head.weight)
    head.float()
    x_train = train_features.float()
    x_eval = eval_features.float()
    y_train = train_labels.to(device)
    m_train = train_mask.to(device)
    y_eval = eval_labels.to(device)
    m_eval = eval_mask.to(device)
    optimizer = torch.optim.LBFGS(
        head.parameters(),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=1e-10,
        history_size=min(50, max_iter),
        line_search_fn="strong_wolfe",
    )
    calls = 0

    def closure() -> torch.Tensor:
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        logits = mask_invalid_logits(head(x_train), m_train)
        loss = F.cross_entropy(logits, y_train)
        if l2:
            loss = loss + 0.5 * l2 * head.weight.square().sum()
        loss.backward()
        return loss

    start = time.monotonic()
    optimizer.step(closure)
    elapsed = time.monotonic() - start
    with torch.inference_mode():
        train_logits = head(x_train)
        eval_logits = head(x_eval)
        train_acc = masked_accuracy(train_logits, y_train, m_train)
        eval_acc = masked_accuracy(eval_logits, y_eval, m_eval)
        train_loss = float(
            F.cross_entropy(mask_invalid_logits(train_logits, m_train), y_train).item()
        )
        eval_loss = float(
            F.cross_entropy(mask_invalid_logits(eval_logits, m_eval), y_eval).item()
        )
    return {
        "weight": head.weight.detach().cpu(),
        "train_accuracy": train_acc,
        "eval_accuracy": eval_acc,
        "train_cross_entropy": train_loss,
        "eval_cross_entropy": eval_loss,
        "closure_calls": calls,
        "elapsed_seconds": elapsed,
    }


def stratified_fold_ids(
    labels: Sequence[int], choice_counts: Sequence[int], n_folds: int, seed: int
) -> list[int]:
    """Assign deterministic folds within (choice-count, answer-position) strata."""
    if len(labels) != len(choice_counts):
        raise ValueError("labels and choice_counts must have equal length")
    if n_folds < 2 or n_folds > len(labels):
        raise ValueError("invalid number of folds")
    strata: dict[tuple[int, int], list[int]] = {}
    for index, (label, count) in enumerate(zip(labels, choice_counts)):
        if not 0 <= int(label) < int(count):
            raise ValueError("label is outside its valid choices")
        strata.setdefault((int(count), int(label)), []).append(index)
    rng = random.Random(seed)
    folds = [-1] * len(labels)
    offset = 0
    for key in sorted(strata):
        indices = strata[key]
        rng.shuffle(indices)
        for local_position, index in enumerate(indices):
            folds[index] = (offset + local_position) % n_folds
        offset = (offset + len(indices)) % n_folds
    if any(item < 0 for item in folds):
        raise AssertionError("incomplete fold assignment")
    return folds
