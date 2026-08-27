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
