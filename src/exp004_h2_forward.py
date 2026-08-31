"""Numerically compatible DeBERTa path forward for EXP-004 H2.

The canonical head artifact was fitted on the fragmented experiment's FP16
branch stack: every transformer block computes in FP32, then its entire hidden
state is rounded to FP16 before the next block.  This boundary rounding is part
of the operational model and must also be used by H2 search and prefix caches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class PreparedDebertaInput:
    x0: torch.Tensor
    attention_mask: torch.Tensor
    relative_pos: torch.Tensor | None


class H2ModularDeberta:
    """Arbitrary one-based layer paths with per-block FP16 state boundaries."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.encoder = model.encoder
        self.layers = self.encoder.layer
        self.relative_embeddings = self.encoder.get_rel_embedding()

    @torch.inference_mode()
    def prepare(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> PreparedDebertaInput:
        x0 = self.model.embeddings(input_ids, mask=attention_mask).float()
        expanded_mask = self.encoder.get_attention_mask(attention_mask)
        relative_pos = self.encoder.get_rel_pos(x0)
        return PreparedDebertaInput(x0, expanded_mask, relative_pos)

    @torch.inference_mode()
    def apply_layer(
        self,
        hidden: torch.Tensor,
        prepared: PreparedDebertaInput,
        layer_id: int,
    ) -> torch.Tensor:
        if not 1 <= int(layer_id) <= len(self.layers):
            raise ValueError(f"layer id {layer_id} outside 1..{len(self.layers)}")
        output, _ = self.layers[int(layer_id) - 1](
            hidden.float(),
            prepared.attention_mask,
            relative_pos=prepared.relative_pos,
            rel_embeddings=self.relative_embeddings,
        )
        return output.to(torch.float16)

    @torch.inference_mode()
    def forward_from_prefix(
        self,
        prepared: PreparedDebertaInput,
        path: Sequence[int],
        *,
        cached_prefix_length: int = 0,
        cached_hidden: torch.Tensor | None = None,
        capture_prefixes: bool = False,
    ) -> tuple[torch.Tensor, dict[tuple[int, ...], torch.Tensor]]:
        """Evaluate a suffix and optionally retain each exact FP16 prefix state."""
        route = tuple(int(layer) for layer in path)
        if not route:
            raise ValueError("H2 paths cannot be empty")
        if not 0 <= cached_prefix_length <= len(route):
            raise ValueError("cached prefix length is outside the path")
        if cached_prefix_length:
            if cached_hidden is None:
                raise ValueError("non-empty cached prefix requires a hidden state")
            if tuple(cached_hidden.shape) != tuple(prepared.x0.shape):
                raise ValueError("cached hidden shape does not match prepared input")
            hidden = cached_hidden
        else:
            if cached_hidden is not None:
                raise ValueError("cached hidden supplied for an empty prefix")
            hidden = prepared.x0
        captured: dict[tuple[int, ...], torch.Tensor] = {}
        for offset, layer_id in enumerate(route[cached_prefix_length:], start=cached_prefix_length):
            hidden = self.apply_layer(hidden, prepared, layer_id)
            if capture_prefixes:
                captured[route[: offset + 1]] = hidden
        return hidden, captured


@torch.inference_mode()
def fixed_head_logits(
    terminal_hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Score the branch-stack CLS using the artifact's FP64 ridge head."""
    if terminal_hidden.ndim != 3:
        raise ValueError("terminal hidden must have shape [batch, tokens, hidden]")
    if weight.ndim != 2 or bias.ndim != 1 or weight.shape[1] != bias.shape[0]:
        raise ValueError("invalid fixed-head shapes")
    return terminal_hidden[:, 0].to(weight.dtype) @ weight + bias


def gold_rank_from_logits(logits: torch.Tensor, gold_class: int) -> int:
    """One-based rank with deterministic lower-class-ID tie-breaking.

    This convention guarantees ``rank == 1`` exactly when PyTorch ``argmax``
    predicts the gold class, including the otherwise unlikely exact-tie case.
    """
    if logits.ndim != 1 or not 0 <= int(gold_class) < logits.numel():
        raise ValueError("invalid logits or gold class")
    gold_class = int(gold_class)
    gold_value = logits[gold_class]
    greater = (logits > gold_value).sum()
    tied_lower_id = (logits[:gold_class] == gold_value).sum()
    return int((greater + tied_lower_id + 1).item())


class SamplePathEvaluator:
    """Exact sample-local path-result and FP16 hidden-prefix cache."""

    def __init__(
        self,
        executor: H2ModularDeberta,
        prepared: PreparedDebertaInput,
        weight: torch.Tensor,
        bias: torch.Tensor,
        gold_class: int,
    ):
        self.executor = executor
        self.prepared = prepared
        self.weight = weight
        self.bias = bias
        self.gold_class = int(gold_class)
        self.prefix_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self.result_cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self.calls = 0
        self.exact_result_cache_hits = 0
        self.transformer_blocks_executed = 0

    @torch.inference_mode()
    def evaluate(self, path: Sequence[int]) -> dict[str, Any]:
        route = tuple(int(layer) for layer in path)
        self.calls += 1
        cached_result = self.result_cache.get(route)
        if cached_result is not None:
            self.exact_result_cache_hits += 1
            return {
                **cached_result,
                "exact_result_cache_hit": True,
                "cached_prefix_length": len(route),
                "executed_transformer_blocks": 0,
                "new_prefixes": 0,
            }

        prefix_length, cached_hidden = longest_cached_prefix(route, self.prefix_cache)
        hidden, captured = self.executor.forward_from_prefix(
            self.prepared,
            route,
            cached_prefix_length=prefix_length,
            cached_hidden=cached_hidden,
            capture_prefixes=True,
        )
        new_prefixes = 0
        for prefix, state in captured.items():
            if prefix not in self.prefix_cache:
                self.prefix_cache[prefix] = state
                new_prefixes += 1
        executed = len(route) - prefix_length
        self.transformer_blocks_executed += executed
        logits = fixed_head_logits(hidden, self.weight, self.bias)[0]
        predicted = int(logits.argmax().item())
        rank = gold_rank_from_logits(logits, self.gold_class)
        base = {
            "path": list(route),
            "predicted_class": predicted,
            "gold_class": self.gold_class,
            "correct": predicted == self.gold_class,
            "gold_rank": rank,
        }
        self.result_cache[route] = base
        return {
            **base,
            "exact_result_cache_hit": False,
            "cached_prefix_length": prefix_length,
            "executed_transformer_blocks": executed,
            "new_prefixes": new_prefixes,
        }

    def summary(self) -> dict[str, Any]:
        prefix_bytes = sum(
            state.numel() * state.element_size() for state in self.prefix_cache.values()
        )
        return {
            "evaluation_calls": self.calls,
            "unique_paths": len(self.result_cache),
            "exact_result_cache_hits": self.exact_result_cache_hits,
            "transformer_blocks_executed": self.transformer_blocks_executed,
            "prefix_cache_entries": len(self.prefix_cache),
            "prefix_cache_bytes": prefix_bytes,
        }


def longest_cached_prefix(
    path: Sequence[int], cache: Mapping[tuple[int, ...], torch.Tensor]
) -> tuple[int, torch.Tensor | None]:
    route = tuple(int(layer) for layer in path)
    for depth in range(len(route), 0, -1):
        state = cache.get(route[:depth])
        if state is not None:
            return depth, state
    return 0, None


def canonical_cycle_path(length: int, n_layers: int = 12) -> tuple[int, ...]:
    if length < 1 or n_layers < 1:
        raise ValueError("path length and layer count must be positive")
    return tuple(index % n_layers + 1 for index in range(length))


def h2_simulation_counts(
    *,
    grid_size: int = 49,
    tuning_samples: int = 60,
    test_samples: int = 4500,
    simulations_per_search: int = 200,
) -> dict[str, int]:
    tuning_one = grid_size * tuning_samples * simulations_per_search
    test_one = test_samples * simulations_per_search
    return {
        "primary_tuning": tuning_one,
        "binary_tuning": tuning_one,
        "primary_test": test_one,
        "binary_test": test_one,
        "random_test": test_one,
        "total": 2 * tuning_one + 3 * test_one,
    }


def runtime_hours(simulations: int, milliseconds_per_simulation: float) -> float:
    if simulations < 0 or milliseconds_per_simulation < 0:
        raise ValueError("runtime inputs cannot be negative")
    return simulations * milliseconds_per_simulation / 3_600_000.0
