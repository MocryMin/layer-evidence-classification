"""Numerically compatible DeBERTa path forward for EXP-004 H2.

The canonical head artifact was fitted on the fragmented experiment's FP16
branch stack: every transformer block computes in FP32, then its entire hidden
state is rounded to FP16 before the next block.  This boundary rounding is part
of the operational model and must also be used by H2 search and prefix caches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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
