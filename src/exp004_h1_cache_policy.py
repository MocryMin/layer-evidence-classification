"""Cost-aware GPU/SSD/HDD hierarchy layered over the durable prefix trie."""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from src.exp004_h1_cache import GlobalPrefixCache, prefix_key


COMPLETE_DISK_STATUSES = {"ssd", "hdd"}


@dataclass(frozen=True)
class CacheCostModel:
    """Live-machine latency model; all alternatives are numerically equivalent."""

    common_seconds: float = 1.38
    full_intercept_seconds: float = 0.0430876
    full_layer_seconds: float = 0.970368
    cached_intercept_seconds: float = 0.7618
    cached_layer_seconds: float = 0.937175
    gpu_intercept_seconds: float = 0.15
    ssd_read_seconds: float = 0.9383006
    hdd_read_seconds: float = 20.5440527
    minimum_fractional_saving: float = 0.30

    def full_seconds(self, path_length: int) -> float:
        return self.common_seconds + self.full_intercept_seconds + self.full_layer_seconds * path_length

    def cached_seconds(self, path_length: int, prefix_depth: int, tier: str) -> float:
        if not 0 < prefix_depth <= path_length:
            raise ValueError("prefix depth must be within the path")
        residual = path_length - prefix_depth
        if tier == "gpu":
            return self.common_seconds + self.gpu_intercept_seconds + self.cached_layer_seconds * residual
        read_seconds = {"ssd": self.ssd_read_seconds, "hdd": self.hdd_read_seconds}[tier]
        return self.common_seconds + read_seconds + self.cached_intercept_seconds + self.cached_layer_seconds * residual

    def qualifies(self, path_length: int, prefix_depth: int, tier: str) -> bool:
        threshold = (1.0 - self.minimum_fractional_saving) * self.full_seconds(path_length)
        return self.cached_seconds(path_length, prefix_depth, tier) <= threshold

    def minimum_depth(self, path_length: int, tier: str) -> int | None:
        for depth in range(1, path_length + 1):
            if self.qualifies(path_length, depth, tier):
                return depth
        return None


class GpuPrefixCache:
    """Clean L1 cache. SSD/HDD always retain the durable backing page."""

    def __init__(self, *, cap_bytes: int, device: torch.device, config_hash: str) -> None:
        self.cap_bytes = int(cap_bytes)
        self.device = device
        self.config_hash = str(config_hash)
        self.entries: dict[str, dict[str, Any]] = {}
        self.clock = 0
        self.hits = 0
        self.loads = 0
        self.evictions = 0

    def contains(self, path: Sequence[int]) -> bool:
        return prefix_key(path) in self.entries

    def touch(self, path: Sequence[int]) -> dict[str, Any]:
        key = prefix_key(path)
        self.clock += 1
        self.entries[key]["last_access"] = self.clock
        self.hits += 1
        return {"key": key, "path": list(path), "depth": len(path), "cache_status": "gpu"}

    def bytes_used(self) -> int:
        return sum(int(entry["bytes"]) for entry in self.entries.values())

    def _evict_until(self, required_bytes: int) -> None:
        if required_bytes > self.cap_bytes:
            raise RuntimeError("one prefix node exceeds the GPU L1 capacity")
        while self.bytes_used() + required_bytes > self.cap_bytes and self.entries:
            victim = min(self.entries, key=lambda key: (self.entries[key]["last_access"], key))
            del self.entries[victim]
            self.evictions += 1

    def load_from_disk(self, path: Sequence[int], disk_cache: GlobalPrefixCache) -> dict[str, Any]:
        key = prefix_key(path)
        if key in self.entries:
            return self.touch(path)
        item = disk_cache.node(path)
        if item is None or item["cache_status"] not in COMPLETE_DISK_STATUSES:
            raise RuntimeError("GPU admission requires a complete durable cache page")
        self._evict_until(int(item["cache_bytes"]))
        shards: dict[tuple[str, int, int], list[torch.Tensor]] = {}
        for shard_path in sorted(disk_cache.payload_root(path).glob("*/*.pt")):
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            if payload["config_hash"] != self.config_hash or payload["prefix"] != list(path):
                raise RuntimeError(f"GPU cache validation failed for {shard_path}")
            shard_id = (str(payload["split"]), int(payload["start"]), int(payload["end"]))
            shards[shard_id] = [
                tensor.to(device=self.device, dtype=torch.bfloat16)
                for tensor in payload["hidden_batches"]
            ]
        if not shards:
            raise RuntimeError("refusing to admit an empty GPU cache page")
        actual_bytes = sum(
            tensor.numel() * tensor.element_size()
            for batches in shards.values()
            for tensor in batches
        )
        self._evict_until(actual_bytes)
        self.clock += 1
        self.entries[key] = {
            "path": list(path),
            "shards": shards,
            "bytes": actual_bytes,
            "last_access": self.clock,
        }
        self.loads += 1
        return {"key": key, "path": list(path), "depth": len(path), "cache_status": "gpu"}

    def shard_batches(self, path: Sequence[int], split: str, start: int, end: int) -> list[torch.Tensor]:
        return self.entries[prefix_key(path)]["shards"][(str(split), int(start), int(end))]

    def stats(self) -> dict[str, Any]:
        return {
            "nodes": len(self.entries),
            "bytes": self.bytes_used(),
            "cap_bytes": self.cap_bytes,
            "hits": self.hits,
            "loads": self.loads,
            "evictions": self.evictions,
        }


def complete_disk_prefixes(cache: GlobalPrefixCache, path: Sequence[int]) -> list[dict[str, Any]]:
    matches = []
    for depth in range(1, len(path) + 1):
        item = cache.node(path[:depth])
        if item is not None and item["cache_status"] in COMPLETE_DISK_STATUSES:
            matches.append(item)
    return matches


def select_cache_plan(
    path: Sequence[int],
    *,
    disk_cache: GlobalPrefixCache,
    gpu_cache: GpuPrefixCache,
    cost_model: CacheCostModel,
) -> dict[str, Any]:
    """Select one tier only when it predicts at least 30% end-to-end saving."""
    length = len(path)
    full = cost_model.full_seconds(length)
    options: list[dict[str, Any]] = []
    for depth in range(1, length + 1):
        prefix = list(path[:depth])
        if gpu_cache.contains(prefix):
            options.append({
                "tier": "gpu", "path": prefix, "depth": depth,
                "seconds": cost_model.cached_seconds(length, depth, "gpu"),
            })
    for item in complete_disk_prefixes(disk_cache, path):
        tier = str(item["cache_status"])
        options.append({
            "tier": tier, "path": item["path"], "depth": int(item["depth"]),
            "seconds": cost_model.cached_seconds(length, int(item["depth"]), tier),
        })
    threshold = (1.0 - cost_model.minimum_fractional_saving) * full
    qualified = [item for item in options if float(item["seconds"]) <= threshold]
    if not qualified:
        return {
            "action": "recompute", "tier": None, "path": [], "depth": 0,
            "predicted_seconds": full, "full_seconds": full,
            "predicted_fractional_saving": 0.0,
        }
    selected = min(qualified, key=lambda item: (float(item["seconds"]), -int(item["depth"])))
    predicted = float(selected["seconds"])
    return {
        "action": "cache", "tier": selected["tier"], "path": selected["path"],
        "depth": selected["depth"], "predicted_seconds": predicted,
        "full_seconds": full, "predicted_fractional_saving": 1.0 - predicted / full,
    }


class HddPromotionTracker:
    """Promote only after a page is reloaded from HDD twice in one session."""

    def __init__(self, *, threshold: int = 2) -> None:
        self.threshold = int(threshold)
        self.loads: dict[str, int] = {}

    def observe_and_maybe_promote(
        self, path: Sequence[int], cache: GlobalPrefixCache
    ) -> dict[str, Any]:
        key = prefix_key(path)
        self.loads[key] = self.loads.get(key, 0) + 1
        if self.loads[key] < self.threshold:
            return {"promoted": False, "qualified_loads": self.loads[key]}
        item = cache.node(path)
        if item is None or item["cache_status"] != "hdd":
            return {"promoted": False, "qualified_loads": self.loads[key], "reason": "not_hdd"}
        source = cache.hdd_root / key
        destination = cache.ssd_root / key
        temporary = cache.ssd_root / f".{key}.promoting-{os.getpid()}"
        t0 = time.monotonic()
        if temporary.exists():
            cache._safe_remove_tree(temporary, cache.ssd_root)
        shutil.copytree(source, temporary)
        if sum(p.stat().st_size for p in temporary.rglob("*") if p.is_file()) != item["cache_bytes"]:
            cache._safe_remove_tree(temporary, cache.ssd_root)
            raise RuntimeError("HDD-to-SSD promotion size mismatch")
        os.replace(temporary, destination)
        cache._safe_remove_tree(source, cache.hdd_root)
        cache.connection.execute(
            "UPDATE nodes SET cache_status='ssd', last_access_ns=? WHERE key=?",
            (time.time_ns(), key),
        )
        cache.connection.commit()
        cache.enforce_caps()
        return {
            "promoted": True,
            "qualified_loads": self.loads[key],
            "elapsed_seconds": time.monotonic() - t0,
        }
