"""Crash-aware global prefix cache for EXP-004 H1.

The trie registry and the hidden-state payloads are deliberately separate.  A
registered node means that a layer prefix has been traversed; it does not imply
that a reusable payload is resident.  Payload identity is global across search
sources because, under the frozen model/input/batching configuration, the
hidden state is a pure function of the layer-prefix tuple.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


def prefix_key(path: Iterable[int]) -> str:
    values = tuple(int(layer) for layer in path)
    if not values:
        return "root"
    encoded = ",".join(str(layer) for layer in values).encode("ascii")
    return f"px_{hashlib.sha256(encoded).hexdigest()[:20]}"


def _directory_bytes(root: Path) -> int:
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


class GlobalPrefixCache:
    """One global prefix trie with SSD/HDD payload tiers and leaf-LRU eviction."""

    def __init__(
        self,
        *,
        index_path: Path,
        ssd_root: Path,
        hdd_root: Path,
        ssd_cap_bytes: int,
        hdd_cap_bytes: int,
        config_hash: str,
    ) -> None:
        self.index_path = index_path.resolve()
        self.ssd_root = ssd_root.resolve()
        self.hdd_root = hdd_root.resolve()
        self.ssd_cap_bytes = int(ssd_cap_bytes)
        self.hdd_cap_bytes = int(hdd_cap_bytes)
        self.config_hash = str(config_hash)
        if self.ssd_cap_bytes <= 0 or self.hdd_cap_bytes <= 0:
            raise ValueError("cache capacities must be positive")
        if self.ssd_root == self.hdd_root:
            raise ValueError("SSD and HDD cache roots must differ")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.ssd_root.mkdir(parents=True, exist_ok=True)
        self.hdd_root.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.index_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()
        self._validate_identity()
        self.register_path([])

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS nodes (
                key TEXT PRIMARY KEY,
                parent_key TEXT,
                path_json TEXT NOT NULL,
                depth INTEGER NOT NULL,
                traversals INTEGER NOT NULL DEFAULT 0,
                cache_status TEXT NOT NULL DEFAULT 'none',
                cache_bytes INTEGER NOT NULL DEFAULT 0,
                last_access_ns INTEGER NOT NULL DEFAULT 0,
                access_count INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self.connection.commit()

    def _validate_identity(self) -> None:
        expected = {
            "config_hash": self.config_hash,
            "ssd_root": str(self.ssd_root),
            "hdd_root": str(self.hdd_root),
            "eviction": "resident-leaf-lru",
        }
        observed = dict(self.connection.execute("SELECT name, value FROM metadata"))
        if observed:
            for name, value in expected.items():
                if observed.get(name) != value:
                    raise RuntimeError(f"prefix-cache identity changed for {name}")
        else:
            self.connection.executemany(
                "INSERT INTO metadata(name, value) VALUES(?, ?)", expected.items()
            )
            self.connection.commit()

    def register_path(self, path: Sequence[int]) -> None:
        values: list[int] = []
        parent: str | None = None
        for layer in [None, *[int(item) for item in path]]:
            if layer is not None:
                values.append(layer)
            key = prefix_key(values)
            self.connection.execute(
                """
                INSERT INTO nodes(key, parent_key, path_json, depth, traversals)
                VALUES(?, ?, ?, ?, 1)
                ON CONFLICT(key) DO UPDATE SET traversals=traversals+1
                """,
                (key, parent, json.dumps(values), len(values)),
            )
            parent = key
        self.connection.commit()

    def node(self, path: Sequence[int]) -> dict[str, Any] | None:
        row = self.connection.execute(
            """SELECT key, parent_key, path_json, depth, traversals,
                      cache_status, cache_bytes, last_access_ns, access_count
               FROM nodes WHERE key=?""",
            (prefix_key(path),),
        ).fetchone()
        if row is None:
            return None
        return {
            "key": row[0],
            "parent_key": row[1],
            "path": json.loads(row[2]),
            "depth": row[3],
            "traversals": row[4],
            "cache_status": row[5],
            "cache_bytes": row[6],
            "last_access_ns": row[7],
            "access_count": row[8],
        }

    def _tier_root(self, status: str) -> Path:
        if status in {"ssd", "partial_ssd"}:
            return self.ssd_root
        if status == "hdd":
            return self.hdd_root
        raise ValueError(f"status has no payload tier: {status}")

    def payload_root(self, path: Sequence[int], status: str | None = None) -> Path:
        if status is None:
            item = self.node(path)
            if item is None:
                raise KeyError(f"unregistered prefix: {list(path)}")
            status = item["cache_status"]
        return self._tier_root(status) / prefix_key(path)

    def shard_path(
        self, path: Sequence[int], split: str, start: int, end: int, *, writing: bool = False
    ) -> Path:
        item = self.node(path)
        if item is None:
            raise KeyError(f"unregistered prefix: {list(path)}")
        status = item["cache_status"]
        if writing:
            status = "partial_ssd"
        return self.payload_root(path, status) / split / f"shard_{start:05d}_{end:05d}.pt"

    def prepare_write(self, path: Sequence[int]) -> bool:
        """Prepare an SSD target. Return false when a complete payload exists."""
        self.register_path(path)
        item = self.node(path)
        assert item is not None
        if item["cache_status"] in {"ssd", "hdd"}:
            return False
        target = self.ssd_root / item["key"]
        target.mkdir(parents=True, exist_ok=True)
        self.connection.execute(
            "UPDATE nodes SET cache_status='partial_ssd' WHERE key=?", (item["key"],)
        )
        self.connection.commit()
        return True

    def finalize_write(self, path: Sequence[int]) -> None:
        item = self.node(path)
        if item is None or item["cache_status"] != "partial_ssd":
            raise RuntimeError("cannot finalize a cache node that is not partial_ssd")
        root = self.ssd_root / item["key"]
        size = _directory_bytes(root)
        if size <= 0:
            raise RuntimeError("refusing to finalize an empty prefix cache")
        now = time.time_ns()
        self.connection.execute(
            """UPDATE nodes SET cache_status='ssd', cache_bytes=?,
                     last_access_ns=?, access_count=access_count+1 WHERE key=?""",
            (size, now, item["key"]),
        )
        self.connection.commit()
        self.enforce_caps()

    def deepest_complete_prefix(self, path: Sequence[int]) -> dict[str, Any] | None:
        for depth in range(len(path), 0, -1):
            item = self.node(path[:depth])
            if item is not None and item["cache_status"] in {"ssd", "hdd"}:
                now = time.time_ns()
                self.connection.execute(
                    """UPDATE nodes SET last_access_ns=?, access_count=access_count+1
                       WHERE key=?""",
                    (now, item["key"]),
                )
                self.connection.commit()
                item["last_access_ns"] = now
                item["access_count"] += 1
                return item
        return None

    def _resident(self, statuses: tuple[str, ...]) -> list[dict[str, Any]]:
        marks = ",".join("?" for _ in statuses)
        rows = self.connection.execute(
            f"""SELECT key, path_json, cache_status, cache_bytes,
                       last_access_ns, access_count
                FROM nodes WHERE cache_status IN ({marks})""",
            statuses,
        ).fetchall()
        return [
            {
                "key": row[0],
                "path": json.loads(row[1]),
                "cache_status": row[2],
                "cache_bytes": row[3],
                "last_access_ns": row[4],
                "access_count": row[5],
            }
            for row in rows
        ]

    @staticmethod
    def _leaf_candidates(
        candidates: list[dict[str, Any]], all_resident: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        all_keys = {item["key"] for item in all_resident}
        nonleaves: set[str] = set()
        for item in all_resident:
            path = item["path"]
            for depth in range(1, len(path)):
                ancestor = prefix_key(path[:depth])
                if ancestor in all_keys:
                    nonleaves.add(ancestor)
        leaves = [item for item in candidates if item["key"] not in nonleaves]
        return leaves or candidates

    def _safe_remove_tree(self, path: Path, root: Path) -> None:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
        valid_name = resolved_path.name.startswith("px_") or resolved_path.name.startswith(".px_")
        if resolved_path.parent != resolved_root or not valid_name:
            raise RuntimeError(f"unsafe cache deletion target: {resolved_path}")
        if resolved_path.exists():
            shutil.rmtree(resolved_path)

    def _move_ssd_to_hdd(self, item: dict[str, Any]) -> None:
        source = self.ssd_root / item["key"]
        destination = self.hdd_root / item["key"]
        temporary = self.hdd_root / f".{item['key']}.moving-{os.getpid()}"
        if destination.exists():
            self._safe_remove_tree(source, self.ssd_root)
        else:
            if temporary.exists():
                self._safe_remove_tree(temporary, self.hdd_root)
            shutil.copytree(source, temporary)
            if _directory_bytes(temporary) != item["cache_bytes"]:
                raise RuntimeError("cache tier copy size mismatch")
            os.replace(temporary, destination)
            self._safe_remove_tree(source, self.ssd_root)
        self.connection.execute(
            "UPDATE nodes SET cache_status='hdd' WHERE key=?", (item["key"],)
        )
        self.connection.commit()

    def _drop_hdd(self, item: dict[str, Any]) -> None:
        self._safe_remove_tree(self.hdd_root / item["key"], self.hdd_root)
        self.connection.execute(
            """UPDATE nodes SET cache_status='none', cache_bytes=0
               WHERE key=?""",
            (item["key"],),
        )
        self.connection.commit()

    def enforce_caps(self) -> None:
        while True:
            ssd = self._resident(("ssd",))
            if sum(item["cache_bytes"] for item in ssd) <= self.ssd_cap_bytes:
                break
            all_resident = self._resident(("ssd", "hdd"))
            leaves = self._leaf_candidates(ssd, all_resident)
            victim = min(leaves, key=lambda item: (item["last_access_ns"], item["key"]))
            self._move_ssd_to_hdd(victim)
        while True:
            hdd = self._resident(("hdd",))
            if sum(item["cache_bytes"] for item in hdd) <= self.hdd_cap_bytes:
                break
            all_resident = self._resident(("ssd", "hdd"))
            leaves = self._leaf_candidates(hdd, all_resident)
            victim = min(leaves, key=lambda item: (item["last_access_ns"], item["key"]))
            self._drop_hdd(victim)

    def stats(self) -> dict[str, Any]:
        values = {}
        for status in ("none", "partial_ssd", "ssd", "hdd"):
            count, size = self.connection.execute(
                """SELECT COUNT(*), COALESCE(SUM(cache_bytes), 0)
                   FROM nodes WHERE cache_status=?""",
                (status,),
            ).fetchone()
            values[status] = {"nodes": int(count), "bytes": int(size)}
        values["registered_nodes"] = int(
            self.connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        )
        return values
