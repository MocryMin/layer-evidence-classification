#!/usr/bin/env python3
"""Bounded process-level recovery for the frozen EXP-004 H1 discovery.

CUDA driver resets poison a process context and cannot safely be recovered
inside PyTorch.  This supervisor starts a fresh process only when the discovery
has explicitly committed ``failed_resumable`` state.  It stops after repeated
failures without any newly committed artifact progress, preventing a
deterministic bug from becoming an infinite restart loop.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp004_h1 import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp004_h1_frozen.yaml")
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--retry-seconds", type=int, default=10)
    parser.add_argument("--max-stalled-restarts", type=int, default=3)
    return parser.parse_args()


def resolved(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def append_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def progress_fingerprint(output_root: Path) -> tuple[int, int, int]:
    state_path = output_root / "search_state.json"
    if not state_path.is_file():
        return 0, 0, 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    results = sum(1 for _ in (output_root / "results").glob("*.json"))
    shards = sum(1 for _ in (output_root / "features").glob("**/shard_*.pt"))
    return int(state["completed_search_candidates"]), results, shards


def run() -> int:
    args = parse_args()
    if args.retry_seconds < 1 or args.max_stalled_restarts < 1:
        raise ValueError("retry and restart limits must be positive")
    config_path = resolved(args.config).resolve()
    config = load_yaml(config_path)
    output_root = resolved(config["runtime"]["artifact_root"]).resolve()
    hard_stop = datetime.fromisoformat(args.stop_at)
    if hard_stop.tzinfo is None or hard_stop.utcoffset() is None:
        raise ValueError("--stop-at must include an explicit UTC offset")
    soft_stop = hard_stop - timedelta(minutes=int(config["runtime"]["reserve_minutes"]))
    event_path = output_root / "supervisor_events.jsonl"
    child: subprocess.Popen[Any] | None = None
    signal_received: int | None = None

    def forward_signal(signum: int, _frame: Any) -> None:
        nonlocal signal_received
        signal_received = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)
    stalled_restarts = 0
    attempt = 0
    previous = progress_fingerprint(output_root)
    append_event(event_path, "supervisor_started", progress=previous, hard_stop=hard_stop.isoformat())

    while datetime.now().astimezone() < soft_stop:
        attempt += 1
        command = [
            sys.executable,
            str(ROOT / "scripts" / "exp004_h1_discovery.py"),
            "--config",
            str(config_path),
            "--stop-at",
            args.stop_at,
        ]
        if (output_root / "search_state.json").is_file():
            command.append("--resume")
        append_event(event_path, "child_started", attempt=attempt, progress=previous)
        child = subprocess.Popen(command, cwd=ROOT)
        return_code = child.wait()
        child = None
        current = progress_fingerprint(output_root)
        append_event(
            event_path,
            "child_exited",
            attempt=attempt,
            return_code=return_code,
            progress=current,
        )
        if return_code in (0, 75):
            return return_code
        if signal_received is not None:
            return 128 + signal_received
        manifest_path = output_root / "run_manifest.json"
        if not manifest_path.is_file():
            append_event(event_path, "failed_before_manifest", return_code=return_code)
            return return_code or 1
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "failed_resumable":
            append_event(event_path, "non_resumable_exit", status=manifest.get("status"))
            return return_code or 1
        if current > previous:
            stalled_restarts = 0
        else:
            stalled_restarts += 1
        if stalled_restarts >= args.max_stalled_restarts:
            append_event(
                event_path,
                "stalled_restart_limit_reached",
                stalled_restarts=stalled_restarts,
                progress=current,
            )
            return return_code or 1
        previous = current
        remaining = (soft_stop - datetime.now().astimezone()).total_seconds()
        if remaining <= args.retry_seconds:
            break
        time.sleep(args.retry_seconds)

    append_event(event_path, "supervisor_soft_deadline_reached", progress=previous)
    return 75


if __name__ == "__main__":
    raise SystemExit(run())
