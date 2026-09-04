#!/usr/bin/env python3
"""Verify that historical EXP-log artifact pointers resolve in the release."""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "release/public_bundle_spec.json"
POINTERS = ROOT / "release/ARTIFACT_POINTERS.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def expanded_release_paths(spec: dict) -> set[str]:
    paths = {
        item.get("release_path", item["source"])
        for item in spec["files"]
    }
    for file_set in spec.get("file_sets", []):
        matches = sorted(ROOT.glob(file_set["source_glob"]))
        if not matches:
            raise ValueError(
                f"file set matched no files: {file_set['source_glob']}"
            )
        if len(matches) > int(file_set["max_matches"]):
            raise ValueError(
                f"file set exceeds max_matches: {file_set['source_glob']}"
            )
        paths.update(path.relative_to(ROOT).as_posix() for path in matches)
    return paths


def main() -> None:
    spec = read_json(SPEC)
    registry = read_json(POINTERS)
    release_paths = expanded_release_paths(spec)
    failures: list[dict[str, object]] = []
    target_count = 0

    for record in registry["records"]:
        log_path = ROOT / record["log"]
        lines = log_path.read_text(encoding="utf-8").splitlines()
        line_number = int(record["line"])
        if line_number < 1 or line_number > len(lines):
            failures.append({"id": record["id"], "error": "line_out_of_range"})
            continue
        if record["literal"] not in lines[line_number - 1]:
            failures.append({"id": record["id"], "error": "literal_mismatch"})
            continue

        normalized = record.get("normalized_paths") or [
            record["normalized_path"]
        ]
        for target in normalized:
            target_count += 1
            if "*" in target:
                resolved = any(
                    fnmatch.fnmatch(path, target) for path in release_paths
                )
            elif target.endswith("/"):
                resolved = any(path.startswith(target) for path in release_paths)
            else:
                resolved = target in release_paths
            if not resolved:
                failures.append(
                    {
                        "id": record["id"],
                        "error": "unresolved_target",
                        "target": target,
                    }
                )

    result = {
        "pointer_records": len(registry["records"]),
        "normalized_targets": target_count,
        "release_paths": len(release_paths),
        "failures": failures,
        "passed": not failures,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
