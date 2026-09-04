#!/usr/bin/env python3
"""Synchronize the five author logs into the public research-record layer.

The source files remain the writing workspace. Public copies are byte-identical
snapshots with deterministic names and a SHA-256 manifest. The script never
rewrites or deletes source files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "research-logs"

MAPPINGS = (
    (
        "EXP-20260729-001--Verifying Intermediate-LayerRecoverability in "
        "Fine-Grained Intent Classification.md",
        "EXP-001.md",
        "formal_experiment_log",
    ),
    (
        "EXP-20260730-002--Diagnosing Linear-probe mid-layer variance collapse "
        "on DeBERTa-v3-base &CLINC150 (from EXP-001).md",
        "EXP-002.md",
        "formal_experiment_log",
    ),
    (
        "EXP-20260810-003-Restore EXP-001 under a probe protocol that EXP-002 "
        "has shown to be valid.md",
        "EXP-003.md",
        "formal_experiment_log",
    ),
    (
        "EXP-20260821-004--Fixed Readout Head Imposes an Implicit Admissibility "
        "Constraint on Path Searching.md",
        "EXP-004.md",
        "formal_experiment_log",
    ),
    (
        "Post-EXP-003--DirectionResearch-History&Prospects---"
        "EXP-004~006Plan.md",
        "POST-EXP-003-DIRECTION-HISTORY.md",
        "research_direction_history",
    ),
)

CREDENTIAL_PATTERNS = {
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{10,}"),
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{10,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(
        rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the five host-side Markdown logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Snapshot destination (default: repository research-logs/).",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"source directory does not exist: {source_dir}")
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise SystemExit("output directory must not be the source directory")

    records: list[dict[str, object]] = []
    for source_name, destination_name, record_type in MAPPINGS:
        source = source_dir / source_name
        if not source.is_file():
            raise SystemExit(f"required log is missing: {source}")
        data = source.read_bytes()
        if b"\x00" in data:
            raise SystemExit(f"binary/NUL content in Markdown source: {source.name}")
        for label, pattern in CREDENTIAL_PATTERNS.items():
            if pattern.search(data):
                raise SystemExit(
                    f"refusing to publish {source.name}: possible {label}"
                )

        destination = output_dir / destination_name
        atomic_write(destination, data)
        digest = sha256_bytes(data)
        records.append(
            {
                "record_type": record_type,
                "source_basename": source_name,
                "public_path": f"research-logs/{destination_name}",
                "bytes": len(data),
                "sha256": digest,
                "byte_identical_to_source": (
                    destination.read_bytes() == source.read_bytes()
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "policy": (
            "Byte-preserving snapshots; current interpretation is controlled "
            "by docs/EVIDENCE_INDEX.md and corrections by docs/ERRATA.md."
        ),
        "records": records,
    }
    encoded = (
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write(output_dir / "SOURCE_MANIFEST.json", encoded)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "records": len(records),
                "total_bytes": sum(int(r["bytes"]) for r in records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
