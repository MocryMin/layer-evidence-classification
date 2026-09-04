#!/usr/bin/env python3
"""Build the allowlisted EXP-001--004 public evidence bundle.

The builder is intentionally conservative: it accepts only files named in the
release specification, rejects unapproved tensor/database/cache formats and
common credential patterns, exports a sanitized view of selected MLflow runs,
and produces deterministic SHA-256 manifests. It never edits source artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "release/public_bundle_spec.json"
DEFAULT_DB = ROOT / "mlruns.db"
DATASET_CARD = ROOT / "release/HF_DATASET_CARD.md"

FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".safetensors",
    ".npy",
    ".npz",
    ".bin",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".parquet",
    ".arrow",
}
FORBIDDEN_COMPONENTS = {
    "cache",
    "features",
    "heads",
    "models",
    "prefix_cache_ssd",
}
CREDENTIAL_PATTERNS = {
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{10,}"),
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{10,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(
        rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def checked_relative_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"source must be repository-relative: {raw}")
    return path


def public_locations(
    dataset_id: str, revision: str, release_path: str
) -> dict[str, str]:
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(release_path, safe="/")
    return {
        "hf_uri": (
            f"hf://datasets/{dataset_id}@{revision}/{release_path}"
        ),
        "web_url": (
            f"https://huggingface.co/datasets/{dataset_id}/blob/"
            f"{encoded_revision}/{encoded_path}"
        ),
    }


def scan_credentials(path: Path) -> None:
    data = path.read_bytes()
    for label, pattern in CREDENTIAL_PATTERNS.items():
        if pattern.search(data):
            raise ValueError(f"possible {label} in release file: {path}")


def sanitize_string(value: str) -> str:
    replacements = {
        str(ROOT): "<REPO_ROOT>",
        str(Path.home()): "<USER_HOME>",
        "/home/mocry": "<USER_HOME>",
    }
    output = value
    for source, replacement in replacements.items():
        output = output.replace(source, replacement)
    output = re.sub(
        r"(?i)[A-Z]:\\[^\r\n\t\"']+",
        "<LOCAL_WINDOWS_PATH>",
        output,
    )
    return output


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_string(value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    return value


def run_metadata(
    connection: sqlite3.Connection, run_id: str, include_details: bool
) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT r.run_uuid, r.experiment_id, e.name AS experiment_name,
               r.status, r.start_time, r.end_time, r.lifecycle_stage
        FROM runs AS r
        JOIN experiments AS e ON e.experiment_id = r.experiment_id
        WHERE r.run_uuid = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"MLflow run missing from local database: {run_id}")
    record = dict(row)
    if include_details:
        record["params"] = {
            item["key"]: sanitize_string(item["value"])
            for item in connection.execute(
                "SELECT key, value FROM params WHERE run_uuid = ? ORDER BY key",
                (run_id,),
            )
        }
        record["latest_metrics"] = {
            item["key"]: {
                "value": item["value"],
                "step": item["step"],
                "timestamp": item["timestamp"],
            }
            for item in connection.execute(
                """
                SELECT key, value, step, timestamp
                FROM latest_metrics
                WHERE run_uuid = ?
                ORDER BY key
                """,
                (run_id,),
            )
        }
        excluded_tags = {"mlflow.user"}
        record["tags"] = {
            item["key"]: sanitize_string(item["value"])
            for item in connection.execute(
                "SELECT key, value FROM tags WHERE run_uuid = ? ORDER BY key",
                (run_id,),
            )
            if item["key"] not in excluded_tags
        }
    return sanitize(record)


def export_mlflow(spec: dict[str, Any], database: Path) -> bytes:
    if not database.is_file():
        raise ValueError(f"MLflow database missing: {database}")
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        canonical = [
            run_metadata(connection, run_id, include_details=True)
            for run_id in spec["canonical_mlflow_run_ids"]
        ]
        for run in canonical:
            if run["status"] != "FINISHED":
                raise ValueError(
                    f"canonical MLflow run is not FINISHED: {run['run_uuid']}"
                )
        stale = [
            {
                **run_metadata(connection, run_id, include_details=False),
                "export_classification": "stale_source_record_excluded",
            }
            for run_id in spec["stale_mlflow_run_ids"]
        ]
    finally:
        connection.close()
    payload = {
        "schema_version": 1,
        "source": "sanitized export from local mlruns.db",
        "policy": (
            "Canonical FINISHED runs include params/latest metrics/tags. "
            "Abandoned stale RUNNING records are identified but not rewritten."
        ),
        "canonical_runs": canonical,
        "stale_excluded_runs": stale,
    }
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def git_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def git_resolve_commit(revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{revision}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--mlflow-db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the default dist/<bundle_id> destination.",
    )
    parser.add_argument(
        "--tracked-manifest",
        type=Path,
        help=(
            "Override the tracked manifest path; defaults to "
            "release/manifests/<bundle_id>.manifest.json."
        ),
    )
    args = parser.parse_args()

    spec_path = args.spec.expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    required_top = {
        "schema_version",
        "bundle_id",
        "source_revision",
        "hf_dataset_id",
        "canonical_mlflow_run_ids",
        "stale_mlflow_run_ids",
        "files",
        "max_file_bytes",
    }
    missing = required_top.difference(spec)
    if missing:
        raise SystemExit(f"release spec missing keys: {sorted(missing)}")
    if spec["schema_version"] != 1:
        raise SystemExit("unsupported release spec schema")

    dirty_before_build = git_is_dirty()
    if dirty_before_build:
        raise SystemExit(
            "release builds require a clean worktree; commit or remove "
            "tracked changes first"
        )
    head_commit = git_resolve_commit("HEAD")
    source_commit = git_resolve_commit(spec["source_revision"])
    if head_commit != source_commit:
        raise SystemExit(
            "HEAD does not match source_revision: "
            f"HEAD={head_commit}, {spec['source_revision']}={source_commit}. "
            "Build from a dedicated checkout of the tagged source revision."
        )

    bundle_id = spec["bundle_id"]
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (ROOT / "dist" / bundle_id).resolve()
    )
    if output_dir.exists():
        raise SystemExit(
            f"output already exists; refusing to mix releases: {output_dir}"
        )
    if output_dir == ROOT or output_dir in ROOT.parents:
        raise SystemExit("output directory may not be or contain the repository")

    tracked_manifest = (
        args.tracked_manifest.expanduser().resolve()
        if args.tracked_manifest
        else ROOT / "release/manifests" / f"{bundle_id}.manifest.json"
    )
    seen_sources: set[str] = set()
    reserved_release_paths = {
        "README.md",
        "MANIFEST.json",
        "MANIFEST.sha256",
        "mlflow/selected_runs.json",
    }
    seen_release_paths = set(reserved_release_paths)
    expanded_items = list(spec["files"])
    for file_set in spec.get("file_sets", []):
        for key in (
            "source_glob",
            "category",
            "split_sensitivity",
            "rationale",
            "max_matches",
        ):
            if not file_set.get(key):
                raise ValueError(f"release file set missing {key}: {file_set}")
        glob_path = checked_relative_path(file_set["source_glob"])
        matches = sorted(
            path for path in ROOT.glob(glob_path.as_posix()) if path.is_file()
        )
        if not matches:
            raise ValueError(
                f"release file set matched no files: {file_set['source_glob']}"
            )
        if len(matches) > int(file_set["max_matches"]):
            raise ValueError(
                "release file set exceeded max_matches "
                f"({len(matches)}): {file_set['source_glob']}"
            )
        for source in matches:
            expanded_items.append(
                {
                    "source": source.relative_to(ROOT).as_posix(),
                    "category": file_set["category"],
                    "split_sensitivity": file_set["split_sensitivity"],
                    "rationale": file_set["rationale"],
                    "allow_forbidden_suffix": file_set.get(
                        "allow_forbidden_suffix", False
                    ),
                }
            )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{bundle_id}.building-", dir=output_dir.parent
    ) as temporary:
        build_root = Path(temporary)
        manifest_entries: list[dict[str, Any]] = []

        card_destination = build_root / "README.md"
        shutil.copyfile(DATASET_CARD, card_destination)
        manifest_entries.append(
            {
                "source": "release/HF_DATASET_CARD.md",
                "release_path": "README.md",
                "category": "dataset_card",
                "split_sensitivity": "none",
                "rationale": "Public bundle description and boundaries.",
                "bytes": card_destination.stat().st_size,
                "sha256": sha256_file(card_destination),
                **public_locations(
                    spec["hf_dataset_id"], spec["source_revision"], "README.md"
                ),
            }
        )

        for item in expanded_items:
            for key in ("source", "category", "split_sensitivity", "rationale"):
                if not item.get(key):
                    raise ValueError(f"release entry missing {key}: {item}")
            relative = checked_relative_path(item["source"])
            source_key = relative.as_posix()
            if source_key in seen_sources:
                raise ValueError(f"duplicate source in release spec: {source_key}")
            seen_sources.add(source_key)
            if (
                relative.suffix.lower() in FORBIDDEN_SUFFIXES
                and not item.get("allow_forbidden_suffix", False)
            ):
                raise ValueError(f"forbidden release suffix: {source_key}")
            if FORBIDDEN_COMPONENTS.intersection(relative.parts):
                raise ValueError(f"forbidden cache/model component: {source_key}")

            source = (ROOT / relative).resolve()
            if ROOT not in source.parents or not source.is_file():
                raise ValueError(f"selected source is missing or unsafe: {source_key}")
            if source.is_symlink():
                raise ValueError(f"release sources may not be symlinks: {source_key}")
            size = source.stat().st_size
            if size > int(spec["max_file_bytes"]):
                raise ValueError(
                    f"selected source exceeds max_file_bytes ({size}): {source_key}"
                )
            scan_credentials(source)

            destination_relative = checked_relative_path(
                item.get("release_path", source_key)
            )
            destination_key = destination_relative.as_posix()
            if destination_key in seen_release_paths:
                raise ValueError(
                    f"duplicate or reserved release path: {destination_key}"
                )
            seen_release_paths.add(destination_key)
            destination = build_root / destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            manifest_entries.append(
                {
                    "source": source_key,
                    "release_path": destination_relative.as_posix(),
                    "category": item["category"],
                    "split_sensitivity": item["split_sensitivity"],
                    "rationale": item["rationale"],
                    "bytes": size,
                    "sha256": sha256_file(destination),
                    **public_locations(
                        spec["hf_dataset_id"],
                        spec["source_revision"],
                        destination_key,
                    ),
                }
            )

        mlflow_destination = build_root / "mlflow/selected_runs.json"
        atomic_write(
            mlflow_destination,
            export_mlflow(spec, args.mlflow_db.expanduser().resolve()),
        )
        scan_credentials(mlflow_destination)
        manifest_entries.append(
            {
                "source": "generated:mlruns.db canonical-run export",
                "release_path": "mlflow/selected_runs.json",
                "category": "mlflow_export",
                "split_sensitivity": "mixed_aggregate_only",
                "rationale": (
                    "Makes selected local MLflow metadata publicly auditable "
                    "without exposing the live database."
                ),
                "bytes": mlflow_destination.stat().st_size,
                "sha256": sha256_file(mlflow_destination),
                **public_locations(
                    spec["hf_dataset_id"],
                    spec["source_revision"],
                    "mlflow/selected_runs.json",
                ),
            }
        )

        manifest_entries.sort(key=lambda item: item["release_path"])
        manifest = {
            "schema_version": 1,
            "bundle_id": bundle_id,
            "source_revision": spec["source_revision"],
            "hf_dataset_id": spec["hf_dataset_id"],
            "evidence_root_uri": (
                f"hf://datasets/{spec['hf_dataset_id']}@"
                f"{spec['source_revision']}/"
            ),
            "artifacts_root_uri": (
                f"hf://datasets/{spec['hf_dataset_id']}@"
                f"{spec['source_revision']}/artifacts/"
            ),
            "exp_root_uri": (
                f"hf://datasets/{spec['hf_dataset_id']}@"
                f"{spec['source_revision']}/artifacts/"
            ),
            "selection_spec": "release/public_bundle_spec.json",
            "entry_count": len(manifest_entries),
            "total_payload_bytes": sum(item["bytes"] for item in manifest_entries),
            "entries": manifest_entries,
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        atomic_write(build_root / "MANIFEST.json", manifest_bytes)

        checksum_paths = [
            path
            for path in build_root.rglob("*")
            if path.is_file() and path.name != "MANIFEST.sha256"
        ]
        checksum_lines = [
            f"{sha256_file(path)}  {path.relative_to(build_root).as_posix()}"
            for path in sorted(checksum_paths)
        ]
        atomic_write(
            build_root / "MANIFEST.sha256",
            ("\n".join(checksum_lines) + "\n").encode("utf-8"),
        )

        for path in build_root.rglob("*"):
            if path.is_file():
                scan_credentials(path)
        os.replace(build_root, output_dir)

    atomic_write(tracked_manifest, manifest_bytes)
    result = {
        "bundle_id": bundle_id,
        "output_dir": str(output_dir),
        "tracked_manifest": str(tracked_manifest),
        "entry_count": manifest["entry_count"],
        "total_payload_bytes": manifest["total_payload_bytes"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "git_worktree_dirty_before_build": dirty_before_build,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
