import hashlib
import fnmatch
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "release/public_bundle_spec.json"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def _tracked_paths_at_revision(revision: str) -> frozenset[str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "-z", revision],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return frozenset(
        path.decode("utf-8")
        for path in result.stdout.split(b"\0")
        if path
    )


def _release_source_bytes(source: str, revision: str) -> bytes:
    if source not in _tracked_paths_at_revision(revision):
        return (ROOT / source).read_bytes()
    result = subprocess.run(
        ["git", "show", f"{revision}:{source}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _expanded_release_items(spec):
    items = [dict(entry) for entry in spec["files"]]
    for file_set in spec.get("file_sets", []):
        matches = sorted(ROOT.glob(file_set["source_glob"]))
        assert matches, file_set["source_glob"]
        assert len(matches) <= file_set["max_matches"]
        for path in matches:
            items.append(
                {
                    "source": path.relative_to(ROOT).as_posix(),
                    "category": file_set["category"],
                    "split_sensitivity": file_set["split_sensitivity"],
                    "rationale": file_set["rationale"],
                }
            )
    return items


def test_public_bundle_spec_is_unique_and_resolves():
    spec = _json(SPEC_PATH)
    assert spec["source_revision"] == "exp-001-004-evidence-v1"
    assert "freeze" not in spec["source_revision"].lower()
    items = _expanded_release_items(spec)
    sources = [entry["source"] for entry in items]
    assert len(sources) == len(set(sources))
    release_paths = [entry.get("release_path", entry["source"]) for entry in items]
    assert len(release_paths) == len(set(release_paths))
    assert len(spec["canonical_mlflow_run_ids"]) == len(
        set(spec["canonical_mlflow_run_ids"])
    )
    assert set(spec["canonical_mlflow_run_ids"]).isdisjoint(
        spec["stale_mlflow_run_ids"]
    )
    for entry in items:
        assert entry["category"]
        assert entry["split_sensitivity"]
        assert entry["rationale"]
        path = ROOT / entry["source"]
        assert path.is_file(), entry["source"]
        assert path.stat().st_size <= spec["max_file_bytes"]


def test_author_log_snapshots_match_source_manifest():
    manifest = _json(ROOT / "research-logs/SOURCE_MANIFEST.json")
    assert len(manifest["records"]) == 5
    for record in manifest["records"]:
        path = ROOT / record["public_path"]
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]
        assert record["byte_identical_to_source"] is True


def test_tracked_release_manifest_matches_tagged_sources():
    spec = _json(SPEC_PATH)
    manifest = _json(
        ROOT
        / "release/manifests"
        / f"{spec['bundle_id']}.manifest.json"
    )
    assert manifest["bundle_id"] == spec["bundle_id"]
    assert manifest["source_revision"] == spec["source_revision"]
    by_source = {
        entry["source"]: entry
        for entry in manifest["entries"]
        if not entry["source"].startswith("generated:")
        and entry["source"] != "release/HF_DATASET_CARD.md"
    }
    expected = {
        entry["source"]: entry.get("release_path", entry["source"])
        for entry in _expanded_release_items(spec)
    }
    assert set(by_source) == set(expected)
    for source, entry in by_source.items():
        source_bytes = _release_source_bytes(source, spec["source_revision"])
        assert entry["bytes"] == len(source_bytes)
        assert entry["sha256"] == hashlib.sha256(source_bytes).hexdigest()
        assert entry["release_path"] == expected[source]
        assert entry["hf_uri"].endswith("/" + entry["release_path"])
        assert entry["web_url"].startswith(
            "https://huggingface.co/datasets/"
        )


def test_public_artifact_paths_are_exact_and_log_pointers_resolve():
    spec = _json(SPEC_PATH)
    items = _expanded_release_items(spec)
    release_paths = {
        entry.get("release_path", entry["source"]): entry["source"]
        for entry in items
    }
    for entry in items:
        if entry["source"].startswith("artifacts/"):
            assert entry.get("release_path", entry["source"]) == entry["source"]

    registry = _json(ROOT / "release/ARTIFACT_POINTERS.json")
    assert len({record["id"] for record in registry["records"]}) == len(
        registry["records"]
    )
    for record in registry["records"]:
        log_lines = (ROOT / record["log"]).read_text(encoding="utf-8").splitlines()
        assert record["literal"] in log_lines[record["line"] - 1], record["id"]
        normalized = record.get("normalized_paths") or [record["normalized_path"]]
        for path in normalized:
            if "*" in path:
                assert any(fnmatch.fnmatch(item, path) for item in release_paths), (
                    record["id"],
                    path,
                )
            elif path.endswith("/"):
                assert any(item.startswith(path) for item in release_paths), (
                    record["id"],
                    path,
                )
            else:
                assert path in release_paths, (record["id"], path)


def test_public_directory_names_and_release_revision_are_stable():
    assert not (ROOT / "agent-BuildReports/freeze-20260810").exists()
    assert (
        ROOT
        / "agent-BuildReports/experiments/EXP-001-002-supporting-evidence"
    ).is_dir()
    roots = _json(ROOT / "release/ARTIFACT_ROOTS.json")
    assert roots["release_revision"] == "exp-001-004-evidence-v1"
    assert roots["artifacts_root_uri"].endswith("/artifacts/")
    assert roots["exp_root_uri"] == roots["artifacts_root_uri"]


def test_h1_release_is_labelled_train_discovery_only():
    for relative in (
        "artifacts/EXP-20260827-004-h1-discovery/run_manifest.json",
        "artifacts/EXP-20260828-004-h1-sourcewise-rerun/run_manifest.json",
    ):
        manifest = _json(ROOT / relative)
        assert manifest["official_hypothesis_evidence"] == "train_discovery_only"
        assert manifest["validation_accessed"] is False
        assert manifest["test_accessed"] is False


def test_h2_release_reconstructs_preregistered_pass():
    root = ROOT / "artifacts/EXP-20260831-004-h2-mcts-v2"
    final = _json(root / "final_summary.json")
    gate = _json(root / "test_access_gate.json")
    primary = final["test"]["primary_rank"]
    assert final["acceptance"] == {
        "epsilon_R_short": 0.1,
        "epsilon_R_recov": 0.1,
        "efficiency": True,
        "recoverability": True,
        "H2": True,
    }
    assert primary["n_short"] == 3775
    assert primary["n_pos"] == 3986
    assert primary["n_recov"] == 449
    assert primary["n_neg"] == 514
    assert primary["R_short_wilson_two_sided_95_lower"] > 0.10
    assert primary["R_recov_wilson_two_sided_95_lower"] > 0.10
    assert gate["reason"] == "both primary and binary tuning selections are complete"


def test_new_navigation_markdown_has_no_broken_relative_links():
    files = [
        ROOT / "README.md",
        ROOT / "docs/README.md",
        ROOT / "docs/EVIDENCE_INDEX.md",
        ROOT / "docs/REPRODUCIBILITY.md",
        ROOT / "docs/ARTIFACT_REGISTRY.md",
        ROOT / "docs/PUBLIC_ARTIFACTS.md",
        ROOT / "docs/ERRATA.md",
        ROOT / "docs/MAINTAINER_GUIDE.zh-CN.md",
        ROOT / "proposal/README.md",
        ROOT / "research-logs/README.md",
        ROOT / "release/README.md",
        ROOT / "agent-BuildReports/fragmented-experiments/README.md",
    ]
    markdown_link = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for source in files:
        for raw_target in markdown_link.findall(source.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (source.parent / target).resolve()
            assert resolved.exists(), f"{source.relative_to(ROOT)} -> {target}"
