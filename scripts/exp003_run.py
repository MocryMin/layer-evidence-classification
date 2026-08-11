"""EXP-20260810-003: run the validated-probe recoverability verification.

Orchestrates the three probe families (centered_plain, ln_plain, ridge) on the
frozen DeBERTa-v3-base / CLINC150 cache from EXP-001, collects the EXP-001 §4
metrics, computes recoverability / oracle / D_JS, runs the H1/H1'/H2 judgement
with 10-seed bootstrap CI (gradient probes) or point estimates (ridge), and
applies the plan §3.2 cross-family accept protocol.

Usage:
    python -u scripts/exp003_run.py
    python -u scripts/exp003_run.py --smoke   # 1 seed, 3 layers, 50 ep (pipeline check)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.exp003 import (  # noqa: E402
    analyze_gradient_family,
    analyze_ridge_family,
    cross_family_accept,
    load_caches,
    load_config,
    run_gradient_family,
    run_ridge_family,
)
from src.seeding import enable_determinism  # noqa: E402


def get_git_info() -> tuple[str, bool]:
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip() != ""
    except Exception:
        commit, dirty = "unknown", True
    return commit, dirty


def save_gradient_arrays(family_results: dict, out_dir: Path) -> dict:
    """Persist per-seed predictions + logits as .npy; return a JSON-safe copy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    logit_dir = out_dir / "logits"
    pred_dir.mkdir(exist_ok=True)
    logit_dir.mkdir(exist_ok=True)

    json_safe = {"family": family_results["family"], "per_seed": {}}
    for seed, sd in family_results["per_seed"].items():
        np.save(pred_dir / f"seed_{seed}_test.npy", sd["test_pred"])
        np.save(logit_dir / f"seed_{seed}_test.npy", sd["test_logits"])
        json_safe["per_seed"][seed] = {
            "per_layer": sd["per_layer"],
            "recoverability": sd["recoverability"],
        }
    return json_safe


def save_ridge_arrays(family_results: dict, out_dir: Path) -> dict:
    """Persist ridge predictions as .npy; return a JSON-safe copy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "predictions_test.npy", family_results["test_pred"])
    return {
        "family": "ridge",
        "per_layer": family_results["per_layer"],
        "recoverability": family_results["recoverability"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="quick pipeline check: 1 seed, layers {1,6,12}, 50 ep")
    args = ap.parse_args()

    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()

    # smoke override
    if args.smoke:
        cfg["seeds"] = [17]
        cfg["probe_layers"] = [1, 6, 12]
        cfg["training"]["min_epochs"] = 5
        cfg["training"]["max_epochs"] = 50
        cfg["training"]["patience"] = 20
        cfg["artifact_root"] = "artifacts/EXP-20260810-003-smoke"

    artifact_root = PROJECT_ROOT / cfg["artifact_root"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    git_commit, git_dirty = get_git_info()
    seeds = cfg["seeds"]
    layers = cfg["probe_layers"]
    final_layer = cfg["final_layer"]
    n_classes = cfg["n_classes"]

    print(f"[EXP-003] device={device} seeds={seeds} layers={layers}")
    print(f"[EXP-003] artifact_root={artifact_root}")
    print(f"[EXP-003] git={git_commit} dirty={git_dirty}")
    if args.smoke:
        print("[EXP-003] SMOKE MODE (1 seed, 3 layers, 50ep)")

    # MLflow setup
    import mlflow
    from src.mlflow_utils import resolve_tracking_uri
    uri = resolve_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow_experiment_name"])

    # load cache + copy label maps
    cache_dir = PROJECT_ROOT / cfg["cache_dir"]
    label_maps_dir = PROJECT_ROOT / cfg["label_maps_dir"]
    print(f"[EXP-003] loading cache from {cache_dir}")
    caches = load_caches(cache_dir, device)
    for fn in ["label2id.json", "id2label.json", "seeds.json"]:
        src = label_maps_dir / fn
        if src.exists():
            shutil.copy2(src, artifact_root / fn)
    # save resolved run config
    with open(artifact_root / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    all_results: dict = {"config": cfg, "git_commit": git_commit, "git_dirty": git_dirty}
    analysis: dict = {}

    with mlflow.start_run(run_name=cfg["experiment_id"]) as parent:
        mlflow.log_param("experiment_id", cfg["experiment_id"])
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("git_dirty", git_dirty)
        mlflow.log_param("seeds", ",".join(map(str, seeds)))
        mlflow.log_param("layers", ",".join(map(str, layers)))
        mlflow.log_param("smoke", args.smoke)

        # --- centered plain (primary) ---
        print("\n=== centered_plain (primary) ===")
        cp_results = run_gradient_family(
            caches, "centered_plain", cfg["probe_families"]["centered_plain"],
            cfg["training"], seeds, layers, n_classes, device,
        )
        cp_json = save_gradient_arrays(cp_results, artifact_root / "centered_plain")
        all_results["centered_plain"] = cp_json
        if not args.smoke:
            cp_analysis = analyze_gradient_family(
                cp_results, seeds, final_layer,
                cfg["epsilon_1"], cfg["epsilon_2"],
                cfg["bootstrap_resamples"], cfg["bootstrap_ci"],
            )
            analysis["centered_plain"] = cp_analysis
            mlflow.log_metric("centered_candidate_test_acc",
                              cp_results["per_seed"][seeds[0]]["per_layer"][str(cp_analysis["candidate_layer"])]["test_acc"])

        # --- LN plain (robustness control) ---
        print("\n=== ln_plain (robustness control) ===")
        ln_results = run_gradient_family(
            caches, "ln_plain", cfg["probe_families"]["ln_plain"],
            cfg["training"], seeds, layers, n_classes, device,
        )
        ln_json = save_gradient_arrays(ln_results, artifact_root / "ln_plain")
        all_results["ln_plain"] = ln_json
        if not args.smoke:
            ln_analysis = analyze_gradient_family(
                ln_results, seeds, final_layer,
                cfg["epsilon_1"], cfg["epsilon_2"],
                cfg["bootstrap_resamples"], cfg["bootstrap_ci"],
            )
            analysis["ln_plain"] = ln_analysis

        # --- ridge (solver reference) ---
        print("\n=== ridge (solver reference) ===")
        rg_results = run_ridge_family(
            caches, cfg["probe_families"]["ridge"], layers, n_classes,
        )
        rg_json = save_ridge_arrays(rg_results, artifact_root / "ridge")
        all_results["ridge"] = rg_json
        if not args.smoke:
            rg_analysis = analyze_ridge_family(
                rg_results, final_layer, cfg["epsilon_1"], cfg["epsilon_2"],
            )
            analysis["ridge"] = rg_analysis

        # --- cross-family accept protocol ---
        if not args.smoke:
            accept = cross_family_accept(
                analysis["centered_plain"], analysis["ln_plain"], analysis["ridge"],
            )
            all_results["analysis"] = analysis
            all_results["accept_protocol"] = accept
            mlflow.log_dict(accept, "accept_protocol.json")

        # save results.json
        results_path = artifact_root / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=_json_default)
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(artifact_root / "run_config.yaml"))

    print(f"\n[EXP-003] results saved to {results_path}")

    if not args.smoke:
        print("\n=== HYPOTHESIS JUDGEMENT (cross-family accept protocol) ===")
        for h in ["h1", "h1_prime", "h2"]:
            v = all_results["accept_protocol"][h]
            print(f"  {h}: centered={v['primary_centered']} ln={v['ln_control']} "
                  f"ridge={v['ridge_reference']} -> {v['verdict']}")
        print("\n=== convergence summary ===")
        for fam in ["centered_plain", "ln_plain"]:
            c = all_results["analysis"][fam]["convergence"]
            print(f"  {fam}: {c['n_nonconverged']}/{c['n_runs']} non-converged "
                  f"({c['nonconverged_fraction']:.0%})")


def _json_default(obj):
    import math
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"not serialisable: {type(obj)}")


if __name__ == "__main__":
    main()
