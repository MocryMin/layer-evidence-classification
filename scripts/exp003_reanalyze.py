"""EXP-20260810-003: reanalysis pass - recompute class-wise metrics from saved predictions.

The training run saved per-seed test predictions/logits (.npy) and scalar
metrics (results.json), but the class-wise recoverability R_{l,c} / H_{l,c}
were dropped during JSON serialisation. This script recomputes the full
recoverability block (including R_{l,c}, H_{l,c}, n_err_c, n_rec_c, D_JS) from
the saved predictions + test labels, without retraining, and writes
``classwise_summary.json`` per probe family.

Usage:
    python -u scripts/exp003_reanalyze.py
    python -u scripts/exp003_reanalyze.py --artifact-root artifacts/EXP-20260810-003-smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.exp003 import load_caches  # noqa: E402
from src.metrics import recoverability  # noqa: E402

FAMILIES = ["centered_plain", "ln_plain", "ridge"]
SEEDS = [17, 29, 43, 59, 71, 89, 101, 127, 149, 173]
N_CLASSES = 150
N_LAYERS = 12


def _summarize(recs: list[dict]) -> dict:
    """Aggregate a list of per-seed recoverability dicts -> mean/std summary."""
    def agg(field, k=1):
        vals = [r[field] for r in recs]
        if field in ("R_lc", "H_lc"):
            # per (layer, class) mean/std of ratios
            keys = set()
            for r in recs:
                keys |= set(r[field].keys())
            out = {}
            for kk in sorted(keys):
                ratios = []
                for r in recs:
                    if kk in r[field]:
                        num, den, ratio = r[field][kk]
                        if ratio is not None:
                            ratios.append(ratio)
                    else:
                        ratios.append(float("nan"))
                arr = np.array([x for x in ratios if not np.isnan(x)], dtype=float)
                out[str(kk)] = {
                    "mean": float(arr.mean()) if len(arr) else None,
                    "std": float(arr.std()) if len(arr) else None,
                    "n_seeds": int(len(arr)),
                }
            return out
        vals = [r[field] for r in recs]
        arr = np.array(vals, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    # ratio dicts: {layer: (num, den, ratio)}
    def agg_ratio_dict(field):
        keys = set()
        for r in recs:
            keys |= set(r[field].keys())
        out = {}
        for kk in sorted(keys):
            nums, dens, ratios = [], [], []
            for r in recs:
                if kk in r[field]:
                    num, den, ratio = r[field][kk]
                    nums.append(num)
                    dens.append(den)
                    if ratio is not None:
                        ratios.append(ratio)
            arr = np.array(ratios, dtype=float)
            out[kk] = {
                "num_mean": float(np.mean(nums)),
                "den_mean": float(np.mean(dens)),
                "ratio_mean": float(arr.mean()) if len(arr) else None,
                "ratio_std": float(arr.std()) if len(arr) else None,
                "n_seeds": int(len(arr)),
            }
        return out

    return {
        "acc_L": agg("acc_L"),
        "acc_oracle": agg("acc_oracle"),
        "R_oracle": agg("R_oracle"),
        "oracle_gain": agg("oracle_gain"),
        "d_js_class": agg("d_js_class"),
        "n_err_c": agg("n_err_c"),
        "n_rec_c": agg("n_rec_c"),
        "R_l": agg_ratio_dict("R_l"),
        "H_l": agg_ratio_dict("H_l"),
        "R_lc": agg("R_lc"),
        "H_lc": agg("H_lc"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", default="artifacts/EXP-20260810-003")
    args = ap.parse_args()

    root = PROJECT_ROOT / args.artifact_root
    device = torch.device("cpu")
    cache_dir = PROJECT_ROOT / "artifacts/EXP-20260729-001/cache"
    caches = load_caches(cache_dir, device)
    te_y = caches["test"]["labels"].cpu().numpy()
    print(f"[reanalyze] root={root} n_test={len(te_y)}")

    for fam in FAMILIES:
        fam_dir = root / fam
        if not fam_dir.exists():
            print(f"  skip {fam}: {fam_dir} not found")
            continue

        if fam == "ridge":
            pred = np.load(fam_dir / "predictions_test.npy")  # (12, N)
            layer_correct = (pred == te_y)
            rec = recoverability(layer_correct, pred, te_y, N_LAYERS - 1, N_CLASSES)
            out = {
                "family": "ridge",
                "n_seeds": 1,
                "seeds": [],
                "per_seed": {"1": {"recoverability": _jsonify(rec)}},
                "summary": _summarize([rec]),
            }
        else:
            per_seed = {}
            recs = []
            for s in SEEDS:
                pred = np.load(fam_dir / "predictions" / f"seed_{s}_test.npy")
                layer_correct = (pred == te_y)
                rec = recoverability(layer_correct, pred, te_y, N_LAYERS - 1, N_CLASSES)
                per_seed[str(s)] = {"recoverability": _jsonify(rec)}
                recs.append(rec)
            out = {
                "family": fam,
                "n_seeds": len(SEEDS),
                "seeds": SEEDS,
                "per_seed": per_seed,
                "summary": _summarize(recs),
            }

        out_path = fam_dir / "classwise_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=_json_default)
        print(f"  {fam}: saved {out_path}")

    # also regenerate results.json with R_lc/H_lc? No - keep results.json as-is;
    # classwise_summary.json is the authoritative class-wise source.


def _jsonify(rec: dict) -> dict:
    """Convert a recoverability dict to JSON-safe form (fractions preserved)."""
    import math

    def cr(r):
        num, den, ratio = r
        return {"num": int(num), "den": int(den), "ratio": None if math.isnan(ratio) else float(ratio)}

    return {
        "acc_L": rec["acc_L"],
        "acc_oracle": rec["acc_oracle"],
        "R_oracle": rec["R_oracle"],
        "num_R_oracle": rec["num_R_oracle"],
        "denom_R_oracle": rec["denom_R_oracle"],
        "oracle_gain": rec["oracle_gain"],
        "R_l": {str(l): cr(r) for l, r in rec["R_l"].items()},
        "H_l": {str(l): cr(r) for l, r in rec["H_l"].items()},
        "R_lc": {str(k): cr(v) for k, v in rec["R_lc"].items()},
        "H_lc": {str(k): cr(v) for k, v in rec["H_lc"].items()},
        "d_js_class": rec["d_js_class"],
        "n_err_c": rec["n_err_c"].tolist(),
        "n_rec_c": rec["n_rec_c"].tolist(),
    }


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serialisable: {type(obj)}")


if __name__ == "__main__":
    main()
