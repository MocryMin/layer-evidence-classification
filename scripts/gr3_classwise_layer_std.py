"""gr3 step-1 addendum: per-class recoverability spread across layers.

For each class c:  R_{l,c} over layers -> std_c (spread of recoverability
across the stack) and the sorted-curve drop structure (sort R_{l,c} desc;
largest consecutive drop + its position) - is recoverability for a class
concentrated in a few layers (early sharp drop) or diffuse (flat curve)?

Reads analysis.json (unsimplified num/den fractions), writes
layer_std_analysis.json + {model}_layerstd.csv next to it. CPU only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "artifacts/fragmented-experiments/class-wiseRecoverabilityWOS46985_260818_01"


def parse_fraction(s: str) -> float:
    num, den = s.split("/")
    return float(num) / float(den) if float(den) > 0 else np.nan


def analyse(model: str, res: dict, l2_names, l2_to_l1, l1_names) -> dict:
    rlc = res["classwise"]["R_lc"]          # (L,) list of 134 fraction strings
    L = len(rlc)
    n_cls = len(rlc[0])
    M = np.array([[parse_fraction(rlc[l][c]) for c in range(n_cls)]
                  for l in range(L)])       # (L, C)
    n_err = np.array(res["classwise"]["n_err_c"], dtype=float)

    std_c = np.nanstd(M, axis=0)            # spread over layers
    mean_c = np.nanmean(M, axis=0)
    r_max = np.nanmax(M, axis=0)
    argmax = np.nanargmax(M, axis=0) + 1

    # sorted-curve drop structure per class
    sorted_curves = np.sort(M, axis=0)[::-1]          # (L, C) desc
    drops = -np.diff(sorted_curves, axis=0)           # (L-1, C) r_k - r_{k+1}
    with np.errstate(invalid="ignore"):
        drop_pos = np.nanargmax(drops, axis=0) + 1    # 1-indexed position k
        drop_mag = np.nanmax(drops, axis=0)
    top_gap = sorted_curves[0] - sorted_curves[1]     # r1 - r2

    valid = n_err >= 10
    n_valid = int(valid.sum())
    out = dict(
        model=model, L=L, n_classes=n_cls,
        n_valid=int(valid.sum()), min_n_err=10,
        std_c=dict(mean=float(np.nanmean(std_c[valid])),
                   median=float(np.nanmedian(std_c[valid])),
                   min=float(np.nanmin(std_c[valid])),
                   max=float(np.nanmax(std_c[valid]))),
        corr_std_n_err=dict(
            pearson=float(np.corrcoef(std_c[valid], n_err[valid])[0, 1]),
            spearman=float(np.corrcoef(np.argsort(np.argsort(std_c[valid])),
                                       np.argsort(np.argsort(n_err[valid])))[0, 1])),
        # where does the largest sorted drop happen (hist over classes)
        drop_pos_hist={str(k): int((drop_pos[valid] == k).sum())
                       for k in range(1, L)},
        drop_mag=dict(mean=float(np.nanmean(drop_mag[valid])),
                      median=float(np.nanmedian(drop_mag[valid]))),
        # early-sharp-drop classes: max drop at position <=3 and >= 0.25
        sharp=[{"class": str(l2_names[c]), "l1": l1_names[int(l2_to_l1[c])],
                "n_err": int(n_err[c]), "std": float(std_c[c]),
                "r_max": float(r_max[c]), "argmax": int(argmax[c]),
                "drop_mag": float(drop_mag[c]), "drop_pos": int(drop_pos[c]),
                "top_gap": float(top_gap[c])}
               for c in range(n_cls)
               if valid[c] and drop_pos[c] <= 3 and drop_mag[c] >= 0.25],
        flat=[{"class": str(l2_names[c]), "l1": l1_names[int(l2_to_l1[c])],
               "n_err": int(n_err[c]), "std": float(std_c[c]),
               "r_max": float(r_max[c]), "argmax": int(argmax[c])}
              for c in range(n_cls)
              if valid[c] and std_c[c] <= 0.05],
        # per-class rows for CSV
        rows=[{"class": str(l2_names[c]), "l1": l1_names[int(l2_to_l1[c])],
               "n_err": int(n_err[c]), "std": float(std_c[c]),
               "mean": float(mean_c[c]), "r_max": float(r_max[c]),
               "argmax_layer": int(argmax[c]), "drop_mag": float(drop_mag[c]),
               "drop_pos": int(drop_pos[c]), "top_gap": float(top_gap[c])}
              for c in range(n_cls)],
    )
    return out


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from gr3_classwise_recoverability import load_class_names

    d = json.load(open(D / "analysis.json"))
    _, l2_to_l1 = load_class_names()      # (134,) primary L1 index
    l2_names = d["l2_names"]
    l1_names = ["Computer Science", "ECE", "Psychology", "MAE", "Civil",
                "Medical", "Biochemistry"]
    out = {}
    for model in ("deberta", "modernbert"):
        res = d["results"][model]
        a = analyse(model, res, l2_names, l2_to_l1, l1_names)
        out[model] = a
        with open(D / f"{model}_layerstd.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(a["rows"][0].keys()))
            w.writeheader(); w.writerows(a["rows"])
        print(f"[{model}] std mean {a['std_c']['mean']:.4f} "
              f"median {a['std_c']['median']:.4f} "
              f"drop_pos hist {a['drop_pos_hist']}")
        print(f"  sharp-drop classes: {len(a['sharp'])}; flat classes: {len(a['flat'])}")
    json.dump(out, open(D / "layer_std_analysis.json", "w"), indent=1)
    print(f"[done] {D / 'layer_std_analysis.json'}")


if __name__ == "__main__":
    main()
