"""WOS-46985 train/val/test split, reproducing the HYDRA (EMNLP 2025) counts.

The HF mirror (data/raw/wos46985/wos46895.parquet) ships NO official split:
the original kk7nc GitHub repo is gone (404). HYDRA (EMNLP 2025) formally
uses a 30,070 / 7,518 / 9,397 train/val/test split (sum = 46,985 = the full
dataset). The exact per-document assignment is unavailable, so we reproduce
the COUNTS with a plain random permutation at a fixed seed (17). If the
original indices are ever recovered, replace `split_indices.npz` and re-run
the WOS experiments (exp 3-5 of fragmented_exp_gr1).

Outputs under data/processed/wos46985/ (gitignored):
- wos46985_split.npz: text (46985,), labels (46985, 141) uint8 one-hot
  (7 L1 domains + 134 L2 subcategories), train_idx / val_idx / test_idx (int64)
- split_summary.json: split sizes, per-L1-domain counts, per-L2-class
  counts, min/max L2 counts per split

Usage:
    python scripts/wos46985_split.py            # seed 17
    python scripts/wos46985_split.py --seed 42  # documented alternative
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/wos46985/wos46895.parquet"
OUT_DIR = ROOT / "data/processed/wos46985"

TRAIN_N, VAL_N, TEST_N = 30070, 7518, 9397  # HYDRA (EMNLP 2025) counts
N_L1, N_L2 = 7, 134  # label = [L1 domain | L2 subcategory] one-hot (141-dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    df = pd.read_parquet(RAW)
    n = len(df)
    assert n == TRAIN_N + VAL_N + TEST_N, f"counts sum {TRAIN_N + VAL_N + TEST_N} != {n}"
    labels = np.asarray(df["label"].tolist(), dtype=np.uint8)
    assert labels.shape == (n, N_L1 + N_L2)
    assert (labels.sum(1) == 2).all()  # exactly one L1 + one L2 per row

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    train_idx, val_idx, test_idx = (
        perm[:TRAIN_N], perm[TRAIN_N:TRAIN_N + VAL_N], perm[TRAIN_N + VAL_N:])
    assert train_idx.shape[0] == TRAIN_N and val_idx.shape[0] == VAL_N \
        and test_idx.shape[0] == TEST_N

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "wos46985_split.npz",
        text=df["text"].to_numpy(),
        labels=labels,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )

    l1_names = [d[0] for d in df["label_description"]]
    l2_ids = labels[:, N_L1:].argmax(1)  # per-row L2 id (one-hot block)
    summary = {
        "source": str(RAW),
        "split": "plain random, fixed seed (HYDRA counts only; original "
                 "per-document assignment unavailable)",
        "seed": args.seed,
        "counts": {"train": TRAIN_N, "val": VAL_N, "test": TEST_N},
        "n_docs": n,
        "per_split": {},
    }
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        l2 = Counter(l2_ids[idx])
        summary["per_split"][name] = {
            "n": int(len(idx)),
            "n_l2_classes_present": len(l2),
            "min_l2_count": min(l2.values()),
            "max_l2_count": max(l2.values()),
            "l1_domains": dict(Counter(l1_names[i] for i in idx.tolist())),
        }
    with open(OUT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        s = summary["per_split"][name]
        print(f"{name:5s} n={s['n']:6d}  L2 classes={s['n_l2_classes_present']:3d}  "
              f"min/max L2 count={s['min_l2_count']}/{s['max_l2_count']}")
    print(f"[split] saved {OUT_DIR / 'wos46985_split.npz'} (seed={args.seed})")


if __name__ == "__main__":
    main()
