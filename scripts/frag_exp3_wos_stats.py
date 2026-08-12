"""gr1 exp3: WOS-46985 basic statistics (HYDRA-count split 30070/7518/9397).

Counts class distribution (7 L1 domains, 134 L2), text length distribution
(chars + deberta tokens), and sanity checks on train/val/test.

Usage:
    python scripts/frag_exp3_wos_stats.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.fragmented import (  # noqa: E402
    REPORT_ROOT, WOS_N_L1, WOS_N_L2, exp_dir, git_state,
)

EXP_NAME = "WOS46985Features_260812_03"
RAW = ROOT / "data/raw/wos46985/wos46895.parquet"
SPLIT = ROOT / "data/processed/wos46985/wos46985_split.npz"


def token_lengths(texts: list[str], batch_size: int = 512) -> np.ndarray:
    """Token-id counts per text via the frozen deberta tokenizer (no truncation)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/deberta-v3-base"))
    lens = np.empty(len(texts), dtype=np.int64)
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], add_special_tokens=True)
        lens[s:s + batch_size] = [len(x) for x in enc["input_ids"]]
    return lens


def main():
    df = pd.read_parquet(RAW)
    split = np.load(SPLIT, allow_pickle=True)  # text column is object-dtype
    texts = split["text"]
    labels = split["labels"]  # (N, 141) uint8
    # one-hot blocks: exactly one 1 in each block (verified below), so argmax
    # gives the class id exactly (np.flatnonzero on the transpose is WRONG —
    # the mod does not recover the row class id)
    l1 = labels[:, :WOS_N_L1].argmax(1)
    l2 = labels[:, WOS_N_L1:].argmax(1)
    l1_names = [d[0].strip() for d in df["label_description"]]
    l2_names = [d[1].strip() for d in df["label_description"]]

    print("[exp3] computing char + token lengths ...")
    char_lens = np.array([len(t) for t in texts], dtype=np.int64)
    tok_lens = token_lengths(texts.tolist())

    summary = {
        "experiment": EXP_NAME, "date": "2026-08-12",
        "reporting_model": "deepseek-v4-flash",
        "git": git_state(),
        "source": str(RAW),
        "split": str(SPLIT),
        "split_note": "HYDRA (EMNLP 2025) counts 30070/7518/9397, plain random "
                      "seed 17 (scripts/wos46985_split.py)",
        "n_docs": int(len(texts)),
        "sanity": {
            "n_cols": list(df.columns),
            "duplicate_texts": int(df["text"].duplicated().sum()),
            "label_sum_always_2": bool((labels.sum(1) == 2).all()),
            "l1_bits_sum_1": bool((labels[:, :WOS_N_L1].sum(1) == 1).all()),
            "l2_bits_sum_1": bool((labels[:, WOS_N_L1:].sum(1) == 1).all()),
        },
        "per_split": {},
    }

    for name, key in [("train", "train_idx"), ("validation", "val_idx"),
                      ("test", "test_idx")]:
        idx = split[key]
        chars, toks = char_lens[idx], tok_lens[idx]
        l1c = Counter(l1[idx].tolist())
        l2c = Counter(l2[idx].tolist())
        per = {
            "n": int(len(idx)),
            "chars": _length_stats(chars),
            "tokens": _length_stats(toks),
            "l1_domains": {str(k): {"count": v, "name": l1_names[idx[np.where(l1[idx] == k)[0][0]]]}
                           for k, v in sorted(l1c.items())},
            "l2_n_classes_present": len(l2c),
            "l2_min_count": min(l2c.values()),
            "l2_max_count": max(l2c.values()),
            "l2_counts": {str(k): v for k, v in sorted(l2c.items())},
            "l2_top20": [{"id": k, "count": v, "name": l2_names[idx[np.where(l2[idx] == k)[0][0]]]}
                         for k, v in l2c.most_common(20)],
            "l2_rarest10": [{"id": k, "count": v, "name": l2_names[idx[np.where(l2[idx] == k)[0][0]]]}
                            for k, v in l2c.most_common()[-10:]],
        }
        summary["per_split"][name] = per
        print(f"  {name}: n={per['n']} chars mean={per['chars']['mean']:.0f} "
              f"tokens mean={per['tokens']['mean']:.0f} L2 classes={len(l2c)}")

    # L1 x L2 coverage across the full set
    l1l2 = Counter(zip(l1.tolist(), l2.tolist()))
    summary["l1_x_l2"] = {
        "n_pairs": len(l1l2),
        "l1": {str(k): v for k, v in Counter(l1.tolist()).items()},
    }

    d = exp_dir(EXP_NAME)
    (d / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(d / "lengths.npz", char_lens=char_lens, tok_lens=tok_lens,
             l1=l1, l2=l2)

    _write_plots(d, char_lens, tok_lens, split, l1)
    rpt = _write_report(d, summary)
    print(f"[done] artifacts: {d}")
    print(f"[done] report:    {rpt}")


def _length_stats(a: np.ndarray) -> dict:
    q = np.quantile(a, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {"min": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "max": float(q[4]),
            "mean": float(a.mean()), "std": float(a.std())}


def _write_plots(d, char_lens, tok_lens, split, l1):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib unavailable — skipping PNGs")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"train": "#4C72B0", "validation": "#55A868", "test": "#C44E52"}
    for name, idx in [("train", split["train_idx"]), ("validation", split["val_idx"]),
                      ("test", split["test_idx"])]:
        axes[0].hist(char_lens[idx], bins=80, alpha=0.55, label=name,
                     color=colors[name])
        axes[1].hist(tok_lens[idx], bins=80, alpha=0.55, label=name,
                     color=colors[name])
    axes[0].set_title("char length")
    axes[1].set_title("deberta token length")
    for ax in axes:
        ax.set_xlabel("length"); ax.legend()
    fig.tight_layout()
    fig.savefig(d / "length_distribution.png", dpi=120)
    plt.close(fig)


def _write_report(d, summary) -> Path:
    rpt = []
    rpt.append(f"# {EXP_NAME} — WOS-46985 dataset statistics\n")
    rpt.append(f"Date: 2026-08-12 · Group: `plans/fragmented_exp_gr1.md` · "
               f"Reporting model: deepseek-v4-flash · Git: "
               f"`{summary['git']['commit']}` (dirty={summary['git']['dirty']})\n")
    rpt.append("## Config\n")
    rpt.append(f"- Source: `data/raw/wos46985/wos46895.parquet` (HF mirror "
               f"`jesse-tong/wos46985`; original kk7nc repo 404)")
    rpt.append(f"- Split: {summary['split_note']}; artifact "
               f"`data/processed/wos46985/wos46985_split.npz`")
    rpt.append(f"- n = {summary['n_docs']} docs, 3 columns: `text`, `label` "
               f"(141-dim one-hot = 7 L1 + 134 L2, exactly 2 ones/row), "
               f"`label_description` ([domain, subcategory])\n")
    rpt.append("## 1. Sanity\n")
    rpt.append(f"- duplicate texts: {summary['sanity']['duplicate_texts']} · "
               f"label sum == 2 always: {summary['sanity']['label_sum_always_2']} · "
               f"L1 bit == 1 always: {summary['sanity']['l1_bits_sum_1']} · "
               f"L2 bit == 1 always: {summary['sanity']['l2_bits_sum_1']}\n")
    rpt.append("## 2. Per-split length distribution\n")
    rows = []
    for name, s in summary["per_split"].items():
        for what in ("chars", "tokens"):
            st = s[what]
            rows.append([name, what, f"{st['mean']:.0f}", f"{st['median']:.0f}",
                         f"{st['min']:.0f}", f"{st['p75']:.0f}", f"{st['max']:.0f}"])
    rpt.append("| split | unit | mean | median | min | p75 | max |")
    rpt.append("|-------|------|-----:|-------:|----:|----:|----:|")
    rpt.extend(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |"
               for r in rows)
    rpt.append(f"\nHistograms: `artifacts/fragmented-experiments/{EXP_NAME}/"
               f"length_distribution.png`\n")
    rpt.append("## 3. Class distribution (L1 domains)\n")
    for name, s in summary["per_split"].items():
        rpt.append(f"- **{name}** (n={s['n']}): " + ", ".join(
            f"{v['name']} {v['count']} ({v['count'] / s['n'] * 100:.1f}%)"
            for v in s["l1_domains"].values()) )
    rpt.append("\n## 4. Class distribution (L2, 134 classes)\n")
    for name, s in summary["per_split"].items():
        rpt.append(f"- **{name}**: {s['l2_n_classes_present']}/134 classes present · "
                   f"min/max per class = {s['l2_min_count']}/{s['l2_max_count']}")
    rpt.append("\nTop-20 L2 classes (train):\n")
    rpt.append("| id | count | name |")
    rpt.append("|----|------:|------|")
    for c in summary["per_split"]["train"]["l2_top20"]:
        rpt.append(f"| {c['id']} | {c['count']} | {c['name']} |")
    rpt.append("\nRarest-10 L2 classes (train):\n")
    rpt.append("| id | count | name |")
    rpt.append("|----|------:|------|")
    for c in summary["per_split"]["train"]["l2_rarest10"]:
        rpt.append(f"| {c['id']} | {c['count']} | {c['name']} |")
    rpt.append("\n## 5. Artifacts\n")
    rpt.append(f"- `artifacts/fragmented-experiments/{EXP_NAME}/`: `results.json` "
               f"(full per-split L2 counts), `lengths.npz`, `length_distribution.png`.\n")
    rpt.append("## 6. Observations\n")
    rpt.append("- Data collection point — no hypothesis. Full per-L2-count "
               "vectors are in `results.json` for downstream experiments.")
    path = REPORT_ROOT / f"{EXP_NAME}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rpt) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
