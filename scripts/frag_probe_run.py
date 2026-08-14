"""Run one gr1 probe experiment (exp 1/2/4/5) end-to-end.

Single seed 17, frozen backbone, per-layer readout (last-token or CLS):
1. feature extraction (cached, resumable) -> 2. variance/collapse check ->
3. lr smoke on one mid layer -> 4. gradient family (plain | centered_plain)
   + LN plain + ridge grid over all layers, max 10k ep ES on val ->
5. recoverability + class-wise summary -> 6. report + artifacts.

Usage:
    python scripts/frag_probe_run.py --experiment qwen3_clinc
    python scripts/frag_probe_run.py --experiment modernbert_wos --smoke
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
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from src.fragmented import (  # noqa: E402
    CLINC_PROMPT, MAX_EPOCHS, MIN_DELTA, MIN_EPOCHS, PATIENCE, RIDGE_ALPHAS, SEED,
    classwise_summary, exp_dir, extract_features, git_state, judge_collapse,
    layer_variance_stats, load_cache_split, load_clinc_plus, load_wos_46985,
    lr_smoke, run_gradient_family_frag, run_ridge_family_frag, save_cache,
    save_probe_results, write_probe_report,
)
from src.seeding import enable_determinism  # noqa: E402

EXPERIMENTS = {
    "qwen3_clinc": dict(
        exp_name="Qwen3Emb0p6bExp1Ver_260812_01",
        title="Qwen3-Embedding-0.6B last-token readout on CLINC150 — "
              "side verification of EXP-001/003",
        model_name="Qwen3-Embedding-0.6B", model_path="models/Qwen3-Embedding-0.6B",
        pooling="last_token", n_layers=28, final_layer=28, smoke_layer=14,
        dataset="clinc", max_length=512, batch_size=32),
    "modernbert_clinc": dict(
        exp_name="ModernBERTBaseExp1Ver_260812_02",
        title="modernBERT-base CLS readout on CLINC150 — side verification "
              "of EXP-001/003",
        model_name="modernbert-base", model_path="models/modernbert-base",
        pooling="cls", n_layers=22, final_layer=22, smoke_layer=11,
        dataset="clinc", max_length=512, batch_size=32),
    "deberta_wos": dict(
        exp_name="DeBERTaV3BaseWOS46985Baseline_260812_04",
        title="DeBERTa-v3-base CLS baseline on WOS-46985 (134 L2)",
        model_name="deberta-v3-base", model_path="models/deberta-v3-base",
        pooling="cls", n_layers=12, final_layer=12, smoke_layer=6,
        dataset="wos", max_length=512, batch_size=64),
    "modernbert_wos": dict(
        exp_name="ModernBERTBaseWOS46985Baseline_260812_05",
        title="modernBERT-base CLS baseline on WOS-46985 (134 L2)",
        model_name="modernbert-base", model_path="models/modernbert-base",
        pooling="cls", n_layers=22, final_layer=22, smoke_layer=11,
        dataset="wos", max_length=512, batch_size=32),
    # experiment name per user instruction (Qwen2Emb0p6...); the local model is
    # Qwen3-Embedding-0.6B (no Qwen2-Embedding-0.6B exists on disk)
    "qwen_wos": dict(
        exp_name="Qwen2Emb0p6WOS46985Baseline_260814_01",
        title="Qwen3-Embedding-0.6B last-token baseline on WOS-46985 (134 L2)",
        model_name="Qwen3-Embedding-0.6B", model_path="models/Qwen3-Embedding-0.6B",
        pooling="last_token", n_layers=28, final_layer=28, smoke_layer=14,
        dataset="wos", max_length=512, batch_size=32, date="2026-08-14"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subsets, 50 ep, 2 layers (cache_smoke dir)")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    args = ap.parse_args()

    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecfg = EXPERIMENTS[args.experiment]
    exp_name = ecfg["exp_name"]
    d = exp_dir(exp_name)
    cache_dir = d / ("cache_smoke" if args.smoke else "cache")
    print(f"[{exp_name}] device={device} smoke={args.smoke}")

    # ---- data ----------------------------------------------------------- #
    if ecfg["dataset"] == "clinc":
        data = load_clinc_plus()
        dataset_note = "CLINC150 plus config, OOS (intent 42) dropped, " \
                       "EXP-001 id2label mapping"
        prompt = CLINC_PROMPT
    else:
        data = load_wos_46985()
        dataset_note = "WOS-46985, HYDRA-count split 30070/7518/9397 (seed 17, " \
                       "plain random), 134 L2 classes"
        prompt = None
    n_classes = 150 if ecfg["dataset"] == "clinc" else 134
    split_sizes = {k: len(v[0]) for k, v in data.items()}
    print(f"[data] splits={split_sizes}")

    # ---- extraction (resumable; smoke uses its own cache dir) ----------- #
    hidden, labels = {}, {}
    model = tok = None
    for k in data:
        cache_file = cache_dir / f"{k}_hidden.npz"
        if cache_file.exists():
            h, l = load_cache_split(cache_dir, k)
            if h.shape[0] == split_sizes[k] and (l == data[k][1]).all():
                hidden[k], labels[k] = h, l
                continue
        if model is None:
            print(f"[extract] model={ecfg['model_path']} pooling={ecfg['pooling']} "
                  f"max_length={ecfg['max_length']}")
            model = AutoModel.from_pretrained(str(ROOT / ecfg["model_path"]))
            tok = AutoTokenizer.from_pretrained(str(ROOT / ecfg["model_path"]))
        texts = data[k][0] if not args.smoke else data[k][0][:200]
        y = data[k][1] if not args.smoke else data[k][1][:200]
        feat = extract_features(model, tok, texts, ecfg["pooling"],
                                ecfg["max_length"], ecfg["batch_size"],
                                device, prompt)
        save_cache(cache_dir, k, feat, y)
        hidden[k], labels[k] = feat, y
        print(f"  [extract] {k}: {feat.shape}")

    # ---- variance / collapse (union of splits) -------------------------- #
    union = np.concatenate([hidden[k] for k in ("train", "validation", "test")],
                           axis=0)
    stats = layer_variance_stats(union)
    decision = judge_collapse(stats)
    print(f"[variance] min inter_std={decision['min_inter_std']:.3e} "
          f"@L{decision['min_layer']} -> "
          f"{'COLLAPSED' if decision['collapsed'] else 'healthy'}")

    grad_family = {
        "name": decision["gradient_family"],
        "head_type": "plain",
        "centering": "train_mean" if decision["collapsed"] else "none",
    }
    ln_family = {"name": "ln_plain", "head_type": "ln", "centering": "none"}
    training = dict(lr=1e-2, weight_decay=decision["weight_decay"], grad_clip=0.0,
                    min_epochs=MIN_EPOCHS,
                    max_epochs=50 if args.smoke else MAX_EPOCHS,
                    patience=PATIENCE, min_delta=MIN_DELTA, seed=SEED)

    layers = args.layers or list(range(1, ecfg["n_layers"] + 1))
    if args.smoke:
        # cheap smoke: first two requested layers + final layer (recoverability)
        layers = sorted(set(layers[:2] + [ecfg["final_layer"]]))
    print(f"[probe] layers={layers} gradient={grad_family['name']} "
          f"wd={training['weight_decay']:g}")

    # ---- lr smoke ------------------------------------------------------- #
    lr_smoke_res = {}
    if not args.smoke:
        lr_smoke_res = lr_smoke(hidden, labels, [grad_family, ln_family],
                                ecfg["smoke_layer"], [1e-2, 1e-3], training,
                                layers, ecfg["final_layer"], n_classes, device)

    # ---- probe families ------------------------------------------------- #
    results = {
        "experiment": exp_name, "date": ecfg.get("date", "2026-08-12"),
        "reporting_model": "deepseek-v4-flash",
        "git": git_state(),
        "config": {
            "title": ecfg["title"], "dataset": ecfg["dataset"],
            "dataset_note": dataset_note, "split_sizes": split_sizes,
            "n_classes": n_classes, "model_name": ecfg["model_name"],
            "pooling": ecfg["pooling"], "max_length": ecfg["max_length"],
            "truncation": "right", "cache_dtype": "float16",
            "families": [grad_family["name"], "ln_plain", "ridge"],
            "training": training, "ridge_alphas": RIDGE_ALPHAS, "seed": SEED,
            "final_layer": ecfg["final_layer"],
        },
        "variance": {"stats": stats, "decision": decision},
        "lr_smoke": lr_smoke_res,
        "families": {},
    }

    def run_and_record(fam_cfg, tcfg):
        fam = run_gradient_family_frag(hidden, labels, fam_cfg, tcfg, layers,
                                       ecfg["final_layer"], n_classes, device)
        fam["classwise"] = classwise_summary(fam["recoverability"])
        results["families"][fam_cfg["name"]] = fam
        save_probe_results(exp_name, results)  # incremental checkpoint
        print(f"  [save] {fam_cfg['name']} done")

    for fam_cfg in (grad_family, ln_family):
        tcfg = {**training}
        if not args.smoke:
            tcfg["lr"] = lr_smoke_res[fam_cfg["name"]]["chosen_lr"]
        run_and_record(fam_cfg, tcfg)

    if not args.smoke:
        ridge = run_ridge_family_frag(hidden, labels, RIDGE_ALPHAS, layers,
                                      ecfg["final_layer"], n_classes)
        ridge["classwise"] = classwise_summary(ridge["recoverability"])
        results["families"]["ridge"] = ridge
        save_probe_results(exp_name, results)
        print("  [save] ridge done")

    if args.smoke:
        print(f"[smoke] OK — {exp_name} layers={layers} "
              f"families={list(results['families'])}")
        return

    rpt = write_probe_report(exp_name, results,
                             date=ecfg.get("date", "2026-08-12"),
                             group="user_exp_plans/fragmented_exp_gr1.md")
    print(f"[done] artifacts: {d}")
    print(f"[done] report:    {rpt}")


if __name__ == "__main__":
    main()
