"""Shared infrastructure for fragmented experiments (gr1) — data collection & direction exploration.

Single-point experiments with no hypothesis; this module provides the
dataset-agnostic pieces (AGENT_PROTOCOL.md §9):

- CLINC150 (plus config, OOS intent-42 dropped, EXP-001 id2label map) and
  WOS-46985 (HYDRA-count split) loaders;
- frozen-backbone per-layer feature extraction (CLS or last-token pooling);
- per-layer variance stats + collapse judgement (EXP-002 conventions);
- lr smoke and single-seed probe suites (gradient families + ridge grid)
  reusing :mod:`src.probe` / :mod:`src.exp003` primitives;
- artifact + report writing under
  ``artifacts/fragmented-experiments/<exp_name>/`` and
  ``agent-BuildReports/fragmented-experiments/<exp_name>.md``.
"""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT
from .exp003 import _recoverability_to_json, fit_ridge_grid
from .heads import build_head
from .metrics import aggregate_metrics, confusion_matrix, macro_f1, recoverability
from .probe import train_probe_fullbatch_es
from .seeding import enable_determinism

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/fragmented-experiments"
REPORT_ROOT = PROJECT_ROOT / "agent-BuildReports/fragmented-experiments"

# EXP-002 conventions: CLS inter-sample std < 1e-3 == collapsed readout.
COLLAPSE_INTER_STD = 1e-3
RIDGE_ALPHAS = [0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1, 10, 100]
# gr1 probe budget ("1w ep"), EXP-003 early-stop rule
MIN_EPOCHS, MAX_EPOCHS, PATIENCE, MIN_DELTA = 100, 10000, 100, 1.0e-4
SEED = 17
CLINC_PROMPT = "Classify the intent: {utterance}"
CLINC_OOS_ID = 42
WOS_N_L1, WOS_N_L2 = 7, 134


def git_state() -> dict:
    """Current commit + dirty flag (recorded in every fragmented artifact)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = "unknown", True
    return {"commit": commit, "dirty": dirty}


def exp_dir(name: str) -> Path:
    d = ARTIFACT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #
def load_clinc_plus() -> dict:
    """CLINC150 plus config, OOS (intent 42) dropped, EXP-001 label mapping.

    Reproduces exactly the EXP-001/003 cache population: same parquet row
    order, same filter, probe ids = sorted original ids of id2label.json.
    Returns ``{"train": (texts, y), "validation": ..., "test": ...}``.
    """
    import pandas as pd

    root = PROJECT_ROOT / "data/raw/clinc_oos/plus"
    # id2label.json maps probe ids 0..149 (contiguous) to intent names; the
    # OOS intent (original id 42) was dropped and ids above it shifted down
    # by one. Probe id = orig - 1 if orig > 42 else orig.
    remap = {orig: orig - 1 if orig > 42 else orig
             for orig in range(0, 151) if orig != 42}
    assert len(remap) == 150 and 42 not in remap

    def load(f: str) -> tuple[list[str], np.ndarray]:
        df = pd.read_parquet(root / f)
        df = df[df["intent"] != CLINC_OOS_ID].reset_index(drop=True)
        y = df["intent"].map(remap).to_numpy()
        return df["text"].tolist(), y

    return {
        "train": load("train-00000-of-00001.parquet"),
        "validation": load("validation-00000-of-00001.parquet"),
        "test": load("test-00000-of-00001.parquet"),
    }


def load_wos_46985() -> dict:
    """WOS-46985 with the HYDRA-count split (30070/7518/9397, seed 17).

    Labels are the 134 L2 subcategory ids (drop the 7 L1 domain bits).
    Returns ``{"train": (texts, y), ...}`` with y in [0, 134).
    """
    # text column is a pandas str -> object-dtype array (pickled inside npz)
    split = np.load(PROJECT_ROOT / "data/processed/wos46985/wos46985_split.npz",
                    allow_pickle=True)
    texts = split["text"]
    l2 = split["labels"][:, WOS_N_L1:]  # (N, 134)
    assert (l2.sum(1) == 1).all()
    y = l2.argmax(1)
    out = {}
    for name, key in [("train", "train_idx"), ("validation", "val_idx"),
                      ("test", "test_idx")]:
        idx = split[key]
        out[name] = (texts[idx].tolist(), y[idx])
    return out


# --------------------------------------------------------------------------- #
# Feature extraction (frozen backbone)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    pooling: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    prompt: str | None = None,
) -> np.ndarray:
    """Per-layer pooled hidden states, (N, L, D) float16.

    ``pooling="cls"`` takes position 0 (requires right-side padding);
    ``pooling="last_token"`` takes the last non-pad token per sample
    (padding-side agnostic). Truncation is from the right (keeps the text
    head). Hidden states 1..L of the frozen backbone.
    """
    if prompt is not None:
        texts = [prompt.format(utterance=t) for t in texts]
    model = model.to(device).eval()
    chunks = []
    for s in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[s:s + batch_size], padding="longest", truncation=True,
            max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        out = model(**enc, output_hidden_states=True)
        feat = []
        for h in out.hidden_states[1:]:  # skip embeddings
            if pooling == "cls":
                f = h[:, 0]
            elif pooling == "last_token":
                pos = enc["attention_mask"].sum(dim=1) - 1
                f = h[torch.arange(h.shape[0], device=h.device), pos]
            else:
                raise ValueError(f"unknown pooling {pooling}")
            feat.append(f)
        # cast to float32 first: some backbones (qwen3) emit bfloat16 hidden states
        chunks.append(torch.stack(feat, dim=1).float().cpu().numpy().astype(np.float16))
    return np.concatenate(chunks, axis=0)


def save_cache(d: Path, split: str, feat: np.ndarray, y: np.ndarray) -> None:
    """Persist one split's features + labels (resumable extraction)."""
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{split}_hidden.npz", hidden=feat, labels=y.astype(np.int16))


def load_cache_split(d: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(d / f"{split}_hidden.npz")
    return z["hidden"], z["labels"]


# --------------------------------------------------------------------------- #
# Variance / collapse analysis
# --------------------------------------------------------------------------- #
def layer_variance_stats(feat: np.ndarray) -> list[dict]:
    """Per-layer inter-sample std, participation ratio, top-1 var fraction.

    ``feat``: (N, L, D) — variance is computed on the union of all splits.
    """
    out = []
    for l in range(feat.shape[1]):
        x = feat[:, l].astype(np.float64)
        inter_std = float(x.std(axis=0).mean())
        cov = np.cov(x, rowvar=False)
        eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
        trace = float(eig.sum())
        pr = float((eig.sum() ** 2) / ((eig ** 2).sum())) if trace > 0 else 0.0
        top1 = float(eig.max() / trace) if trace > 0 else 0.0
        out.append({"inter_std": inter_std, "participation_ratio": pr,
                    "top1_var_frac": top1})
    return out


def judge_collapse(stats: list[dict]) -> dict:
    """Collapse judgement (EXP-002): any layer with inter_std < 1e-3.

    Returns the decision dict; family choice follows the gr1 plan: healthy ->
    plain (+ wd=1e-2), collapsed -> centered plain (+ wd=0). LN plain always
    runs; ridge always runs.
    """
    inter_stds = [s["inter_std"] for s in stats]
    collapsed = min(inter_stds) < COLLAPSE_INTER_STD
    return {
        "collapsed": collapsed,
        "min_inter_std": min(inter_stds),
        "min_layer": int(np.argmin(inter_stds)) + 1,
        "threshold": COLLAPSE_INTER_STD,
        "gradient_family": "centered_plain" if collapsed else "plain",
        "weight_decay": 0.0 if collapsed else 1.0e-2,
        "criterion": "min inter-sample std < 1e-3 => collapsed readout "
                     "(EXP-002 convention); wd=1e-2 if healthy else wd=0 (gr1 plan)",
    }


# --------------------------------------------------------------------------- #
# Probe suites (single seed)
# --------------------------------------------------------------------------- #
def run_gradient_family_frag(
    hidden: dict,
    labels: dict,
    family: dict,
    training: dict,
    layers: list[int],
    final_layer: int,
    n_classes: int,
    device: torch.device,
) -> dict:
    """Single-seed gradient probe family over ``layers`` (generic in_dim).

    Mirrors EXP-003's run_gradient_family with seed fixed by ``training``.
    """
    head_type = family["head_type"]
    centering = family.get("centering", "none")
    tr_h = torch.as_tensor(hidden["train"], dtype=torch.float32, device=device)
    va_h = torch.as_tensor(hidden["validation"], dtype=torch.float32, device=device)
    te_h = torch.as_tensor(hidden["test"], dtype=torch.float32, device=device)
    tr_y = torch.as_tensor(labels["train"], dtype=torch.long, device=device)
    va_y = torch.as_tensor(labels["validation"], dtype=torch.long, device=device)
    te_y = torch.as_tensor(labels["test"], dtype=torch.long, device=device)
    train_means = tr_h.mean(dim=0) if centering == "train_mean" else None
    in_dim = tr_h.shape[2]

    per_layer: dict[int, dict] = {}
    test_preds: list[np.ndarray] = []
    test_logits_list: list[np.ndarray] = []
    for layer in layers:
        tx = tr_h[:, layer - 1, :].contiguous()
        vx = va_h[:, layer - 1, :].contiguous()
        tex = te_h[:, layer - 1, :].contiguous()
        if train_means is not None:
            mu = train_means[layer - 1]
            tx, vx, tex = tx - mu, vx - mu, tex - mu

        head = build_head(head_type, in_dim=in_dim, n_classes=n_classes)
        res = train_probe_fullbatch_es(
            head=head, train_x=tx, train_y=tr_y, val_x=vx, val_y=va_y,
            lr=training["lr"], weight_decay=training["weight_decay"],
            grad_clip=training.get("grad_clip", 0.0),
            min_epochs=training["min_epochs"], max_epochs=training["max_epochs"],
            patience=training["patience"], min_delta=training["min_delta"],
            seed=training["seed"], device=device)

        head.load_state_dict(res["best_state"])
        head.to(device).eval()
        with torch.no_grad():
            te_logits = torch.cat(
                [head(tex[s:s + 1024]) for s in range(0, tex.shape[0], 1024)], dim=0
            ).cpu().numpy().astype(np.float32)
        te_pred = te_logits.argmax(1)
        te_labels_np = te_y.cpu().numpy()
        agg = aggregate_metrics(te_logits, te_labels_np, n_classes, n_bins=10)

        per_layer[layer] = {
            "best_epoch": res["best_epoch"],
            "best_val_acc": res["best_val_acc"],
            "best_val_nll": res["best_val_nll"],
            "final_val_acc": res["final_val_acc"],
            "converged": res["converged"],
            "stop_reason": res["stop_reason"],
            "n_epochs_run": len(res["history"]),
            "test_acc": agg["accuracy"],
            "test_macro_f1": agg["macro_f1"],
            "test_nll": agg["nll"],
            "test_ece": agg["ece"],
        }
        test_preds.append(te_pred.astype(np.int16))
        test_logits_list.append(te_logits.astype(np.float16))
        print(f"  [{family['name']}] L{layer:2d}: val={res['best_val_acc']:.4f}"
              f"@{res['best_epoch']} test={agg['accuracy']:.4f}"
              f" ({res['stop_reason']})")

    te_labels_np = te_y.cpu().numpy()
    layer_pred = np.stack(test_preds)
    rec = recoverability(
        layer_correct=(layer_pred == te_labels_np), layer_pred=layer_pred,
        labels=te_labels_np, final_layer_idx=layers.index(final_layer),
        n_classes=n_classes)
    return {
        "family": family["name"],
        "per_layer": {str(l): per_layer[l] for l in layers},
        "test_pred": layer_pred,
        "test_logits": np.stack(test_logits_list),
        "recoverability": _recoverability_to_json(rec),
    }


def run_ridge_family_frag(
    hidden: dict,
    labels: dict,
    alphas: list[float],
    layers: list[int],
    final_layer: int,
    n_classes: int,
) -> dict:
    """Ridge alpha grid per layer (deterministic; OLS for alpha=0)."""
    tr_h = hidden["train"].astype(np.float64)
    va_h = hidden["validation"].astype(np.float64)
    te_h = hidden["test"].astype(np.float64)
    tr_y, va_y, te_y = labels["train"], labels["validation"], labels["test"]

    per_layer: dict[int, dict] = {}
    test_preds: list[np.ndarray] = []
    for layer in layers:
        res = fit_ridge_grid(tr_h[:, layer - 1], tr_y, va_h[:, layer - 1], va_y,
                             te_h[:, layer - 1], alphas, n_classes)
        te_pred = res["test_pred"]
        cm = confusion_matrix(te_y, te_pred, n_classes)
        per_layer[layer] = {
            "best_alpha": res["best_alpha"],
            "best_val_acc": res["best_val_acc"],
            "test_acc": float((te_pred == te_y).mean()),
            "test_macro_f1": macro_f1(cm),
            "per_alpha": {str(a): v for a, v in res["per_alpha"].items()},
        }
        test_preds.append(te_pred.astype(np.int16))
        print(f"  [ridge] L{layer:2d}: alpha={res['best_alpha']:g} "
              f"val={res['best_val_acc']:.4f} test={per_layer[layer]['test_acc']:.4f}")

    layer_pred = np.stack(test_preds)
    rec = recoverability(
        layer_correct=(layer_pred == te_y), layer_pred=layer_pred,
        labels=te_y, final_layer_idx=layers.index(final_layer),
        n_classes=n_classes)
    return {
        "family": "ridge",
        "per_layer": {str(l): per_layer[l] for l in layers},
        "test_pred": layer_pred,
        "recoverability": _recoverability_to_json(rec),
    }


def lr_smoke(
    hidden: dict, labels: dict, family_cfgs: list[dict], mid_layer: int,
    lrs: list[float], training: dict, layers_all: list[int], final_layer: int,
    n_classes: int, device: torch.device,
) -> dict:
    """Quick 500-ep lr sweep on one mid layer; returns chosen lr per family."""
    training = {**training, "max_epochs": 500}
    chosen: dict[str, dict] = {}
    for fam in family_cfgs:
        results = {}
        for lr in lrs:
            cfg = {**training, "lr": lr}
            res = run_gradient_family_frag(
                hidden, labels, fam, cfg, [mid_layer], final_layer,
                n_classes, device)
            results[f"{lr:g}"] = res["per_layer"][str(mid_layer)]["best_val_acc"]
        best_lr = max(lrs, key=lambda l: (results[f"{l:g}"], -l))
        chosen[fam["name"]] = {"chosen_lr": best_lr, "val_acc": results}
        print(f"  [lr-smoke {fam['name']}] {results} -> lr={best_lr:g}")
    return chosen


# --------------------------------------------------------------------------- #
# Class-wise summary (EXP-003 §4 conventions)
# --------------------------------------------------------------------------- #
def classwise_summary(rec_json: dict) -> dict:
    """Per-class max recoverability across mid layers + coverage counts."""
    per_class_max: dict[int, float] = {}
    for k, v in rec_json["R_lc"].items():
        _, c = ast.literal_eval(k)
        if v["den"] > 0:
            per_class_max[c] = max(per_class_max.get(c, 0.0), v["ratio"])
    return {
        "n_classes": len(per_class_max),
        "coverage": len(per_class_max),
        "n_R_ge_0_5": sum(1 for r in per_class_max.values() if r >= 0.5),
        "n_R_ge_0_8": sum(1 for r in per_class_max.values() if r >= 0.8),
        "n_R_ge_1_0": sum(1 for r in per_class_max.values() if r >= 1.0),
        "definition": "R_max(c) = max over mid layers of R_lc ratio (den>0); "
                      "counts of classes with R_max >= 0.5/0.8/1.0",
    }


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #
def _fmt_table(rows: list[list], header: list[str]) -> str:
    w = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    line = "| " + " | ".join(f"{h:<{w[i]}}" for i, h in enumerate(header)) + " |"
    sep = "|" + "|".join("-" * (wi + 2) for wi in w) + "|"
    body = "\n".join(
        "| " + " | ".join(f"{str(c):<{w[i]}}" for i, c in enumerate(r)) + " |"
        for r in rows)
    return f"{line}\n{sep}\n{body}"


def write_probe_report(exp_name: str, results: dict) -> Path:
    """Generate the single-point report md from a probe results dict."""
    cfg = results["config"]
    rpt = []
    rpt.append(f"# {exp_name} — {cfg['title']}\n")
    rpt.append(f"Date: 2026-08-12 · Group: `plans/fragmented_exp_gr1.md` · "
               f"Reporting model: deepseek-v4-flash · Git: `{results['git']['commit']}`"
               f" (dirty={results['git']['dirty']}) · Single seed {cfg['seed']}\n")

    rpt.append("## Config\n")
    rpt.append(f"- Dataset: {cfg['dataset']} ({cfg['dataset_note']}) · "
               f"{cfg['n_classes']} classes · splits {cfg['split_sizes']}")
    rpt.append(f"- Model: {cfg['model_name']} (frozen) · pooling `{cfg['pooling']}` · "
               f"max_length {cfg['max_length']} · truncation right · cache float16")
    rpt.append(f"- Probe families: {', '.join(cfg['families'])}")
    rpt.append(f"- Training: full-batch AdamW, lr={cfg['training']['lr']:g}, "
               f"wd={cfg['training']['weight_decay']:g}, Xavier init, "
               f"min_ep {cfg['training']['min_epochs']}/max {cfg['training']['max_epochs']}/"
               f"patience {cfg['training']['patience']}/min_delta {cfg['training']['min_delta']:g} "
               f"(early stop on val acc)")
    rpt.append(f"- Ridge grid: {cfg['ridge_alphas']} (alpha=0 -> OLS lstsq), "
               f"alpha by val acc, test once\n")

    rpt.append("## 1. Readout variance (collapse check)\n")
    rpt.append("| layer | inter_std | PR | top1 frac |")
    rpt.append("|------:|----------:|----:|----------:|")
    for i, s in enumerate(results["variance"]["stats"], start=1):
        rpt.append(f"| {i} | {s['inter_std']:.3e} | {s['participation_ratio']:.2f} "
                   f"| {s['top1_var_frac']:.3f} |")
    j = results["variance"]["decision"]
    rpt.append(f"\n**Judgement:** {'COLLAPSED' if j['collapsed'] else 'healthy'} "
               f"(min inter_std {j['min_inter_std']:.3e} @ L{j['min_layer']}; "
               f"threshold {j['threshold']:g}). Gradient family = `{j['gradient_family']}`, "
               f"wd = {j['weight_decay']:g}.\n")

    if results.get("lr_smoke"):
        rpt.append("## 2. LR smoke (500 ep, one mid layer)\n")
        for fam, d in results["lr_smoke"].items():
            rpt.append(f"- `{fam}`: { {k: f'{v:.4f}' for k, v in d['val_acc'].items()} }"
                       f" -> lr = {d['chosen_lr']:g}")

    for fam_name, fam in results["families"].items():
        rpt.append(f"\n## 3. {fam_name} — layer-wise results\n")
        rows = []
        for l in sorted(int(x) for x in fam["per_layer"]):
            p = fam["per_layer"][str(l)]
            rows.append([l, p["best_val_acc"], p["test_acc"], p.get("test_macro_f1", "-"),
                         p.get("best_epoch", "-"), p.get("stop_reason", "-")])
        rpt.append(_fmt_table(rows, ["layer", "val_acc", "test_acc", "macro_f1",
                                     "best_ep", "stop"]))
        rec = fam["recoverability"]
        rpt.append(f"\n- Recoverability vs final layer: oracle gain "
                   f"**+{rec['oracle_gain']:.4f}** (acc_L {rec['acc_L']:.4f} -> "
                   f"acc_oracle {rec['acc_oracle']:.4f}), R_oracle "
                   f"{rec['R_oracle']:.4f} ({rec['num_R_oracle']}/{rec['denom_R_oracle']}), "
                   f"D_JS {rec['d_js_class']:.4f}")
        cs = fam["classwise"]
        rpt.append(f"- Class-wise: coverage {cs['coverage']}/{cfg['n_classes']} classes; "
                   f"R_max >= 0.5/0.8/1.0: {cs['n_R_ge_0_5']}/{cs['n_R_ge_0_8']}/"
                   f"{cs['n_R_ge_1_0']}")

    rpt.append("\n## 4. Artifacts\n")
    rpt.append(f"- `artifacts/fragmented-experiments/{exp_name}/`: `results.json`, "
               f"`cache/` (features float16), `<family>_test_pred.npy` "
               f"(L×N int16), `<family>_test_logits.npy` (gradient families, "
               f"L×N×C float16), `<family>_classwise_summary.json`.")
    rpt.append(f"- Full per-layer val histories are not persisted (fragmented "
               f"records scalars only); see `results.json` for all metrics.\n")
    rpt.append("## 5. Observations\n")
    rpt.append("- _Data collection point — no hypothesis; observations are "
               "recorded in results.json and reproduced by the report tables._")

    path = REPORT_ROOT / f"{exp_name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rpt) + "\n", encoding="utf-8")
    return path


def save_probe_results(exp_name: str, results: dict) -> Path:
    d = exp_dir(exp_name)
    # persist npy arrays, strip them from the JSON payload, then dump JSON
    for fam_name, fam in results["families"].items():
        # idempotent: arrays may already have been popped by an earlier call
        pred = fam.pop("test_pred", None)
        if pred is not None:
            np.save(d / f"{fam_name}_test_pred.npy", pred)
        logits = fam.pop("test_logits", None)
        if logits is not None:
            np.save(d / f"{fam_name}_test_logits.npy", logits)
        (d / f"{fam_name}_classwise_summary.json").write_text(
            json.dumps(fam["classwise"], indent=2), encoding="utf-8")
    (d / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return d
