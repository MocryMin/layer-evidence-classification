"""Pipeline orchestration for EXP-20260729-001.

Stages:
  cache_all_splits  -> one-pass frozen-backbone CLS cache (float16 safetensors)
  run_smoke_lr      -> validation-only lr selection on representative layers
  run_full_experiment -> 12 layers x 10 seeds head training + one-time test eval
  run_analysis      -> recoverability / oracle / D_JS / bootstrap CI / judgement

Entrypoints in scripts/ are thin wrappers around these functions.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT, Config
from .artifact import ArtifactPaths, write_resolved_config
from .data import build_label_maps, load_split, save_label_maps, tokenise_split
from .head import predict_with_checkpoint, train_head
from .metrics import aggregate_metrics, per_sample_metrics, recoverability
from .seeding import enable_determinism, seed_all


def get_git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
            ).strip()
            != ""
        )
    except Exception:
        commit, dirty = "unknown", True
    return commit, dirty


def _load_model(cfg: Config, device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(cfg.model_abs_path))
    model = AutoModel.from_pretrained(str(cfg.model_abs_path), dtype=torch.float32)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return tok, model


# --------------------------------------------------------------------------- #
# Stage 1: cache
# --------------------------------------------------------------------------- #
def cache_all_splits(cfg: Config, device: torch.device, paths: ArtifactPaths) -> dict:
    from .cache import cache_split, save_cache, write_manifest

    paths.ensure()
    label2id, id2label, in_scope_ids = build_label_maps(
        cfg.dataset_abs_path, cfg.dataset_config, cfg.drop_oos_label
    )
    save_label_maps(label2id, id2label, paths.root)
    with open(paths.seeds_file(), "w", encoding="utf-8") as f:
        json.dump({"seeds": cfg.seeds}, f)

    tok, model = _load_model(cfg, device)

    manifest = {
        "experiment_id": cfg.experiment_id,
        "model_name": cfg.model_name,
        "model_path": str(cfg.model_abs_path),
        "dataset_config": cfg.dataset_config,
        "drop_oos_label": cfg.drop_oos_label,
        "n_classes": cfg.n_classes,
        "n_transformer_layers": cfg.n_transformer_layers,
        "hidden_state_offset": cfg.hidden_state_offset,
        "cls_token_index": cfg.cls_token_index,
        "cache_dtype": cfg.cache_dtype,
        "splits": {},
    }

    for split in ["train", "validation", "test"]:
        ds = load_split(
            cfg.dataset_abs_path, cfg.dataset_config, split,
            cfg.drop_oos_label, in_scope_ids,
        )
        tok_out = tokenise_split(ds, tok, cfg.prompt, cfg.max_length, cfg.truncation)
        cache = cache_split(
            model, tok, tok_out, device,
            batch_size=128,
            n_layers=cfg.n_transformer_layers,
            hidden_state_offset=cfg.hidden_state_offset,
            cls_index=cfg.cls_token_index,
        )
        hidden = cache["hidden"]
        assert hidden.shape == (len(ds), cfg.n_transformer_layers, cfg.head_input_dim), (
            f"{split}: hidden shape {hidden.shape} != ({len(ds)}, {cfg.n_transformer_layers}, {cfg.head_input_dim})"
        )
        # fail loudly on NaN/Inf
        if not torch.isfinite(hidden).all():
            raise RuntimeError(f"NaN/Inf in cached hidden states for split {split}")
        entry = save_cache(cache, paths.cache_file(split), cfg.cache_dtype)
        manifest["splits"][split] = {
            **entry,
            "n_samples": len(ds),
            "n_layers_cached": cfg.n_transformer_layers,
        }
        print(f"  cached {split}: {hidden.shape} -> {paths.cache_file(split).name}")

    write_manifest(manifest, paths.manifest())
    return manifest


def _load_cache_tensors(paths: ArtifactPaths, device: torch.device, dtype: torch.dtype) -> dict:
    from .cache import load_cache

    out = {}
    for split in ["train", "validation", "test"]:
        c = load_cache(paths.cache_file(split), device=device, dtype=dtype)
        out[split] = c
    return out


# --------------------------------------------------------------------------- #
# Stage 2: lr smoke test (validation-only)
# --------------------------------------------------------------------------- #
def run_smoke_lr(cfg: Config, device: torch.device, paths: ArtifactPaths) -> dict:
    caches = _load_cache_tensors(paths, device, torch.float32)
    train_h = caches["train"]["hidden"]  # (Ntrain, 12, 768)
    val_h = caches["validation"]["hidden"]
    train_y = caches["train"]["labels"].to(device)
    val_y = caches["validation"]["labels"].to(device)

    candidates = list(cfg.lr_smoke_test["candidates"])
    layers = list(cfg.lr_smoke_test["layers"])
    seed = int(cfg.lr_smoke_test["seed"])

    results = []
    for lr in candidates:
        per_layer = {}
        for layer in layers:
            tx = train_h[:, layer - 1, :]
            vx = val_h[:, layer - 1, :]
            res = train_head(
                train_x=tx, train_y=train_y, val_x=vx, val_y=val_y,
                seed=seed, lr=lr, epochs=cfg.epochs, batch_size=cfg.batch_size,
                weight_decay=cfg.weight_decay, grad_clip=cfg.gradient_clip_norm,
                in_dim=cfg.head_input_dim, n_classes=cfg.n_classes,
                device=device, ckpt_dir=None, save_checkpoints=False,
            )
            per_layer[layer] = {"best_val_acc": res["best_val_acc"], "best_val_nll": res["best_val_nll"]}
            print(f"    lr={lr} layer={layer}: val_acc={res['best_val_acc']:.4f} val_nll={res['best_val_nll']:.4f}")
        accs = [per_layer[l]["best_val_acc"] for l in layers]
        nlls = [per_layer[l]["best_val_nll"] for l in layers]
        results.append({
            "lr": lr,
            "mean_val_acc": float(np.mean(accs)),
            "mean_val_nll": float(np.mean(nlls)),
            "per_layer": {str(l): per_layer[l] for l in layers},
        })

    # select: max mean_val_acc, tie-break min mean_val_nll
    results.sort(key=lambda r: (-r["mean_val_acc"], r["mean_val_nll"]))
    selected_lr = results[0]["lr"]

    out = {
        "candidates": candidates,
        "layers": layers,
        "seed": seed,
        "test_involved": False,
        "selection_rule": "max mean validation accuracy; tie-break min mean validation NLL",
        "results": results,
        "selected_lr": selected_lr,
    }
    with open(paths.smoke_results(), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    write_resolved_config(cfg, selected_lr, paths.run_config())
    print(f"\n  selected lr = {selected_lr} "
          f"(mean_val_acc={results[0]['mean_val_acc']:.4f}, mean_val_nll={results[0]['mean_val_nll']:.4f})")
    return out
