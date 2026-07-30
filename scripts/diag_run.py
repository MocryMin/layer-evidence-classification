"""EXP-20260729-002 diagnostic runner.

Phases (each writes to artifacts/EXP-20260729-002/<phase>/):
  stats         : per-layer variance / anisotropy statistics (task 2, core finding)
  lr_grid       : unrestricted lr sweep of the plain linear probe (task 1)
  ln_ablation   : plain / LN / norm-only / affine-only across layers (task 3a)
  optimizer     : AdamW / LBFGS / SGD on the plain linear probe (task 3d)
  no_prompt     : re-cache without instruction prefix + stats + probe (task 3b)
  ft_backbone   : full-FT backbone, re-cache + stats + probe (task 3c)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.config import load_diag_config  # noqa: E402
from src.seeding import enable_determinism, seed_all  # noqa: E402
from src.diag import (  # noqa: E402
    head_ablation, layerwise_feature_stats, lr_grid_probe, optimizer_comparison,
    save_stats_json, cache_variant, load_variant_cache,
)


def setup():
    cfg = load_diag_config()
    if cfg.raw.get("deterministic", True):
        enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.artifact_path.mkdir(parents=True, exist_ok=True)
    return cfg, device


def _save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"  saved {path}")


def _exp001_cache_dir(cfg):
    return Path("artifacts/EXP-20260729-001/cache")


def _load_exp001_cache(cfg, device):
    from src.cache import load_cache
    d = _exp001_cache_dir(cfg)
    out = {}
    for split in ["train", "validation", "test"]:
        out[split] = load_cache(d / f"{split}_hidden.safetensors", device=device, dtype=torch.float32)
    return out


# --------------------------------------------------------------------------- #
def phase_stats(cfg, device):
    """Task 2: per-layer variance/anisotropy statistics on the frozen-backbone cache."""
    caches = _load_exp001_cache(cfg, device)
    out_dir = cfg.artifact_path / "02_variance_collapse"
    rows = layerwise_feature_stats(caches["train"], cfg.n_classes, split="train")
    save_stats_json(rows, out_dir / "feature_stats_train_with_prompt.json")
    # also test split
    rows_test = layerwise_feature_stats(caches["test"], cfg.n_classes, split="test")
    save_stats_json(rows_test, out_dir / "feature_stats_test_with_prompt.json")

    print("\n=== per-layer feature stats (train, with-prompt, frozen backbone) ===")
    hdr = f"{'layer':>5} | {'inter_std':>10} | {'within_std':>11} | {'mean_norm':>10} | {'class_sig':>9} | {'part_ratio':>10} | {'top1_var':>9} | {'top10_var':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['layer']:>5} | {r['inter_sample_std_mean']:>10.6f} | {r['within_sample_std_mean']:>11.5f} | "
              f"{r['mean_norm']:>10.3f} | {r['class_signal_ratio']:>9.4f} | {r['participation_ratio']:>10.3f} | "
              f"{r['top1_var_frac']:>9.4f} | {r['top10_var_frac']:>10.4f}")


# --------------------------------------------------------------------------- #
def phase_lr_grid(cfg, device):
    """Task 1: unrestricted lr grid on the plain linear probe."""
    from src.mlflow_utils import setup_mlflow
    import mlflow
    setup_mlflow(cfg)
    caches = _load_exp001_cache(cfg, device)
    out_dir = cfg.artifact_path / "01_lr_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=f"{cfg.experiment_id}_lr_grid") as run:
        mlflow.log_param("phase", "lr_grid")
        mlflow.log_param("head_type", "plain")
        mlflow.log_param("optimizer", "adamw")
        mlflow.log_param("lr_grid", cfg.lr_grid)
        mlflow.log_param("layers", cfg.representative_layers)
        res = lr_grid_probe(
            caches["train"], caches["validation"], cfg.lr_grid, cfg.representative_layers,
            head_type="plain", optimizer="adamw", epochs=cfg.epochs,
            batch_size=cfg.batch_size, weight_decay=cfg.weight_decay,
            grad_clip=cfg.gradient_clip_norm, seed=cfg.seed, device=device,
            in_dim=cfg.head_input_dim, n_classes=cfg.n_classes,
        )
        _save(out_dir / "lr_grid_plain_adamw.json", res)
        # best lr per layer
        best_per_layer = {}
        for layer in cfg.representative_layers:
            cand = [(lr, res[f"layer_{layer}_lr_{lr:g}"]["best_val_acc"])
                    for lr in cfg.lr_grid]
            best_lr, best_acc = max(cand, key=lambda x: x[1])
            best_per_layer[layer] = {"best_lr": best_lr, "best_val_acc": best_acc}
            mlflow.log_metric(f"best_val_acc_layer_{layer}", best_acc)
        _save(out_dir / "best_lr_per_layer.json", best_per_layer)
        print("\n=== best lr per layer (plain linear probe) ===")
        for l, v in best_per_layer.items():
            print(f"  layer {l}: best_lr={v['best_lr']:g} best_val_acc={v['best_val_acc']:.4f}")


# --------------------------------------------------------------------------- #
def phase_ln_ablation(cfg, device):
    """Task 3a: plain / LN / norm-only / affine-only across all layers at lr=1e-2."""
    from src.mlflow_utils import setup_mlflow
    import mlflow
    setup_mlflow(cfg)
    caches = _load_exp001_cache(cfg, device)
    out_dir = cfg.artifact_path / "03a_ln_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    lr = 1.0e-2
    with mlflow.start_run(run_name=f"{cfg.experiment_id}_ln_ablation") as run:
        mlflow.log_param("phase", "ln_ablation")
        mlflow.log_param("lr", lr)
        res = head_ablation(
            caches["train"], caches["validation"],
            ["plain", "ln", "norm_only", "affine_only"], cfg.probe_layers,
            lr=lr, optimizer="adamw", epochs=cfg.epochs, batch_size=cfg.batch_size,
            weight_decay=cfg.weight_decay, grad_clip=cfg.gradient_clip_norm,
            seed=cfg.seed, device=device, in_dim=cfg.head_input_dim,
            n_classes=cfg.n_classes,
        )
        _save(out_dir / "ln_ablation_lr1e-2.json", res)
        for k, v in res.items():
            mlflow.log_metric(f"{k}_best_val_acc", v["best_val_acc"])


# --------------------------------------------------------------------------- #
def phase_optimizer(cfg, device):
    """Task 3d: AdamW / LBFGS / SGD on the plain linear probe."""
    from src.mlflow_utils import setup_mlflow
    import mlflow
    setup_mlflow(cfg)
    caches = _load_exp001_cache(cfg, device)
    out_dir = cfg.artifact_path / "03d_optimizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=f"{cfg.experiment_id}_optimizer") as run:
        mlflow.log_param("phase", "optimizer")
        all_res = {}
        # AdamW and SGD at lr=1e-2 on representative layers
        for opt in ["adamw", "sgd"]:
            r = lr_grid_probe(
                caches["train"], caches["validation"], [1e-2, 1e-1], cfg.representative_layers,
                head_type="plain", optimizer=opt, epochs=cfg.epochs, batch_size=cfg.batch_size,
                weight_decay=cfg.weight_decay, grad_clip=cfg.gradient_clip_norm,
                seed=cfg.seed, device=device, in_dim=cfg.head_input_dim, n_classes=cfg.n_classes,
            )
            all_res[opt] = r
        # LBFGS full-batch, fewer epochs (it converges fast), lr=1.0
        lbfgs_epochs = min(cfg.epochs, 30)
        r = lr_grid_probe(
            caches["train"], caches["validation"], [1.0], cfg.representative_layers,
            head_type="plain", optimizer="lbfgs", epochs=lbfgs_epochs, batch_size=cfg.batch_size,
            weight_decay=cfg.weight_decay, grad_clip=0.0, seed=cfg.seed, device=device,
            in_dim=cfg.head_input_dim, n_classes=cfg.n_classes, lbfgs_cfg=cfg.lbfgs,
        )
        all_res["lbfgs"] = r
        _save(out_dir / "optimizer_comparison.json", all_res)
        for opt, rd in all_res.items():
            for k, v in rd.items():
                mlflow.log_metric(f"{opt}_{k}_best_val_acc", v["best_val_acc"])


# --------------------------------------------------------------------------- #
def phase_no_prompt(cfg, device):
    """Task 3b: re-cache without the instruction prefix; stats + plain probe."""
    from src.mlflow_utils import setup_mlflow
    import mlflow
    setup_mlflow(cfg)
    out_dir = cfg.artifact_path / "03b_no_prompt"
    cache_dir = out_dir / "cache"
    with mlflow.start_run(run_name=f"{cfg.experiment_id}_no_prompt") as run:
        mlflow.log_param("phase", "no_prompt")
        mlflow.log_param("prompt", cfg.prompt_pure)
        print("  caching with pure-utterance prompt (no instruction)...")
        cache_variant(
            cfg.model_abs_path, cfg.prompt_pure, cfg.dataset_abs_path, cfg.dataset_config,
            cfg.drop_oos_label, cfg.n_classes, cfg.max_length, cfg.truncation,
            cfg.n_transformer_layers, cfg.hidden_state_offset, cfg.cls_token_index,
            cfg.cache_dtype, device, cache_dir,
        )
        vc = load_variant_cache(cache_dir, device)
        rows = layerwise_feature_stats(vc["train"], cfg.n_classes, split="train")
        save_stats_json(rows, out_dir / "feature_stats_train_no_prompt.json")
        mlflow.log_param("inter_std_layer6", next(r["inter_sample_std_mean"] for r in rows if r["layer"] == 6))
        # plain linear probe at lr=1e-2 on all layers + a small lr grid on representative layers
        res = lr_grid_probe(
            vc["train"], vc["validation"], [1e-3, 1e-2, 1e-1], cfg.probe_layers,
            head_type="plain", optimizer="adamw", epochs=cfg.epochs, batch_size=cfg.batch_size,
            weight_decay=cfg.weight_decay, grad_clip=cfg.gradient_clip_norm,
            seed=cfg.seed, device=device, in_dim=cfg.head_input_dim, n_classes=cfg.n_classes,
        )
        _save(out_dir / "plain_probe_no_prompt.json", res)
        print("\n=== no-prompt per-layer stats ===")
        for r in rows:
            print(f"  layer {r['layer']:2d}: inter_std={r['inter_sample_std_mean']:.6f} "
                  f"part_ratio={r['participation_ratio']:.3f} class_sig={r['class_signal_ratio']:.4f}")


# --------------------------------------------------------------------------- #
def phase_ft_backbone(cfg, device):
    """Task 3c: full-FT backbone, re-cache, stats + plain/LN probe."""
    from src.mlflow_utils import setup_mlflow
    from src.finetune import finetune_backbone
    import mlflow
    setup_mlflow(cfg)
    out_dir = cfg.artifact_path / "03c_ft_backbone"
    cache_dir = out_dir / "cache"
    with mlflow.start_run(run_name=f"{cfg.experiment_id}_ft_backbone") as run:
        mlflow.log_param("phase", "ft_backbone")
        print("  full fine-tuning backbone on CLINC150...")
        ft = finetune_backbone(cfg, device)
        mlflow.log_param("ft_save_path", ft["save_path"])
        mlflow.log_metric("ft_final_val_acc", ft["final_val_acc"])
        mlflow.log_metric("ft_final_test_acc", ft["final_test_acc"])
        _save(out_dir / "ft_history.json", ft["history"])
        print(f"  FT done: val_acc={ft['final_val_acc']:.4f} test_acc={ft['final_test_acc']:.4f}")

        print("  caching FT backbone hidden states...")
        ft_path = Path(cfg.finetune["save_backbone"])
        cache_variant(
            ft_path, cfg.prompt_with_instruction, cfg.dataset_abs_path, cfg.dataset_config,
            cfg.drop_oos_label, cfg.n_classes, cfg.max_length, cfg.truncation,
            cfg.n_transformer_layers, cfg.hidden_state_offset, cfg.cls_token_index,
            cfg.cache_dtype, device, cache_dir,
        )
        vc = load_variant_cache(cache_dir, device)
        rows = layerwise_feature_stats(vc["train"], cfg.n_classes, split="train")
        save_stats_json(rows, out_dir / "feature_stats_train_ft.json")
        for r in rows:
            mlflow.log_metric(f"ft_inter_std_layer_{r['layer']}", r["inter_sample_std_mean"])
        # plain + LN probe at lr=1e-2 on all layers
        res_plain = lr_grid_probe(
            vc["train"], vc["validation"], [1e-2], cfg.probe_layers,
            head_type="plain", optimizer="adamw", epochs=cfg.epochs, batch_size=cfg.batch_size,
            weight_decay=cfg.weight_decay, grad_clip=cfg.gradient_clip_norm,
            seed=cfg.seed, device=device, in_dim=cfg.head_input_dim, n_classes=cfg.n_classes,
        )
        _save(out_dir / "plain_probe_ft.json", res_plain)
        res_ln = head_ablation(
            vc["train"], vc["validation"], ["plain", "ln"], cfg.probe_layers,
            lr=1e-2, optimizer="adamw", epochs=cfg.epochs, batch_size=cfg.batch_size,
            weight_decay=cfg.weight_decay, grad_clip=cfg.gradient_clip_norm,
            seed=cfg.seed, device=device, in_dim=cfg.head_input_dim, n_classes=cfg.n_classes,
        )
        _save(out_dir / "plain_ln_probe_ft.json", res_ln)
        print("\n=== FT-backbone per-layer probe (plain) ===")
        for layer in cfg.probe_layers:
            v = res_plain[f"layer_{layer}_lr_0.01"]
            print(f"  layer {layer:2d}: plain best_val_acc={v['best_val_acc']:.4f}")


PHASES = {
    "stats": phase_stats,
    "lr_grid": phase_lr_grid,
    "ln_ablation": phase_ln_ablation,
    "optimizer": phase_optimizer,
    "no_prompt": phase_no_prompt,
    "ft_backbone": phase_ft_backbone,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=list(PHASES.keys()))
    args = ap.parse_args()
    cfg, device = setup()
    print(f"[diag] phase={args.phase} device={device} exp={cfg.experiment_id}")
    PHASES[args.phase](cfg, device)


if __name__ == "__main__":
    main()
