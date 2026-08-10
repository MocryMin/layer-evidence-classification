# EXP-001 / EXP-002 Freeze Manifest

- **Freeze tag:** `exp-001-002-freeze-20260810`
- **Generated:** 2026-08-10T19:37:39
- **Git remote:** `git@github.com:MocryMin/layer-evidence-classification.git`
- **HF dataset URI:** `PENDING (filled after upload, step 9)`
- **Experiments:** EXP-20260729-001, EXP-20260729-002

## What is preserved where

| Location | Contents |
|---|---|
| **GitHub** (this repo, at the freeze tag) | code (`src/`, `scripts/`), configs, tests, requirements, experiment reports (`agent-BuildReports/experiments/`), this manifest + small summaries (`agent-BuildReports/freeze-20260810/`), PROJECT_STATUS/RESOURCES/AGENT_PROTOCOL |
| **HuggingFace** (private dataset, see URI above) | 38 artifact result files, FT-backbone best checkpoint (8 files), frozen mlruns.db snapshot |
| **Not uploaded** (rebuildable) | 12 CLS safetensors caches (2 GB) - rebuild via `scripts/cache_hidden.py` |

## Not applicable (never produced by these experiments)

- **predictions_logits:** never generated; experiments record aggregate metrics + per-epoch val_history, not per-sample outputs
- **plots:** never generated; results are JSON/markdown tables, no figure files
- **probe_checkpoints:** probe heads never checkpointed; results recorded as JSON

## HF upload contents — included artifact files

| path | size | sha256 (first 12) |
|---|---:|---|
| `artifacts/EXP-20260729-001/README.md` | 4 KB | `e2b907c4946a` |
| `artifacts/EXP-20260729-001/cache/cache_manifest.json` | 2 KB | `c3b92bc96bcf` |
| `artifacts/EXP-20260729-001/id2label.json` | 4 KB | `0985fcdeeeef` |
| `artifacts/EXP-20260729-001/label2id.json` | 3 KB | `a405cfbe8001` |
| `artifacts/EXP-20260729-001/plain_probe_mainline/results.json` | 2 MB | `6e2d33887a5d` |
| `artifacts/EXP-20260729-001/run_config.yaml` | 2 KB | `3cc339104641` |
| `artifacts/EXP-20260729-001/seeds.json` | 55 B | `02136675771a` |
| `artifacts/EXP-20260729-001/smoke_lr_results.json` | 2 KB | `09d9be0f54d7` |
| `artifacts/EXP-20260729-002/01_lr_grid/best_lr_per_layer.json` | 378 B | `55adddfd21b8` |
| `artifacts/EXP-20260729-002/01_lr_grid/lr_grid_plain_adamw.json` | 12 KB | `ec42ab1eaefe` |
| `artifacts/EXP-20260729-002/02_variance_collapse/feature_stats_test_with_prompt.json` | 8 KB | `61d549363a8e` |
| `artifacts/EXP-20260729-002/02_variance_collapse/feature_stats_train_with_prompt.json` | 8 KB | `1e7870f06c9c` |
| `artifacts/EXP-20260729-002/02_variance_collapse/per_layer_collapse_summary.json` | 8 KB | `b5b63006306b` |
| `artifacts/EXP-20260729-002/03a_ln_ablation/ln_ablation_lr1e-2.json` | 9 KB | `665eea108c0a` |
| `artifacts/EXP-20260729-002/03a_ln_ablation/transformed_feature_stats.json` | 17 KB | `4f455abeb800` |
| `artifacts/EXP-20260729-002/03b_no_prompt/cache/cache_manifest.json` | 2 KB | `3b1b52738e3d` |
| `artifacts/EXP-20260729-002/03b_no_prompt/feature_stats_train_no_prompt.json` | 8 KB | `e02d872eadf6` |
| `artifacts/EXP-20260729-002/03b_no_prompt/plain_probe_no_prompt.json` | 9 KB | `b396b56074a0` |
| `artifacts/EXP-20260729-002/03c_ft_backbone/cache/cache_manifest.json` | 2 KB | `9e326657ce95` |
| `artifacts/EXP-20260729-002/03c_ft_backbone/feature_stats_train_ft.json` | 8 KB | `f6d9184db769` |
| `artifacts/EXP-20260729-002/03c_ft_backbone/ft_history.json` | 1 KB | `8beb7eaa04e2` |
| `artifacts/EXP-20260729-002/03c_ft_backbone/plain_ln_probe_ft.json` | 4 KB | `2af48bb51d36` |
| `artifacts/EXP-20260729-002/03c_ft_backbone/plain_probe_ft.json` | 3 KB | `47ad278ca43f` |
| `artifacts/EXP-20260729-002/03d_optimizer/optimizer_comparison.json` | 6 KB | `ca1b4ac05c22` |
| `artifacts/EXP-20260729-002/03e_fp16_control/result.json` | 564 B | `dec425188a3e` |
| `artifacts/EXP-20260729-002/03f_ridge/ridge_results.json` | 3 KB | `4c481f1e7752` |
| `artifacts/EXP-20260729-002/03g_ridge_alpha_grid/cache_ft_noprompt/cache_manifest.json` | 2 KB | `00a5c523466c` |
| `artifacts/EXP-20260729-002/03g_ridge_alpha_grid/ols_reference.json` | 1 KB | `aceae04232d1` |
| `artifacts/EXP-20260729-002/03g_ridge_alpha_grid/ridge_base_noprompt.json` | 16 KB | `1037ffbfc403` |
| `artifacts/EXP-20260729-002/03g_ridge_alpha_grid/ridge_base_withprompt.json` | 16 KB | `8f8a88da26b2` |
| `artifacts/EXP-20260729-002/03g_ridge_alpha_grid/ridge_ft_noprompt.json` | 16 KB | `06e38eafb735` |
| `artifacts/EXP-20260729-002/03g_ridge_alpha_grid/summary.json` | 8 KB | `4fb562f32e18` |
| `artifacts/EXP-20260729-002/03h_adam_ablation/adam_ablation.json` | 11 MB | `7439a04fa233` |
| `artifacts/EXP-20260729-002/04_mlp_probe/dead_relu_diagnosis.json` | 36 KB | `55506692aad4` |
| `artifacts/EXP-20260729-002/04_mlp_probe/mlp_probe_lr1e-2.json` | 29 KB | `fab851a91bc7` |
| `artifacts/EXP-20260729-002/05_act_ablation/act_ablation.json` | 31 KB | `879c499fa84f` |
| `artifacts/EXP-20260729-002/06_token_variance/token_variance_check.json` | 8 KB | `961ecb90d378` |
| `artifacts/EXP-20260729-002/README.md` | 8 KB | `3abe253fa3e8` |

## HF upload contents — best checkpoint (FT backbone)

| path | size | sha256 (first 12) |
|---|---:|---|
| `models/deberta-v3-base-clinc150-ft/config.json` | 905 B | `1785cec98744` |
| `models/deberta-v3-base-clinc150-ft/ft_head.pt` | 452 KB | `64df208cedb1` |
| `models/deberta-v3-base-clinc150-ft/ft_history.json` | 1 KB | `8beb7eaa04e2` |
| `models/deberta-v3-base-clinc150-ft/id2label.json` | 4 KB | `0985fcdeeeef` |
| `models/deberta-v3-base-clinc150-ft/label2id.json` | 3 KB | `a405cfbe8001` |
| `models/deberta-v3-base-clinc150-ft/model.safetensors` | 701 MB | `7e0f5397577f` |
| `models/deberta-v3-base-clinc150-ft/tokenizer.json` | 8 MB | `1305924e6107` |
| `models/deberta-v3-base-clinc150-ft/tokenizer_config.json` | 536 B | `1121ad33ecd9` |

**mlruns snapshot:** `mlruns_exp-001-002_freeze_20260810.db` (772 KB), sha256 `3be0508b7cba…`, source: mlruns.db (SQLite, VACUUM-d, integrity_check=ok), db rows: {'experiments': 2, 'runs': 8, 'metrics': 44, 'params': 21, 'latest_metrics': 44}

## Excluded from upload — rebuildable caches

| path | size | sha256 (first 12) |
|---|---:|---|
| `artifacts/EXP-20260729-001/cache/test_hidden.safetensors` | 79 MB | `8acd014d2445` |
| `artifacts/EXP-20260729-001/cache/train_hidden.safetensors` | 264 MB | `c1e0d3ee6156` |
| `artifacts/EXP-20260729-001/cache/validation_hidden.safetensors` | 53 MB | `3129f3bb0c2e` |
| `artifacts/EXP-20260729-002/03b_no_prompt/cache/test_hidden.safetensors` | 79 MB | `6362c5fc5ad1` |
| `artifacts/EXP-20260729-002/03b_no_prompt/cache/train_hidden.safetensors` | 264 MB | `c8bcf11a9c61` |
| `artifacts/EXP-20260729-002/03b_no_prompt/cache/validation_hidden.safetensors` | 53 MB | `5964c2126bd5` |
| `artifacts/EXP-20260729-002/03c_ft_backbone/cache/test_hidden.safetensors` | 79 MB | `ae88dc15f00d` |
| `artifacts/EXP-20260729-002/03c_ft_backbone/cache/train_hidden.safetensors` | 264 MB | `8a9c69916193` |
| `artifacts/EXP-20260729-002/03c_ft_backbone/cache/validation_hidden.safetensors` | 53 MB | `07c3ea98b4f3` |
| `artifacts/EXP-20260729-002/03g_ridge_alpha_grid/cache_ft_noprompt/test_hidden.safetensors` | 79 MB | `a6a9d2574872` |
| `artifacts/EXP-20260729-002/03g_ridge_alpha_grid/cache_ft_noprompt/train_hidden.safetensors` | 264 MB | `2b68fadec7c4` |
| `artifacts/EXP-20260729-002/03g_ridge_alpha_grid/cache_ft_noprompt/validation_hidden.safetensors` | 53 MB | `415961be6882` |

**Total excluded:** 2 GB. frozen-backbone CLS safetensors caches, cheaply rebuildable via scripts/cache_hidden.py.

## Reproduction

- Code + config + reports: `git clone` this repo, checkout tag `exp-001-002-freeze-20260810`.
- Artifacts + mlruns snapshot: download from the HF dataset URI above (private).
- CLS caches (not uploaded): rebuild with `python -u scripts/cache_hidden.py` (frozen `microsoft/deberta-v3-base`, CLS pooling, layers 1..12, fp16 safetensors). Verify against the `cache_manifest.json` sha256 listed in the HF upload.
- FT backbone: either download from HF (`best_checkpoints/deberta-v3-base-clinc150-ft/`) or re-run `src/finetune.py`.
- Full write-up: `agent-BuildReports/experiments/EXP-20260729-002--linear-probe-midlayer-collapse.md`.
