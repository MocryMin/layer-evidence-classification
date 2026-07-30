# EXP-20260729-002 - Linear-probe mid-layer variance collapse on DeBERTa-v3-base / CLINC150

- **Date**: 2026-07-29
- **State**: IN PROGRESS (diagnostic campaign; EXP-20260729-001 mainline paused)
- **Parent**: EXP-20260729-001
- **Git commit**: _(filled at finalisation)_
- **MLflow experiment**: `EXP-20260729-002` (`sqlite:///mlruns.db`)
- **Artifact root**: `artifacts/EXP-20260729-002/`

## Question

EXP-20260729-001 trains a **pure linear probe** (`W x + b`, frozen backbone) on the
CLS token of each DeBERTa-v3-base layer to test whether mid layers are
non-inferior / superior to the final layer and possess recoverability (H1/H1'/H2).
A validation-only lr smoke test revealed that mid layers stay at near-random
accuracy for every tested lr. This experiment characterises **why**, and tests
four candidate remedies.

## Setup

- Backbone: `microsoft/deberta-v3-base` (frozen, 12 layers, hidden 768), CLS pooling.
- Dataset: CLINC150 `plus` config, OOS (label 42) dropped, 150 in-scope classes,
  splits 15000/3000/4500. Prompt: `Classify the intent: {utterance}` (left truncation, max_length 512).
- Cache: one-pass frozen-backbone forward, CLS of layers 1..12, float16 safetensors
  (shared with EXP-001: `artifacts/EXP-20260729-001/cache/`).
- Probe: linear head 768->150 (+bias), Xavier-uniform weight, zero bias; AdamW,
  wd 0.01, grad-clip 1.0, 100 epochs, batch 256, seed 17 (val split only for
  diagnostics here; test not used for selection).
- lr grid (task 1, unrestricted): {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3, 10}.
- Head variants (task 3a): `plain`, `ln` (LayerNorm+linear), `norm_only`
  (normalise, no affine), `affine_only` (gamma*x+beta, no normalise).
- Optimizers (task 3d): AdamW, SGD, LBFGS (full-batch, strong-Wolfe).
- No-prompt variant (task 3b): re-cache with pure utterance (no instruction prefix).
- FT backbone (task 3c): full fine-tune backbone+head on last-layer CLS, save backbone,
  re-probe.

## Results

### Task 2 - Mid-layer CLS variance collapse (core finding)

Per-layer feature statistics on the frozen-backbone train cache (with-prompt):

| layer | inter-sample std | within-sample std | mean norm | class-signal ratio | participation ratio | top-1 var frac | top-10 var frac |
|------:|-----------------:|------------------:|----------:|-------------------:|--------------------:|---------------:|----------------:|
| 1  | 0.00364 | 0.239 | 6.62  | 0.324 | 1.05  | 0.978 | 0.993 |
| 2  | 0.00127 | 0.067 | 1.86  | 0.414 | 1.55  | 0.801 | 0.909 |
| 3  | 0.00611 | 0.100 | 2.78  | 0.340 | 1.11  | 0.949 | 0.974 |
| 4  | 0.00055 | 0.179 | 4.96  | 0.279 | 3.54  | 0.506 | 0.791 |
| 5  | 0.00055 | 0.164 | 4.55  | 0.427 | 2.70  | 0.601 | 0.825 |
| 6  | 0.00020 | 0.148 | 4.12  | 0.186 | 2.39  | 0.627 | 0.890 |
| 7  | 0.00028 | 0.176 | 4.87  | 0.566 | 17.24 | 0.190 | 0.543 |
| 8  | 0.00022 | 0.208 | 5.76  | 0.355 | 10.08 | 0.281 | 0.622 |
| 9  | 0.00117 | 0.116 | 3.21  | 0.616 | 22.45 | 0.138 | 0.536 |
| 10 | 0.00050 | 0.240 | 6.66  | 0.499 | 25.08 | 0.145 | 0.482 |
| 11 | 0.02454 | 0.285 | 7.93  | 0.279 | 2.24  | 0.664 | 0.833 |
| 12 | 0.02035 | 0.696 | 19.28 | 0.369 | 7.10  | 0.319 | 0.721 |

_(raw JSON: `artifacts/EXP-20260729-002/02_variance_collapse/feature_stats_train_with_prompt.json`)_

### Task 1 - Unrestricted lr grid (plain linear probe)

_(to be filled from `01_lr_grid/lr_grid_plain_adamw.json`)_

### Task 3a - LN ablation (plain / ln / norm_only / affine_only)

_(to be filled from `03a_ln_ablation/ln_ablation_lr1e-2.json`)_

### Task 3b - No-instruction prompt

_(to be filled from `03b_no_prompt/`)_

### Task 3c - Fine-tuned backbone

_(to be filled from `03c_ft_backbone/`)_

### Task 3d - Optimizer (AdamW / SGD / LBFGS)

_(to be filled from `03d_optimizer/optimizer_comparison.json`)_

## Observations

_(filled after phases complete)_

## Interpretation, alternatives, limitations

_(filled after phases complete)_

## Decision

_(filled after phases complete)_

## Next action

_(filled after phases complete)_
