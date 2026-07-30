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

Plain linear probe, AdamW, 100 epochs, seed 17, validation accuracy. Best lr per
representative layer (full curve in `01_lr_grid/lr_grid_plain_adamw.json`):

| layer | best lr | best val acc | note |
|------:|--------|-------------:|------|
| 1  | 1e-2 | 0.135 | near-random (150 classes -> 1/150=0.0067) |
| 3  | 1e-2 | 0.437 | weak |
| 6  | 1e-3 | 0.028 | random |
| 9  | 1e-2 | 0.583 | mediocre |
| 12 | 1e-2 | 0.789 | best, but still undertrained |

Layer 12 lr curve: 1e-6->0.007, 1e-5->0.076, 1e-4->0.329, 1e-3->0.607,
1e-2->0.789, 1e-1->0.557, 3e-1->0.166, 1->0.046, 3->0.028, 10->0.026 (peaks at
1e-2, then overshoots). Layer 6 stays at 0.007-0.028 for lr 1e-6..3e-1.

**Finding**: mid layers (1, 6) stay near-random at every lr; the collapse is
lr-invariant. The best global lr is 1e-2 (used for the ablations below). Even at
its best lr the final layer only reaches 0.789 and is still climbing at epoch 100,
i.e. undertrained - the plain probe is structurally hampered, not just mis-tuned.

### Task 3a - LN ablation (plain / ln / norm_only / affine_only)

All heads, AdamW, lr=1e-2, 100 epochs, seed 17, validation accuracy
(`03a_ln_ablation/ln_ablation_lr1e-2.json`):

| layer | plain | ln | norm_only | affine_only | inter-sample std |
|------:|------:|-----:|----------:|------------:|-----------------:|
| 1  | 0.135 | 0.806 | 0.244 | 0.728 | 0.00364 |
| 2  | 0.237 | 0.836 | 0.594 | 0.745 | 0.00127 |
| 3  | 0.437 | 0.876 | 0.686 | 0.848 | 0.00611 |
| 4  | 0.081 | 0.730 | 0.164 | 0.488 | 0.00055 |
| 5  | 0.124 | 0.776 | 0.257 | 0.586 | 0.00055 |
| 6  | 0.027 | 0.651 | 0.043 | 0.119 | 0.00020 |
| 7  | 0.064 | 0.901 | 0.270 | 0.642 | 0.00028 |
| 8  | 0.028 | 0.831 | 0.093 | 0.332 | 0.00022 |
| 9  | 0.583 | 0.844 | 0.804 | 0.838 | 0.00117 |
| 10 | 0.174 | 0.897 | 0.412 | 0.777 | 0.00050 |
| 11 | 0.801 | 0.867 | 0.818 | 0.870 | 0.02454 |
| 12 | 0.789 | 0.870 | 0.791 | 0.868 | 0.02035 |

**Findings**:
- `plain` collapses exactly on the low-inter-sample-std layers (4-8, 10): 0.03-0.17.
- `ln` works on **every** layer (0.65-0.90), including the most collapsed (layer 6: 0.651, layer 8: 0.831).
- `norm_only` (normalise, no affine) tracks `plain` - it fails wherever `plain` fails. Per-sample
  normalisation does not remove the cross-sample constant, so it does not fix the conditioning.
- `affine_only` (gamma*x+beta, no normalise) is a reparameterised linear classifier; it rescues the
  *mildly* collapsed layers (1,2,3,9,10,11,12: 0.73-0.87) but still fails on the *severely* collapsed
  ones (4,5,6,7,8: 0.12-0.64). Per-dim rescaling partially fixes conditioning but cannot amplify the
  tiny signal at layers 4-8.
- With `ln`, mid layers 7 (0.901) and 10 (0.897) exceed the final layer 12 (0.870) - i.e. the
  H1' / H2 regime that EXP-001 wants to test only becomes visible once the probe can actually fit.

The earlier layer-6-only ablation overstated affine_only's failure: affine_only fails on the *most*
collapsed layer (6) but works on milder layers; full LN is the only AdamW variant that works on all.

### Task 3b - No-instruction prompt

_(to be filled from `03b_no_prompt/`)_

### Task 3c - Fine-tuned backbone

_(to be filled from `03c_ft_backbone/`)_

### Task 3d - Optimizer (AdamW / SGD / LBFGS)

Plain linear probe, 100 epochs (LBFGS: 30 epochs, full-batch, strong-Wolfe),
seed 17, validation accuracy (`03d_optimizer/optimizer_comparison.json`):

| layer | AdamW (1e-2) | SGD (1e-2) | SGD (1e-1) | LBFGS (1.0) |
|------:|-------------:|-----------:|-----------:|------------:|
| 1  | 0.135 | 0.018 | 0.012 | **0.789** |
| 3  | 0.437 | 0.022 | 0.014 | **0.862** |
| 6  | 0.027 | 0.007 | 0.007 | **0.410** |
| 9  | 0.583 | 0.013 | 0.009 | **0.831** |
| 12 | 0.789 | 0.035 | 0.021 | **0.812** |

**Findings**:
- SGD (non-adaptive first-order) fails on **every** layer, including the final layer (0.035) - worse
  than AdamW everywhere.
- AdamW (adaptive first-order) fails on mid layers (1, 6) but fits 9/12.
- LBFGS (quasi-Newton, second-order) fits **every** layer (0.41-0.86), rescuing mid layers
  dramatically (layer 1: 0.135 -> 0.789; layer 6: 0.027 -> 0.410).
- With LBFGS, layer 3 (0.862) > layer 12 (0.812) - H1' (mid superior) holds.

This is the decisive result: the mid-layer signal **is** linearly extractable. The plain-probe
collapse under AdamW/SGD is an **ill-conditioning / optimisation** artifact, not absence of signal.
LBFGS navigates the badly-conditioned landscape (dominated by the near-constant component) that
first-order methods cannot. Layer 6 - the most collapsed - is only partially recoverable by LBFGS
(0.41, vs LN's 0.65), consistent with it carrying the weakest linear signal.

## Observations

_(filled after phases complete)_

## Interpretation, alternatives, limitations

_(filled after phases complete)_

## Decision

_(filled after phases complete)_

## Next action

_(filled after phases complete)_
