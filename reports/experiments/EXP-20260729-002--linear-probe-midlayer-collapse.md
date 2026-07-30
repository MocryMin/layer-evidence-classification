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

Re-cached with the pure-utterance prompt `{utterance}` (no `Classify the intent:`
prefix) and repeated the stats + plain probe (`03b_no_prompt/`).

Inter-sample std, with-prompt -> no-prompt (selected layers):

| layer | with-prompt | no-prompt |
|------:|------------:|----------:|
| 1  | 0.00364 | 0.00654 |
| 6  | 0.00020 | 0.00028 |
| 9  | 0.00117 | 0.00206 |
| 12 | 0.02035 | 0.02669 |

Plain linear probe best val acc, with-prompt -> no-prompt:

| layer | with-prompt | no-prompt |
|------:|------------:|----------:|
| 1  | 0.135 | 0.268 |
| 6  | 0.027 | 0.076 |
| 9  | 0.583 | 0.736 |
| 12 | 0.789 | 0.803 |

**Finding**: removing the instruction prefix slightly *increases* inter-sample std
and slightly improves the plain probe (layers 1, 9 notably), but the collapse
pattern is unchanged - layers 4-8 still fall to 0.04-0.22. The prompt is not the
root cause; it marginally worsens the collapse. The mid-layer CLS anisotropy is an
intrinsic property of the frozen DeBERTa-v3-base backbone, not an artefact of the
input framing.

### Task 3c - Fine-tuned backbone

Full fine-tune (backbone + last-layer CLS head, AdamW lr=2e-5, 5 epochs, bs=32),
saved to `models/deberta-v3-base-clinc150-ft/`, re-cached, re-probed
(`03c_ft_backbone/`). FT history: val/test acc 0.910/0.902 -> 0.961/0.956 ->
0.963/0.960 -> 0.970/0.963 -> **0.973/0.967**.

FT-backbone feature stats (inter-sample std / class-signal ratio, frozen -> FT):

| layer | inter_std (frozen -> FT) | class-signal (frozen -> FT) |
|------:|--------------------------|-----------------------------|
| 6  | 0.00020 -> 0.00020 | 0.19 -> 0.46 |
| 8  | 0.00022 -> 0.00044 | 0.36 -> 3.00 |
| 11 | 0.0245 -> 0.210 | 0.28 -> 7.33 |
| 12 | 0.0203 -> 0.875 | 0.37 -> 12.84 |

Plain / LN probe on the FT backbone (val acc), with frozen-backbone values for
reference:

| layer | plain(FT) | ln(FT) | plain(frozen) | ln(frozen) |
|------:|----------:|-------:|--------------:|-----------:|
| 1  | 0.142 | 0.813 | 0.135 | 0.806 |
| 3  | 0.527 | 0.895 | 0.437 | 0.876 |
| 6  | 0.072 | 0.931 | 0.027 | 0.651 |
| 7  | 0.574 | 0.966 | 0.064 | 0.901 |
| 8  | 0.481 | 0.969 | 0.028 | 0.831 |
| 9  | 0.970 | 0.969 | 0.583 | 0.844 |
| 12 | 0.969 | 0.972 | 0.789 | 0.870 |

**Findings**:
- FT last layer reaches 0.967 test acc - as predicted, far above 0.90, so the test
  set has only ~150 final-layer errors: class-wise recoverability (R_{l,c}) is
  statistically starved. The frozen-backbone regime is where recoverability is
  meaningful.
- FT does **not** fix the mid-layer CLS variance collapse: inter_sample_std for
  layers 1-8 stays at 0.0002-0.004. FT concentrates task signal in the upper
  layers (layer 11/12 inter_std and class-signal explode) and raises the mid-layer
  class-signal *ratio* (layer 8: 0.36 -> 3.0) without raising the absolute
  variance.
- Plain probe on FT backbone: upper layers (9-12) become excellent (~0.97), mid
  layers (1-8) stay collapsed (0.07-0.57) - the ill-conditioning persists.
- LN on the FT backbone rescues every layer to 0.79-0.97 (layer 6: 0.651 -> 0.931,
  layer 8: 0.831 -> 0.969): FT strengthened the mid-layer signal and LN extracts it.
- With FT+LN, layers 7-12 all sit at ~0.97 - layer-wise differences (the substrate
  of H1'/H2) largely vanish near ceiling.

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

1. **The collapse is real and structural.** Frozen DeBERTa-v3-base mid-layer CLS
   representations have inter-sample std of 0.0002-0.006 (vs 0.020-0.025 at the
   final layer), with a dominant near-constant component (within-sample std
   0.07-0.24, participation ratio 1-3.5 for layers 1-6). The CLS vector has a
   fixed "shape" that is nearly identical across samples; the class signal lives
   in tiny per-sample deviations.

2. **But the signal is present and linearly extractable.** LBFGS (plain linear
   head, no feature transform) fits every layer to 0.41-0.86 (layer 1: 0.135 ->
   0.789; layer 6: 0.027 -> 0.410). The AdamW/SGD failure is therefore an
   optimisation/ill-conditioning artefact, not absence of linear signal.

3. **It is a conditioning problem with two independent fixes.** The near-constant
   component makes the linear-probe loss landscape badly conditioned: the
   class-discriminative direction has vanishing gradient relative to the constant
   direction. Two routes resolve it:
   - *Second-order optimiser* (LBFGS) - navigates the ill-conditioned landscape
     directly, no feature transform.
   - *LayerNorm head* (normalise + affine) - reshapes features into a
     well-conditioned space, so first-order AdamW works.
   First-order methods without conditioning fixes (AdamW/SGD plain,
   norm-only, affine-only) fail on the severely collapsed layers.

4. **LN's benefit is normalisation + affine in synergy, not either alone.**
   norm-only (per-sample normalise, no affine) fails because per-sample
   normalisation does not remove the *cross-sample* constant. affine-only
   (per-dim rescale, no normalise) rescues mildly-collapsed layers but not
   severely-collapsed ones (cannot amplify the 0.0002-scale signal). Full LN does
   both: normalisation amplifies the signal to a learnable scale, affine cancels
   the residual constant and selects dims.

5. **The collapse is intrinsic to the frozen backbone, not the input framing.**
   Removing the instruction prefix marginally raises inter-sample std and helps
   the plain probe a little (layer 1: 0.135 -> 0.268) but leaves the collapse
   pattern intact.

6. **FT does not repair the mid-layer collapse but concentrates signal upstream.**
   FT drives the last layer to 0.967 test acc and explodes layers 11-12 variance,
   while mid-layer inter-sample std stays at 0.0002-0.004. FT raises the mid-layer
   class-signal *ratio* (extractable by LN to 0.93-0.97) but leaves the absolute
   variance - and hence the plain-probe collapse - intact.

7. **Mid layers can exceed the final layer once the probe fits.** With LBFGS,
   layer 3 (0.862) > layer 12 (0.812); with LN, layers 7/10 (0.90) > layer 12
   (0.87). The H1'/H2 regime that EXP-001 targets is only visible when the probe
   actually fits.

## Interpretation, alternatives, limitations

- The pure-linear-probe protocol (`linear_with_bias`, AdamW) in EXP-001 would
  produce a **false negative** for H1/H1'/H2: mid layers would appear
  unprobeable (~1-13%) not because they lack recoverable signal, but because
  first-order optimisation cannot navigate the ill-conditioned mid-layer CLS
  landscape. Any layer-comparison conclusion drawn from such a probe is an
  artefact of the optimiser, not of the representations.
- Two minimal remedies preserve the "frozen backbone + linear readout" spirit:
  (a) keep the plain linear head but switch the optimiser to **LBFGS**; (b) keep
  AdamW but add a **LayerNorm** before the linear head. Both make every layer
  fittable. LN is the cheaper integration (one extra module, AdamW unchanged);
  LBFGS is the purer "linear probe" (no nonlinearity) but is full-batch and
  slower, and is somewhat weaker on the most-collapsed layer (layer 6: 0.41 vs
  LN's 0.65 on the frozen backbone).
- A fixed cross-sample standardisation (linear, non-learned) is a third option
  that would keep the probe strictly linear; it was not run here and is left as a
  follow-up.
- Limitations: single seed (17) for the diagnostics; representative-layer grids
  for the lr search; LBFGS run for 30 epochs with default strong-Wolfe (may
  underperform a tuned schedule); FT is a single 5-epoch run. These suffice to
  establish the qualitative finding (collapse + conditioning, not absence of
  signal) but the exact numbers are point estimates.

## Decision

- **EXP-001 mainline must use a probe that actually fits.** Adopt **LN head +
  AdamW (lr=1e-2)** as the default probe for the recoverability experiment (all
  12 layers fittable on the frozen backbone; cheapest integration; mid layers
  become competitive with the final layer, enabling H1'/H2). Record the
  head-type change and lr in the resolved run config. Keep the frozen-backbone
  regime (not FT) so that enough final-layer errors remain for class-wise
  recoverability.
- The variance-collapse finding is recorded as a standalone methodological result
  (this report): *first-order linear probing of frozen transformer mid-layer CLS
  yields false negatives due to ill-conditioning; use second-order optimisation
  or feature normalisation.*

## Next action

1. Amend the EXP-001 protocol: `head_type: layernorm + linear_with_bias`,
   `learning_rate: 1e-2` (frozen-backbone), and re-run the (LN-head) lr smoke test
   to confirm 1e-2 is selected.
2. Proceed with the 12-layer × 10-seed frozen-backbone recoverability run using
   the LN head, then compute R_l / R_{l,c} / oracle / D_JS and the H1/H1'/H2
   judgements.
3. (Optional follow-up) test fixed cross-sample standardisation as a strictly
   linear alternative to LN.
