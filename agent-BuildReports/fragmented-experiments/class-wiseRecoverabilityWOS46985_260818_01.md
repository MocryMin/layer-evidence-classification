# class-wiseRecoverabilityWOS46985_260818_01 - layer + class-wise recoverability (EXP-001 defs) on WOS-46985 baselines

Date: 2026-08-18 · Group: `user_exp_plans/fragmented_exp_gr3.md` (step 1) · Reporting model: deepseek-v4-flash · Git: `9a6cc9d` (report analysis; inputs from gr1 baselines at `32b4ac1`) · Single seed 17

## Config

- Inputs: gr1 baselines `DeBERTaV3BaseWOS46985Baseline_260812_04` (12 layers) and `ModernBERTBaseWOS46985Baseline_260812_05` (22 layers), **ridge family**, test split (9397, 134 L2 classes).
- All quantities recomputed from `ridge_test_pred.npy` + test labels and **cross-checked exactly** against the baselines' `results.json` (R_l, H_l, R_oracle, acc_oracle, D_JS; script `scripts/gr3_classwise_recoverability.py`).
- Definitions (EXP-001): `R_l = P(ŷ_l=y | ŷ_L≠y)`, `H_l = P(ŷ_l≠y | ŷ_L=y)`, `R_{l,c} = P(ŷ_l=y | y=c, ŷ_L≠c)`, `H_{l,c}` symmetric on harm side, `R_oracle = P(∃l<L: ŷ_l=y | ŷ_L≠y)`, `D_JS = JS(e_c‖r_c)` (eq. 12-16). "mid" = layers 1..L-1.
- L1 domain per L2 class recovered from the raw parquet (`label_description`); 3 L2 classes are cross-listed to 2 L1 domains (primary used).

## 1. DeBERTa-v3-base (final layer acc 0.3003, 6575 test errors)

| l | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|----|----|
| R_l | .347 | .354 | .283 | .308 | **.355** | .335 | .290 | .270 | .230 | .194 | .146 |
| H_l | .244 | .219 | .216 | .196 | **.170** | .171 | .186 | .175 | .180 | .247 | .228 |

- Oracle: acc 0.7382, gain +0.4379, **R_oracle 0.6259** (4115/6575); D_JS 0.0359.
- Best single mid layer (L5) recovers 0.355/0.626 = **57% of oracle** - no single layer dominates; recoverability is spread across the stack.
- Class-wise: R_max≥0.5 for **61/134** classes, ≥0.8 for 8; coverage 134/134. Argmax-layer histogram concentrates early-mid: L1×38, L2×24, L5×28, others ≤15; **no class has its best recoverer at L10-L11**.
- Top classes: Emergency Contraception 0.88 (L1), Suspension Bridge 0.86 (L1), Overactive Bladder 0.82 (L1), False memories 0.82 (L5), Atrial Fibrillation 0.82 (L4). Bottom (n_err≥10): Kidney Health 0/24, Voltage law 0/15, Structured Storage 0/13, Senior Health 0/15, Anxiety 0.02/43.
- corr(n_err_c, R_max) = −0.15 (Pearson) / −0.20 (Spearman): classes with more final-layer errors are *not* more recoverable.
- L1 domains (R_oracle / weighted-best layer): Civil **0.756** (L5, weighted R 0.514) > biochemistry 0.685 (L5) ≈ Psychology 0.660 (L5) ≈ MAE 0.659 (L5) > CS 0.611 (L6) > ECE 0.582 (L1) ≈ Medical 0.585 (L2, weighted 0.380). Medical is the weakest-yet-largest error mass (2368 errors).

## 2. modernBERT-base (final layer acc 0.4858, 4832 test errors)

| l | 1 | 2 | 3 | 4 | 5 | 6 | ... | 20 | 21 |
|---|---|---|---|---|---|---|-----|----|----|
| R_l | **.223** | .177 | .148 | .127 | .106 | .099 | ↓ | .067 | .058 |
| H_l | **.186** | .234 | .310 | .324 | .391 | .410 | ∪ | .232 | .221 |

- Oracle: acc 0.7116, gain +0.2258, R_oracle 0.4392 (2122/4832); D_JS 0.0484.
- **R_l is strictly monotonically decreasing in depth** - layer 1 is the strongest recoverer for the stack overall (argmax L1 for **87/134** classes) and also the least harmful (H_1 0.186; harm peaks ~0.41 at L5-6). Consistent with gr1: modernBERT's signal lives in its early layers.
- Class-wise: R_max≥0.5 only **12/134**, ≥0.8 for 0 - far more concentrated-low than DeBERTa. Top: Kidney Health 0.78 (L1), Hereditary Angioedema 0.73 (L1), Skin Care 0.71 (L1).
- Domains: biochemistry 0.580 > Civil 0.494 ≈ MAE 0.472 ≈ Psychology 0.467 > ECE 0.430 > CS 0.388 ≈ Medical 0.390.

## 3. Cross-model

- Per-class R_max correlation (n_err≥10): **+0.626** - substantially shared, partially complementary.
- Largest flips toward modernBERT: **Kidney Health** (deb 0.000 -> mb 0.783), Senior Health (0 -> 0.13), Electrical generator (0.48 -> 0.58). Toward DeBERTa: Crohn's Disease (0.76 vs 0.18), Remote Sensing (0.71 vs 0.19), Overactive Bladder (0.82 vs 0.31), False memories (0.82 vs 0.33).
- Mean mid-layer harm: DeBERTa 0.203 vs modernBERT 0.324 - modernBERT's mid layers break correct finals more while recovering less (outside L1).

## 4. Artifacts

`artifacts/fragmented-experiments/class-wiseRecoverabilityWOS46985_260818_01/`: `analysis.json` (full R_lc/H_lc unsimplified fractions per EXP-001 convention, per-class n/n_err/n_rec, R_max, argmax, domains), `{deberta,modernbert}_classwise.csv`.

## 5. Observations

- _Data collection point - no hypothesis; observations are supported by the outputs above._
- On WOS-46985 (134-class, acc ~0.30/0.49) the per-class error budget is large enough for stable class-wise fractions (mean ~49 errors/class DeBERTa, ~36 modernBERT) - the CLINC sparsity limitation does not apply.
- The two backbones show qualitatively different recoverability geometry: DeBERTa has a mid-layer sweet spot (L5 best recoverer AND least harm, R_oracle 0.63 spread across the stack), modernBERT has a strict early-layer gradient (L1 dominates; deeper layers both recover less and harm more).
- Recoverability is broad, not concentrated (D_JS 0.036/0.048), and is not explained by error mass (negative corr with n_err). Domain structure matters: Civil/biochemistry recover best on both backbones; Medical - the largest error mass - recovers worst on both.
- Per-class complementarity across backbones (corr 0.63, hard flips like Kidney Health 0->0.78) suggests recoverability is partly a function of class, not only of backbone geometry.
