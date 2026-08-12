# Project Status

## Current state

EXP-20260810-003 (validated-probe recoverability verification) is **complete**
(report at
`agent-BuildReports/experiments/EXP-20260810-003--validated-probe-recoverability.md`).
**$H_1$, $H_1'$, $H_2$ are all very strongly supported** across all three probe
families (centered plain, LN plain, ridge). Mid layers are non-inferior to,
superior to, and recoverable over the final layer on frozen DeBERTa-v3-base /
CLINC150. The EXP-001 mainline pause is lifted; its verification is complete.

EXP-20260729-001 (original plain-probe mainline) is **superseded** by EXP-003:
its pure-linear-probe protocol gave false negatives on mid layers (optimisation
collapse). EXP-20260729-002 diagnosed the cause and is **complete** (report at
`agent-BuildReports/experiments/EXP-20260729-002--linear-probe-midlayer-collapse.md`).

Key finding: frozen DeBERTa-v3-base mid-layer CLS has near-zero inter-sample
variance (a near-constant component), so a plain linear probe under first-order
AdamW/SGD collapses to near-random - but the class signal IS linearly
extractable. The collapse is an ill-conditioning/optimisation artefact, not
absence of signal. **Task 03g (RidgeClassifier α grid)** showed a closed-form
least-squares solve at α≈1e-6 (OLS) recovers mid layers to 0.917 (L6) / ≈0.94
(L7-10), beating every CE probe tried - so the "collapse" is a probe-methodology
artefact, not a feature property. **Task 03h (Adam ablation)** decomposed the
AdamW failure into four factors (none fundamental): mini-batch noise (dominant;
batch=256 fails for every init/wd), wd=0.01 (caps acc at ≈0.55), no early
stopping + overparameterised head (CE peaks at ep8 then overfits), and slow
convergence from Xavier. **OLS-init + full-batch CE + early stop peaks at 0.919
@ep8** (above OLS, with calibrated probabilities) - the tested probe for EXP-001.

## Latest valid result

**EXP-003 (validated probes, 10 seeds, test accuracy):**
- **Centered plain (primary, candidate L11):** L11 0.862 > L12 0.852
  (d2=+0.009, CI [+0.008,+0.010]); oracle gain 0.115, R_oracle 0.78, D_JS 0.020.
  92% of runs non-converged (hit 1000ep); signal still clear.
- **LN plain (control, candidate L10):** L10 0.875 > L12 0.839
  (d2=+0.036, CI [+0.033,+0.039]); oracle gain 0.143, R_oracle 0.89, D_JS 0.007.
- **Ridge (reference, candidate L10, α=0 OLS):** L10 0.926 > L12 0.885
  (d2=+0.041); oracle gain 0.097, R_oracle 0.84. L7-L10 all 0.925-0.928.
- **Cross-family verdict: H1/H1'/H2 all very_strong.** Recoverability is broad
  (D_JS 0.007-0.020), not concentrated in a few classes. L6 is the hardest mid
  layer (centered 0.295, LN 0.602, ridge 0.914).
- Artifacts: `artifacts/EXP-20260810-003/results.json` (gitignored; README
  documents layout). MLflow run `364bf897628d443189b1ae6f1288fc6e`.

EXP-002 diagnostics (seed 17, frozen backbone unless noted), validation accuracy:
- Plain linear probe (AdamW, best lr 1e-2): layers 1/6 fail (0.13/0.03), layer 12 = 0.79.
- **EXP-001 mainline plain probe (lr=1e-3, 10 seeds, 100ep - post-hoc, saved):**
  full 12-layer collapse curve. Mid layers 4-8 near-random (L6 0.026 ± 0.002,
  L8 0.035), L12 0.608 ± 0.003; small std => collapse is seed-robust. Test tracks
  val. (`artifacts/EXP-20260729-001/plain_probe_mainline/results.json`; reproduces
  `smoke_lr_results.json` on seed-17 {1,6,12}.)
- LBFGS plain probe (30 epochs): every layer 0.41-0.86 (layer 6: 0.03 -> 0.41; layer 3 > layer 12).
- LN head (AdamW, 1e-2): every layer 0.65-0.90 (layer 6: 0.65; layers 7/10 > layer 12).
- norm-only fails on mid layers; affine-only rescues mild layers but not severe ones.
- **Feature stats + acc per layer in one table:**
  `artifacts/EXP-20260729-002/02_variance_collapse/per_layer_collapse_summary.json`
  (inter_std / participation_ratio / top1 + plain-AdamW / LBFGS / OLS / LN acc).
- **RidgeClassifier α=1e-6 (≈OLS, task 03g): every layer 0.79-0.94 (layer 6:
  0.917; layers 7-10 ≈0.94 > layer 12 ≈0.90).** Best linear accuracy on the
  frozen base. α=10 fails (over-regularised); best α is 1e-6 for all mid layers.
  Prompt-independent (with-prompt 0.917 vs no-prompt 0.912 at L6).
- **Adam ablation (task 03h, layer 6):** mini-batch 256 fails for every init/wd
  (OLS-init 0.917->0.61 wd0 / 0.023 wd0.01; Xavier 0.03-0.04). Full-batch rescues:
  OLS-init + full-batch CE peaks **0.919 @ep8** (then overfits to 0.79 by 20k);
  Xavier + full-batch wd=0 climbs to 0.70 @20k (still slow), wd=0.01 plateaus 0.55.
- **MLP probe (task 04):** matched-parameter MLP (919r, r=128 ~= plain params)
  fails on EVERY layer incl. L12 (uniform-prediction collapse, all lr/r) - the
  constant CLS component + ReLU forms a uniform attractor. Dead-ReLU diagnosis:
  features are predominantly POSITIVE (L6: 85% dims mu-delta>0) - the pervasive
  dead ReLU (46% -> 100% within epoch 1, batch 29/58) is driven by shared
  logits across samples (|logits[0]-logits[1]|=0) -> aligned gradient, not by
  negative features. Centering the features (fixed linear transform) rescues:
  centered plain L6 0.317, centered MLP r=256 L6 0.737 / L7 0.913 > L12 - but
  OLS (0.917 L6) still wins. Collapse is NOT a linearity limitation.
- **Activation ablation (task 05):** none/relu/leaky/gelu/sigmoid all fail on
  raw features (L6 best 0.007-0.010, all r). relu/gelu dead-lock at uniform
  (loss=ln(150), neg=1.00); leaky stalls 0.8 within uniform (slope too weak);
  sigmoid (bounded-saturating) dead-locks at uniform on 8/12 layers, stalls
  4.86-5.54 on the rest (still chance acc) - all four activation families
  covered; 2-layer linear (no act) cannot dead-lock but diverges (loss
  7.9-47.6) and fails even at L12. The uniform attractor is NOT ReLU-specific
  - it is a property of the near-constant features; only the 1-layer
  bias-carrying head escapes it.
- **Token-position check (task 06): the compression is CLS-specific.** On a
  2000-sample subset (CLS values reproduce full-set stats), non-CLS tokens
  have healthy inter-sample std at every layer - at L6 (CLS 2e-4) all non-CLS
  positions are >= 0.11 (500-2000x larger, zero compressed positions). The
  mid-layer representation space is NOT collapsed; only the CLS pooling token
  is. Follow-up: mean-pooling readout may escape the collapse.
- Removing the instruction prompt does not fix the collapse (intrinsic to the backbone).
- Full-FT backbone: last layer 0.967 test acc (too few errors for class-wise
  recoverability); FT does not fix mid-layer collapse but LN on the FT backbone
  reaches 0.93-0.97 on mid layers.

## Freeze (2026-08-10)

EXP-001/002 evidence is frozen for reproducibility:

- **Git tag** `exp-001-002-freeze-20260810` (on `5ad36f4`): code, configs, reports,
  tests, small summaries, manifest. `main` advanced to `f55481b` (HF URI recorded).
- **HF private dataset** `MocryMin/lec-exp-001-002-freeze` (759 MB, 51 files): all
  artifact result JSONs, FT-backbone best checkpoint, frozen `mlruns.db` snapshot,
  manifest + dataset README. 1.66 GB rebuildable CLS caches excluded.
- **Manifest:** `agent-BuildReports/freeze-20260810/manifest.md` (sha256 + sizes).
- Per-sample predictions/logits, plots, and probe checkpoints were never produced
  (experiments record aggregate JSON metrics + per-epoch histories) - marked N/A.

## Convergence probe (2026-08-12, follow-up to EXP-003 §5)

Variable-max_ep probe (`scripts/exp003_convergence_probe.py`, artifacts under
`artifacts/EXP-20260810-003/convergence_probe/`): one long run per
(family, layer) at seed 17, max_ep=20000, early stop disabled; budgets
replayed from the full history.

**Result: all 24 runs converge within 20000 ep.** Centered plain 857-8954 ep,
LN 625-4057 ep (2-4x faster). Converged accuracies far exceed the 1000ep cap
(e.g. centered L6 0.295->0.762, L7->0.923 > L12 0.884; LN L6 0.602->0.881).
The 1000ep non-convergence is slow convergence (~10-20x beyond the cap for
mid layers), not a probe pathology; the §2 superiority verdict strengthens
with convergence. min_delta=1e-4/patience=100 early stopping is strict and
can cut off slow tail gains (centered L4 would stop at 8954 with ~0.72 vs
0.807 at 20000).

## Active blockers

None. EXP-003 is complete; $H_{1-2}$ are accepted (very strong). EXP-001's
plain-probe protocol is superseded - no further plain-probe runs needed.

## Next action

$H_{1-2}$ stand. Per the EXP-001 plan §3, the next steps (await user direction):
- Scale to `modernBERT-base` and `modernBERT-large` as future major targets.
- Consider non-CLS readouts (mean-pooling) - EXP-002 task 06 showed non-CLS
  tokens do not collapse at mid layers.
- `Qwen3-Embedding-0.6B` side verification (pending since EXP-001).

## Important paths

- `agent-BuildReports/experiments/EXP-20260810-003--validated-probe-recoverability.md` - EXP-003 report
- `artifacts/EXP-20260810-003/` - EXP-003 data (gitignored; README documents layout)
- `agent-BuildReports/experiments/EXP-20260729-002--linear-probe-midlayer-collapse.md` - diagnostic report
- `artifacts/EXP-20260729-002/` - diagnostic data (gitignored; README documents layout)
- `artifacts/EXP-20260729-001/cache/` - shared frozen-backbone CLS cache (reused by EXP-003)
- `configs/exp003_config.yaml` - EXP-003 config; `configs/diag_config.yaml` - EXP-002; `configs/exp_config.yaml` - EXP-001
- `models/deberta-v3-base-clinc150-ft/` - FT backbone (task 3c, gitignored)
- `plans/EXP-20260729-001--*/` - EXP-001 RP log + AgentProtocol
- `plans/EXP-20260810-003-*/` - EXP-003 plan + AgentProtocol
