# Project Status

## Current state

EXP-20260729-001 (mainline recoverability run) is **paused**: its pure-linear-probe
protocol was found to give false negatives on mid layers. Diagnostic campaign
EXP-20260729-002 characterised the cause and is **complete** (report at
`reports/experiments/EXP-20260729-002--linear-probe-midlayer-collapse.md`).

Key finding: frozen DeBERTa-v3-base mid-layer CLS has near-zero inter-sample
variance (a near-constant component), so a plain linear probe under first-order
AdamW/SGD collapses to near-random - but the class signal IS linearly
extractable. The collapse is an ill-conditioning/optimisation artefact, not
absence of signal. **Task 03g (RidgeClassifier α grid) upgraded this:** a
closed-form least-squares solve at α≈1e-6 (OLS) recovers mid layers to 0.917
(L6) / ≈0.94 (L7-10), beating every CE probe tried (LBFGS 0.41, LN+AdamW 0.65,
plain AdamW 0.03) - so the "collapse" is a probe-methodology artefact, not a
feature property. The prior 03f "Ridge fails" was an α-mis-scaling artefact
(only tried α≥0.1). Prompt (with/without instruction) barely matters at α=1e-6.

## Latest valid result

EXP-002 diagnostics (seed 17, frozen backbone unless noted), validation accuracy:
- Plain linear probe (AdamW, best lr 1e-2): layers 1/6 fail (0.13/0.03), layer 12 = 0.79.
- LBFGS plain probe (30 epochs): every layer 0.41-0.86 (layer 6: 0.03 -> 0.41; layer 3 > layer 12).
- LN head (AdamW, 1e-2): every layer 0.65-0.90 (layer 6: 0.65; layers 7/10 > layer 12).
- norm-only fails on mid layers; affine-only rescues mild layers but not severe ones.
- **RidgeClassifier α=1e-6 (≈OLS, task 03g): every layer 0.79-0.94 (layer 6:
  0.917; layers 7-10 ≈0.94 > layer 12 ≈0.90).** Best linear accuracy on the
  frozen base. α=10 fails (over-regularised); best α is 1e-6 for all mid layers.
  Prompt-independent (with-prompt 0.917 vs no-prompt 0.912 at L6).
- Removing the instruction prompt does not fix the collapse (intrinsic to the backbone).
- Full-FT backbone: last layer 0.967 test acc (too few errors for class-wise
  recoverability); FT does not fix mid-layer collapse but LN on the FT backbone
  reaches 0.93-0.97 on mid layers.

## Active blockers

None for the diagnostics. EXP-001 mainline needs a protocol amendment before
resuming (see decision). **Probe choice reopened by 03g**: OLS gives the best
mid-layer accuracy but no calibrated probabilities (the recoverability metrics
NLL/ECE/D_JS need a softmax-CE probe). Candidate: OLS-init + CE fine-tune (untested).

## Next action

Resolve the probe choice (user decision): (a) OLS-init + CE fine-tune, (b) CE
probe run to convergence (more LBFGS epochs / from OLS init) to test whether CE
closes the gap to OLS, or (c) fall back to LN head + AdamW (lr=1e-2, fits all
layers but understates mid-layer recoverability at L6: 0.65 vs OLS 0.917). Then
re-run the probe-specific lr smoke test and proceed with the 12-layer x 10-seed
recoverability run and H1/H1'/H2 judgement.

## Important paths

- `reports/experiments/EXP-20260729-002--linear-probe-midlayer-collapse.md` - diagnostic report
- `artifacts/EXP-20260729-002/` - diagnostic data (gitignored; README documents layout)
- `artifacts/EXP-20260729-001/cache/` - shared frozen-backbone CLS cache
- `configs/diag_config.yaml` - EXP-002 config; `configs/exp_config.yaml` - EXP-001 config
- `models/deberta-v3-base-clinc150-ft/` - FT backbone (task 3c, gitignored)
- `plans/EXP-20260729-001--*/` - EXP-001 RP log + AgentProtocol
