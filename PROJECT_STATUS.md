# Project Status

## Current state

EXP-20260729-001 (mainline recoverability run) is **paused**: its pure-linear-probe
protocol was found to give false negatives on mid layers. Diagnostic campaign
EXP-20260729-002 characterised the cause and is **complete** (report at
`reports/experiments/EXP-20260729-002--linear-probe-midlayer-collapse.md`).

Key finding: frozen DeBERTa-v3-base mid-layer CLS has near-zero inter-sample
variance (a near-constant component), so a plain linear probe under first-order
AdamW/SGD collapses to near-random - but the class signal IS linearly
extractable (LBFGS or a LayerNorm head recovers it). The collapse is an
ill-conditioning/optimisation artefact, not absence of signal.

## Latest valid result

EXP-002 diagnostics (seed 17, frozen backbone unless noted), validation accuracy:
- Plain linear probe (AdamW, best lr 1e-2): layers 1/6 fail (0.13/0.03), layer 12 = 0.79.
- LBFGS plain probe: every layer 0.41-0.86 (layer 6: 0.03 -> 0.41; layer 3 > layer 12).
- LN head (AdamW, 1e-2): every layer 0.65-0.90 (layer 6: 0.65; layers 7/10 > layer 12).
- norm-only fails on mid layers; affine-only rescues mild layers but not severe ones.
- Removing the instruction prompt does not fix the collapse (intrinsic to the backbone).
- Full-FT backbone: last layer 0.967 test acc (too few errors for class-wise
  recoverability); FT does not fix mid-layer collapse but LN on the FT backbone
  reaches 0.93-0.97 on mid layers.

## Active blockers

None for the diagnostics. EXP-001 mainline needs a protocol amendment before
resuming (see decision).

## Next action

Amend EXP-001 protocol to `head_type: layernorm + linear_with_bias`,
`learning_rate: 1e-2` (frozen backbone), re-run the LN-head lr smoke test, then
proceed with the 12-layer x 10-seed recoverability run and H1/H1'/H2 judgement.
Optional follow-up: test fixed cross-sample standardisation as a strictly-linear
probe alternative.

## Important paths

- `reports/experiments/EXP-20260729-002--linear-probe-midlayer-collapse.md` - diagnostic report
- `artifacts/EXP-20260729-002/` - diagnostic data (gitignored; README documents layout)
- `artifacts/EXP-20260729-001/cache/` - shared frozen-backbone CLS cache
- `configs/diag_config.yaml` - EXP-002 config; `configs/exp_config.yaml` - EXP-001 config
- `models/deberta-v3-base-clinc150-ft/` - FT backbone (task 3c, gitignored)
- `plans/EXP-20260729-001--*/` - EXP-001 RP log + AgentProtocol
