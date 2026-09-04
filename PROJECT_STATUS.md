# Project status

Last audited: 2026-09-04 (Asia/Shanghai)

## Main research cycle

EXP-001--004 are closed as the first proposal-building research cycle. “Closed”
means that no further computation is currently planned for those experiment
packages; it does not make every result equally confirmatory.

| Study | Operational status | Evidence maturity | Canonical record |
|---|---|---|---|
| EXP-001 | complete; protocol superseded | historical | `research-logs/EXP-001.md` |
| EXP-002 | complete | diagnostic | `research-logs/EXP-002.md` |
| EXP-003 | complete | confirmatory within DeBERTa-v3-base × CLINC150 | `research-logs/EXP-003.md` |
| EXP-004 H1 | computation complete | train-discovery existence evidence; cross-split confirmation not run | `agent-BuildReports/experiments/EXP-20260828-004-H1-agent-report.md` |
| EXP-004 H2 | complete | confirmatory within DeBERTa-v3-base × CLINC150 | `agent-BuildReports/experiments/EXP-20260831-004-H2-agent-report.md` |

The public-facing interpretation is controlled by
`docs/EVIDENCE_INDEX.md`. Historical logs are preserved, and known wording or
status discrepancies are documented in `docs/ERRATA.md` rather than silently
rewritten.

## Headline results

- EXP-002: strong class signal remains linearly extractable from compressed
  DeBERTa-v3-base intermediate CLS features; the original near-chance AdamW
  result was an optimization/readout false negative.
- EXP-003: all three accepted probe families support intermediate-layer
  non-inferiority, superiority, and recovery of final-layer errors. Oracle
  gains are 0.115 (centered plain), 0.143 (LN plain), and 0.097 (ridge).
- EXP-004 H1: 586 unique discovery-good paths and 150 unique adaptive
  discovery-stage readability-gap witnesses were identified after exact path
  deduplication. `repeat_L28` is an independent strong control witness. These
  counts are not an IID prevalence estimate and were not cross-split confirmed.
- EXP-004 H2: rank-reward MCTS achieved `R_short=0.9471` (Wilson lower 0.9397)
  and `R_recov=0.8735` (Wilson lower 0.8420) on 4,500 untouched test samples;
  both preregistered lower-bound thresholds were 0.10. The same-budget random
  control was also strong, limiting any claim that MCTS is necessary.

## Record integrity

- Local Git `main` contains the finished EXP-004 implementation and reports.
- The reviewed GitHub evidence layer is public at source tag
  `exp-001-004-evidence-v1` (commit
  `6081e8ecf316400e3b69e820be0c557c7457a763`). Anonymous access was verified
  on 2026-09-04.
- The curated HF dataset `MocryMin/lec-exp-001-004-evidence` is public at
  revision `exp-001-004-evidence-v1` (HF commit
  `7aa5ed568738ad006808830387b85e5a9eebb50e`). Anonymous access was verified
  for the dataset card, artifact root, manifest, and a representative artifact.
- The local MLflow database contained 19 historical runs at the start of this
  audit. Three abandoned early runs have stale `RUNNING` state. They are retained
  as historical records and excluded from canonical-run exports rather than
  retrospectively rewritten. Two missing H1 discovery records were subsequently
  registered from their frozen artifacts, bringing the ledger to 21 runs.
- Full local artifacts occupy approximately 272 GB. The majority is rebuildable
  Llama prefix-state caching, not evidence that should be uploaded wholesale.
- A scan of the full pre-release Git history and the current candidate tree
  found no HF/GitHub/AWS credential or private-key pattern. A publication
  review is still required for every new release.

## Immediate next phase

1. Freeze the first 2--4 page LaTeX research proposal using the claim IDs in
   `docs/EVIDENCE_INDEX.md`.
2. Begin first-round supervisor outreach using the RP as the entry point and
   this repository as optional supporting material.
3. Keep future experiments under new experiment IDs and publish them as new
   versioned evidence releases rather than moving the EXP-001--004 anchors.
