---
pretty_name: Layer Evidence Classification — EXP-001–004 evidence
tags:
- reproducibility
- interpretability
- early-exit
- dynamic-depth
- mlflow
---

# Layer Evidence Classification: EXP-001–004 evidence

This is a curated audit package for the research record at
https://github.com/MocryMin/layer-evidence-classification, frozen at Git tag
`exp-001-004-evidence-v1`.

It is not a training dataset. It contains author logs, objective experiment
reports, frozen configurations, selected aggregate or audit-level result files,
an exported view of canonical MLflow runs, and a SHA-256 manifest.

Repository artifact paths are preserved at the dataset root. The stable roots
are recorded in `ARTIFACT_ROOTS.json`; every artifact pointer in the four
author experiment logs is normalized by `ARTIFACT_POINTERS.json`.

## Supported claims

The exact current claims and limitations are defined in
`docs/EVIDENCE_INDEX.md`. In brief, the package supports:

- diagnosis of probe/readout false negatives in intermediate representations;
- confirmatory intermediate-layer recoverability results in
  DeBERTa-v3-base × CLINC150;
- train-discovery evidence for path-specific readability gaps in
  Llama-3.2-3B-Instruct × ARC-Easy;
- confirmatory sample-wise fixed-head path-search results in
  DeBERTa-v3-base × CLINC150.

## Deliberate exclusions

The package does not redistribute upstream datasets, model weights, fine-tuned
checkpoints, raw hidden states, per-sample logits/predictions, Llama prefix
caches, per-path heads, the live MLflow database, or the 2.7 million H2 raw
simulation records. These remain rebuildable or privately archived.

## Verification

`MANIFEST.json` records the repository-relative source, release path, role,
split sensitivity, byte size, SHA-256 digest, immutable HF URI, and web URL of
every included file.
`MANIFEST.sha256` can be checked with a standard SHA-256 utility.

## Evidence status

“EXP-001–004” identifies the completed first research cycle, not a uniform
claim of confirmatory maturity. EXP-004 H1 is explicitly labelled
discovery/preliminary because official validation/test confirmation was not
run. EXP-004 H2 is confirmatory within the stated single-model/task design.

## License

No reuse license has yet been granted for the project-authored material. The
files are provided for public research review. Upstream models and datasets are
not included and remain governed by their own terms.
