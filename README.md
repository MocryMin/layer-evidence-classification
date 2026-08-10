# Layer Evidence Classification

Research code for studying intermediate-layer evidence and corrective
utility in text classification.

## Status

Early-stage research prototype. Research questions, methods, and
experimental conclusions are subject to change.

## Environment

See the local global environment documentation and tracked project
configuration.

## Reproducibility

Experiments are tracked with Git, configuration files, MLflow, and
structured experiment reports.

## Frozen artifacts (EXP-001 / EXP-002, 2026-08-10)

Code, configs, reports, tests, and small summary metrics are frozen at Git
tag `exp-001-002-freeze-20260810`. Full experiment artifacts (result JSONs,
the fine-tuned backbone checkpoint, and a frozen MLflow `mlruns.db` snapshot)
are archived in a private HuggingFace dataset:

- **HF dataset:** https://huggingface.co/datasets/MocryMin/lec-exp-001-002-freeze (private)
- **Freeze tag:** `exp-001-002-freeze-20260810`
- **Manifest:** `agent-BuildReports/freeze-20260810/manifest.md` (full file list with sha256 + sizes)

Rebuildable CLS safetensors caches (1.66 GB) are excluded from the archive;
rebuild them via `scripts/cache_hidden.py`. Each cache set has a
`cache_manifest.json` (archived) with sha256 to verify rebuilds.