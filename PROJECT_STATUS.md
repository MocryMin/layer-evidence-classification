# Project Status

## Current state

Repository initialised and pushed to `origin/main`. Pilot model
(`deberta-v3-base`), transfer model (`Qwen3-Embedding-0.6B`), and
dataset (CLINC150 / `clinc_oos`) are downloaded and verified loadable.
No experiment has been run yet.

## Latest valid result

None (no training/evaluation yet).

## Active blockers

- `sentencepiece` is not installed in the shared `ai-env`. The
  `deberta-v3-base` tokenizer (`spm.model`) cannot be instantiated
  without it. Model weights themselves load fine. Installing it needs
  explicit user approval per `AGENT_PROTOCOL.md` §3.

## Next action

Decide whether to install `sentencepiece` (or use a tokenizer that does
not need it), then define the first experiment config and a smoke-test
training/eval run. Record resources in `PROJECT_RESOURCES.md` (done).

## Important paths

- `AGENT_PROTOCOL.md` - stable workflow rules
- `PROJECT_RESOURCES.md` - registered models and datasets
- `models/deberta-v3-base/` - pilot encoder (gitignored)
- `models/Qwen3-Embedding-0.6B/` - transfer embedder (gitignored)
- `data/raw/clinc_oos/` - CLINC150 parquet (gitignored; label 42 = OOS)
