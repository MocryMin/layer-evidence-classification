# Project Status

## Current state

Repository initialised and pushed to `origin/main`. Pilot model
(`deberta-v3-base`), transfer model (`Qwen3-Embedding-0.6B`), and
dataset (CLINC150 / `clinc_oos`) are downloaded and verified loadable.
`sentencepiece==0.2.2` installed for the deberta tokenizer. No
experiment has been run yet.

## Latest valid result

None (no training/evaluation yet).

## Active blockers

None.

## Next action

Define the first experiment config and a smoke-test training/eval run
on CLINC150 using `deberta-v3-base` as the pilot encoder.

## Important paths

- `AGENT_PROTOCOL.md` - stable workflow rules
- `PROJECT_RESOURCES.md` - registered models and datasets
- `requirements.txt` - project-specific dependencies
- `models/deberta-v3-base/` - pilot encoder (gitignored)
- `models/Qwen3-Embedding-0.6B/` - transfer embedder (gitignored)
- `data/raw/clinc_oos/` - CLINC150 parquet (gitignored; label 42 = OOS)
