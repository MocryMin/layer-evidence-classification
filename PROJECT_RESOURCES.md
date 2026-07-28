# Project Resources

Reusable models, datasets, and other resources for this project. Check
this file before downloading anything from the internet. Record only
resources likely to be reused across sessions.

Fields: `name | type | path | source | status | notes`

---

## Models

| name | type | path | source | status | notes |
|------|------|------|--------|--------|-------|
| deberta-v3-base | model (encoder) | `models/deberta-v3-base/` | `microsoft/deberta-v3-base` (HF) | verified | Pilot model. 183.8M params, 12 layers, hidden 768, `model_type=deberta-v2`. Weights load via `AutoModel`; forward pass OK. Repo has no `model.safetensors`; weights are `pytorch_model.bin` (371 MB). Redundant `tf_model.h5` / `rust_model.ot` were NOT downloaded. Tokenizer (`spm.model`, `DebertaV2Tokenizer`, vocab 128000) loads via `sentencepiece==0.2.2` (installed in shared env; see `requirements.txt`). A spurious `fix_mistral_regex` warning from transformers 5.12.1 is a false positive; tokenization output is correct. |
| Qwen3-Embedding-0.6B | model (embedder) | `models/Qwen3-Embedding-0.6B/` | `Qwen/Qwen3-Embedding-0.6B` (HF), migrated from `~/projects/lora/models/` | verified | Transfer model. 595.8M params, 28 layers, hidden 1024, `model_type=qwen3`. `model.safetensors` (1.19 GB). Ships sentence-transformers config (`1_Pooling/`, `modules.json`, `config_sentence_transformers.json`); usable as a bi-encoder. Tokenizer loads without extra deps. Apache-2.0. |

## Datasets

| name | type | path | source | status | notes |
|------|------|------|--------|--------|-------|
| CLINC150 (clinc_oos) | dataset | `data/raw/clinc_oos/` | `clinc/clinc_oos` (HF dataset, official) | verified | Intent classification + OOS. 150 in-scope intents + 1 OOS label (**label id 42 = `oos`**). Three configs: `small` (train 7600 / val 3100 / test 5500), `imbalanced` (train 10625), `plus` (train 15250). All configs share val (3100) and test (5500). Parquet format, columns `text` (string) + `intent` (class_label 0-150 with names in README). CC-BY-3.0. OOS counts: small train 100 / val 100 / test 1000; plus train 250. |

## Notes

- Model weights and datasets are gitignored (`models/`, `data/raw/`) and are NOT committed, per `AGENT_PROTOCOL.md` §4.
- `sentencepiece==0.2.2` was added to the shared `ai-env` for the deberta-v3-base tokenizer; tracked in `requirements.txt`.
