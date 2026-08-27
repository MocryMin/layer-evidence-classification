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
| modernbert-base | model (encoder) | `models/modernbert-base/` | `answerdotai/modernbert-base` (HF) | verified | Future major target (EXP-001 plan §3). 149.0M params, **22 layers**, hidden 768, `model_type=modernbert` (no token_type_ids). `model.safetensors` (598.6 MB) + config + tokenizer.json (2.1 MB) downloaded; ONNX files and `pytorch_model.bin` NOT downloaded. Loads via `AutoModel`; forward pass OK. Note: the repo's safetensors includes classification-head weights (`decoder.bias`, `head.dense.*`, `head.norm.*`) - reported UNEXPECTED by AutoModel, harmless for encoder use. |
| Llama-3.2-3B-Instruct | model (causal LM) | `models/Llama-3.2-3B-Instruct/` | `meta-llama/Llama-3.2-3B-Instruct` (HF) | verified | EXP-004 target. Local two-shard safetensors are complete (about 6.1 GB); 28 decoder layers, hidden 3072, vocab 128256, BF16, tied embeddings. Config/tokenizer/chat template load offline with Transformers 5.12.1. A--E at the assistant generation boundary are single tokens with IDs 32--36. Subject to the bundled Llama 3.2 Community License. |

## Datasets

| name | type | path | source | status | notes |
|------|------|------|--------|--------|-------|
| CLINC150 (clinc_oos) | dataset | `data/raw/clinc_oos/` | `clinc/clinc_oos` (HF dataset, official) | verified | Intent classification + OOS. 150 in-scope intents + 1 OOS label (**label id 42 = `oos`**). Three configs: `small` (train 7600 / val 3100 / test 5500), `imbalanced` (train 10625), `plus` (train 15250). All configs share val (3100) and test (5500). Parquet format, columns `text` (string) + `intent` (class_label 0-150 with names in README). CC-BY-3.0. OOS counts: small train 100 / val 100 / test 1000; plus train 250. |
| WOS-46985 | dataset | `data/raw/wos46985/wos46895.parquet` | `jesse-tong/wos46985` (HF) | verified | Web of Science title+abstract classification. 46985 docs, 3 columns: `text`, `label` (141-dim one-hot = 7 domains + 134 subcategories, exactly 2 ones per row), `label_description` ([domain, subcategory] names). Domains: Medical 14625, Psychology 7142, CS 6514, ECE 5483, biochemistry 5687, Civil 4237, MAE 3297. NOTE: file named `wos46895.parquet` (mirror typo); 36.5 MB. Original kk7nc GitHub repo is gone (404) - this HF mirror is the source. Split: mirror ships NO official split; HYDRA (EMNLP 2025) formally uses 30,070/7,518/9,397 train/val/test (sum=46985) - reproduced as plain random, seed 17 (per-doc assignment is ours; original indices unavailable). Artifact `data/processed/wos46985/wos46985_split.npz` + `split_summary.json` via `scripts/wos46985_split.py`. |
| WOS-CT | dataset | `data/raw/wos-ct/CT/` | `marcelsun/wos_hierarchical_multi_label_text_classification` (HF) | verified | du Toit & Dunaiski (2024, arXiv:2411.19119) **citation-based** hierarchical variant. 65200 samples (train 45640 / dev 9780 / test 9780). `CT_{train,dev,test}.json` = JSONL, each line `{token: "title, abstract", label: [depth0, depth1]}`; `depth2label.pt` (depth 0: 10 labels, depth 1: 326 labels), `path_list.pt`, `slot.pt`, `value2slot.pt`, `value_dict.pt` = hierarchical class tree. CC-BY-4.0. Sibling variants JT (journal-based) / JTF (filtered) NOT downloaded. |
| ARC-Easy | dataset | `data/raw/arc_easy/ARC-Easy/` | `allenai/ai2_arc`, config `ARC-Easy` (HF dataset mirror) | verified | EXP-004 target. Local Parquet splits: train 2251 / validation 570 / test 2376. Samples contain 3--5 choices (over 99% are four-choice); maximum task-head width is therefore 5. The qualification protocol relabels options by position A--E and masks nonexistent choices. No network access is required. |

## Notes

- Model weights and datasets are gitignored (`models/`, `data/raw/`) and are NOT committed, per `AGENT_PROTOCOL.md` §4.
- `sentencepiece==0.2.2` was added to the shared `ai-env` for the deberta-v3-base tokenizer; tracked in `requirements.txt`.
