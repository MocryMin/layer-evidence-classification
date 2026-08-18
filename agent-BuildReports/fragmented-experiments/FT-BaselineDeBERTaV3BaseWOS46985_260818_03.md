# FT-BaselineDeBERTaV3BaseWOS46985_260818_03 — fine-tuning baseline on DeBERTa-v3-base / WOS-46985 (LN-plain head, full + attention-only)

Date: 2026-08-18 · Plan: `user_exp_plans/fragmented_exp_gr3.md` step 3 · Inputs: `DeBERTaV3BaseWOS46985Baseline_260812_04` (frozen baseline, cache) · Script: `scripts/gr3_ft_baseline.py` (stages probe/smoke/ft/analyze) · Single seed 17 · max_length 512 · same HYDRA split (30070/7518/9397)

## 1. Config

- Model: deberta-v3-base, fp32 training; head = **LN-plain** (LayerNorm + linear, 768→134), initialised from the stage-0 probe on the frozen backbone, **trained jointly** with the backbone.
- Optimizer: AdamW, **wd=0.01**, **bs=32 (microbs 8 × grad-accum 4)**, linear warmup 10% + linear decay, grad clip 1.0, **5 epochs**, val eval **4×/epoch**, best checkpoint by final-layer val acc (test evaluated once on the best ckpt).
- Variants: **full** (all 183.8M params) and **attn** (attention-only: `attention.self` q/k/v_proj + `attention.output.dense`, 28.3M backbone params; embeddings, rel_embeddings, FFN, all LayerNorms frozen). Head always trainable.
- Stage 0 probe protocol (gr1): full-batch AdamW lr=0.01 wd=0, ES patience 100 / max 10k ep, on the baseline cache L12.

## 2. Stage 0 — init probe (crosscheck)

LN-plain on frozen L12 (recomputed from the baseline cache): val 0.2934 vs baseline 0.2954 (**dev 0.0020**, GPU nondeterminism — two reruns gave 0.2934/0.2938; the baseline itself ran on GPU, so exact reproduction is not expected). Protocol reproduces; head saved as warm start.

## 3. Stage 1 — lr smoke (250 steps, full unfreeze)

| lr | 5e-6 | 1e-5 | 2e-5 | 5e-5 | 1e-4 | 2e-4 | 5e-4 |
|---|---|---|---|---|---|---|---|
| val @250 | 0.313 | 0.362 | 0.391 | 0.453 | 0.510 | **0.556** | 0.099 (diverged) |

Smoke picked 2e-4, but the **full 5-epoch run at 2e-4 was unstable** (val oscillated 0.645→0.156→0.635→0.299→0.040 over epochs 1-2; killed at step 1645). **Fallback lr = 1e-4** — short-horizon smoke overestimates the stable lr for full FT. All subsequent runs use 1e-4.

## 4. Stage 2 — fine-tuning

| variant | trainable | best val acc (@step) | test @ best | val−test gap |
|---|---|---|---|---|
| full | 183.8M | **0.8341** (@4700, ep5 end) | **0.8316** | 0.003 |
| attn | 28.3M + head | 0.8219 (@4700) | 0.8123 | 0.010 |

- Both runs climbed monotonically through all 5 epochs (full: 0.328→0.566→0.702→0.734→…→0.834; attn: 0.348→0.500→0.675→…→0.822), no early-stopping plateau — 5 epochs may be slightly underfit.
- vs the frozen LN-plain probe (val 0.2934 / test 0.2852): **+0.54 / +0.55 test** on final-layer readout.
- Attention-only reaches 97% of full FT's accuracy with 15% of the trainable parameters (28.3M/183.8M).

## 5. Stage 3a — variance compression (EXP-002 def: per-layer inter_std, threshold 1e-3)

| layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | below 1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen | .0029 | .0008 | .0010 | .0007 | .0007 | **.0002** | .0005 | .0004 | .0008 | .0010 | .0438 | .0395 | 8/12 |
| ft full | .0039 | .0008 | .0013 | .0011 | .0023 | **.0005** | .0018 | .0007 | .0021 | .0022 | **.0890** | .0129 | 3/12 |
| ft attn | .0033 | .0009 | .0009 | .0005 | .0014 | **.0004** | .0011 | .0004 | .0012 | .0014 | .0190 | .0157 | 5/12 |

- **Judgement: no additional variance compression from fine-tuning.** The frozen DeBERTa-WOS baseline was *already* compressed in mid layers by the 1e-3 criterion (8/12 layers, min 2e-4 — the same signature that made the gr1 baseline pick the "collapsed" gradient-family config; contrast modernBERT-WOS healthy min 0.178). FT slightly *widens* mid-layer activation spread (L3-10) and strongly spreads L11 (0.044→0.089 full), the layer directly feeding the classifier, as it becomes class-discriminative. The collapse signature on WOS-DeBERTa is a pre-norm baseline trait, not an FT effect.

## 6. Stage 3b — all-layer probes on the fine-tuned backbone (LN-plain + ridge, gr1 protocol)

**LN-plain test acc:**

| layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen | .447 | .439 | .411 | .389 | .434 | .383 | .389 | .352 | .346 | .308 | .311 | .285 |
| ft full | .461 | .493 | .596 | .627 | .755 | .783 | .801 | .807 | .804 | .797 | .814 | **.832** |
| ft attn | .447 | .466 | .529 | .598 | .750 | .771 | .794 | .791 | .786 | .788 | .797 | **.814** |

**Ridge test acc:**

| layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen | .470 | .482 | .433 | .457 | .498 | .484 | .447 | .436 | .407 | .362 | .334 | .300 |
| ft full | .488 | .533 | .602 | .698 | .781 | .794 | .797 | .813 | .819 | .813 | .824 | **.833** |
| ft attn | .472 | .507 | .546 | .691 | .762 | .781 | .781 | .797 | .796 | .793 | .797 | **.807** |

- **The layer-accuracy curve flips from mid-hump to monotone**: frozen DeBERTa-WOS peaks at L5 (ridge .504) and degrades to L12 (.300); after FT (either variant) accuracy increases monotonically L1→L12, and the final layer becomes the single best readout. FT has "used up" the mid-layer advantage — the EXP-001 pattern.
- Recoverability on the remaining errors: frozen ridge R_oracle 0.6259 (6575 errors); **ft full ln R_oracle 0.5327** (oracle gain +0.089 over acc_L 0.832), ridge 0.4736; ft attn ln R_oracle 0.5776. Mid layers still recover ~half of the final-layer errors even after FT, but the error budget is 4× smaller.
- attn mirrors full at every layer (gap 0.02-0.03), including the monotone shape — attention alone re-organizes the whole stack.

## 7. Artifacts

`artifacts/fragmented-experiments/FT-BaselineDeBERTaV3BaseWOS46985_260818_03/`: `probe_init.{pt,json}`, `smoke.json` (incl. stability note + final_lr), `ft_full/`, `ft_attn/` (backbone+tokenizer+`ft_head.pt`+`ft_history.json`), `analysis/{full,attn}_results.json` (variance + ln/ridge per-layer + recoverability), `analysis/{full,attn}_ridge_test_pred.npy`, per-split feature caches under `ft_{variant}/cache/`.

## 8. Observations

- _Data collection point — no hypothesis; observations are supported by the outputs above._
- Final-layer optimization target reached: 0.285 → **0.832 test** (LN-plain head, jointly trained, full FT). Attention-only is a near-parity cheap variant (0.812, 15% params).
- On WOS-DeBERTa, variance compression (1e-3 inter_std) is a pre-existing pre-norm signature of the frozen model, not an FT consequence; FT instead spreads the pre-classifier layer L11.
- FT converts the mid-hump probe curve into a monotone top-heavy curve in both variants — recoverability of mid layers persists (~0.5 R_oracle) but the final layer stops needing them.
- lr note: 250-step smoke selected 2e-4, which destabilised the 5-epoch run; 1e-4 stable and monotone. Future FT smokes on this family should test longer horizons or include a stability window.
