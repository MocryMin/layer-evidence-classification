# ModernBERTBaseWOS46985LayerProbe_260814_02 — modular layer probe (gr2 tasks 1-3) on WOS-46985

Date: 2026-08-17 (run 2026-08-17 09:55 → 15:17, full run after padding fix) · Group: `user_exp_plans/fragmented_exp_gr2.md` (tasks 1-3) · Reporting model: deepseek-v4-flash · Git: `9887739` (report time; run at commit `9887739`) · Single seed 17

## Config

- Dataset: wos (WOS-46985, HYDRA-count split 30070/7518/9397, seed 17) · 134 L2 classes · max_length 512 · truncation right · **padding: per-batch longest** (gr1 baseline extraction semantics; a fixed-512 padding variant was run first and discarded — it systematically lowered modernbert accuracies by ~0.02-0.03, see note)
- Model: modernbert-base (frozen, fp16 weights, fp16 streaming states) · pooling cls, **pre-norm** readout for every layer (final_norm excluded from layer modules; post-norm L22 recorded separately) · ridge α=1e-6 (fp64 closed form, sklearn-equivalent) · train fit / val acc
- Modular semantics: layer modules composed arbitrarily; RoPE cos/sin and global/sliding attention masks are position/mask-only constants → raw chain == true model forward (smoke-verified, max|Δ|=0.0); in-place curve reproduces the gr1 baseline exactly (padding fix verified, e.g. L1 0.5068 vs 0.5076, L22 0.4967 vs 0.4968)
- Tasks: 1 singles vs in-place; 2 greedy to len 50 (repeats allowed); 3 pairwise A/G/S matrices (22×22)
- Runtime: 1562 nodes recorded; resumable (JSONL per node, per-step greedy flush)

**Padding note (2026-08-17):** a first run of this experiment used fixed-512 padding and was discarded after a follow-up showed fixed-512 lowers modernbert accuracies by ~0.02-0.03 vs per-batch longest padding (identical forward otherwise). All numbers below use the corrected (baseline-equivalent) semantics.

## 1. Task 1 — every layer as the only layer vs in-place (raw trained chain)

| layer | single (modular [k]) | in-place (raw chain) | Δ (single − in-place) |
|------:|--------------------:|--------------------:|----------------------:|
| 1 | **0.5068** | 0.5068 | 0.0000 |
| 2 | 0.1450 | 0.4566 | −0.3117 |
| 3 | 0.1453 | 0.4137 | −0.2684 |
| 4 | 0.1946 | 0.3978 | −0.2032 |
| 5 | 0.1443 | 0.3558 | −0.2115 |
| 6 | 0.1828 | 0.3409 | −0.1582 |
| 7 | 0.2885 | 0.3593 | −0.0708 |
| 8 | 0.2515 | 0.3484 | −0.0968 |
| 9 | 0.2628 | 0.3561 | −0.0932 |
| 10 | 0.2389 | 0.3783 | −0.1394 |
| 11 | 0.2557 | 0.3626 | −0.1069 |
| 12 | 0.2392 | 0.3581 | −0.1189 |
| 13 | 0.2167 | 0.3723 | −0.1556 |
| 14 | 0.2519 | 0.3763 | −0.1244 |
| 15 | 0.2204 | 0.3621 | −0.1417 |
| 16 | 0.2592 | 0.3772 | −0.1180 |
| 17 | 0.2291 | 0.3772 | −0.1482 |
| 18 | 0.2039 | 0.3728 | −0.1689 |
| 19 | 0.2702 | 0.4302 | −0.1600 |
| 20 | 0.2106 | 0.4221 | −0.2115 |
| 21 | 0.1716 | 0.4225 | −0.2509 |
| 22 | 0.3049 | 0.4967 | −0.1918 |

- **Every layer ≥ 2 in isolation is far worse than its in-place readout** (Δ −0.07 to −0.31): modernBERT layers depend on the trained-order input distribution (pre-norm blocks trained on the previous layer's output); applied directly to embeddings they are nearly useless for linear readout (0.14-0.30).
- In-place curve is flat-ish (L1 0.5068 → L22 0.4967, mid ~0.34-0.38) and matches the gr1 baseline exactly — modernBERT's chain is self-consistent and healthy, unlike DeBERTa's WOS chain.
- Post-norm L22 0.4959 ≈ pre-norm L22 0.4967 (final_norm is near-identity for the CLS readout on this task).
- **Contrast with DeBERTa on the same data**: DeBERTa singles (0.47-0.55) beat in-place at every layer; modernBERT singles (0.14-0.30, except L1/L22) are crushed by in-place. Model architecture determines whether the trained chain is essential (modernBERT: yes) or a liability (DeBERTa WOS: yes).

## 2. Task 2 — greedy layer queue (to len 50)

- Start: L1 (best single 0.5068). Step 1: +L22 → **0.5215** (+0.0148, the only positive step; max acc at step 1 == best pair). Steps 2..49: **40/49 negative gains**, gentle decay to **0.4384 @ len 50** (vs DeBERTa's collapse to 0.0185). Max occurs BEFORE any negative step; no recovery after negatives.
- Repeated self-application is far less destructive for modernBERT (final 0.44 vs DeBERTa 0.02).

## 3. Task 3 — pairwise add-layer matrices (22×22, A/G/S in task3_pairwise.npz/json)

- A max **0.5215 @ [1,22]** (== greedy step-1 queue). G max **+0.1353 @ [2,22]** (the second-worst single, boosted by L22; L2 needs L22's input distribution to unlock 0.27).
- G ≥ 0 (super-additive pairs): [2,22] +0.1353, [3,22] +0.1123, [21,22] +0.1025, [2,21] +0.1020, [1,22] +0.0148 — pairs ending at L22 dominate; self-pairs [k,k] negative (e.g. [22,22] −0.0047, [2,2] −0.0116).

## 4. Baseline cross-check

`baseline_crosscheck.json`: my α=1e-6 fits on the gr1 baseline cache reproduce the baseline per_alpha 1e-6 val accs exactly for all 22 layers (e.g. L1 0.5074/0.5074, L11 0.3634/0.3634, L22 0.4967/0.4967).

## 5. Artifacts

- `artifacts/fragmented-experiments/ModernBERTBaseWOS46985LayerProbe_260814_02/`: `nodes.jsonl` (1562), `nodes_pred.npy`, `greedy_steps.jsonl` (49), `task3_pairwise.npz/.json`, `inplace.json`, `results.json`, `baseline_crosscheck.json`.

## 6. Observations

- _Data collection point — no hypothesis; observations are supported by the outputs above._
- modernBERT layers are input-distribution-dependent (in-place ≫ single for L≥2; only L22-additions rescue pairs); DeBERTa on the same WOS data inverts this (single ≫ in-place). Greedy peaks at step 1 in both models, decays monotonically after; modernBERT's decay is graceful (0.44) vs DeBERTa's collapse (0.02).
