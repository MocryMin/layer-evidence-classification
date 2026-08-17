# DeBERTaV3BaseWOS46985LayerProbe_260814_01 — modular layer probe (gr2 tasks 1-3) on WOS-46985

Date: 2026-08-17 (run 2026-08-14 13:36 → 2026-08-15 16:13, resumable) · Group: `user_exp_plans/fragmented_exp_gr2.md` (tasks 1-3) · Reporting model: deepseek-v4-flash · Git: `9887739` (report time; run finalized at `7a95605` 2026-08-15, began at `c7b6a2b`) · Single seed 17

## Config

- Dataset: wos (WOS-46985, HYDRA-count split 30070/7518/9397, seed 17) · 134 L2 classes · max_length 512 · truncation right · **padding: fixed 512** (see note below)
- Model: deberta-v3-base (frozen, fp16 weights, fp16 streaming states) · pooling cls (pre-norm readout) · ridge α=1e-6 (fp64 closed form, sklearn-equivalent) · train fit / val acc
- Modular semantics: layer modules composed arbitrarily; rel_pos position-only constant, no conv, no final norm → raw chain == true model forward (smoke-verified, max|Δ|=0.0); fp16 chain vs fp32 baseline within 0.0009 acc
- Tasks: 1 singles vs in-place; 2 greedy to len 50 (repeats allowed); 3 pairwise A/G/S matrices
- Runtime: 732 nodes recorded; deadline-guarded + resumable (JSONL per node, per-step greedy flush)

**Padding note:** this run used fixed-512 padding. A follow-up check (2026-08-17) showed fixed-512 padding lowers **modernbert** accuracies by ~0.02-0.03 but leaves **deberta** unaffected — the DeBERTa in-place curve reproduces the gr1 baseline ridge α=1e-6 within 0.0009 (baseline_crosscheck.json), so this run stands as-is.

## 1. Task 1 — every layer as the only layer vs in-place (raw trained chain)

| layer | single (modular [k]) | in-place (raw chain) | Δ (single − in-place) |
|------:|--------------------:|--------------------:|----------------------:|
| 1 | 0.4690 | 0.4690 | 0.0000 |
| 2 | 0.5217 | 0.4848 | +0.0369 |
| 3 | 0.5086 | 0.4354 | +0.0732 |
| 4 | 0.5072 | 0.4582 | +0.0490 |
| 5 | 0.5243 | 0.5040 | +0.0203 |
| 6 | 0.5294 | 0.4844 | +0.0450 |
| 7 | 0.5333 | 0.4507 | +0.0826 |
| 8 | 0.5460 | 0.4409 | +0.1051 |
| 9 | **0.5517** | 0.4119 | +0.1398 |
| 10 | 0.5289 | 0.3631 | +0.1658 |
| 11 | 0.5426 | 0.3452 | +0.1974 |
| 12 | 0.4918 | 0.3134 | +0.1784 |

- Every layer in isolation (directly from embeddings) **beats its in-place readout**, with the gap growing with depth (L12: +0.178). The trained-order chain progressively degrades each layer's linear readout utility on WOS.
- **Opposite of CLINC gr2**, where in-place layers (0.917-0.942) beat singles (0.825-0.875). Dataset-dependent.
- In-place curve reproduces the gr1 baseline ridge (L5 best 0.5040 vs baseline 0.5070/0.5031 — within fp16/α-grid noise).

## 2. Task 2 — greedy layer queue (to len 50)

- Start: L9 (best single 0.5517). Greedy steps 1..49 (49/49 completed).
- **Max acc 0.5375 at step 1** (queue [9,2]); **43/49 steps had negative gain**; monotone decay to **0.0185 @ len 50**. Max occurs BEFORE any negative step; no recovery after negatives.
- Step 1 already loses (−0.0142): the best single-layer extension [9,2] is worse than L9 alone. Greedy is a pure downhill run here — no additive layer helps.
- Early steps: [9,2]→[9,2,9]→[9,2,9,5]→[9,2,9,5,9]→… (layer 9 recurs; acc collapses toward 0.02 with repeated self-application).

## 3. Task 3 — pairwise add-layer matrices (12×12, A/G/S in task3_pairwise.npz/json)

- A max **0.5565 @ [1,9]** (best pair) — slightly above the best single (0.5517); G max **+0.0875 @ [1,9]**.
- G ≥ 0 (super-additive pairs): top gains [1,9] +0.0875, [1,2] +0.0527, [2,9] +0.0156, [1,11] +0.0129; most pairs with i≠j are near 0 or negative; self-pairs [k,k] all strongly negative (e.g. [9,9] −0.0169, [12,12] −0.0203).

## 4. Baseline cross-check

`baseline_crosscheck.json`: my α=1e-6 fits on the gr1 baseline cache reproduce the baseline per_alpha 1e-6 val accs **exactly** for all 12 layers (e.g. L1 0.4698/0.4698, L5 0.5031/0.5031, L12 0.3140/0.3140).

## 5. Artifacts

- `artifacts/fragmented-experiments/DeBERTaV3BaseWOS46985LayerProbe_260814_01/`: `nodes.jsonl` (732), `nodes_pred.npy`, `greedy_steps.jsonl` (49), `task3_pairwise.npz/.json`, `inplace.json`, `results.json`, `baseline_crosscheck.json`.

## 6. Observations

- _Data collection point — no hypothesis; observations are supported by the outputs above._
- On WOS, DeBERTa's trained-order chain is a liability for every layer readout (single-modular > in-place, gap growing with depth), inverting the CLINC gr2 picture; no pair (except [1,9], barely) beats the best single; greedy decays monotonically from the start.
