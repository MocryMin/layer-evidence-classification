# mudularized_layer_probe_260813_01 — modularized layer probe on DeBERTa-v3-base × CLINC150

Date: 2026-08-13 (run 23:48 → 05:48 next day, ≈6.0 h) · Group: `plans/fragmented_exp_gr2.md` · Reporting model: deepseek-v4-flash · Git: `3cf29fb` (dirty=False) · Single seed 17

## Config

- Dataset: clinc (CLINC150 plus, OOS intent 42 dropped, EXP-001 id2label mapping) · 150 classes · splits {train: 15000, validation: 3000} · **val-only evaluation** (train fit, val acc, per plan; no test access)
- Model: deberta-v3-base (frozen) · layer modules composed into **arbitrary repeatable sequences** ("paths"), CLS readout at the tail · max_length 38 (observed max, no truncation) · prompt `Classify the intent: {utterance}` (EXP-003)
- Probe: ridge classifier α=1e-6, closed-form fp64 eigen-solve — **bit-equal to sklearn `RidgeClassifier(alpha=1e-6, fit_intercept=True, solver='svd')`** (EXP-003 config; smoke-verified below)
- Modular semantics: rel_pos is position-only (constant across applications) and deberta-v3-base has no encoder conv, so layer composition == the raw pipeline; the raw chain `[1..12]` is **bit-equal to a true model forward** (max|Δ| = 0.0, fp32)
- Branch stack stores fp16 states (rounding 0.0078 max|Δ| on the raw chain; stack-vs-scratch val-acc Δ ≤ 0.004, smoke)
- Prefix-trie reuse: 4500 random paths (Σ lengths 33,625) processed via 22,046 distinct nodes (incl. 156 task-1/3 nodes + greedy candidates) — ~34% fewer forwards than per-path evaluation; the path set is generated first, only computation order changes
- Tasks: 1) single-layer `[i]` vs in-raw-place, 2) greedy queue to len 50 (repeats allowed), 3) pairwise gain matrices, 4) n=4500 random paths, uniform len 3–12, uniform layers 1–12
- Deadline 07:30 not hit (run finished 05:48); resumable checkpoints (nodes.jsonl / greedy_steps.jsonl)

## 0. Validation (smoke, 2000/1000 random subset)

| check | result |
|-------|--------|
| embeddings (mine vs model) max\|Δ\| | 0.0 |
| modular raw chain fp32 vs true model L12 max\|Δ\| | 0.0 |
| modular raw chain fp16 stack vs true model max\|Δ\| | 0.0078 |
| torch ridge == sklearn on raw L1 / L6 / L12 (val acc) | 0.734 / 0.827 / 0.771 — exact match |
| stack(fp16) vs scratch(fp32) val acc on 5 random paths | Δ ≤ 0.004 |

Full-data cross-check: in-place α=1e-6 val acc vs EXP-003 ridge (α=1e-6, fp16 cache): L6 0.9173/0.9170, L10 0.9420/0.9420, L12 0.9047/0.9030 (Δ ≤ 0.0017; the L12 gap ≈ their fp16-cache rounding).

## 1. Task 1 — single-layer-only vs in-raw-place (val acc, macro_f1)

| L | single | inplace | Δ | f1_single |
|--:|-------:|--------:|-----:|----------:|
| 1 | 0.8250 | 0.8330 | −0.0080 | 0.8189 |
| 2 | 0.8750 | 0.8710 | **+0.0040** | 0.8723 |
| 3 | 0.8460 | 0.8923 | −0.0463 | 0.8414 |
| 4 | 0.8260 | 0.8860 | −0.0600 | 0.8202 |
| 5 | 0.8727 | 0.9130 | −0.0403 | 0.8708 |
| 6 | 0.8627 | 0.9173 | −0.0547 | 0.8587 |
| 7 | 0.8593 | 0.9390 | −0.0797 | 0.8554 |
| 8 | 0.8567 | 0.9423 | −0.0857 | 0.8529 |
| 9 | 0.8490 | 0.9370 | −0.0880 | 0.8449 |
| 10 | 0.8477 | 0.9420 | −0.0943 | 0.8439 |
| 11 | 0.8600 | 0.9120 | −0.0520 | 0.8565 |
| 12 | 0.8613 | 0.9047 | −0.0433 | 0.8580 |

- Every layer ALONE is a strong classifier (0.825–0.875); the model's representational power is not a product of depth alone.
- Only L2 is better alone than in-place (+0.0040). The largest isolation losses are at the layers that are best in-place (L7–10, −0.0797…−0.0943) — mid-layer strength depends on receiving the trained input distribution.

## 2. Task 2 — greedy layer queue (start = best single L2, max len 50)

- Step 1 appends L11 → **0.8827** (the task-3 argmax pair `[2,11]`, +0.0077 over single L2 0.8750). Step 4 (+L1 → `[2,11,1,8,1]` 0.8790, +0.0030) is the only later positive step.
- **47/49 steps have negative gain; max acc 0.8827 occurs at step 1 — before any negative step, so the max does NOT exist after a negative step.** From step 4 on, acc decays monotonically to **0.2183** at len 50.
- First steps: `[2]` 0.8750 → `+11` 0.8827 → `+1` 0.8777 → `+8` 0.8760 → `+1` 0.8790 → `+1` 0.8720 → `+8` 0.8537 → `+1` 0.8460 … (full chain in `greedy_steps.jsonl`). The queue oscillates through L1/L7/L8/L9/L11 — greedy never revisits the peak.

## 3. Task 3 — pairwise add-layer gains (12×12, in `task3_pairwise.npz/.json`)

- A = val acc of `[i,j]`: mean 0.8490, **max `[2,11]` 0.8827**, min `[1,4]` 0.7680.
- Top pairs: `[2,11]` 0.8827, `[2,5]` 0.8813, `[2,7]` 0.8807, `[2,10]` 0.8777, `[2,9]` 0.8770, `[2,12]` 0.8753, `[3,12]` 0.8750, `[3,9]` 0.8730 — 6/8 start with L2. Bottom: `[12,5]` 0.7937, `[12,2]` 0.7890, `[12,12]` 0.7783, `[1,4]` 0.7680.
- G_ij = A_ij − A_i (gain of j as 2nd layer): max **+0.0463 for `[1,2]`** (the raw-pipeline prefix) — the largest add-layer gain of any pair.
- S_ij = A_ij − max(A_i, A_j): **42/144 pairs are super-additive** (beat the best single layer); top `[1,1]` +0.0367 (L1∘L1 0.8617 vs L1 alone 0.8250), `[3,9]` +0.0240, `[3,12]` +0.0137.

## 4. Task 4 — random paths (n = 4500, uniform len 3–12, repeats allowed)

- Val acc: mean 0.6241, median 0.6740, **max 0.9093** (`[1,2,6,4,9]`), min 0.0280.
- Per length (mean / median / max / n): the decline with length is monotone.

| len | n | mean | median | max |
|----:|---:|------:|-------:|-----:|
| 3 | 461 | 0.8365 | 0.8477 | 0.8953 |
| 4 | 454 | 0.7997 | 0.8240 | 0.9047 |
| 5 | 438 | 0.7619 | 0.7995 | 0.9093 |
| 6 | 471 | 0.7025 | 0.7420 | 0.8977 |
| 7 | 431 | 0.6445 | 0.6857 | 0.8823 |
| 8 | 463 | 0.5817 | 0.6163 | 0.8897 |
| 9 | 473 | 0.5480 | 0.5723 | 0.8847 |
| 10 | 419 | 0.4843 | 0.4910 | 0.8823 |
| 11 | 438 | 0.4545 | 0.4558 | 0.8277 |
| 12 | 452 | 0.4136 | 0.3970 | 0.8423 |

- Top-10 paths: `[1,2,6,4,9]` 0.9093, `[1,2,7,10]` 0.9047, `[1,2,3,10,6]` 0.9030, `[1,3,5,11,10,5]` 0.8977, `[2,3,5]` 0.8953, `[1,2,5,6]` 0.8940, `[1,5,8,12]` 0.8927, `[1,3,5]` 0.8920 ×2, `[2,2,9]` 0.8913 — **8/10 start with `[1,2]`** (the raw-pipeline prefix).
- Repeats degrade: 3523/4500 paths contain a repeated layer — mean acc 0.5791 vs **0.7865** for repeat-free paths.
- Tail layer matters little (per-tail-layer means 0.5738–0.6558).

## 5. Artifacts

- `artifacts/fragmented-experiments/mudularized_layer_probe_260813_01/`: `results.json` (config + task summaries), `nodes.jsonl` (22,046 nodes: path/len/tail_layer/tasks/val_acc/macro_f1/ts), `nodes_pred.npy` (22,046×3000 int16, rows aligned with nodes.jsonl), `random_paths.json` (4500 generated paths), `task3_pairwise.npz` + `.json` (A/G/S), `inplace.json` (12-layer in-place), `greedy_steps.jsonl` (49 steps incl. per-candidate accs).
- Smoke validation numbers are reproduced in §0 (the smoke artifact dir was cleaned before the full run).

## 6. Observations

- _Data collection point — no hypothesis; observations are supported directly by the outputs above._
- Single layers are strong standalone classifiers; a second layer adds little (+0.0077 best, 42/144 super-additive pairs) and longer stacks decay monotonically — greedy peaks at len 2 and never recovers (47/49 negative steps).
- The trained order is special: the raw prefix `[1,2,…]` is near-optimal among all compositions (largest pairwise gain +0.0463; 8/10 top random paths start with `[1,2]`; best random path 0.9093 still below in-place L8 0.9423).
- Repeating layers hurts (0.579 vs 0.787 mean); layers 7–10 are the strongest in-place but lose the most in isolation (input-distribution dependence).
