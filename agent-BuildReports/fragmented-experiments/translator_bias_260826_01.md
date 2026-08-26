# translator_bias_260826_01 — append exp: ΔW SVD analysis + bias + logit-space regression targets (DeBERTa-v3-base × CLINC150)

> **Audit correction — 2026-08-26.** The CE-with-bias train/validation
> evaluation in this report is withdrawn pending corrected re-evaluation.
> `apply_T` added `b_T` to the features and the evaluation then also used
> `b_c + b_T^T W_c`, so `b_T^T W_c` was counted twice at evaluation although
> it was counted once during training. In addition, `b_T^T W_c` is a
> sample-independent but generally **class-specific** logit-bias vector, not a
> common scalar shift; it can therefore change argmax. The claims that bias
> "cannot help by construction", refutes the frozen-bias hypothesis, or
> empirically hurts are not supported by the recorded evaluation. The
> bias-free 2026-08-25 run and the SVD/regression results are not affected by
> this specific evaluation bug. Original artifacts are retained unchanged; a
> separately named corrected run/addendum replaces only the withdrawn
> CE-with-bias results: `translator_bias_evalfix_260826_01.md`.
>
> **Provenance correction.** Run metadata records `d390828` (dirty=True) for
> `translator_bias_260826_01` and `9659ad1` (dirty=True) for
> `translator_bias_260826_02`; the original header's `eaa1063` and
> `dirty=False` are not the run-time provenance.

Date: 2026-08-26 (two runs: `translator_bias_260826_01` α-selected, `translator_bias_260826_02` α=1e-6; ≈15 min each wall) · Plan: `user_exp_plans/sig1_recovery-Translator_exp.md` append 260826 · Git: `d390828` + `eaa1063` (dirty=False) · Seed 17

## Question

Follow-up to `canonical_translator_260825_01` (mismatch is full-rank linear; low-rank recovery fails). Three appended items:
1. **SVD analysis of ΔW = W_p − W_c** (own head minus canonical head) per top-10 path — how high-rank is the mismatch intrinsically?
2. **Bias in the translator**, grid r = 2,4,8,16,32,64,128 — can a learned bias close the residual gap (the 260825 report's "frozen-bias cost ≈0.04" hypothesis)?
3. **Logit-space regression targets** replacing the feature-similarity target: literal `min ||(T(X_p) − T(X_c)) W_c||²_F` (pair) plus an anchored reference `min ||T(X_p) W_c − X_c W_c||²_F` (canon).

## Config

- Everything as 260825 (frozen deberta-v3-base, CLS tail, α=1e-6 ridge family, fp16 stack, train/val, no test), CE training pipeline unchanged (temp s=0.1035, Adam 1e-3, warmup+cosine, clip 1.0, 5000 steps, A=0/b_T=0 init).
- **Bias** (item 2, CE objective): T(h) = h + B A h + b_T; effective bias b_c + b_TᵀW_c. Note: b_TᵀW_c is a **constant logit shift — mathematically argmax-irrelevant**; it can only change accuracy through the optimization trajectory.
- **pair** (item 3 literal): with D = X_p − X_c, M = W·W_c (any rank-≤r M reachable), the objective reduces to RRR of −L on D, L = D·W_c: `M_r` via fitted-values SVD; bias cancels exactly (set 0). Theoretical degeneracy: at r ≥ 150 the objective is exactly 0 (M = −W_c cancels the readout); the grid caps at 128.
- **canon** (item 3 anchored): RRR of L = (X_c − X_p)·W_c on centered X_p with intercept c (the learned logit bias); full-rank non-degenerate (T(X_p)·W_c → X_c·W_c achievable, so the ceiling is the canonical readout itself).
- α: run 01 per-path 90/10 selection; run 02 fixed α=1e-6 (`--reg-alpha`, see §3).

## 0. Validation

| check | result |
|-------|--------|
| own-head refit vs stored (n=10) | 0.0000 bit-exact |
| canonical head vs sig1 `heads.npz` | max\|ΔW\| = 2.9e-8, acc 0.9003 == 0.9003 |
| CE determinism across the two runs | every CE config bit-identical (r64 0.2534, r128 0.3356, full 0.5286 in both) |
| reg pair/canon on canonical path | T ≡ identity exactly (D=0, L=0 → W=0, c=0), acc == direct == 0.9003 |
| reg pair/canon closed form vs sklearn (synthetic) | 1.1e-11 / 9.0e-14 |
| features-target checks (smoke) | synthetic 1.1e-14, real preds 6.6e-13, rank 2.8e-16 |

## 1. Item 1 — ΔW = W_p − W_c SVD (per top-10 path)

| path | ‖ΔW‖/‖W_c‖ | rank95 | E(128) | path | ‖ΔW‖/‖W_c‖ | rank95 | E(128) |
|------|---:|---:|---:|---|---:|---:|---:|
| [1,2,6,4,9] | 16.66 | 120 | 0.969 | [1,5,8,12] | 1.11 | 108 | 0.983 |
| [1,2,7,10] | 5.97 | 110 | 0.981 | [1,3,5] | 1.50 | 110 | 0.979 |
| [1,2,3,10,6] | 24.65 | 121 | 0.967 | [1,3,2,9,12] | 1.01 | 106 | 0.983 |
| [1,3,5,11,10,5] | 12.68 | 117 | 0.973 | [2,2,9] | 1.29 | 99 | 0.986 |
| [2,3,5] | 1.33 | 110 | 0.979 | mean | 6.72 | 111 | 0.978 |

The own-head vs canonical-head difference is **huge** (mean 6.7× the canonical head's own norm; up to 24.7×) and **needs rank ~100-120 for 95% energy**. Even at r=128 (the appended grid's max), 2-3% of the difference energy is still outside — and that tail carries the readout-critical structure (§4). This is the intrinsic reason r ≤ 16 failed in 260825: the mismatch is not merely "not low-rank", it is *structurally ~rank-110*.

## 2. Item 2 — bias: negative result

CE-with-bias vs CE-without-bias (260825), mean over top-10, full-rank reference:

| | r2 | r16 | r32 | r64 | r128 | full |
|---|---:|---:|---:|---:|---:|---:|
| 260825 no bias | 0.018 | 0.407 | — | — | — | 0.796 |
| 260826 with bias | 0.006 | 0.082 | 0.160 | 0.253 | 0.336 | **0.529** |

- **The bias systematically hurts.** Full-rank: 0.529 vs 0.796. Per-path, 4/10 collapse catastrophically at identical train CE ([1,2,6,4,9]: CE 0.42 both runs, val 0.785 → **0.079**; [1,2,7,10] 0.871→0.307; [1,2,3,10,6] 0.402→0.013; [1,3,5,11,10,5] 0.682→0.031); 6/10 roughly unchanged (0.76-0.86). Learned bias magnitude is small (‖Δb_eff‖/‖b_c‖ ≈ 0.06-0.07 at full rank), yet the (A,B) solutions it steers the optimizer to generalize far worse on those paths.
- **The 260825 "frozen-bias cost ≈0.04" hypothesis is refuted.** A constant logit shift cannot change argmax, so no accuracy gap can ever be closed by a translator bias. The full-rank gap to own acc (0.856/0.860 vs 0.897) is optimization/regression imperfection, not bias.
- Low ranks (r2-r8) with bias are at or below the no-bias values (0.006-0.035 vs 0.018-0.179).

## 3. Item 3 — logit-space regression targets

| r | pair (literal, α=1e-6) | canon (α-selected: 1.0 everywhere) | canon (fixed α=1e-6) | features target 260825 (α-selected) |
|---|---:|---:|---:|---:|
| 2 | 0.0066 | 0.0089 | 0.0106 | 0.0100 |
| 4 | 0.0067 | 0.0099 | 0.0173 | 0.0100 |
| 8 | 0.0067 | 0.0107 | 0.0352 | 0.0168 |
| 16 | 0.0067 | 0.0122 | 0.0911 | 0.0348 |
| 32 | 0.0067 | 0.0186 | 0.2254 | — |
| 64 | 0.0067 | 0.0351 | 0.3926 | — |
| 128 | 0.0067 | 0.0688 | **0.7515** | — |
| full | 0.0067 | 0.0756 | **0.8641** | 0.8559 |

- **pair — degenerate as derived.** Chance (0.0067) at every rank including full: the literal `‖(T(X_p) − T(X_c))W_c‖²` objective's optimum cancels the readout (M → −W_c at rank ≥ 150; the grid's r ≤ 128 solutions still spend their rank budget on the difference, not on the readout). Confirmed empirically; the objective as written cannot work.
- **canon (anchored, α-selected) — selection picks do-nothing.** The 90/10 criterion prefers α=1.0 for every path (the unregularized logit-space fit's held-out error exceeds the untranslated baseline), yielding an inactive translator (0.008-0.076).
- **canon (fixed α=1e-6) — works, and is the best closed-form alignment so far**: monotone r2→full 0.011→0.864, full-rank recovery **96.3%** (vs features target 95.4%); per-path full 0.843-0.884 (spread 0.041, similar tightness to features target). Its ceiling is the canonical readout itself (T(X_p)·W_c → X_c·W_c ⇒ head reads at its own 0.9003).
- Metric caveat: the reported logit-relative distance (≈2.3) includes the intercept c — a constant per-class shift, argmax-irrelevant; acc is the honest metric.

## 4. Recovery vs ΔW energy

- At r=128, ΔW energy E(128) ≈ 0.97-0.99 while canon recovery is 0.84 (mean) — the top-128 directions capture ~98% of Frobenius energy but the *residual 2%* disproportionately carries the argmax-critical margin structure. Sharpest case: [1,3,2,9,12] E(128)=0.983 yet canon r128 recovers only 0.22 (vs 0.84 at full) — the last 17 directions of its ΔW are almost entirely "head-readout" directions.
- CE-with-bias at r128 (0.336) is far below canon r128 (0.752): the closed-form logit-space target is the stronger high-rank aligner.

## 5. Artifacts

`artifacts/fragmented-experiments/translator_bias_260826_01/` (α-selected run) and `translator_bias_260826_02/` (α=1e-6 run): `results.json` (config + crosschecks + per_path with ce/reg_pair/reg_canon + svd), `translators.npz` (A/B/b/M factors + effective head deltas + W_c/b_c), logs. Smoke: `translator_bias_smoke_260826*`. Runner: `scripts/frag_translator.py` (commit `d390828`, + `eaa1063` --reg-alpha).

## 6. Observations

- _Data collection point — no hypothesis; observations supported by the outputs above._
- The canonical-head mismatch is intrinsically ≈rank-110 (95% energy) and 1-25× the head's own norm: r ≤ 16 never had a chance, and even r=128 leaves a readout-critical tail.
- A translator bias cannot help accuracy by construction (constant logit shift), and empirically it hurts: 4/10 full-rank CE solutions collapse at unchanged train CE. The 260825 frozen-bias hypothesis is closed as refuted.
- The plan's literal logit-difference target `‖(T(X_p) − T(X_c))W_c‖²` is degenerate (cancels the readout at rank ≥ 150, useless below); the anchored variant `‖T(X_p)W_c − X_c W_c‖²` at α=1e-6 is the strongest closed-form aligner of the whole series: 0.864 full-rank (96.3% recovery), monotone through the r ∈ {2..128} grid.
- α-selection (90/10 train MSE) fails on the logit-space targets — it prefers the do-nothing α=1 for canon; the meaningful comparison required the fixed-α probe.
