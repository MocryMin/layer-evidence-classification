# mudularized_layer_probe_260813_01 — data analysis

Date: 2026-08-14 · Plan: `user_exp_plans/gr2_data_analysis_plan.md` (user-written) · Reporting model: deepseek-v4-flash · Git: `986a447` (dirty=True, analysis script/report uncommitted at write time) · Inputs: `artifacts/fragmented-experiments/mudularized_layer_probe_260813_01/` (nodes.jsonl 22,046 + nodes_pred.npy + random_paths.json) · Outputs: `…/mudularized_layer_probe_260813_01/analysis/` · Script: `scripts/gr2_analysis.py`

## 1. Class-conditioned transition utility

Definition: `Gain_{(i,j)|c} = acc(i,j | y=c) − acc(i | y=c)`, `var_c(Gain)` over the 150 classes → 12×12 matrix (`task1_class_gain_var.npy/.json`, mean-gain matrix included).

- var_c(Gain): mean 0.00644, min 0.00258, **max 0.01696 at `[5,6]`**; top pairs `[5,6]` 0.0170, `[1,10]` 0.0144, `[5,5]` 0.0141, `[12,3]` 0.0139, `[12,12]` 0.0138, `[1,4]` 0.0135.
- **Noise floor**: each class has only 20 val samples → per-class acc SE ≈ 0.08, so var of an acc *difference* from noise alone is up to ~0.0127 (independence bound). The observed mean (0.0064) sits at/below that floor. **The class-conditioned utility of transitions is not distinguishable from sampling noise at n=20/class** — no pair shows a strong class-specific gain pattern; the matrix is delivered as data, to be re-read only with per-class denominators of ~100+.

## 2. Path feature regression (task-4 paths)

4,445 unique paths after removing 55 exact duplicates. Features per path: len, repeat_count (adjacent repeats; `repeat_count_all` = extra occurrences also stored), backward_jump_count, canonical_adjacent_edge_count, longest_canonical_run, start_with_1, start_with_12, distinct_layer_count, distinct_ratio, repeat_ratio/(len−1), backward_jump_ratio/(len−1), canonical_edge_ratio/(len−1), longest_canonical_run_ratio, start_layer, tail_layer (CSV: `task2_path_features.csv`). `acc^res(P) = acc(P) − mean acc over same-length paths` (μ_k, σ_k in `task2_regressions.json`).

**a) Linear: len/start_layer/tail_layer on raw acc** — R² in-sample 0.4615, held-out 0.4725. Nearly half of raw accuracy variance is explained by three features (dominantly length).

**b) Linear: all features on acc^res** — R² in-sample 0.0607, held-out 0.0213. Weak: after length residualization, path structure is barely linearly predictable.

**c) RandomForest (500 trees) on acc^res** — R² OOB 0.1923, held-out 0.1229. The non-linear model recovers real structure the linear one misses. Importances: **start_layer 0.310**, tail_layer 0.141, distinct_ratio 0.098, backward_jump_ratio 0.084, len 0.063, longest_canonical_run_ratio 0.062, distinct_layer_count 0.059, repeat_ratio 0.056.

Correlations with acc^res (sorted by |r|):

| feature | r | feature | r |
|---|---|---|---|
| start_with_1 | +0.206 | repeat_ratio | −0.053 |
| start_layer | −0.136 | tail_layer | +0.041 |
| start_with_12 | +0.082 | backward_jump_ratio | −0.021 |
| distinct_ratio | +0.020 | canonical_edge_ratio | +0.011 |

**What explains performance:** raw acc is dominated by length (monotone decline, R²≈0.46). After removing length, the strongest remaining signal is the **start position** — starting at layer 1 helps (start_with_1 +0.21; RF's top feature is start_layer) — i.e., receiving the trained input distribution (embeddings) early matters, consistent with task-1's finding that layers are input-distribution-dependent. Local structure (repeats, jumps, canonical edges) explains little beyond this; repeats are mildly negative.

## 3. Weighted bigram/trigram vocabulary

Definition: paths ranked by acc; per occurrence weight = `acc^res/σ_|P|/(len−1)` for bigrams, `/(len−2)` for trigrams (aggregated over deduped paths; `task3_vocab.json`).

- Bigrams (top): `[1,11]` 17.82, `[2,7]` 14.77, `[2,8]` 13.89, `[2,6]` 13.07, `[1,3]` 13.02, `[1,2]` 12.65, `[2,11]` 12.42, `[2,5]` 11.78, `[11,1]` 11.76, `[11,2]` 11.32.
- Bigrams (bottom): `[12,12]` −37.88, `[4,4]` −22.20, `[4,12]` −19.85, `[12,3]` −19.12, `[12,2]` −19.00 — self-repeats and layer-12-involving edges are the most damaging per occurrence.
- Trigrams (top): `[2,6,9]` 3.49, `[1,2,11]` 3.35, `[2,2,11]` 3.25, `[5,2,8]` 2.94, `[1,5,2]` 2.93, `[2,8,11]` 2.92, `[2,6,8]` 2.89, `[1,11,4]` 2.73, `[1,10,5]` 2.72, `[11,4,2]` 2.70.
- Trigrams (bottom): `[4,12,12]` −8.44, `[12,12,2]` −7.95, `[12,3,2]` −6.93, `[12,12,5]` −6.32, `[12,3,3]` −5.77.

## 4. Observations

- _Data collection point — no hypothesis; observations are supported by the outputs above._
- Transition utility (task 1) is at the noise floor of the current per-class sample size (20 val/class); a class-wise claim would need per-class denominators ≥ ~100 (e.g., the WOS-46985 134-L2 setting from gr1, where den_c median 50 and 115/134 classes ≥ 30).
- Length is the master variable (R² ≈ 0.46 alone); residual performance is start-position-driven, not local-structure-driven — the vocabulary's most negative edges (`[12,12]`, `[4,4]`, `[4,12]`…) agree with the repeat/backward findings of the main report.
