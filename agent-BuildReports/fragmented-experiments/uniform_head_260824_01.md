# uniform_head_260824_01 - uniform head over modularized paths (DeBERTa-v3-base × CLINC150)

Date: 2026-08-24 (run 09:33 -> 12:15, ≈2.8 h GPU) · Group: `user_exp_plans/fragmented_exp_sig1.md` · Git: `16a97b9` (dirty=False) · Single seed 17

## Config

- Question: in `mudularized_layer_probe_260813_01` every path had its OWN ridge head. Here one **uniform head** is shared by k paths, fitted on the POOLED train features of the member paths with **equal weights**, and evaluated per path on val.
- **Plan amendment (user, 2026-08-24):** the plan's "acc weighted train loss" was dropped - path acc is a posteriori (val) information and must not enter training. Equal per-path weights instead.
- Grid: k ∈ {1, 10, 50, 100, 200, 1000} × strategy
  1. **top_k** - best k paths by stored own-head val acc (selection leaky by design; leak isolated by 3),
  2. **random_k** - seeded nested shuffle of the pool (default_rng(17)); k=1 -> canonical `[1..12]`,
  3. **unleaky_k** - seeded nested shuffle of ranks 101–4445 (top-100 excluded, default_rng(18)).
- Pool: the 4445 unique random-tagged nodes of the source experiment, ranked by stored val acc (ties broken by path). Union of all 18 config members + canonical = 2393 nodes (16,163 trie edges), forwarded in two prefix-trie passes (A: train stats; B: val scoring).
- Head: ridge α=1e-6 closed-form fp64 on pooled sufficient stats (`G = ΣXᵀX − N·μμᵀ`, `H = ΣXᵀY − N·μȳᵀ`), i.e. sklearn `RidgeClassifier(alpha=1e-6, fit_intercept=True, solver='svd')` on the literally stacked data. CLS tail readout, train 15000 fit / val 3000 eval, fp16 branch stack, batch 512 - all as in the source experiment. No test access.
- gap = uniform_acc − own_acc(recomputed this run, same forward); stored own accs crosschecked separately.

## 0. Validation

| check | result |
|-------|--------|
| pooled closed form vs sklearn on stacked features (smoke, top_3) | acc **exact match**, max\|ΔW\| 1.25e-5 (eigh vs svd path noise) |
| k=1 identity (uniform head == own head) | gap exactly 0.0000 for all three k=1 configs |
| own head refit vs stored own acc (n=2392) | mean and max abs diff **0.0000** - bit-exact deterministic re-forward |
| canonical `[1..12]` own refit (fp16 stack) vs true-forward in-place L12 | 0.9003 vs 0.9047 (Δ0.0044, fp16 stack rounding envelope) |

## 1. Main grid (member-path means)

| config | k | mean own | mean uniform | gap mean ± std |
|--------|--:|---------:|-------------:|---------------:|
| top_1 | 1 | 0.9093 | 0.9093 | +0.0000 |
| top_10 | 10 | 0.8971 | 0.5284 | −0.3687 ± 0.338 |
| top_50 | 50 | 0.8870 | 0.2526 | −0.6344 ± 0.294 |
| top_100 | 100 | 0.8813 | 0.1983 | −0.6830 ± 0.241 |
| top_200 | 200 | 0.8745 | 0.1616 | −0.7129 ± 0.203 |
| top_1000 | 1000 | 0.8456 | 0.0749 | −0.7707 ± 0.099 |
| random_1 (canonical) | 1 | 0.9003 | 0.9003 | +0.0000 |
| random_10 | 10 | 0.6215 | 0.1705 | −0.4510 ± 0.284 |
| random_100 | 100 | 0.6116 | 0.0657 | −0.5459 ± 0.204 |
| random_1000 | 1000 | 0.6193 | 0.0208 | −0.5985 ± 0.202 |
| unleaky_1 | 1 | 0.7690 | 0.7690 | +0.0000 |
| unleaky_10 | 10 | 0.6907 | 0.3161 | −0.3747 ± 0.255 |
| unleaky_100 | 100 | 0.6195 | 0.0686 | −0.5508 ± 0.219 |
| unleaky_1000 | 1000 | 0.6062 | 0.0211 | −0.5851 ± 0.208 |

Chance = 1/150 ≈ 0.0067. Full 18-row table in `results.json/per_config`; random/unleaky intermediate k are monotone between the shown endpoints.

## 2. Top-10 paths under top-strategy heads (uniform acc)

| rank | path | own | top_1 | top_10 | top_50 | top_100 | top_200 | top_1000 |
|--:|------|----:|------:|-------:|-------:|--------:|---------:|---------:|
| 0 | [1,2,6,4,9] | 0.9093 | 0.909 | 0.169 | 0.021 | 0.016 | 0.005 | 0.007 |
| 1 | [1,2,7,10] | 0.9047 | 0.007 | 0.283 | 0.020 | 0.007 | 0.011 | 0.014 |
| 2 | [1,2,3,10,6] | 0.9030 | 0.009 | 0.026 | 0.007 | 0.007 | 0.014 | 0.007 |
| 3 | [1,3,5,11,10,5] | 0.8977 | 0.007 | 0.039 | 0.009 | 0.009 | 0.008 | 0.009 |
| 4 | [2,3,5] | 0.8953 | 0.006 | 0.794 | 0.261 | 0.189 | 0.152 | 0.062 |
| 5 | [1,2,5,6] | 0.8940 | 0.012 | 0.785 | 0.260 | 0.198 | 0.069 | 0.076 |
| 6 | [1,5,8,12] | 0.8927 | 0.007 | 0.795 | 0.639 | 0.468 | 0.207 | 0.051 |
| 7 | [1,3,5] | 0.8920 | 0.004 | 0.783 | 0.224 | 0.143 | 0.067 | 0.046 |
| 8 | [1,3,2,9,12] | 0.8913 | 0.006 | 0.842 | 0.685 | 0.517 | 0.292 | 0.082 |
| 9 | [2,2,9] | 0.8913 | 0.023 | 0.769 | 0.268 | 0.211 | 0.068 | 0.036 |

Under random_*/unleaky_* heads all ten paths are at chance (≤0.05) at every k (per-path detail in `results.json/top10.per_path`).

## 3. Top-100 reference set: gap mean ± std under every head

| strategy | k=1 | k=10 | k=50 | k=100 | k=200 | k=1000 |
|----------|----:|-----:|-----:|------:|------:|-------:|
| top | −0.864 ± 0.087 | −0.807 ± 0.186 | −0.742 ± 0.236 | −0.683 ± 0.241 | −0.741 ± 0.192 | −0.818 ± 0.091 |
| random | −0.874 ± 0.007 | −0.874 ± 0.008 | −0.870 ± 0.010 | −0.869 ± 0.011 | −0.868 ± 0.012 | −0.860 ± 0.027 |
| unleaky | −0.875 ± 0.008 | −0.873 ± 0.009 | −0.871 ± 0.009 | −0.871 ± 0.009 | −0.868 ± 0.015 | −0.864 ± 0.018 |

random/unleaky stds near zero = everything at chance, uniformly; top heads have real spread (the bimodality of §4).

## 4. Structure in the collapse

- **Bimodal ruin, not uniform dilution.** Under the top_10 head, its own members split: 6/10 retain 0.77–0.84 (gap −0.05…−0.12) while 4/10 fall to 0.03–0.28. The best single transfer anywhere (k≥10 heads) is [1,3,2,9,12] at 0.8417 under top_10 (own 0.8913). Survivors vs collapsed does not follow path prefix, length, or own-acc order (rank-0 [1,2,6,4,9] collapses to 0.169 under the head its own selection helped build). Unverified hypothesis: pooled-Gram domination by high-norm feature blocks - the fit serves whichever paths dominate the second-moment, others get crushed.
- **Monotone capacity decay for survivors.** e.g. [1,5,8,12]: 0.795 (k=10) -> 0.639 -> 0.468 -> 0.207 -> 0.051 (k=1000); [1,3,2,9,12]: 0.842 -> 0.685 -> 0.517 -> 0.292 -> 0.082. Every path is at chance under every k=1000 head.
- **Elite paths cluster; random pools serve nobody.** top heads beat random/unleaky heads at every k (member mean 0.5284 vs 0.1705/0.3161 at k=10; 0.0749 vs ~0.021 at k=1000) and only top heads give the top-100 reference any spread (§3). The elite cluster shares composition (8/10 start [1,2] or [1,3], all len 3–6), so pooling them is partially coherent; random/unleaky pools are too heterogeneous to serve anyone, including their own members.
- **Canonical is not privileged.** [1..12] own 0.9003; every head except its own k=1 head reads it at chance (≤0.015). The raw-pipeline representation is just another incompatible member.
- **Pairwise scale of the effect:** even k=2-style sharing fails on average - a single path's head reads any OTHER path at ~0.007 (smoke), and by k=10 the pooled mean is already −0.37 (top) / −0.45 (random).

## 5. Artifacts

`artifacts/fragmented-experiments/uniform_head_260824_01/`: `results.json` (config + per_config + top10/top100 summaries + crosschecks), `per_node.jsonl` (2393 rows: path/rank/own_stored/own_recomp/uniform acc under all 18 heads), `heads.npz` (18 heads W/b fp64). Smoke validation dir: `uniform_head_smoke_260824/` (sklearn stacked-equality + k=1 identity). Runner: `scripts/frag_uniform_head.py` (commit `16a97b9`). Runtime: pass A (train stats) 8461 s + head fits + pass B (val scoring) 1609 s.

## 6. Observations

- _Data collection point - no hypothesis; observations are supported directly by the outputs above._
- A single linear readout cannot serve multiple layer compositions: per-path own heads are mutually incompatible linear maps, and pooling them does not average - it annihilates (chance-level for almost every path-head pair; best case 0.84 vs 0.89 own with only 10 similar elite paths).
- The incompatibility is much stronger than "different layers need different heads": it holds within the elite cluster (same prefixes, lengths, accuracies) and pairs already fail.
- Selection by val acc (leaky) is visible but irrelevant to the conclusion: it buys partial coherence for the selected cluster only, and everything decays to chance by k=1000.
