# canonical_translator_260825_01 — per-path low-rank translators in front of the canonical head (DeBERTa-v3-base × CLINC150)

Date: 2026-08-25 (run ≈9 min wall; features 34 s + 55 CE configs ≈8.3 min + closed-form reg) · Plan: `user_exp_plans/sig1_recovery-Translator_exp.md` (user-written) · Git: `34da9c7` (dirty=False) · Single seed 17

## Question

sig1 (`uniform_head_260824_01`) showed a shared head cannot serve multiple path features (collapse to chance). Hypothesis to test: the failure is a **geometry mismatch** of the different paths' hidden features, and a tiny residual low-rank translator per top-10 path, `T_P(h) = h + B_P A_P h` (A ∈ R^{r×768}, B ∈ R^{768×r}, bias-free), should pull `h_P` back into a form the **canonical classification head** can read out. Canonical head (user-confirmed): ridge α=1e-6 on canonical path [1..12] CLS train features (== sig1 `random_1` head, val 0.9003).

## Config

- Pool/heads/features: exactly as `mudularized_layer_probe_260813_01` / sig1 — CLINC150, CLS tail, α=1e-6 fp64 ridge closed form, fp16 branch stack, seed 17, batch 512, train 15000 fit / val 3000 eval, no test access. Paths = sig1 top-10 by stored own val acc + canonical itself.
- **Effective-head identity**: logits = `hᵀ(W_c + Aᵀ(BᵀW_c)) + b_c` — the translated head is the canonical head plus a **rank-≤r correction with the bias frozen at b_c**. Param counts: r=2 3072, r=4 6144 (= 768×8 = 4/75 of the 768×150 = 115200 head, plan's number), r=8 12288, r=16 24576, full-rank 589824 (reference).
- **Objective ce** (end-metric): CE through the frozen head with fixed temperature s = 0.1035 (std of canonical-head logits on canonical train features, the ridge ±1 target scale; argmax-invariant — raw CE has logits at scale ~100 where Adam diverges and LBFGS line search fails, diag-verified). Adam fp32 full-batch, lr 1e-3 fixed (an 800-step lr selection ranks the post-init transient, not final quality — 3e-4 underconverges, 3e-3 unstable), warmup 250 → cosine decay to 0.1×, grad-norm clip 1.0, fixed 5000-step budget (plateau rule misfires on the post-init excursion). A=0 init (T starts exactly at identity), B ~ N(0, 1/r). Val never enters training.
- **Objective reg** (closed-form, the literal "pull back to canonical form"): reduced-rank regression `min ‖X_c − X_P − X_P W‖² s.t. rank(W) ≤ r`, `W_r = W_full V_r V_rᵀ` (V_r = top-r right singular vectors of fitted values `X_P W_full`), ridge α per path from {1e-6, 1e-4, 1e-2, 1} by 90/10 train-split reconstruction MSE. Optimization-free.
- Metrics: val acc (direct = chance baseline 0.0068, own = 0.8971 ceiling), recovery = (acc − direct)/(own − direct), and feature proximity cos/rel-L2 of `T(h_P)` vs same-utterance canonical features (before: cos 0.07–0.97).

## 0. Validation

| check | result |
|-------|--------|
| own-head refit vs stored (n=10) | max abs diff **0.0000** (bit-exact deterministic forward, as sig1) |
| canonical head vs sig1 `heads.npz` random_1 | max\|ΔW\| = 2.9e-8, val acc ours 0.9003 == sig1 0.9003 |
| fit_head vs fit_ridge_torch (smoke, canonical) | 0.0 |
| CE identity sanity (canonical path, r=2) | 0.9033 vs own 0.9003 (**above** — CE can even improve the ridge head's own decisions) |
| reg closed form vs sklearn (smoke): synthetic weights / real train preds | 1.1e-14 / 6.6e-13 (the earlier 1e-6 alpha weight-mismatch was uncentered-Gram ill-conditioning + sklearn `coef_` transpose convention) |
| reg rank: svdvals(W_r)[r:] / s₁ | 2.8e-16 (r=2) |
| reg canonical identity | W ≡ 0 exactly, acc == direct == 0.9003 |
| CE full-rank overfit caveat (canonical path) | train CE 0.025 but val 0.7947 — 589k-param reference config overfits; low-r canonical stays ≥ own |

## 1. Main result — mean over the top-10 paths (direct = chance 0.0068, own = 0.8971)

| r | params | % of head | ce acc | ce recovery | ce cos→canon | reg acc | reg recovery | reg cos→canon |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3072 | 2.7% | 0.0184 | +0.013 | 0.305 | 0.0100 | +0.004 | 0.9974 |
| 4 | 6144 | 5.3% (4/75) | 0.0550 | +0.054 | 0.294 | 0.0100 | +0.004 | 0.9980 |
| 8 | 12288 | 10.7% | 0.1789 | +0.194 | 0.312 | 0.0168 | +0.011 | 0.9984 |
| 16 | 24576 | 21.3% | 0.4074 | +0.452 | 0.325 | 0.0348 | +0.031 | 0.9988 |
| full | 589824 | 5.1× | **0.7961** | +0.887 | 0.403 | **0.8559** | +0.954 | 0.9996 |

## 2. Per-path detail (own / direct / cos-before / ce-r16 / ce-full / reg-r16 / reg-full / reg-α)

| rank | path | own | direct | cos₀ | ce r16 | ce full | reg r16 | reg full | α |
|--:|------|----:|-------:|------:|-------:|--------:|--------:|---------:|-----:|
| 0 | [1,2,6,4,9] | 0.9093 | 0.0067 | 0.081 | 0.101 | 0.785 | 0.032 | 0.873 | 1e-6 |
| 1 | [1,2,7,10] | 0.9047 | 0.0067 | 0.083 | 0.191 | 0.871 | 0.019 | 0.878 | 1e-4 |
| 2 | [1,2,3,10,6] | 0.9030 | 0.0067 | 0.078 | 0.082 | 0.402† | 0.189 | 0.875 | 1e-6 |
| 3 | [1,3,5,11,10,5] | 0.8977 | 0.0067 | 0.082 | 0.106 | 0.682† | 0.050 | 0.857 | 1e-6 |
| 4 | [2,3,5] | 0.8953 | 0.0067 | 0.079 | 0.596 | 0.876 | 0.009 | 0.865 | 1e-4 |
| 5 | [1,2,5,6] | 0.8940 | 0.0067 | 0.089 | 0.595 | 0.900 | 0.013 | 0.844 | 1e-2 |
| 6 | [1,5,8,12] | 0.8927 | 0.0090 | **0.966** | **0.709** | 0.856 | 0.017 | 0.824 | 1e-2 |
| 7 | [1,3,5] | 0.8920 | 0.0067 | 0.072 | 0.583 | 0.886 | 0.007 | 0.871 | 1e-4 |
| 8 | [1,3,2,9,12] | 0.8913 | 0.0057 | **0.787** | 0.579 | 0.842 | 0.008 | 0.839 | 1e-4 |
| 9 | [2,2,9] | 0.8913 | 0.0067 | 0.093 | 0.531 | 0.861 | 0.004 | 0.832 | 1e-2 |
| — | [1..12] canon | 0.9003 | 0.9003 | 1.000 | 0.874 | 0.795 | 0.900 | 0.900 | 1e-6 |

† CE-full underconverged (final train CE 2.83 / 0.93 vs <0.5 on the other 8; closed-form reg-full on the same paths: 0.875 / 0.857).

## 3. Structure

- **The tiny-translator hypothesis is refuted.** At the plan's own scale (r=4, 4/75 of the head) recovery is 5.4% (CE) / 0.4% (reg) of the own-minus-direct gap; even r=16 (21% of head) reaches only 45% (CE) / 3% (reg). The geometry mismatch is **not low-rank**.
- **But the mismatch IS linear.** A full-rank residual translation (bias still frozen at b_c) recovers 95% of the gap (reg full 0.856 vs own 0.897); on the 8 CE-full converged paths, CE full 0.860 — the two objectives converge to the same ≈0.86 ceiling. sig1's uniform head does not fail because paths are unalignable; it fails because a *single shared* head with no per-path adapter is asked to serve features that differ by a **full-rank linear map**.
- **Two geometries dissociated (the cos story).** reg at r=2 already moves cos→canonical to 0.997 (bulk geometry is rank-2 fixable) yet readout stays at chance; CE at full rank reads out at 0.80–0.86 with cos only 0.24–0.40 (features are NOT canonical-like at all). The head has a 618-dim null space (768→150): "canonical-like" and "head-readable" are different geometries, and the regression objective spends its rank budget on the former (useless for readout) while CE spends it on the latter.
- **Closer bulk geometry needs less rank.** cos₀ splits the elite cluster: 8 paths nearly orthogonal to canonical (cos 0.07–0.09) vs [1,5,8,12] (0.966) and [1,3,2,9,12] (0.787) — the two containing layer 12. CE r16 correlates +0.507 with cos₀ (best 0.709 for [1,5,8,12]). The readout gap to close is larger when the bulk geometry is far away.
- **Full-rank translator cost is uniform and small.** reg-full per-path 0.824–0.878 (spread 0.054, own spread 0.018): the frozen bias b_c + regression imperfection cost ≈0.04 across the board, no path specially damaged.
- **Optimization honesty.** CE-full underconverged on 2/10 paths (train CE 2.83/0.93); the closed-form reg-full on those same paths (0.875/0.857) shows it is an optimizer artifact, not representational. The dual-objective design pays off exactly here.

## 4. Artifacts

`artifacts/fragmented-experiments/canonical_translator_260825_01/`: `results.json` (config + crosschecks + per_path with both objectives, summary per r), `translators.npz` (W_c/b_c + per-config A/B factors + effective head deltas + reg W_full, fp32), `canonical_translator_260825_01.log`. Smoke: `canonical_translator_smoke_260825/` (subsets; all machine checks pass). Runner: `scripts/frag_translator.py` (commit `34da9c7`).

## 5. Observations

- _Data collection point — no hypothesis; observations are supported directly by the outputs above._
- A rank-≤16 residual correction of the canonical head cannot read out any of the elite paths (≤45% recovery even at r=16); at r=4, the plan's scale, it is ~chance.
- A full-rank residual correction can: ≈0.86 mean (95% of the own-head gap) through the frozen canonical head, from both an end-to-end CE and a closed-form feature regression — a shared canonical head is *usable* by all top-10 paths given per-path full linear adapters.
- "Pull back to the canonical form" (feature regression) and "become readable by the canonical head" (CE) are different objectives with different rank requirements: the low-rank regression pulls the bulk geometry (cos 0.998 at r=2) to no readout benefit; readout requires the discriminative alignment, which is full-rank.
- Open question this run leaves: where between r=16 and r=768 does recovery transition (reg jumps 3% → 95%)? A finer rank sweep (r ∈ {32, 64, 128, 256}) on the stored `W_full` would locate it (requires re-forwarding train features only, ~1 min).
