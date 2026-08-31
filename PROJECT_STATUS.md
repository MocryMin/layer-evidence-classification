# Project Status

## Current state

**EXP-004 H2 operational semantics — SPECIFIED, NOT RUN (2026-08-31).**
The user-approved implementation details live in
`configs/exp004_h2_mcts_v1.yaml` and the pure CPU reference implementation
`src/exp004_h2_mcts.py`; the main EXP-004 document remains at the scientific
protocol level.  Unit tests cover action indexing/repetition, root and visit
semantics, UCB, random tuning/control, grid tie-breaking, Wilson lower bounds,
and canonical/alternative path reporting.  No H2 validation/test data or GPU
run has been accessed.  Next action: build and benchmark the model-forward
runner against this frozen interface before enabling validation or test.

**EXP-004 H1 engineering qualification + structured pilot — COMPLETE
(2026-08-27; non-confirmatory).** A new Llama-3.2-3B-Instruct × ARC-Easy
runner now supports arbitrary repeatable decoder paths, valid-choice masking,
required timezone-aware stop times, atomic 64-sample shards, signal-safe stop
and resume. Modular canonical features/logits are bit-exact to native forward;
the full 1750/501 official-train qualification uses about 6.13 GiB peak CUDA.
`D_fit`-only 5-fold CV selected task-head L2=0.3 (CV 0.8949±0.0156; refit
discover 0.9182 vs native canonical 0.9042). A fixed 84-path near-canonical
pilot found 57 good paths and one audited provisional collapse:
`[1,...,28,28]` task=0.9182 but native=0.2675 (chance=0.2500; absolute gap
0.6367). Raw forward and refit reproduce bit-exact at the original batch size.
This is **not official H1 evidence**: prompt/search protocol was provisional,
only one source supplied a witness, and validation/test were not accessed.
Report: `agent-BuildReports/experiments/EXP-20260827-004-h1-structured-pilot.md`;
artifacts: `artifacts/EXP-20260827-004-h1-{qualification,
head-qualification-v2,structured-pilot}/`.

**EXP-004 design phase — ACTIVE (2026-08-26).** The main question is no longer
generic path vocabulary construction.  It is whether a frozen canonical
readout imposes an implicit compatibility/admissibility constraint on
alternative layer-path search.  The current design target is:

- H1: task-wise existence + prevalence of paths with a strong path-specific
  low-parameter diagnostic head but weak canonical/native readout;
- H2: sample-wise existence of shorter-correct or error-recovering paths under
  the frozen canonical head, using a prior-work-style MCTS protocol plus a
  matched random-search control;
- ACO is deferred to a later matched search-algorithm comparison; routing
  vocabulary / 2-gram minimisation is deferred to EXP-005.

The immediate work is prior-related review and a complete EXP-004 protocol
document before experiments begin.  The host-side EXP-004 draft still reflects
an older ACO-heavy design and is not yet the execution specification.

**Translator-bias audit correction (2026-08-26).** The original bias-on
evaluation counted `b_T @ W_c` twice and incorrectly described its
class-specific logit shift as argmax-invariant.  `scripts/frag_translator.py`
is corrected and covered by three algebraic regression tests (full suite
55/55).  A deterministic correction run preserves all 541 translator arrays
bit-exactly and changes only evaluation: mean CE accuracy over the top-10 paths
is r2/r4/r8/r16/r32/r64/r128/full =
0.0273/0.0788/0.2281/0.4195/0.6170/0.7160/0.7437/0.7900.  Report:
`agent-BuildReports/fragmented-experiments/translator_bias_evalfix_260826_01.md`;
artifact: `artifacts/fragmented-experiments/translator_bias_evalfix_260826_01/`.

**gr2 tasks 1-3 × WOS-46985 runs — COMPLETE** (runner
`scripts/frag_modular_probe_wos.py`; reports:
`agent-BuildReports/fragmented-experiments/{DeBERTaV3BaseWOS46985LayerProbe_260814_01,
ModernBERTBaseWOS46985LayerProbe_260814_02}.md`; C:
`Qwen3Emb0p6bWOS46985Baseline_260814_01.md`). Key results:
- **DeBERTa**: singles (best L9 0.5517) beat in-place at EVERY layer (gap
  +0.04..+0.18 growing with depth — opposite of CLINC gr2); greedy decays
  monotonically 0.5375→0.0185, 43/49 negative steps; best pair [1,9] 0.5565.
- **ModernBERT**: inverse — singles ≪ in-place for L≥2 (L2: 0.145 vs 0.457;
  layers are input-distribution-dependent), in-place matches baseline
  exactly (L1 0.5068/L22 0.4967); greedy max 0.5215 @ step 1 then gentle
  decay to 0.4384; best pair [1,22].
- **Qwen3 baseline**: L28 ridge 0.6536 / ln_plain 0.6403 / plain 0.6152 val
  (~2× DeBERTa CLS baseline); oracle gain +0.17..+0.29.
- Method note: fixed-512 padding systematically lowers modernBERT accs by
  ~0.02-0.03 (deberta insensitive); runner switched to per-batch longest
  padding (commit `9887739`), ModernBERT run redone with corrected
  semantics.

**Fragmented experiments gr2** (`user_exp_plans/fragmented_exp_gr2.md`) is **complete**
(2026-08-13→14, commit `3cf29fb`+report commits): one data-collection
experiment — `mudularized_layer_probe_260813_01`, DeBERTa-v3-base layer
modules composed into arbitrary repeatable sequences (CLS readout at tail,
ridge α=1e-6, CLINC150, val-only): 22,046 distinct path nodes (single,
pairwise, greedy-to-50, 4,500 random paths). Report
`agent-BuildReports/fragmented-experiments/mudularized_layer_probe_260813_01.md`;
follow-up data analysis per the user's plan
(`user_exp_plans/gr2_data_analysis_plan.md`, commit `fad165e`, report
`agent-BuildReports/fragmented-experiments/mudularized_layer_probe_260813_01_analysis.md`):
class-conditioned transition utility is at the n=20/class noise floor;
length dominates raw acc (R² 0.46), after residualization start_layer is the
main signal (RF OOB R² 0.19); vocab top bigram `[1,11]`, worst `[12,12]`.
Artifacts `artifacts/fragmented-experiments/mudularized_layer_probe_260813_01/`
(see the gr2 section below).

**Fragmented experiments gr1** (`user_exp_plans/fragmented_exp_gr1.md`) is **complete**
(2026-08-12, commit `9504dc8`): 5 single-point data-collection experiments —
side verification of EXP-001/003 on Qwen3-Embedding-0.6B (last-token) and
modernBERT-base (CLS) over CLINC150, plus WOS-46985 statistics and 134-L2
baselines on DeBERTa-v3-base and modernBERT-base (HYDRA-count split
30,070/7,518/9,397, seed 17). Reports under
`agent-BuildReports/fragmented-experiments/*_260812_*.md`, artifacts under
`artifacts/fragmented-experiments/` (see the gr1 section below).

EXP-20260810-003 (validated-probe recoverability verification) is **complete**
(report at
`agent-BuildReports/experiments/EXP-20260810-003--validated-probe-recoverability.md`).
**$H_1$, $H_1'$, $H_2$ are all very strongly supported** across all three probe
families (centered plain, LN plain, ridge). Mid layers are non-inferior to,
superior to, and recoverable over the final layer on frozen DeBERTa-v3-base /
CLINC150. The EXP-001 mainline pause is lifted; its verification is complete.

EXP-20260729-001 (original plain-probe mainline) is **superseded** by EXP-003:
its pure-linear-probe protocol gave false negatives on mid layers (optimisation
collapse). EXP-20260729-002 diagnosed the cause and is **complete** (report at
`agent-BuildReports/experiments/EXP-20260729-002--linear-probe-midlayer-collapse.md`).

Key finding: frozen DeBERTa-v3-base mid-layer CLS has near-zero inter-sample
variance (a near-constant component), so a plain linear probe under first-order
AdamW/SGD collapses to near-random - but the class signal IS linearly
extractable. The collapse is an ill-conditioning/optimisation artefact, not
absence of signal. **Task 03g (RidgeClassifier α grid)** showed a closed-form
least-squares solve at α≈1e-6 (OLS) recovers mid layers to 0.917 (L6) / ≈0.94
(L7-10), beating every CE probe tried - so the "collapse" is a probe-methodology
artefact, not a feature property. **Task 03h (Adam ablation)** decomposed the
AdamW failure into four factors (none fundamental): mini-batch noise (dominant;
batch=256 fails for every init/wd), wd=0.01 (caps acc at ≈0.55), no early
stopping + overparameterised head (CE peaks at ep8 then overfits), and slow
convergence from Xavier. **OLS-init + full-batch CE + early stop peaks at 0.919
@ep8** (above OLS, with calibrated probabilities) - the tested probe for EXP-001.

## Latest valid result

**EXP-003 (validated probes, 10 seeds, test accuracy):**
- **Centered plain (primary, candidate L11):** L11 0.862 > L12 0.852
  (d2=+0.009, CI [+0.008,+0.010]); oracle gain 0.115, R_oracle 0.78, D_JS 0.020.
  92% of runs non-converged (hit 1000ep); signal still clear.
- **LN plain (control, candidate L10):** L10 0.875 > L12 0.839
  (d2=+0.036, CI [+0.033,+0.039]); oracle gain 0.143, R_oracle 0.89, D_JS 0.007.
- **Ridge (reference, candidate L10, α=0 OLS):** L10 0.926 > L12 0.885
  (d2=+0.041); oracle gain 0.097, R_oracle 0.84. L7-L10 all 0.925-0.928.
- **Cross-family verdict: H1/H1'/H2 all very_strong.** Recoverability is broad
  (D_JS 0.007-0.020), not concentrated in a few classes. L6 is the hardest mid
  layer (centered 0.295, LN 0.602, ridge 0.914).
- Artifacts: `artifacts/EXP-20260810-003/results.json` (gitignored; README
  documents layout). MLflow run `364bf897628d443189b1ae6f1288fc6e`.

EXP-002 diagnostics (seed 17, frozen backbone unless noted), validation accuracy:
- Plain linear probe (AdamW, best lr 1e-2): layers 1/6 fail (0.13/0.03), layer 12 = 0.79.
- **EXP-001 mainline plain probe (lr=1e-3, 10 seeds, 100ep - post-hoc, saved):**
  full 12-layer collapse curve. Mid layers 4-8 near-random (L6 0.026 ± 0.002,
  L8 0.035), L12 0.608 ± 0.003; small std => collapse is seed-robust. Test tracks
  val. (`artifacts/EXP-20260729-001/plain_probe_mainline/results.json`; reproduces
  `smoke_lr_results.json` on seed-17 {1,6,12}.)
- LBFGS plain probe (30 epochs): every layer 0.41-0.86 (layer 6: 0.03 -> 0.41; layer 3 > layer 12).
- LN head (AdamW, 1e-2): every layer 0.65-0.90 (layer 6: 0.65; layers 7/10 > layer 12).
- norm-only fails on mid layers; affine-only rescues mild layers but not severe ones.
- **Feature stats + acc per layer in one table:**
  `artifacts/EXP-20260729-002/02_variance_collapse/per_layer_collapse_summary.json`
  (inter_std / participation_ratio / top1 + plain-AdamW / LBFGS / OLS / LN acc).
- **RidgeClassifier α=1e-6 (≈OLS, task 03g): every layer 0.79-0.94 (layer 6:
  0.917; layers 7-10 ≈0.94 > layer 12 ≈0.90).** Best linear accuracy on the
  frozen base. α=10 fails (over-regularised); best α is 1e-6 for all mid layers.
  Prompt-independent (with-prompt 0.917 vs no-prompt 0.912 at L6).
- **Adam ablation (task 03h, layer 6):** mini-batch 256 fails for every init/wd
  (OLS-init 0.917->0.61 wd0 / 0.023 wd0.01; Xavier 0.03-0.04). Full-batch rescues:
  OLS-init + full-batch CE peaks **0.919 @ep8** (then overfits to 0.79 by 20k);
  Xavier + full-batch wd=0 climbs to 0.70 @20k (still slow), wd=0.01 plateaus 0.55.
- **MLP probe (task 04):** matched-parameter MLP (919r, r=128 ~= plain params)
  fails on EVERY layer incl. L12 (uniform-prediction collapse, all lr/r) - the
  constant CLS component + ReLU forms a uniform attractor. Dead-ReLU diagnosis:
  features are predominantly POSITIVE (L6: 85% dims mu-delta>0) - the pervasive
  dead ReLU (46% -> 100% within epoch 1, batch 29/58) is driven by shared
  logits across samples (|logits[0]-logits[1]|=0) -> aligned gradient, not by
  negative features. Centering the features (fixed linear transform) rescues:
  centered plain L6 0.317, centered MLP r=256 L6 0.737 / L7 0.913 > L12 - but
  OLS (0.917 L6) still wins. Collapse is NOT a linearity limitation.
- **Activation ablation (task 05):** none/relu/leaky/gelu/sigmoid all fail on
  raw features (L6 best 0.007-0.010, all r). relu/gelu dead-lock at uniform
  (loss=ln(150), neg=1.00); leaky stalls 0.8 within uniform (slope too weak);
  sigmoid (bounded-saturating) dead-locks at uniform on 8/12 layers, stalls
  4.86-5.54 on the rest (still chance acc) - all four activation families
  covered; 2-layer linear (no act) cannot dead-lock but diverges (loss
  7.9-47.6) and fails even at L12. The uniform attractor is NOT ReLU-specific
  - it is a property of the near-constant features; only the 1-layer
  bias-carrying head escapes it.
- **Token-position check (task 06): the compression is CLS-specific.** On a
  2000-sample subset (CLS values reproduce full-set stats), non-CLS tokens
  have healthy inter-sample std at every layer - at L6 (CLS 2e-4) all non-CLS
  positions are >= 0.11 (500-2000x larger, zero compressed positions). The
  mid-layer representation space is NOT collapsed; only the CLS pooling token
  is. Follow-up: mean-pooling readout may escape the collapse.
- Removing the instruction prompt does not fix the collapse (intrinsic to the backbone).
- Full-FT backbone: last layer 0.967 test acc (too few errors for class-wise
  recoverability); FT does not fix mid-layer collapse but LN on the FT backbone
  reaches 0.93-0.97 on mid layers.

## Freeze (2026-08-10)

EXP-001/002 evidence is frozen for reproducibility:

- **Git tag** `exp-001-002-freeze-20260810` (on `5ad36f4`): code, configs, reports,
  tests, small summaries, manifest. `main` advanced to `f55481b` (HF URI recorded).
- **HF private dataset** `MocryMin/lec-exp-001-002-freeze` (759 MB, 51 files): all
  artifact result JSONs, FT-backbone best checkpoint, frozen `mlruns.db` snapshot,
  manifest + dataset README. 1.66 GB rebuildable CLS caches excluded.
- **Manifest:** `agent-BuildReports/freeze-20260810/manifest.md` (sha256 + sizes).
- Per-sample predictions/logits, plots, and probe checkpoints were never produced
  (experiments record aggregate JSON metrics + per-epoch histories) - marked N/A.

## Convergence probe (2026-08-12, follow-up to EXP-003 §5)

Variable-max_ep probe (`scripts/exp003_convergence_probe.py`, artifacts under
`artifacts/EXP-20260810-003/convergence_probe/`): one long run per
(family, layer) at seed 17, max_ep=20000, early stop disabled; budgets
replayed from the full history.

**Result: all 24 runs converge within 20000 ep.** Centered plain 857-8954 ep,
LN 625-4057 ep (2-4x faster). Converged accuracies far exceed the 1000ep cap
(e.g. centered L6 0.295->0.762, L7->0.923 > L12 0.884; LN L6 0.602->0.881).
The 1000ep non-convergence is slow convergence (~10-20x beyond the cap for
mid layers), not a probe pathology; the §2 superiority verdict strengthens
with convergence. min_delta=1e-4/patience=100 early stopping is strict and
can cut off slow tail gains (centered L4 would stop at 8954 with ~0.72 vs
0.807 at 20000).

## Fragmented experiments gr1 (2026-08-12, complete)

Single seed 17, frozen backbones, full-batch AdamW (10k ep, ES on val),
probe families: gradient (plain | centered_plain by variance verdict) +
LN plain + ridge grid. Reports/artifacts per AGENT_PROTOCOL §9.

- **exp1 Qwen3Emb0p6bExp1Ver** (Qwen3-Emb-0.6B last-token x CLINC150):
  healthy variance (min 0.030). L28 plain 0.9524 / ln 0.9580 / ridge 0.9536;
  ridge mids L23 0.9556 / L25 0.9567 edge above L28. Oracle gain +0.041/+0.036/+0.035.
- **exp2 ModernBERTBaseExp1Ver** (modernBERT CLS x CLINC150): healthy; final
  L22 weak (plain 0.8253 vs deberta 0.852-0.885, qwen3 0.952). **Early layers
  best: plain L2 0.8704 > L22**; L16 dip 0.697. Strongest oracle gains yet:
  +0.146/+0.149/+0.117, D_JS 0.009-0.021, coverage 146/150.
- **exp3 WOS46985Features**: 46985 docs; chars mean 1376 (median 1354), tokens
  mean 262 (median 250, max 1463); L1: Medical 31.1%, Psychology 15.2%, CS
  13.8%, biochemistry 12.1%, ECE 11.7%, Civil 9.0%, MAE 7.0%; all 134 L2
  classes present in every split; no duplicates.
- **exp4 DeBERTaV3BaseWOS46985Baseline** (deberta CLS x WOS 134-L2):
  **mid-layer collapse REPRODUCES on WOS** (inter_std 2.4e-4 @ L6). L11 >
  L12 under centered (0.320 > 0.281) and ln (0.311 > 0.285); ridge L5 0.4978
  >> L12 0.3003 (+0.197). Oracle gains +0.363/+0.508/+0.438, R_oracle
  0.51/0.71/0.63, coverage 134/134.
- **exp5 ModernBERTBaseWOS46985Baseline** (modernBERT CLS x WOS 134-L2):
  healthy; plain L1 0.4881 ≈ L22 0.4974, mids ~0.35. Oracle gains
  +0.292/+0.308/+0.226, coverage 134/134.

Cross-cutting: recoverability (H2) positive on every backbone/dataset; the
deberta CLS collapse is dataset-independent; modernBERT CLS is healthy but a
weak readout - its early layers carry the strongest signal; all 10k-ep
gradient runs converged (early_stop).

## Fragmented experiments gr2 (2026-08-13→14, complete)

`mudularized_layer_probe_260813_01` — DeBERTa-v3-base layer modules composed
into arbitrary repeatable sequences, CLS readout at tail, ridge α=1e-6
(closed-form fp64, bit-equal to sklearn solver='svd'; modular raw chain
bit-equal to the true model forward), CLINC150, train fit / val acc, seed 17,
22,046 path nodes (prefix-trie reuse). Key numbers:

- **Task 1 (single-layer-only vs in-place):** every layer ALONE is a strong
  classifier (0.825–0.875 val); only L2 beats its in-place self (+0.004);
  layers best in-place lose most in isolation (L7–10: −0.0797…−0.0943).
  In-place α=1e-6 reproduces EXP-003 (L6 0.9173/0.9170, L10 0.9420/0.9420,
  L12 0.9047/0.9030).
- **Task 2 (greedy to len 50):** starts L2 (0.8750), step 1 +L11 → **0.8827**
  (peak, == task-3 argmax), then 47/49 negative steps, monotone decay to
  0.2183 at len 50; **max acc occurs BEFORE any negative step** (no recovery).
- **Task 3 (pairwise):** A mean 0.8490, max `[2,11]` 0.8827; G max +0.0463
  for `[1,2]` (raw-pipeline prefix); 42/144 pairs super-additive (top `[1,1]`
  +0.0367).
- **Task 4 (4500 random paths, len 3–12):** acc mean 0.6241, max 0.9093
  (`[1,2,6,4,9]`); monotone decline with length (len3 0.8365 → len12 0.4136);
  8/10 top paths start with `[1,2]`; repeats degrade (0.5791 vs 0.7865);
  tail layer matters little (0.57–0.66).

Infra: `scripts/frag_modular_probe.py` (arbitrary layer-sequence composition,
trie DFS with branch-stack fp16 cache, resumable JSONL, wall-clock deadline).

## Active blockers

- EXP-004's official prompt, exact multi-source generator/temperature/max path
  length, validation-confirmation wording and H2 controls are still being
  finalised. The infrastructure and non-confirmatory H1 pilot are valid, but do
  not start an official EXP-004 run from the provisional configs.
- Recent fragmented work remains local-only until the Git/report correction
  commits and a new artifact freeze are synchronised.
- EXP-003 is complete; its historical manual log is not on the critical path
  for EXP-004 and is not being reopened during the current deadline.

## Next action

1. Complete the EXP-004 prior map and freeze the official H1/H2 protocol.
2. Incorporate the qualified implementation facts (A--E mask, final-RMSNorm
   shared input, D_fit-only head CV, L2=0.3 candidate, required stop-at) without
   treating the structured-pilot witness as confirmatory evidence.
3. After protocol freeze, create a new official run ID/config and run H1
   discovery in explicitly timed resumable segments; validation/test remain
   locked until the candidate set and all thresholds are frozen.
4. Commit and synchronise the translator evaluation correction and recent
   local Git history; create a checksummed private freeze for recent artifacts.
5. Keep historical EXP-003 statistical/log amendments as a separate
   non-blocking addendum; do not divert the EXP-004 deadline to rewriting it.

## Important paths

- `agent-BuildReports/experiments/EXP-20260810-003--validated-probe-recoverability.md` - EXP-003 report
- `artifacts/EXP-20260810-003/` - EXP-003 data (gitignored; README documents layout)
- `agent-BuildReports/experiments/EXP-20260729-002--linear-probe-midlayer-collapse.md` - diagnostic report
- `artifacts/EXP-20260729-002/` - diagnostic data (gitignored; README documents layout)
- `artifacts/EXP-20260729-001/cache/` - shared frozen-backbone CLS cache (reused by EXP-003)
- `configs/exp003_config.yaml` - EXP-003 config; `configs/diag_config.yaml` - EXP-002; `configs/exp_config.yaml` - EXP-001
- `models/deberta-v3-base-clinc150-ft/` - FT backbone (task 3c, gitignored)
- `user_exp_plans/EXP-20260729-001--*/` - EXP-001 RP log + AgentProtocol
- `user_exp_plans/EXP-20260810-003-*/` - EXP-003 plan + AgentProtocol
- `user_exp_plans/fragmented_exp_gr1.md` - gr1 plan; `agent-BuildReports/fragmented-experiments/` - gr1 reports; `artifacts/fragmented-experiments/` - gr1 artifacts (gitignored)
- `user_exp_plans/fragmented_exp_gr2.md` - gr2 plan; `scripts/frag_modular_probe.py` - gr2 runner; `agent-BuildReports/fragmented-experiments/mudularized_layer_probe_260813_01.md` - gr2 report; `artifacts/fragmented-experiments/mudularized_layer_probe_260813_01/` - gr2 artifacts (gitignored); `user_exp_plans/gr2_data_analysis_plan.md` + `scripts/gr2_analysis.py` + `mudularized_layer_probe_260813_01_analysis.md` - gr2 data analysis
