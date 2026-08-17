# DeBERTaV3BaseWOS46985LayerProbe_260814_01 - gr3 step-2 data analysis (class-conditioned transition utility, TEST split)

Date: 2026-08-18 · Plan: `user_exp_plans/fragmented_exp_gr3.md` step 2 + `user_exp_plans/gr2_data_analysis_plan.md` item 1 (only item, per user instruction; items 2-3 skipped) · Inputs: `artifacts/fragmented-experiments/DeBERTaV3BaseWOS46985LayerProbe_260814_01/` (original val run + new test replay: `nodes_test.jsonl`/`nodes_test_pred.npy`, 156 nodes) · Outputs: `…/analysis_gr3/task1_class_gain.{json,npz}` · Scripts: `scripts/frag_modular_probe_wos.py` (`--replay-test`), `scripts/gr3_transition_utility_wos.py`

## 0. Test replay (prerequisite compute)

The original layer probe never persisted ridge weights, so test-split evaluation required re-forwarding train to refit (identical deterministic pipeline: per-batch longest padding, fp16 streaming, ridge α=1e-6 closed-form fp64) and scoring test. All 156 nodes re-run: 12 singles + 144 pairs (incl. (i,i) self-compositions, same convention as the original run).

**Correctness crosscheck**: each single's recomputed val acc vs the original run's val acc (assert <0.002): max dev **0.0003** (L1/L2), 10/12 at ≤0.0001. The refit reproduces the original run.

## 1. Population-level structure on test (context)

Singles (test acc): L1 0.4697 · L2 0.5179 · L3 0.5056 · L4 0.5028 · L5 0.5247 · L6 0.5316 · L7 0.5374 · L8 0.5473 · **L9 0.5561 (best)** · L10 0.5335 · L11 0.5488 · L12 0.4887. Same ordering as val (val best also L9, 0.5517).

Pair gains `A(i,j) − acc(i)`:

- Only **30/144 positive**. Best: **(1,9) +0.092 → 0.5615** (marginally above the best single), (1,11) +0.077, (1,12) +0.076, (1,8) +0.059; row-4 pairs +0.03…+0.05. Worst: (12,3) −0.185, (5,7)/(8,4) −0.147.
- Confirms the gr2 decay finding on test: appending a second layer to a mid/late single almost always hurts; **early layer L1 is the unique single that benefits from late-layer appends**.
- Generalization check: corr(test gain, val gain) over 144 pairs = **0.538**, sign agreement 49% - pair-level specifics only partially transfer (the greedy search on val partly overfits), but the L1-row structure replicates.

## 2. Class-conditioned transition utility (the step-2 question)

Definition (gr2 plan item 1): `Gain_{(i,j)|c} = acc(i,j | y=c) − acc(i | y=c)`, statistics over the 134 test classes (mean class n = **70**, min 7; ~3.5× CLINC's 20/class).

**var_c(Gain)**: mean 0.00738 (sd ≈ 0.086), max **0.02567 at (1,4)**, diag (self-compositions) mean 0.00816. Val side-check (original run preds): mean 0.00774 - consistent.

**Noise reference**: independent-probe binomial bound at p=0.45 → var0 = 0.00913. Mean var_c/var0 = **0.81** - observed variance is *below* the independence noise bound, which by itself would read as "at the noise floor" (the CLINC gr2 conclusion).

**But split-half reliability says the signal is real.** Repeated half-splits of each class's test samples (3 reps, classes n≥20, seed 17):

- corr between halves, per-cell centered gains: **0.52** (0.518 mean)
- corr between halves, class effects (mean over 132 off-diag pairs): **0.65**

Pure sampling noise is independent across halves; a correlation of ~0.5 cannot come from noise. Reconciling the two views: var0 assumes the two probes' errors are independent, but all probes share the same backbone and most errors, so the *actual* correlated sampling variance is ~3-4× smaller than var0. Variance accounting: split-half r ≈ 0.52 with half-sample noise e½ ≈ 2·e_full implies true class-conditioned heterogeneity ≈ **2/3 of observed var_c** (≈0.005), sampling ≈ 1/3 (≈0.002).

**Upgrade over gr2/CLINC**: with n≈70/class the class-conditioned transition utility is no longer unmeasurable - it is **small but reliable** (CLINC at n=20 was genuinely at the noise floor).

## 3. Which classes carry the structure

Class effect = mean over 132 off-diag pairs of the class-centered gain (positive = the class systematically benefits from second-layer appends more than its single's population does).

**Most systematic** (naive t over pairs, inflated by cross-pair correlation - magnitudes are the reliable part):

- helped: **Child abuse +0.03** (t 9.9, n=81), **Image processing** +0.03 (n=101), Geotextile (n=94), Attention (n=99); (Lorentz force law t 9.5 but n=7 - excluded from claims)
- hurt: **Atrial Fibrillation −0.03** (n=63), Asthma (n=68), Diabetes (n=70), Stealth Technology (−0.075, the largest single magnitude, n=27), Enzymology (n=119)

**L1-domain aggregation** (unweighted mean over member classes):

| L1 domain | mean class effect | n classes |
|---|---|---|
| Computer Science | +0.0088 | 17 |
| ECE | +0.0086 | 16 |
| Biochemistry | +0.0076 | 9 |
| Psychology | +0.0017 | 19 |
| Civil | −0.0026 | 12 |
| MAE | −0.0035 | 9 |
| **Medical** | **−0.0062** | 52 |

Layer appends systematically **damage Medical classes** (the domain with the largest error mass, cf. step-1 report) and favor CS/ECE. No class-size bias: corr(class effect, log n) = −0.013.

**Most heterogeneous across pairs** (var over pairs of centered gain, i.e. whose transition response depends on *which* layer is appended): Microcontroller (ECE, 0.042), Kidney Health (Medical, 0.036 - note: also the step-1 cross-model recoverability flip case), Healthy Sleep, Parkinson's Disease (+0.035 effect), Stealth Technology.

## 4. Conclusions

1. Test replay validated (≤0.0003 dev); the gr2 decay + L1-exception structure holds on test; best pair (1,9) 0.5615 ≈ best single L9 0.5561.
2. Class-conditioned transition heterogeneity is **real but small**: observed var 0.0074, of which ≈2/3 is genuine signal (split-half 0.52/0.65), 1/3 correlated sampling noise. The naive independence noise bound (var0=0.0091) overstates noise because probes share errors.
3. Structure is organized by domain: appends hurt Medical, help CS/ECE; specific reliable effects (Child abuse +, Atrial Fibrillation −, Stealth Technology −0.075).
4. Methodological note for future probes: the independence binomial bound is the wrong null for same-backbone probe families; use split-half reliability (or a shared-error variance model) instead.

## Reproduce

```bash
python scripts/frag_modular_probe_wos.py --model deberta --replay-test --deadline "..."   # 156-node test replay (~75 min GPU)
python scripts/gr3_transition_utility_wos.py                                              # analysis + split-half
```
