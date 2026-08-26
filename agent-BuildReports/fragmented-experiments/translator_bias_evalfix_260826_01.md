# translator_bias_evalfix_260826_01 — corrected CE-with-bias evaluation addendum

Date: 2026-08-26 · Parent report: `translator_bias_260826_01.md` · Corrected
artifact: `artifacts/fragmented-experiments/translator_bias_evalfix_260826_01/`
· Seed 17 · train/validation only; test not accessed

## Status and scope

This addendum replaces only the CE-with-translator-bias accuracy evaluation in
`translator_bias_260826_01.md`.  It does not replace the bias-free 2026-08-25
run, the ΔW SVD analysis, or the closed-form pair/canon regression results.

The original evaluation used transformed features that already contained
`b_T`, then also supplied the effective head bias `b_c + b_T^T W_c`.  Its
logits therefore contained `2 b_T^T W_c`, whereas training contained one copy.
The corrected evaluation is

```text
T(X) @ W_c + b_c
```

with `T(X)` containing `b_T` exactly once.  Equivalently, one may use the
bias-free transformed features with `b_c + b_T^T W_c`, but the two forms must
not be mixed.

The prior interpretation that `b_T^T W_c` is argmax-irrelevant was also
incorrect.  It is constant across samples but generally varies across the 150
classes, so it can change relative logits and argmax.

## Isolation check

The corrected run used the same configuration as
`translator_bias_260826_02` (fixed regression α=1e-6).  All 541 arrays in the
two `translators.npz` files have the same keys and are bit-identical
(`max_abs_array_diff = 0.0`).  Thus the correction isolates evaluation: no
translator state, canonical head, SVD result, or regression state changed.

The new implementation adds three algebraic regression tests, and the full
suite passes 55/55.

## Corrected CE result — mean over top-10 paths

Direct canonical-head baseline on these path features is 0.0068; mean own-head
accuracy is 0.8971.

| rank | original double-count acc | corrected acc | correction | corrected recovery |
|---:|---:|---:|---:|---:|
| 2 | 0.0087 | 0.0273 | +0.0186 | 0.023 |
| 4 | 0.0151 | 0.0788 | +0.0637 | 0.081 |
| 8 | 0.0349 | 0.2281 | +0.1932 | 0.250 |
| 16 | 0.0823 | 0.4195 | +0.3372 | 0.465 |
| 32 | 0.1600 | 0.6170 | +0.4570 | 0.686 |
| 64 | 0.2534 | 0.7160 | +0.4626 | 0.797 |
| 128 | 0.3356 | 0.7437 | +0.4081 | 0.828 |
| full | 0.5286 | 0.7900 | +0.2614 | 0.880 |

## Full-rank detail

| path | original double-count | corrected | corrected train CE | status |
|---|---:|---:|---:|---|
| [1,2,6,4,9] | 0.0793 | 0.7797 | 0.4178 | converged less fully than best paths |
| [1,2,7,10] | 0.3070 | 0.8643 | 0.0922 | converged |
| [1,2,3,10,6] | 0.0133 | 0.3660 | 2.8460 | underconverged |
| [1,3,5,11,10,5] | 0.0310 | 0.6740 | 0.9709 | underconverged |
| [2,3,5] | 0.8033 | 0.8760 | 0.0000 | converged/overfit train |
| [1,2,5,6] | 0.7597 | 0.9000 | 0.0000 | converged/overfit train |
| [1,5,8,12] | 0.8557 | 0.8570 | 0.0011 | converged |
| [1,3,5] | 0.7993 | 0.8910 | 0.0000 | converged/overfit train |
| [1,3,2,9,12] | 0.8237 | 0.8250 | 0.0000 | converged/overfit train |
| [2,2,9] | 0.8137 | 0.8667 | 0.0000 | converged/overfit train |

Excluding the same two underconverged paths identified in the 2026-08-25
bias-free analysis, corrected bias-on full-rank mean is 0.8575 versus 0.8596
for bias-free CE.  This single-seed diagnostic therefore provides no evidence
that adding `b_T` improves the final mean, but it also does not show the
previously claimed bias-induced collapse.  The defensible conclusion is:

> Translator bias is a valid class-specific readout correction and can change
> predictions.  Under this fixed optimization budget it did not improve the
> converged-path mean over the bias-free run; the earlier collapse and the
> claimed mathematical impossibility of improvement were evaluation and
> interpretation errors.

## Provenance

The run was executed from commit `d380b86` with a dirty working tree because
the evaluation fix, its tests, this audit correction, and the pre-existing
user plan append had not yet been committed.  Exact SHA-256 values and the
command are recorded in the corrected artifact's `provenance.json`.  Original
artifacts remain unchanged.

AI reporting statement: Codex identified the evaluation error, implemented the
fix and regression tests, executed the correction run, compared artifacts, and
drafted this objective addendum.  The user did not select or edit results.
