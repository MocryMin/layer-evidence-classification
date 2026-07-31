# EXP-20260729-001 - Intermediate-layer recoverability on DeBERTa-v3-base / CLINC150

- **AI reporting statement**: The Results section below (data tables and numbers)
  was compiled by an AI coding agent (model: **GLM-5.2**, run via the Claude Code
  CLI) from the saved experiment artifacts; the analysis, the remaining sections,
  and the RP demo log are authored by the user. Human-supervised; experimental
  numbers are machine-produced.
- **Date**: 2026-07-29
- **State**: **REDIRECTED** - the frozen-backbone plain-linear-probe mainline was
  paused after the lr smoke test revealed mid-layer collapse. Diagnostics live in
  EXP-20260729-002; the redirected protocol (probe fix) is recorded in
  EXP-20260729-003.
- **Redirects to**: EXP-20260729-002 (diagnostics), EXP-20260729-003 (redirected mainline)
- **Git commit**: `271f873` (this report; EXP-001 data + reproduction script at `38ffc00`)
- **MLflow experiment**: `EXP-20260729-001` (`sqlite:///mlruns.db`)
- **Artifact root**: `artifacts/EXP-20260729-001/` (gitignored; see its `README.md`)

> Sections 1-5 and the analysis prose are authored by the user. The Results
> section below records the data only.

## 6. Results
> `For anyone who wanna reproduce the result, the global config is listed at ./run_config.yaml`.

We first launched a grid smoke test to determine the best `lr` for our config, with which we selected $1e-3$ as our temporary `lr`.
> `Corresponding artifact path: ./smoke_lr_results.json`

| lr | L1 | L6 | L12 | mean |
|---:|---:|---:|---:|---:|
| 1e-5 | 0.009 | 0.007 | 0.076 | 0.031 |
| 1e-4 | 0.031 | 0.023 | 0.329 | 0.127 |
| **1e-3** | 0.067 | 0.028 | 0.607 | **0.234** (selected) |

*(validation-only, test split not involved; selection rule: max mean validation accuracy, tie-break min mean validation NLL.)*

Then we pre-computed all the hidden states, used AdamW and trained the $W_E^L$. After that we went on to train the mid layers. Then we found the performance of several mid layers fellback to nearly random classify.

| layer | val acc (10 seeds) | test acc |
|---:|---:|---:|
| 1 | 0.071 ± 0.005 | 0.068 |
| 2 | 0.129 ± 0.002 | 0.144 |
| 3 | 0.243 ± 0.005 | 0.252 |
| 4 | 0.064 ± 0.002 | 0.064 |
| 5 | 0.091 ± 0.004 | 0.092 |
| 6 | **0.026 ± 0.002** | 0.025 |
| 7 | 0.076 ± 0.012 | 0.074 |
| 8 | 0.035 ± 0.006 | 0.034 |
| 9 | 0.494 ± 0.003 | 0.487 |
| 10 | 0.158 ± 0.014 | 0.162 |
| 11 | 0.574 ± 0.005 | 0.558 |
| 12 | 0.608 ± 0.003 | 0.585 |

> `Corresponding artifact path: ./plain_probe_mainline/results.json` (12 layers x 10 seeds, lr=1e-3, wd=0.01, batch=256, 100ep; reproduces `smoke_lr_results.json` on seed-17 {1,6,12}). Mid layers 4-8 collapse to near-random (L6 0.026) with tiny variance => seed-robust; test tracks val.

*(Phenomenon: the inter-sample std of the CLS at these layers drops to ~2e-4 (L6) vs ~2e-2 at the final layer (L12) - the mid-layer CLS is nearly constant across samples. Feature stats at `../EXP-20260729-002/02_variance_collapse/per_layer_collapse_summary.json`.)*

### 6.1 LN diagnosis

To isolate the effect of the head, plain and LN are compared under a **matched
control** (same lr, seed, epochs, wd, batch; only the head differs):

| layer | plain AdamW (val acc) | LN head (val acc) | plain NLL | LN NLL |
|---:|---:|---:|---:|---:|
| 1 | 0.135 | 0.806 | 4.388 | 0.816 |
| 4 | 0.081 | 0.730 | 4.705 | 1.247 |
| 6 | 0.027 | 0.651 | 4.985 | 1.906 |
| 8 | 0.028 | 0.831 | 4.985 | 0.903 |
| 12 | 0.789 | 0.870 | 0.994 | 0.608 |

> `Matched control: lr=1e-2, seed 17, 100 epochs, wd=0.01, batch=256, AdamW.` Both
> columns are from the same run (`../EXP-20260729-002/03a_ln_ablation/ln_ablation_lr1e-2.json`);
> only the head differs (`linear_with_bias` vs `LayerNorm(768) + linear`). The plain
> column here (lr=1e-2) differs from the mainline collapse table in §6 (lr=1e-3) -
> see note below.

> `Note on lr.` The §6 mainline uses lr=1e-3 (selected by the smoke test). The §6.1
> matched control uses lr=1e-2 because the LN-head probe needs a larger step to move
> off the collapsed features, and a single lr is required for a controlled plain-vs-LN
> comparison. The plain probe collapses at both lr (§6.1 plain at 1e-2: L6 0.027,
> matching the §6 1e-3 result 0.026 within seed noise), so the choice of 1e-2 does not
> mask the collapse - it is the fairer lr for the LN head and keeps the comparison
> single-variable.

**Parameter count:** the `linear_with_bias` head has $768{\times}150 + 150 = 115{,}350$ parameters. The LN head adds `LayerNorm(768)` ($\gamma + \beta$) = $1{,}536$ parameters, i.e. **1.3% of the head**.

