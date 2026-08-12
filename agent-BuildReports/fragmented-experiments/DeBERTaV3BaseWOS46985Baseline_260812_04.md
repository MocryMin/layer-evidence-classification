# DeBERTaV3BaseWOS46985Baseline_260812_04 — DeBERTa-v3-base CLS baseline on WOS-46985 (134 L2)

Date: 2026-08-12 · Group: `plans/fragmented_exp_gr1.md` · Reporting model: deepseek-v4-flash · Git: `32b4ac1` (dirty=False) · Single seed 17

## Config

- Dataset: wos (WOS-46985, HYDRA-count split 30070/7518/9397 (seed 17, plain random), 134 L2 classes) · 134 classes · splits {'train': 30070, 'validation': 7518, 'test': 9397}
- Model: deberta-v3-base (frozen) · pooling `cls` · max_length 512 · truncation right · cache float16
- Probe families: centered_plain, ln_plain, ridge
- Training: full-batch AdamW, lr=0.01, wd=0, Xavier init, min_ep 100/max 10000/patience 100/min_delta 0.0001 (early stop on val acc)
- Ridge grid: [0, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100] (alpha=0 -> OLS lstsq), alpha by val acc, test once

## 1. Readout variance (collapse check)

| layer | inter_std | PR | top1 frac |
|------:|----------:|----:|----------:|
| 1 | 2.851e-03 | 1.31 | 0.875 |
| 2 | 7.721e-04 | 7.25 | 0.286 |
| 3 | 9.670e-04 | 3.51 | 0.510 |
| 4 | 7.328e-04 | 8.06 | 0.306 |
| 5 | 7.034e-04 | 15.87 | 0.196 |
| 6 | 2.431e-04 | 10.21 | 0.275 |
| 7 | 4.732e-04 | 9.81 | 0.297 |
| 8 | 3.989e-04 | 3.73 | 0.512 |
| 9 | 8.022e-04 | 16.79 | 0.202 |
| 10 | 1.024e-03 | 21.09 | 0.164 |
| 11 | 4.378e-02 | 1.34 | 0.861 |
| 12 | 3.951e-02 | 1.32 | 0.871 |

**Judgement:** COLLAPSED (min inter_std 2.431e-04 @ L6; threshold 0.001). Gradient family = `centered_plain`, wd = 0.

## 2. LR smoke (500 ep, one mid layer)

- `centered_plain`: {'0.01': '0.0938', '0.001': '0.0821'} -> lr = 0.01
- `ln_plain`: {'0.01': '0.1737', '0.001': '0.0184'} -> lr = 0.01

## 3. centered_plain — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop       |
|-------|---------|----------|----------|---------|------------|
| 1     | 0.4347  | 0.4218   | 0.3998   | 3321    | early_stop |
| 2     | 0.0971  | 0.0963   | 0.0545   | 7       | early_stop |
| 3     | 0.0625  | 0.0675   | 0.0338   | 7       | early_stop |
| 4     | 0.0801  | 0.0814   | 0.0412   | 11      | early_stop |
| 5     | 0.1510  | 0.1490   | 0.0880   | 6       | early_stop |
| 6     | 0.0952  | 0.0921   | 0.0503   | 4       | early_stop |
| 7     | 0.1233  | 0.1281   | 0.0758   | 6       | early_stop |
| 8     | 0.0825  | 0.0878   | 0.0490   | 6       | early_stop |
| 9     | 0.1258  | 0.1231   | 0.0706   | 7       | early_stop |
| 10    | 0.1268  | 0.1247   | 0.0712   | 10      | early_stop |
| 11    | 0.3359  | 0.3200   | 0.2912   | 1573    | early_stop |
| 12    | 0.2874  | 0.2813   | 0.2510   | 1296    | early_stop |

- Recoverability vs final layer: oracle gain **+0.3633** (acc_L 0.2813 -> acc_oracle 0.6446), R_oracle 0.5055 (3414/6754), D_JS 0.0327
- Class-wise: coverage 134/134 classes; R_max >= 0.5/0.8/1.0: 17/0/0

## 4. ln_plain — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop       |
|-------|---------|----------|----------|---------|------------|
| 1     | 0.4564  | 0.4468   | 0.4322   | 1552    | early_stop |
| 2     | 0.4517  | 0.4390   | 0.4257   | 1695    | early_stop |
| 3     | 0.4219  | 0.4107   | 0.3906   | 1950    | early_stop |
| 4     | 0.3947  | 0.3887   | 0.3643   | 4408    | early_stop |
| 5     | 0.4488  | 0.4342   | 0.4080   | 3091    | early_stop |
| 6     | 0.3899  | 0.3826   | 0.3543   | 4451    | early_stop |
| 7     | 0.4017  | 0.3892   | 0.3606   | 2073    | early_stop |
| 8     | 0.3665  | 0.3521   | 0.3225   | 3254    | early_stop |
| 9     | 0.3582  | 0.3457   | 0.3156   | 1831    | early_stop |
| 10    | 0.3281  | 0.3084   | 0.2812   | 1378    | early_stop |
| 11    | 0.3289  | 0.3110   | 0.2805   | 500     | early_stop |
| 12    | 0.2954  | 0.2852   | 0.2574   | 1176    | early_stop |

- Recoverability vs final layer: oracle gain **+0.5079** (acc_L 0.2852 -> acc_oracle 0.7931), R_oracle 0.7106 (4773/6717), D_JS 0.0062
- Class-wise: coverage 134/134 classes; R_max >= 0.5/0.8/1.0: 43/0/0

## 5. ridge — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop |
|-------|---------|----------|----------|---------|------|
| 1     | 0.4698  | 0.4696   | 0.4140   | -       | -    |
| 2     | 0.4856  | 0.4821   | 0.4314   | -       | -    |
| 3     | 0.4347  | 0.4333   | 0.3711   | -       | -    |
| 4     | 0.4617  | 0.4567   | 0.3937   | -       | -    |
| 5     | 0.5070  | 0.4978   | 0.4390   | -       | -    |
| 6     | 0.4860  | 0.4836   | 0.4215   | -       | -    |
| 7     | 0.4517  | 0.4473   | 0.3802   | -       | -    |
| 8     | 0.4440  | 0.4364   | 0.3667   | -       | -    |
| 9     | 0.4137  | 0.4072   | 0.3350   | -       | -    |
| 10    | 0.3646  | 0.3616   | 0.2923   | -       | -    |
| 11    | 0.3473  | 0.3339   | 0.2712   | -       | -    |
| 12    | 0.3142  | 0.3003   | 0.2341   | -       | -    |

- Recoverability vs final layer: oracle gain **+0.4379** (acc_L 0.3003 -> acc_oracle 0.7382), R_oracle 0.6259 (4115/6575), D_JS 0.0359
- Class-wise: coverage 134/134 classes; R_max >= 0.5/0.8/1.0: 61/8/0

## 4. Artifacts

- `artifacts/fragmented-experiments/DeBERTaV3BaseWOS46985Baseline_260812_04/`: `results.json`, `cache/` (features float16), `<family>_test_pred.npy` (L×N int16), `<family>_test_logits.npy` (gradient families, L×N×C float16), `<family>_classwise_summary.json`.
- Full per-layer val histories are not persisted (fragmented records scalars only); see `results.json` for all metrics.

## 5. Observations

- _Data collection point — no hypothesis; observations are recorded in results.json and reproduced by the report tables._
