# ModernBERTBaseWOS46985Baseline_260812_05 — modernBERT-base CLS baseline on WOS-46985 (134 L2)

Date: 2026-08-12 · Group: `plans/fragmented_exp_gr1.md` · Reporting model: deepseek-v4-flash · Git: `6c7777e` (dirty=False) · Single seed 17

## Config

- Dataset: wos (WOS-46985, HYDRA-count split 30070/7518/9397 (seed 17, plain random), 134 L2 classes) · 134 classes · splits {'train': 30070, 'validation': 7518, 'test': 9397}
- Model: modernbert-base (frozen) · pooling `cls` · max_length 512 · truncation right · cache float16
- Probe families: plain, ln_plain, ridge
- Training: full-batch AdamW, lr=0.01, wd=0.01, Xavier init, min_ep 100/max 10000/patience 100/min_delta 0.0001 (early stop on val acc)
- Ridge grid: [0, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100] (alpha=0 -> OLS lstsq), alpha by val acc, test once

## 1. Readout variance (collapse check)

| layer | inter_std | PR | top1 frac |
|------:|----------:|----:|----------:|
| 1 | 1.783e-01 | 9.99 | 0.243 |
| 2 | 2.858e-01 | 13.11 | 0.205 |
| 3 | 3.819e-01 | 21.72 | 0.129 |
| 4 | 4.404e-01 | 20.72 | 0.141 |
| 5 | 5.379e-01 | 17.61 | 0.148 |
| 6 | 5.347e-01 | 16.55 | 0.178 |
| 7 | 5.746e-01 | 12.46 | 0.232 |
| 8 | 6.066e-01 | 14.32 | 0.210 |
| 9 | 6.359e-01 | 12.32 | 0.238 |
| 10 | 6.816e-01 | 6.81 | 0.361 |
| 11 | 7.238e-01 | 9.51 | 0.294 |
| 12 | 9.375e-01 | 1.85 | 0.732 |
| 13 | 9.885e-01 | 1.87 | 0.728 |
| 14 | 1.007e+00 | 1.88 | 0.727 |
| 15 | 1.025e+00 | 1.87 | 0.729 |
| 16 | 1.251e+00 | 1.62 | 0.784 |
| 17 | 1.433e+00 | 1.89 | 0.723 |
| 18 | 1.631e+00 | 2.23 | 0.662 |
| 19 | 1.968e+00 | 3.08 | 0.553 |
| 20 | 2.192e+00 | 3.66 | 0.500 |
| 21 | 2.365e+00 | 3.74 | 0.494 |
| 22 | 2.742e-01 | 11.72 | 0.225 |

**Judgement:** healthy (min inter_std 1.783e-01 @ L1; threshold 0.001). Gradient family = `plain`, wd = 0.01.

## 2. LR smoke (500 ep, one mid layer)

- `plain`: {'0.01': '0.3618', '0.001': '0.3409'} -> lr = 0.01
- `ln_plain`: {'0.01': '0.3594', '0.001': '0.3422'} -> lr = 0.01

## 3. plain — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop       |
|-------|---------|----------|----------|---------|------------|
| 1     | 0.4890  | 0.4881   | 0.4751   | 1204    | early_stop |
| 2     | 0.4389  | 0.4373   | 0.4260   | 595     | early_stop |
| 3     | 0.3982  | 0.3880   | 0.3725   | 604     | early_stop |
| 4     | 0.3751  | 0.3721   | 0.3537   | 518     | early_stop |
| 5     | 0.3478  | 0.3446   | 0.3259   | 926     | early_stop |
| 6     | 0.3441  | 0.3234   | 0.3053   | 674     | early_stop |
| 7     | 0.3581  | 0.3543   | 0.3336   | 744     | early_stop |
| 8     | 0.3481  | 0.3452   | 0.3224   | 662     | early_stop |
| 9     | 0.3516  | 0.3452   | 0.3239   | 609     | early_stop |
| 10    | 0.3754  | 0.3705   | 0.3499   | 740     | early_stop |
| 11    | 0.3666  | 0.3612   | 0.3417   | 677     | early_stop |
| 12    | 0.3551  | 0.3503   | 0.3297   | 1070    | early_stop |
| 13    | 0.3662  | 0.3593   | 0.3352   | 1072    | early_stop |
| 14    | 0.3599  | 0.3547   | 0.3331   | 931     | early_stop |
| 15    | 0.3512  | 0.3501   | 0.3249   | 983     | early_stop |
| 16    | 0.3767  | 0.3674   | 0.3451   | 974     | early_stop |
| 17    | 0.3692  | 0.3602   | 0.3396   | 1216    | early_stop |
| 18    | 0.3687  | 0.3520   | 0.3297   | 862     | early_stop |
| 19    | 0.3650  | 0.3518   | 0.3254   | 236     | early_stop |
| 20    | 0.4181  | 0.4066   | 0.3842   | 806     | early_stop |
| 21    | 0.4105  | 0.4030   | 0.3770   | 600     | early_stop |
| 22    | 0.5040  | 0.4974   | 0.4822   | 446     | early_stop |

- Recoverability vs final layer: oracle gain **+0.2919** (acc_L 0.4974 -> acc_oracle 0.7893), R_oracle 0.5808 (2743/4723), D_JS 0.0100
- Class-wise: coverage 134/134 classes; R_max >= 0.5/0.8/1.0: 8/1/0

## 4. ln_plain — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop       |
|-------|---------|----------|----------|---------|------------|
| 1     | 0.4796  | 0.4756   | 0.4631   | 434     | early_stop |
| 2     | 0.4318  | 0.4264   | 0.4150   | 235     | early_stop |
| 3     | 0.3908  | 0.3810   | 0.3647   | 293     | early_stop |
| 4     | 0.3718  | 0.3704   | 0.3542   | 314     | early_stop |
| 5     | 0.3353  | 0.3329   | 0.3181   | 309     | early_stop |
| 6     | 0.3317  | 0.3129   | 0.2954   | 189     | early_stop |
| 7     | 0.3478  | 0.3436   | 0.3244   | 264     | early_stop |
| 8     | 0.3424  | 0.3399   | 0.3200   | 258     | early_stop |
| 9     | 0.3456  | 0.3355   | 0.3154   | 204     | early_stop |
| 10    | 0.3634  | 0.3638   | 0.3443   | 280     | early_stop |
| 11    | 0.3594  | 0.3510   | 0.3308   | 230     | early_stop |
| 12    | 0.3478  | 0.3378   | 0.3185   | 333     | early_stop |
| 13    | 0.3577  | 0.3470   | 0.3230   | 289     | early_stop |
| 14    | 0.3569  | 0.3494   | 0.3257   | 289     | early_stop |
| 15    | 0.3408  | 0.3414   | 0.3192   | 311     | early_stop |
| 16    | 0.3642  | 0.3540   | 0.3297   | 279     | early_stop |
| 17    | 0.3618  | 0.3553   | 0.3334   | 314     | early_stop |
| 18    | 0.3606  | 0.3502   | 0.3282   | 285     | early_stop |
| 19    | 0.4254  | 0.4168   | 0.3953   | 270     | early_stop |
| 20    | 0.4238  | 0.4117   | 0.3882   | 244     | early_stop |
| 21    | 0.4231  | 0.4178   | 0.3981   | 300     | early_stop |
| 22    | 0.4943  | 0.4861   | 0.4725   | 214     | early_stop |

- Recoverability vs final layer: oracle gain **+0.3079** (acc_L 0.4861 -> acc_oracle 0.7940), R_oracle 0.5991 (2893/4829), D_JS 0.0080
- Class-wise: coverage 134/134 classes; R_max >= 0.5/0.8/1.0: 8/1/0

## 5. ridge — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop |
|-------|---------|----------|----------|---------|------|
| 1     | 0.5076  | 0.5103   | 0.4586   | -       | -    |
| 2     | 0.4578  | 0.4632   | 0.4119   | -       | -    |
| 3     | 0.4145  | 0.4110   | 0.3575   | -       | -    |
| 4     | 0.3969  | 0.3941   | 0.3390   | -       | -    |
| 5     | 0.3561  | 0.3508   | 0.2938   | -       | -    |
| 6     | 0.3408  | 0.3377   | 0.2800   | -       | -    |
| 7     | 0.3593  | 0.3603   | 0.2993   | -       | -    |
| 8     | 0.3489  | 0.3523   | 0.2917   | -       | -    |
| 9     | 0.3569  | 0.3517   | 0.2893   | -       | -    |
| 10    | 0.3783  | 0.3736   | 0.3082   | -       | -    |
| 11    | 0.3639  | 0.3651   | 0.3031   | -       | -    |
| 12    | 0.3587  | 0.3565   | 0.2956   | -       | -    |
| 13    | 0.3724  | 0.3655   | 0.3026   | -       | -    |
| 14    | 0.3758  | 0.3628   | 0.3023   | -       | -    |
| 15    | 0.3623  | 0.3538   | 0.2934   | -       | -    |
| 16    | 0.3784  | 0.3707   | 0.3058   | -       | -    |
| 17    | 0.3776  | 0.3658   | 0.3008   | -       | -    |
| 18    | 0.3743  | 0.3597   | 0.2941   | -       | -    |
| 19    | 0.4318  | 0.4222   | 0.3565   | -       | -    |
| 20    | 0.4227  | 0.4078   | 0.3430   | -       | -    |
| 21    | 0.4227  | 0.4084   | 0.3412   | -       | -    |
| 22    | 0.4968  | 0.4858   | 0.4272   | -       | -    |

- Recoverability vs final layer: oracle gain **+0.2258** (acc_L 0.4858 -> acc_oracle 0.7116), R_oracle 0.4392 (2122/4832), D_JS 0.0484
- Class-wise: coverage 134/134 classes; R_max >= 0.5/0.8/1.0: 12/0/0

## 4. Artifacts

- `artifacts/fragmented-experiments/ModernBERTBaseWOS46985Baseline_260812_05/`: `results.json`, `cache/` (features float16), `<family>_test_pred.npy` (L×N int16), `<family>_test_logits.npy` (gradient families, L×N×C float16), `<family>_classwise_summary.json`.
- Full per-layer val histories are not persisted (fragmented records scalars only); see `results.json` for all metrics.

## 5. Observations

- _Data collection point — no hypothesis; observations are recorded in results.json and reproduced by the report tables._
