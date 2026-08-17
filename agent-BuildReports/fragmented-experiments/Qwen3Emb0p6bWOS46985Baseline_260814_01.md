# Qwen3Emb0p6bWOS46985Baseline_260814_01 — Qwen3-Embedding-0.6B last-token baseline on WOS-46985 (134 L2)

Date: 2026-08-14 · Group: `user_exp_plans/fragmented_exp_gr1.md` · Reporting model: deepseek-v4-flash · Git: `7a95605` (dirty=False) · Single seed 17

## Config

- Dataset: wos (WOS-46985, HYDRA-count split 30070/7518/9397 (seed 17, plain random), 134 L2 classes) · 134 classes · splits {'train': 30070, 'validation': 7518, 'test': 9397}
- Model: Qwen3-Embedding-0.6B (frozen) · pooling `last_token` · max_length 512 · truncation right · cache float16
- Probe families: plain, ln_plain, ridge
- Training: full-batch AdamW, lr=0.01, wd=0.01, Xavier init, min_ep 100/max 10000/patience 100/min_delta 0.0001 (early stop on val acc)
- Ridge grid: [0, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100] (alpha=0 -> OLS lstsq), alpha by val acc, test once

## 1. Readout variance (collapse check)

| layer | inter_std | PR | top1 frac |
|------:|----------:|----:|----------:|
| 1 | 2.516e-02 | 7.70 | 0.291 |
| 2 | 3.885e-02 | 9.31 | 0.274 |
| 3 | 5.530e-02 | 11.29 | 0.242 |
| 4 | 7.989e-02 | 18.21 | 0.161 |
| 5 | 8.932e-02 | 15.28 | 0.189 |
| 6 | 1.056e-01 | 13.97 | 0.186 |
| 7 | 1.089e-01 | 13.33 | 0.177 |
| 8 | 1.135e-01 | 13.59 | 0.185 |
| 9 | 1.378e-01 | 13.68 | 0.203 |
| 10 | 1.965e-01 | 12.03 | 0.241 |
| 11 | 2.014e-01 | 13.15 | 0.226 |
| 12 | 2.169e-01 | 11.81 | 0.246 |
| 13 | 2.448e-01 | 11.83 | 0.235 |
| 14 | 2.744e-01 | 12.73 | 0.222 |
| 15 | 2.902e-01 | 13.51 | 0.213 |
| 16 | 3.354e-01 | 13.35 | 0.228 |
| 17 | 3.903e-01 | 14.21 | 0.210 |
| 18 | 5.241e-01 | 22.68 | 0.165 |
| 19 | 6.013e-01 | 21.36 | 0.170 |
| 20 | 7.561e-01 | 25.33 | 0.146 |
| 21 | 1.106e+00 | 27.51 | 0.146 |
| 22 | 1.722e+00 | 41.19 | 0.110 |
| 23 | 2.019e+00 | 41.65 | 0.108 |
| 24 | 2.443e+00 | 54.40 | 0.091 |
| 25 | 3.234e+00 | 84.98 | 0.068 |
| 26 | 4.731e+00 | 128.77 | 0.049 |
| 27 | 5.943e+00 | 135.79 | 0.047 |
| 28 | 2.917e+00 | 133.60 | 0.047 |

**Judgement:** healthy (min inter_std 2.516e-02 @ L1; threshold 0.001). Gradient family = `plain`, wd = 0.01.

## 2. LR smoke (500 ep, one mid layer)

- `plain`: {'0.01': '0.4644', '0.001': '0.4391'} -> lr = 0.01
- `ln_plain`: {'0.01': '0.4528', '0.001': '0.4495'} -> lr = 0.01

## 3. plain — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop       |
|-------|---------|----------|----------|---------|------------|
| 1     | 0.4238  | 0.4148   | 0.3867   | 1789    | early_stop |
| 2     | 0.4218  | 0.4150   | 0.3935   | 2419    | early_stop |
| 3     | 0.4016  | 0.3914   | 0.3709   | 2192    | early_stop |
| 4     | 0.4240  | 0.4160   | 0.3969   | 1795    | early_stop |
| 5     | 0.4040  | 0.4075   | 0.3839   | 1693    | early_stop |
| 6     | 0.4287  | 0.4215   | 0.3972   | 978     | early_stop |
| 7     | 0.4413  | 0.4373   | 0.4142   | 1477    | early_stop |
| 8     | 0.5093  | 0.5025   | 0.4791   | 709     | early_stop |
| 9     | 0.4844  | 0.4730   | 0.4512   | 697     | early_stop |
| 10    | 0.4879  | 0.4817   | 0.4549   | 596     | early_stop |
| 11    | 0.4844  | 0.4698   | 0.4469   | 1024    | early_stop |
| 12    | 0.4749  | 0.4629   | 0.4366   | 590     | early_stop |
| 13    | 0.4779  | 0.4556   | 0.4269   | 802     | early_stop |
| 14    | 0.4690  | 0.4562   | 0.4289   | 586     | early_stop |
| 15    | 0.4725  | 0.4560   | 0.4321   | 671     | early_stop |
| 16    | 0.5020  | 0.4803   | 0.4586   | 453     | early_stop |
| 17    | 0.4915  | 0.4757   | 0.4518   | 714     | early_stop |
| 18    | 0.5567  | 0.5472   | 0.5282   | 333     | early_stop |
| 19    | 0.5564  | 0.5507   | 0.5318   | 464     | early_stop |
| 20    | 0.5642  | 0.5561   | 0.5377   | 375     | early_stop |
| 21    | 0.6071  | 0.5997   | 0.5875   | 356     | early_stop |
| 22    | 0.6014  | 0.5897   | 0.5806   | 69      | early_stop |
| 23    | 0.6020  | 0.5847   | 0.5785   | 80      | early_stop |
| 24    | 0.5951  | 0.5840   | 0.5800   | 75      | early_stop |
| 25    | 0.6120  | 0.5932   | 0.5880   | 58      | early_stop |
| 26    | 0.6089  | 0.5937   | 0.5878   | 59      | early_stop |
| 27    | 0.6164  | 0.6070   | 0.6022   | 54      | early_stop |
| 28    | 0.6152  | 0.6049   | 0.5975   | 17      | early_stop |

- Recoverability vs final layer: oracle gain **+0.2890** (acc_L 0.6049 -> acc_oracle 0.8939), R_oracle 0.7315 (2716/3713), D_JS 0.0054
- Class-wise: coverage 134/134 classes; R_max >= 0.5/0.8/1.0: 34/3/0

## 4. ln_plain — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop       |
|-------|---------|----------|----------|---------|------------|
| 1     | 0.4308  | 0.4255   | 0.3999   | 1260    | early_stop |
| 2     | 0.3856  | 0.3832   | 0.3632   | 1083    | early_stop |
| 3     | 0.3643  | 0.3600   | 0.3403   | 1010    | early_stop |
| 4     | 0.3821  | 0.3800   | 0.3612   | 516     | early_stop |
| 5     | 0.3712  | 0.3714   | 0.3473   | 475     | early_stop |
| 6     | 0.4054  | 0.3987   | 0.3763   | 447     | early_stop |
| 7     | 0.4149  | 0.4110   | 0.3887   | 472     | early_stop |
| 8     | 0.4910  | 0.4825   | 0.4593   | 281     | early_stop |
| 9     | 0.4617  | 0.4518   | 0.4289   | 286     | early_stop |
| 10    | 0.4636  | 0.4595   | 0.4355   | 329     | early_stop |
| 11    | 0.4581  | 0.4483   | 0.4222   | 325     | early_stop |
| 12    | 0.4544  | 0.4427   | 0.4154   | 229     | early_stop |
| 13    | 0.4560  | 0.4401   | 0.4119   | 277     | early_stop |
| 14    | 0.4528  | 0.4381   | 0.4117   | 243     | early_stop |
| 15    | 0.4613  | 0.4383   | 0.4152   | 252     | early_stop |
| 16    | 0.4903  | 0.4658   | 0.4415   | 189     | early_stop |
| 17    | 0.4795  | 0.4640   | 0.4397   | 270     | early_stop |
| 18    | 0.5564  | 0.5388   | 0.5188   | 140     | early_stop |
| 19    | 0.5517  | 0.5424   | 0.5237   | 169     | early_stop |
| 20    | 0.5664  | 0.5548   | 0.5374   | 159     | early_stop |
| 21    | 0.6218  | 0.6163   | 0.6093   | 118     | early_stop |
| 22    | 0.6366  | 0.6236   | 0.6168   | 66      | early_stop |
| 23    | 0.6379  | 0.6261   | 0.6198   | 69      | early_stop |
| 24    | 0.6390  | 0.6359   | 0.6286   | 78      | early_stop |
| 25    | 0.6506  | 0.6420   | 0.6352   | 67      | early_stop |
| 26    | 0.6551  | 0.6414   | 0.6346   | 42      | early_stop |
| 27    | 0.6568  | 0.6521   | 0.6434   | 57      | early_stop |
| 28    | 0.6403  | 0.6318   | 0.6231   | 13      | early_stop |

- Recoverability vs final layer: oracle gain **+0.2556** (acc_L 0.6318 -> acc_oracle 0.8874), R_oracle 0.6942 (2402/3460), D_JS 0.0063
- Class-wise: coverage 134/134 classes; R_max >= 0.5/0.8/1.0: 18/0/0

## 5. ridge — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop |
|-------|---------|----------|----------|---------|------|
| 1     | 0.4507  | 0.4548   | 0.3942   | -       | -    |
| 2     | 0.4425  | 0.4428   | 0.3857   | -       | -    |
| 3     | 0.4268  | 0.4257   | 0.3689   | -       | -    |
| 4     | 0.4354  | 0.4265   | 0.3752   | -       | -    |
| 5     | 0.4198  | 0.4203   | 0.3652   | -       | -    |
| 6     | 0.4463  | 0.4422   | 0.3825   | -       | -    |
| 7     | 0.4497  | 0.4427   | 0.3795   | -       | -    |
| 8     | 0.5041  | 0.4980   | 0.4440   | -       | -    |
| 9     | 0.4911  | 0.4796   | 0.4211   | -       | -    |
| 10    | 0.4875  | 0.4796   | 0.4198   | -       | -    |
| 11    | 0.4733  | 0.4576   | 0.3936   | -       | -    |
| 12    | 0.4620  | 0.4470   | 0.3838   | -       | -    |
| 13    | 0.4621  | 0.4457   | 0.3776   | -       | -    |
| 14    | 0.4604  | 0.4419   | 0.3745   | -       | -    |
| 15    | 0.4578  | 0.4439   | 0.3827   | -       | -    |
| 16    | 0.4844  | 0.4680   | 0.4073   | -       | -    |
| 17    | 0.4830  | 0.4711   | 0.4111   | -       | -    |
| 18    | 0.5463  | 0.5343   | 0.4776   | -       | -    |
| 19    | 0.5472  | 0.5362   | 0.4801   | -       | -    |
| 20    | 0.5733  | 0.5567   | 0.5050   | -       | -    |
| 21    | 0.6107  | 0.6005   | 0.5601   | -       | -    |
| 22    | 0.6270  | 0.6101   | 0.5738   | -       | -    |
| 23    | 0.6286  | 0.6154   | 0.5834   | -       | -    |
| 24    | 0.6329  | 0.6242   | 0.5889   | -       | -    |
| 25    | 0.6518  | 0.6446   | 0.6125   | -       | -    |
| 26    | 0.6579  | 0.6522   | 0.6203   | -       | -    |
| 27    | 0.6560  | 0.6543   | 0.6224   | -       | -    |
| 28    | 0.6536  | 0.6525   | 0.6201   | -       | -    |

- Recoverability vs final layer: oracle gain **+0.1742** (acc_L 0.6525 -> acc_oracle 0.8268), R_oracle 0.5014 (1637/3265), D_JS 0.0401
- Class-wise: coverage 134/134 classes; R_max >= 0.5/0.8/1.0: 22/3/0

## 4. Artifacts

- `artifacts/fragmented-experiments/Qwen3Emb0p6bWOS46985Baseline_260814_01/`: `results.json`, `cache/` (features float16), `<family>_test_pred.npy` (L×N int16), `<family>_test_logits.npy` (gradient families, L×N×C float16), `<family>_classwise_summary.json`.
- Full per-layer val histories are not persisted (fragmented records scalars only); see `results.json` for all metrics.

## 5. Observations

- _Data collection point — no hypothesis; observations are recorded in results.json and reproduced by the report tables._
- Final-layer (L28) last-token readout: plain 0.6152 / ln_plain 0.6403 / ridge 0.6536 val — roughly double the DeBERTa-v3 CLS baseline (ridge L12 0.3142) on the same WOS-46985 134-L2 task; mid/deep layers (L18-27) plateau at ~0.60-0.66, recoverability oracle gain ≈ +0.17-0.29 across families.
