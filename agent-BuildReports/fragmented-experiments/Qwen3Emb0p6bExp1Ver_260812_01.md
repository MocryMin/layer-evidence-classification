# Qwen3Emb0p6bExp1Ver_260812_01 — Qwen3-Embedding-0.6B last-token readout on CLINC150 — side verification of EXP-001/003

Date: 2026-08-12 · Group: `plans/fragmented_exp_gr1.md` · Reporting model: deepseek-v4-flash · Git: `309a415` (dirty=True) · Single seed 17

## Config

- Dataset: clinc (CLINC150 plus config, OOS (intent 42) dropped, EXP-001 id2label mapping) · 150 classes · splits {'train': 15000, 'validation': 3000, 'test': 4500}
- Model: Qwen3-Embedding-0.6B (frozen) · pooling `last_token` · max_length 512 · truncation right · cache float16
- Probe families: plain, ln_plain, ridge
- Training: full-batch AdamW, lr=0.01, wd=0.01, Xavier init, min_ep 100/max 10000/patience 100/min_delta 0.0001 (early stop on val acc)
- Ridge grid: [0, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100] (alpha=0 -> OLS lstsq), alpha by val acc, test once

## 1. Readout variance (collapse check)

| layer | inter_std | PR | top1 frac |
|------:|----------:|----:|----------:|
| 1 | 3.026e-02 | 20.61 | 0.148 |
| 2 | 4.442e-02 | 26.87 | 0.119 |
| 3 | 6.282e-02 | 31.20 | 0.108 |
| 4 | 8.448e-02 | 26.23 | 0.137 |
| 5 | 7.113e-02 | 21.80 | 0.157 |
| 6 | 9.045e-02 | 18.66 | 0.177 |
| 7 | 9.958e-02 | 18.42 | 0.169 |
| 8 | 1.242e-01 | 23.95 | 0.145 |
| 9 | 1.342e-01 | 21.67 | 0.152 |
| 10 | 1.959e-01 | 26.58 | 0.112 |
| 11 | 2.284e-01 | 28.42 | 0.112 |
| 12 | 2.469e-01 | 26.29 | 0.112 |
| 13 | 2.603e-01 | 25.60 | 0.114 |
| 14 | 2.632e-01 | 25.48 | 0.115 |
| 15 | 2.698e-01 | 28.27 | 0.108 |
| 16 | 4.012e-01 | 33.81 | 0.099 |
| 17 | 4.647e-01 | 30.62 | 0.111 |
| 18 | 7.991e-01 | 37.14 | 0.086 |
| 19 | 9.847e-01 | 31.58 | 0.102 |
| 20 | 1.259e+00 | 26.90 | 0.124 |
| 21 | 1.826e+00 | 34.04 | 0.090 |
| 22 | 2.876e+00 | 36.21 | 0.080 |
| 23 | 3.668e+00 | 33.88 | 0.085 |
| 24 | 4.509e+00 | 34.47 | 0.080 |
| 25 | 5.092e+00 | 36.95 | 0.079 |
| 26 | 5.989e+00 | 42.37 | 0.072 |
| 27 | 7.020e+00 | 44.69 | 0.071 |
| 28 | 2.007e+00 | 44.50 | 0.072 |

**Judgement:** healthy (min inter_std 3.026e-02 @ L1; threshold 0.001). Gradient family = `plain`, wd = 0.01.

## 2. LR smoke (500 ep, one mid layer)

- `plain`: {'0.01': '0.9197', '0.001': '0.9120'} -> lr = 0.01
- `ln_plain`: {'0.01': '0.9220', '0.001': '0.9140'} -> lr = 0.01

## 3. plain — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop       |
|-------|---------|----------|----------|---------|------------|
| 1     | 0.8553  | 0.8438   | 0.8428   | 1798    | early_stop |
| 2     | 0.8627  | 0.8584   | 0.8579   | 1430    | early_stop |
| 3     | 0.8443  | 0.8336   | 0.8333   | 846     | early_stop |
| 4     | 0.8580  | 0.8462   | 0.8455   | 874     | early_stop |
| 5     | 0.8677  | 0.8516   | 0.8510   | 1412    | early_stop |
| 6     | 0.8727  | 0.8567   | 0.8556   | 784     | early_stop |
| 7     | 0.8727  | 0.8529   | 0.8528   | 999     | early_stop |
| 8     | 0.9280  | 0.9173   | 0.9171   | 978     | early_stop |
| 9     | 0.9270  | 0.9073   | 0.9067   | 559     | early_stop |
| 10    | 0.9330  | 0.9218   | 0.9215   | 469     | early_stop |
| 11    | 0.9317  | 0.9184   | 0.9179   | 553     | early_stop |
| 12    | 0.9273  | 0.9176   | 0.9171   | 796     | early_stop |
| 13    | 0.9327  | 0.9222   | 0.9220   | 875     | early_stop |
| 14    | 0.9233  | 0.9153   | 0.9152   | 726     | early_stop |
| 15    | 0.9287  | 0.9153   | 0.9151   | 331     | early_stop |
| 16    | 0.9430  | 0.9256   | 0.9252   | 443     | early_stop |
| 17    | 0.9347  | 0.9227   | 0.9223   | 583     | early_stop |
| 18    | 0.9473  | 0.9329   | 0.9328   | 390     | early_stop |
| 19    | 0.9460  | 0.9304   | 0.9302   | 388     | early_stop |
| 20    | 0.9377  | 0.9182   | 0.9181   | 117     | early_stop |
| 21    | 0.9527  | 0.9431   | 0.9429   | 204     | early_stop |
| 22    | 0.9540  | 0.9476   | 0.9475   | 125     | early_stop |
| 23    | 0.9533  | 0.9453   | 0.9452   | 112     | early_stop |
| 24    | 0.9503  | 0.9411   | 0.9410   | 272     | early_stop |
| 25    | 0.9537  | 0.9427   | 0.9424   | 89      | early_stop |
| 26    | 0.9547  | 0.9451   | 0.9450   | 81      | early_stop |
| 27    | 0.9527  | 0.9431   | 0.9428   | 199     | early_stop |
| 28    | 0.9623  | 0.9524   | 0.9522   | 386     | early_stop |

- Recoverability vs final layer: oracle gain **+0.0409** (acc_L 0.9524 -> acc_oracle 0.9933), R_oracle 0.8598 (184/214), D_JS 0.0263
- Class-wise: coverage 88/150 classes; R_max >= 0.5/0.8/1.0: 77/57/54

## 4. ln_plain — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop       |
|-------|---------|----------|----------|---------|------------|
| 1     | 0.8263  | 0.8144   | 0.8134   | 495     | early_stop |
| 2     | 0.8403  | 0.8344   | 0.8339   | 978     | early_stop |
| 3     | 0.8257  | 0.8113   | 0.8111   | 424     | early_stop |
| 4     | 0.8387  | 0.8289   | 0.8282   | 451     | early_stop |
| 5     | 0.8480  | 0.8298   | 0.8294   | 451     | early_stop |
| 6     | 0.8610  | 0.8429   | 0.8421   | 464     | early_stop |
| 7     | 0.8533  | 0.8336   | 0.8333   | 330     | early_stop |
| 8     | 0.9163  | 0.9102   | 0.9099   | 375     | early_stop |
| 9     | 0.9207  | 0.9042   | 0.9036   | 498     | early_stop |
| 10    | 0.9290  | 0.9133   | 0.9130   | 364     | early_stop |
| 11    | 0.9290  | 0.9162   | 0.9157   | 346     | early_stop |
| 12    | 0.9243  | 0.9109   | 0.9104   | 366     | early_stop |
| 13    | 0.9297  | 0.9169   | 0.9167   | 462     | early_stop |
| 14    | 0.9220  | 0.9093   | 0.9092   | 347     | early_stop |
| 15    | 0.9333  | 0.9171   | 0.9168   | 601     | early_stop |
| 16    | 0.9367  | 0.9173   | 0.9172   | 111     | early_stop |
| 17    | 0.9360  | 0.9162   | 0.9160   | 182     | early_stop |
| 18    | 0.9493  | 0.9380   | 0.9379   | 178     | early_stop |
| 19    | 0.9530  | 0.9429   | 0.9428   | 316     | early_stop |
| 20    | 0.9507  | 0.9398   | 0.9396   | 351     | early_stop |
| 21    | 0.9650  | 0.9533   | 0.9532   | 221     | early_stop |
| 22    | 0.9690  | 0.9582   | 0.9581   | 230     | early_stop |
| 23    | 0.9677  | 0.9564   | 0.9563   | 126     | early_stop |
| 24    | 0.9653  | 0.9558   | 0.9556   | 110     | early_stop |
| 25    | 0.9673  | 0.9587   | 0.9585   | 215     | early_stop |
| 26    | 0.9673  | 0.9593   | 0.9592   | 159     | early_stop |
| 27    | 0.9667  | 0.9589   | 0.9587   | 190     | early_stop |
| 28    | 0.9663  | 0.9580   | 0.9579   | 70      | early_stop |

- Recoverability vs final layer: oracle gain **+0.0356** (acc_L 0.9580 -> acc_oracle 0.9936), R_oracle 0.8466 (160/189), D_JS 0.0366
- Class-wise: coverage 80/150 classes; R_max >= 0.5/0.8/1.0: 68/50/46

## 5. ridge — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop |
|-------|---------|----------|----------|---------|------|
| 1     | 0.8390  | 0.8387   | 0.8362   | -       | -    |
| 2     | 0.8573  | 0.8602   | 0.8578   | -       | -    |
| 3     | 0.8450  | 0.8516   | 0.8489   | -       | -    |
| 4     | 0.8680  | 0.8638   | 0.8616   | -       | -    |
| 5     | 0.8703  | 0.8638   | 0.8609   | -       | -    |
| 6     | 0.8843  | 0.8811   | 0.8796   | -       | -    |
| 7     | 0.8857  | 0.8798   | 0.8783   | -       | -    |
| 8     | 0.9247  | 0.9182   | 0.9172   | -       | -    |
| 9     | 0.9263  | 0.9216   | 0.9207   | -       | -    |
| 10    | 0.9370  | 0.9287   | 0.9278   | -       | -    |
| 11    | 0.9343  | 0.9309   | 0.9298   | -       | -    |
| 12    | 0.9337  | 0.9267   | 0.9257   | -       | -    |
| 13    | 0.9383  | 0.9260   | 0.9249   | -       | -    |
| 14    | 0.9340  | 0.9251   | 0.9241   | -       | -    |
| 15    | 0.9367  | 0.9269   | 0.9260   | -       | -    |
| 16    | 0.9453  | 0.9282   | 0.9277   | -       | -    |
| 17    | 0.9427  | 0.9282   | 0.9275   | -       | -    |
| 18    | 0.9533  | 0.9369   | 0.9364   | -       | -    |
| 19    | 0.9550  | 0.9396   | 0.9392   | -       | -    |
| 20    | 0.9577  | 0.9427   | 0.9424   | -       | -    |
| 21    | 0.9637  | 0.9476   | 0.9473   | -       | -    |
| 22    | 0.9603  | 0.9524   | 0.9522   | -       | -    |
| 23    | 0.9630  | 0.9556   | 0.9553   | -       | -    |
| 24    | 0.9647  | 0.9540   | 0.9538   | -       | -    |
| 25    | 0.9613  | 0.9567   | 0.9564   | -       | -    |
| 26    | 0.9573  | 0.9529   | 0.9525   | -       | -    |
| 27    | 0.9553  | 0.9544   | 0.9541   | -       | -    |
| 28    | 0.9577  | 0.9536   | 0.9531   | -       | -    |

- Recoverability vs final layer: oracle gain **+0.0353** (acc_L 0.9536 -> acc_oracle 0.9889), R_oracle 0.7608 (159/209), D_JS 0.0458
- Class-wise: coverage 82/150 classes; R_max >= 0.5/0.8/1.0: 62/44/41

## 4. Artifacts

- `artifacts/fragmented-experiments/Qwen3Emb0p6bExp1Ver_260812_01/`: `results.json`, `cache/` (features float16), `<family>_test_pred.npy` (L×N int16), `<family>_test_logits.npy` (gradient families, L×N×C float16), `<family>_classwise_summary.json`.
- Full per-layer val histories are not persisted (fragmented records scalars only); see `results.json` for all metrics.

## 5. Observations

- _Data collection point — no hypothesis; observations are recorded in results.json and reproduced by the report tables._
