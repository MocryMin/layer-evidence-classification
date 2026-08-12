# ModernBERTBaseExp1Ver_260812_02 — modernBERT-base CLS readout on CLINC150 — side verification of EXP-001/003

Date: 2026-08-12 · Group: `plans/fragmented_exp_gr1.md` · Reporting model: deepseek-v4-flash · Git: `bb06e91` (dirty=True) · Single seed 17

## Config

- Dataset: clinc (CLINC150 plus config, OOS (intent 42) dropped, EXP-001 id2label mapping) · 150 classes · splits {'train': 15000, 'validation': 3000, 'test': 4500}
- Model: modernbert-base (frozen) · pooling `cls` · max_length 512 · truncation right · cache float16
- Probe families: plain, ln_plain, ridge
- Training: full-batch AdamW, lr=0.01, wd=0.01, Xavier init, min_ep 100/max 10000/patience 100/min_delta 0.0001 (early stop on val acc)
- Ridge grid: [0, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100] (alpha=0 -> OLS lstsq), alpha by val acc, test once

## 1. Readout variance (collapse check)

| layer | inter_std | PR | top1 frac |
|------:|----------:|----:|----------:|
| 1 | 9.475e-02 | 4.95 | 0.423 |
| 2 | 1.351e-01 | 7.20 | 0.333 |
| 3 | 1.969e-01 | 7.94 | 0.296 |
| 4 | 2.465e-01 | 10.69 | 0.242 |
| 5 | 3.682e-01 | 7.42 | 0.316 |
| 6 | 3.743e-01 | 7.32 | 0.314 |
| 7 | 4.660e-01 | 5.99 | 0.369 |
| 8 | 4.968e-01 | 6.02 | 0.353 |
| 9 | 5.130e-01 | 5.97 | 0.347 |
| 10 | 5.658e-01 | 6.44 | 0.333 |
| 11 | 6.017e-01 | 6.10 | 0.354 |
| 12 | 6.729e-01 | 6.79 | 0.309 |
| 13 | 6.850e-01 | 6.97 | 0.310 |
| 14 | 6.835e-01 | 7.11 | 0.305 |
| 15 | 7.918e-01 | 7.36 | 0.286 |
| 16 | 3.794e+00 | 1.00 | 0.999 |
| 17 | 3.916e+00 | 1.00 | 0.999 |
| 18 | 4.177e+00 | 1.00 | 0.998 |
| 19 | 4.406e+00 | 1.01 | 0.997 |
| 20 | 4.631e+00 | 1.01 | 0.996 |
| 21 | 4.829e+00 | 1.01 | 0.994 |
| 22 | 2.703e-01 | 2.16 | 0.670 |

**Judgement:** healthy (min inter_std 9.475e-02 @ L1; threshold 0.001). Gradient family = `plain`, wd = 0.01.

## 2. LR smoke (500 ep, one mid layer)

- `plain`: {'0.01': '0.7803', '0.001': '0.7407'} -> lr = 0.01
- `ln_plain`: {'0.01': '0.8303', '0.001': '0.7190'} -> lr = 0.01

## 3. plain — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop       |
|-------|---------|----------|----------|---------|------------|
| 1     | 0.8657  | 0.8518   | 0.8511   | 1168    | early_stop |
| 2     | 0.8817  | 0.8704   | 0.8700   | 1002    | early_stop |
| 3     | 0.8647  | 0.8616   | 0.8613   | 1212    | early_stop |
| 4     | 0.8557  | 0.8524   | 0.8514   | 1680    | early_stop |
| 5     | 0.8290  | 0.8176   | 0.8167   | 1178    | early_stop |
| 6     | 0.8330  | 0.8244   | 0.8235   | 1353    | early_stop |
| 7     | 0.8240  | 0.8084   | 0.8072   | 1424    | early_stop |
| 8     | 0.8303  | 0.8140   | 0.8135   | 1420    | early_stop |
| 9     | 0.8213  | 0.8111   | 0.8100   | 1663    | early_stop |
| 10    | 0.8297  | 0.8071   | 0.8059   | 1465    | early_stop |
| 11    | 0.8270  | 0.8171   | 0.8164   | 1756    | early_stop |
| 12    | 0.8280  | 0.8136   | 0.8132   | 1730    | early_stop |
| 13    | 0.8250  | 0.8109   | 0.8100   | 1350    | early_stop |
| 14    | 0.8370  | 0.8131   | 0.8124   | 1611    | early_stop |
| 15    | 0.8263  | 0.8131   | 0.8122   | 1534    | early_stop |
| 16    | 0.7123  | 0.6969   | 0.6952   | 809     | early_stop |
| 17    | 0.7980  | 0.7864   | 0.7846   | 1680    | early_stop |
| 18    | 0.8010  | 0.7887   | 0.7876   | 1828    | early_stop |
| 19    | 0.8037  | 0.7873   | 0.7862   | 1720    | early_stop |
| 20    | 0.8047  | 0.7793   | 0.7782   | 1671    | early_stop |
| 21    | 0.7987  | 0.7767   | 0.7755   | 1603    | early_stop |
| 22    | 0.8347  | 0.8253   | 0.8252   | 943     | early_stop |

- Recoverability vs final layer: oracle gain **+0.1460** (acc_L 0.8253 -> acc_oracle 0.9713), R_oracle 0.8359 (657/786), D_JS 0.0094
- Class-wise: coverage 146/150 classes; R_max >= 0.5/0.8/1.0: 132/75/55

## 4. ln_plain — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop       |
|-------|---------|----------|----------|---------|------------|
| 1     | 0.8623  | 0.8504   | 0.8499   | 508     | early_stop |
| 2     | 0.8777  | 0.8718   | 0.8711   | 451     | early_stop |
| 3     | 0.8613  | 0.8644   | 0.8640   | 612     | early_stop |
| 4     | 0.8487  | 0.8509   | 0.8495   | 617     | early_stop |
| 5     | 0.8327  | 0.8287   | 0.8280   | 717     | early_stop |
| 6     | 0.8313  | 0.8264   | 0.8261   | 712     | early_stop |
| 7     | 0.8300  | 0.8151   | 0.8141   | 702     | early_stop |
| 8     | 0.8343  | 0.8113   | 0.8108   | 506     | early_stop |
| 9     | 0.8277  | 0.8136   | 0.8125   | 627     | early_stop |
| 10    | 0.8297  | 0.8111   | 0.8101   | 497     | early_stop |
| 11    | 0.8340  | 0.8140   | 0.8134   | 693     | early_stop |
| 12    | 0.8350  | 0.8207   | 0.8203   | 1193    | early_stop |
| 13    | 0.8327  | 0.8231   | 0.8229   | 628     | early_stop |
| 14    | 0.8427  | 0.8218   | 0.8211   | 577     | early_stop |
| 15    | 0.8263  | 0.8149   | 0.8142   | 593     | early_stop |
| 16    | 0.7733  | 0.7600   | 0.7617   | 458     | early_stop |
| 17    | 0.7810  | 0.7624   | 0.7613   | 808     | early_stop |
| 18    | 0.7850  | 0.7662   | 0.7654   | 632     | early_stop |
| 19    | 0.7827  | 0.7540   | 0.7533   | 353     | early_stop |
| 20    | 0.7947  | 0.7707   | 0.7696   | 721     | early_stop |
| 21    | 0.8030  | 0.7840   | 0.7829   | 691     | early_stop |
| 22    | 0.8313  | 0.8233   | 0.8230   | 369     | early_stop |

- Recoverability vs final layer: oracle gain **+0.1493** (acc_L 0.8233 -> acc_oracle 0.9727), R_oracle 0.8453 (672/795), D_JS 0.0088
- Class-wise: coverage 145/150 classes; R_max >= 0.5/0.8/1.0: 134/71/53

## 5. ridge — layer-wise results

| layer | val_acc | test_acc | macro_f1 | best_ep | stop |
|-------|---------|----------|----------|---------|------|
| 1     | 0.8587  | 0.8384   | 0.8348   | -       | -    |
| 2     | 0.8720  | 0.8698   | 0.8674   | -       | -    |
| 3     | 0.8817  | 0.8704   | 0.8684   | -       | -    |
| 4     | 0.8740  | 0.8662   | 0.8635   | -       | -    |
| 5     | 0.8650  | 0.8556   | 0.8537   | -       | -    |
| 6     | 0.8660  | 0.8551   | 0.8530   | -       | -    |
| 7     | 0.8670  | 0.8527   | 0.8506   | -       | -    |
| 8     | 0.8640  | 0.8551   | 0.8527   | -       | -    |
| 9     | 0.8677  | 0.8569   | 0.8545   | -       | -    |
| 10    | 0.8757  | 0.8629   | 0.8605   | -       | -    |
| 11    | 0.8803  | 0.8656   | 0.8630   | -       | -    |
| 12    | 0.8810  | 0.8662   | 0.8641   | -       | -    |
| 13    | 0.8800  | 0.8638   | 0.8613   | -       | -    |
| 14    | 0.8843  | 0.8647   | 0.8620   | -       | -    |
| 15    | 0.8707  | 0.8613   | 0.8590   | -       | -    |
| 16    | 0.8703  | 0.8558   | 0.8535   | -       | -    |
| 17    | 0.8543  | 0.8322   | 0.8294   | -       | -    |
| 18    | 0.8480  | 0.8276   | 0.8241   | -       | -    |
| 19    | 0.8537  | 0.8280   | 0.8242   | -       | -    |
| 20    | 0.8420  | 0.8120   | 0.8076   | -       | -    |
| 21    | 0.8243  | 0.8044   | 0.7995   | -       | -    |
| 22    | 0.8563  | 0.8440   | 0.8411   | -       | -    |

- Recoverability vs final layer: oracle gain **+0.1167** (acc_L 0.8440 -> acc_oracle 0.9607), R_oracle 0.7479 (525/702), D_JS 0.0210
- Class-wise: coverage 130/150 classes; R_max >= 0.5/0.8/1.0: 106/56/46

## 4. Artifacts

- `artifacts/fragmented-experiments/ModernBERTBaseExp1Ver_260812_02/`: `results.json`, `cache/` (features float16), `<family>_test_pred.npy` (L×N int16), `<family>_test_logits.npy` (gradient families, L×N×C float16), `<family>_classwise_summary.json`.
- Full per-layer val histories are not persisted (fragmented records scalars only); see `results.json` for all metrics.

## 5. Observations

- _Data collection point — no hypothesis; observations are recorded in results.json and reproduced by the report tables._
