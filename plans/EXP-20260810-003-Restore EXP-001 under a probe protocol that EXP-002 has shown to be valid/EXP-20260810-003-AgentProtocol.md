Inherit global agent protocol. While note the exp-003 specific rules: 

1. Backbone remains frozen.
2. Reuse the exact EXP-001 hidden-state cache and official splits.
3. Centering statistics $μ_l$ are computed from training split only and reused for validation/test.
4. Centered and LN probes use the exact 10 paired seeds defined in EXP-001.
5. Ridge is deterministic and is not repeated over seeds.
6. All hyperparameters/checkpoints are selected using validation only; test is evaluated once.
7. Every probe family compares intermediate layers against its own layer-12 baseline.
8. Primary hypothesis judgement uses Centered Plain; LN and Ridge are robustness/reference analyses.
9. If class-wise recoverability lacks adequate error samples, report it. 