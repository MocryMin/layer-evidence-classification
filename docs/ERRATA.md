# Errata and interpretation notes

Historical research logs are snapshots and are not silently rewritten after an
experiment closes. This file records corrections that affect interpretation or
navigation.

## EXP-004 author log

1. In §5.1, the selected value `0.3` is the task-head **L2 regularization
   coefficient**, not a learning rate. The optimizer was full-batch L-BFGS.
2. In §5.1.1, the candidate with task/native accuracies 0.9062/0.3194 was
   evaluated on `D_discover`, not `D_test`. The H1 run manifest records
   `validation_accessed=false` and `test_accessed=false`.
3. §5.1.3 accepts the **existence** of a readability gap within the completed
   discovery campaign. The stricter agent report states that held-out
   validation/test confirmation and the corrected cross-source condition were
   not completed. Public/RP wording must therefore remain “strong
   train-discovery existence evidence,” not “held-out-confirmed prevalence.”
4. The pooled `150/586 = 25.60%` is a descriptor of a deduplicated candidate
   inventory assembled under two different adaptive samplers. It is not an IID
   population estimate.

## EXP-001

The initial plain-AdamW probe result is retained because it motivated the
diagnostic work. It must not be cited as evidence that intermediate layers lack
information; EXP-002 and EXP-003 supersede that interpretation.

## Record priority

For numerical reconstruction, use machine artifacts and the objective agent
report. For the researcher's scientific motivation and decisions, use the
author log. For current claim strength, use
[`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md).
