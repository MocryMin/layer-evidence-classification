# EXP-004 H1 frozen train-discovery protocol

Frozen on 2026-08-27 before the official adaptive discovery run. The
authoritative human-readable protocol is the host-side EXP-004 document whose
SHA256 is recorded in `configs/exp004_h1_frozen.yaml`; the YAML is the
machine-readable run contract.

## Evidence roles

- `repeat_L28 = [1,...,28,28]` is a preregistered structured H1a control. Its
  engineering-pilot values are not confirmatory. It is excluded from the H1b
  prevalence denominator.
- The five adaptive sources are the H1b discovery population. Canonical is a
  baseline/parent and is also excluded from prevalence.
- Discovery reads the official ARC-Easy train split only: fixed D_fit=1750 and
  D_discover=501. Validation and test are inaccessible to this program.

## Search and stopping

Sources S1--S5 are polled in strict round-robin after deterministic seed
initialisation. Parent probability is `0.75 * softmax(task_accuracy / 0.05) +
0.25 * uniform`; only the path-specific task-head accuracy is visible to the
search policy. Native-head results never affect selection. Paths are globally
deduplicated, bounded to length 36, and proposals receive 100 attempts.

The first campaign collects paths for 12.0 cumulative model-resident GPU hours,
across resumable sessions. It does not stop after any number of good paths or
readability collapses. Safety caps are 5,000 discovered candidates and 200 GiB
of artifacts. Every session additionally requires an absolute user-authorised
stop time with a 15-minute soft-stop reserve.

## Frozen decision rules

A path is good when its task-head discover accuracy is no more than 0.05 below
canonical. A good path has a readability collapse when canonical native
accuracy minus path native accuracy is at least 0.15, or when that gap is at
least 0.50 of the canonical-above-chance margin. Head L2 is fixed at 0.3 from
D_fit-only cross-validation. No discovery result can modify these rules.
