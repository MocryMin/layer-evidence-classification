# Claim-to-evidence index

This page is the authoritative public map from a short research claim to the
records needed to audit it. The underlying logs and artifacts remain historical
records; this index controls the current evidence label and permissible wording.

## Fast reading paths

**Five minutes:** read the repository README, then the “Claim matrix” and
“Claim boundaries” below.

**Twenty minutes:** add the EXP-003 and EXP-004 H1/H2 objective reports.

**Full audit:** follow the configuration, Git revision, MLflow export, and
artifact entries for the claim of interest; verify hashes against the release
manifest.

## Evidence labels

- **Confirmatory within scope:** evaluated under a frozen selection/acceptance
  procedure, while still limited to the stated model, dataset, seed design, and
  task.
- **Diagnostic:** resolves a methodological ambiguity but is not itself the
  final test of the project hypothesis.
- **Discovery/preliminary:** supports existence or motivates a hypothesis, but
  lacks a held-out confirmation or another predeclared condition.
- **Superseded:** retained for research history; its conclusions are not used as
  current evidence without the later correction.

## Claim matrix

| ID | Claim and scope | Evidence label | Headline measurement | Primary human record | Objective report | Configuration / code | Run and artifact evidence |
|---|---|---|---|---|---|---|---|
| C1 | A failed probe is not sufficient evidence that an intermediate representation lacks task information. In frozen DeBERTa-v3-base CLS features on CLINC150, the original mid-layer failure was primarily an optimization/readout-conditioning failure. | Diagnostic | At L6, plain AdamW was near chance, while ridge reached about 0.917 validation accuracy; optimizer, normalization, precision, prompt, nonlinear-head, and token-position controls were run. | [`EXP-002`](../research-logs/EXP-002.md) | [`EXP-002 agent report`](../agent-BuildReports/experiments/EXP-20260729-002--linear-probe-midlayer-collapse.md) | [`diag_config.yaml`](../configs/diag_config.yaml), relevant `scripts/` and `src/` modules | MLflow experiment `EXP-20260729-002`; selected public result files are enumerated by the release manifest. |
| C2 | Under probe protocols shown to fit the representation, intermediate layers can outperform the final layer and recover a large, broadly distributed subset of final-layer errors in DeBERTa-v3-base × CLINC150. | Confirmatory within scope | Centered/LN/ridge candidate layers exceed the final layer; oracle gains are 0.115/0.143/0.097. All three H1/H1′/H2 verdicts are `very_strong`. | [`EXP-003`](../research-logs/EXP-003.md) | [`EXP-003 agent report`](../agent-BuildReports/experiments/EXP-20260810-003--validated-probe-recoverability.md) | [`exp003_config.yaml`](../configs/exp003_config.yaml), [`exp003_run.py`](../scripts/exp003_run.py) | Canonical MLflow run `364bf897628d443189b1ae6f1288fc6e`; local `artifacts/EXP-20260810-003/results.json`; public copy and hash in the release bundle. |
| C3 | A noncanonical Llama-3.2-3B-Instruct path can preserve ARC-Easy task information for a small path-specific linear head while becoming poorly readable to the frozen native A--E readout. | Discovery/preliminary | Deduplicated discovery inventory: 586 good paths, 150 readability-gap witnesses. The independent `repeat_L28` control preserves task accuracy 0.9182 while native accuracy falls from about 0.90 to about 0.27. | [`EXP-004`](../research-logs/EXP-004.md) | [`EXP-004 H1 agent report`](../agent-BuildReports/experiments/EXP-20260828-004-H1-agent-report.md) | [`exp004_h1_frozen.yaml`](../configs/exp004_h1_frozen.yaml), [`source-wise supplement`](../user_exp_plans/EXP-20260828-004-h1-sourcewise-rerun-protocol.md), H1 runner modules | Train-only `D_fit/D_discover`; validation/test not accessed. Canonical H1 discovery MLflow exports and selected witnesses are in the public bundle. |
| C4 | Fixed-head-operational alternative paths are abundant and searchable sample-wise in DeBERTa-v3-base × CLINC150; under the tested 200-simulation budget, rank-reward MCTS finds them more often than binary-reward MCTS or random search. | Confirmatory within scope | Primary test: `R_short=0.9471`, Wilson lower 0.9397; `R_recov=0.8735`, Wilson lower 0.8420. Random also reaches 0.7449/0.6148. | [`EXP-004`](../research-logs/EXP-004.md) | [`EXP-004 H2 agent report`](../agent-BuildReports/experiments/EXP-20260831-004-H2-agent-report.md) | [`exp004_h2_full_v2.yaml`](../configs/exp004_h2_full_v2.yaml), [`exp004_h2_run.py`](../scripts/exp004_h2_run.py), H2 source/tests | MLflow run `2ba85363b2ac44fa9ac5e7318c4e04c6`; local `artifacts/EXP-20260831-004-h2-mcts-v2/`; compact audit files in the public bundle. |

## Research-record map

| Record | Function | Status in the argument |
|---|---|---|
| [`EXP-001`](../research-logs/EXP-001.md) | Original hypothesis and failed plain-probe implementation | Historical and superseded by EXP-002/003. |
| [`EXP-002`](../research-logs/EXP-002.md) | Competing explanations and discriminating diagnostics | Supports C1 and validates later probe choices. |
| [`EXP-003`](../research-logs/EXP-003.md) | Restored confirmatory experiment | Primary support for C2. |
| [`EXP-004`](../research-logs/EXP-004.md) | Readability and fixed-head path search | Supports C3 at discovery level and C4 at confirmatory level. |
| [`Post-EXP-003 direction history`](../research-logs/POST-EXP-003-DIRECTION-HISTORY.md) | Research trajectory and candidate EXP-004--006 directions | Context only; not hypothesis evidence. |
| [`Fragmented experiments`](../agent-BuildReports/fragmented-experiments/README.md) | Robustness checks, model/task probes, and method exploration | Supplementary unless explicitly promoted by a frozen experiment protocol. |

## Claim boundaries

The following stronger statements are **not** supported by the current record:

- that readout mismatch is the only reason arbitrary paths fail;
- that 25.6% is a population prevalence of readability collapse;
- that H1 generalizes beyond the train-derived ARC-Easy discovery splits;
- that MCTS is required to find operational alternative paths;
- that the H2 search is a deployable router (it uses gold-label reward);
- that any claim already generalizes across architectures, datasets, or seeds
  beyond the explicit experiment scope.

For EXP-004 H1, “EXP-004 is complete” denotes closure of the current experiment
package. The public scientific wording remains “strong train-discovery
existence evidence; held-out confirmation not run.” This distinction reconciles
the author log with the stricter objective report without altering either
historical record.

## Proposal citation rule

Every quantitative statement in the RP should carry one of the IDs C1--C4 in
the source or drafting notes. A reader should be able to traverse from that ID
to a human narrative, an objective report, a frozen configuration, a Git
revision, and a checksummed machine result. New claims should not be added to
the RP before they receive an entry here.
