# Layer Evidence Classification

This repository records an independent research project on intermediate-layer
evidence, readout validity, and alternative layer paths in pretrained language
models. The project is being developed as preliminary work for a PhD research
proposal; it is a research record rather than a finished paper.

## Public evidence release

The reviewed EXP-001--004 evidence layer is public at two fixed entry points:

- [GitHub source revision](https://github.com/MocryMin/layer-evidence-classification/tree/exp-001-004-evidence-v1)
  — code, protocols, reports, and evidence navigation;
- [Hugging Face evidence dataset](https://huggingface.co/datasets/MocryMin/lec-exp-001-004-evidence/tree/exp-001-004-evidence-v1)
  — the curated machine-readable evidence package and selected artifacts.

The stable artifact root and the exact log-pointer resolution rule are defined
in [Public artifact paths](docs/PUBLIC_ARTIFACTS.md). The source tag and HF
revision are immutable release anchors; later updates to `main` do not change
the evidence package they identify.

## Research question

When execution depth or layer order is allowed to change, two questions must be
kept separate:

1. does an intermediate representation retain task-relevant information; and
2. can a fixed downstream readout still interpret that representation?

EXP-001--003 establish the methodological motivation for this distinction.
EXP-004 then studies it directly in arbitrary-path search.

## Current evidence at a glance

| Study | Role | Current evidential status | Headline result |
|---|---|---|---|
| EXP-001 | initial study | superseded as a probe protocol | Plain AdamW probes produced false negatives on several intermediate layers. |
| EXP-002 | diagnosis | complete, diagnostic | Closed-form linear probes and normalization recover strong signal, locating the failure in optimization/readout rather than information absence. |
| EXP-003 | verification | complete, confirmatory within one model/task | Three probe families support intermediate-layer non-inferiority, superiority, and error recoverability on DeBERTa-v3-base × CLINC150. |
| EXP-004 H1 | path readability | strong train-discovery evidence; no cross-split confirmation | Noncanonical Llama-3.2-3B paths can retain task-head accuracy while becoming poorly readable to the frozen native head. |
| EXP-004 H2 | path searchability | complete, confirmatory within one model/task | On the untouched test split, rank-reward MCTS found shorter correct paths for 94.71% of canonical-correct samples and recovered 87.35% of canonical errors. |

The status labels above are deliberately conservative. In particular, EXP-004
H1 is not presented as a cross-split prevalence estimate. See the
[claim-to-evidence index](docs/EVIDENCE_INDEX.md) for exact claims, boundaries,
and provenance.

## For prospective supervisors and reviewers

- [Research-proposal workspace](proposal/README.md) — the 2--4 page proposal
  will be the short entry point once its first version is frozen.
- [Claim-to-evidence index](docs/EVIDENCE_INDEX.md) — the fastest route from a
  proposal claim to its log, report, configuration, run, and artifact.
- [Research logs](research-logs/README.md) — author-written chronological
  records, preserved as snapshots rather than retrospectively polished.
- [Objective experiment reports](agent-BuildReports/experiments/) — numerical
  reports reconstructed from machine-written artifacts.
- [Fragmented experiments](agent-BuildReports/fragmented-experiments/README.md)
  — exploratory controls and side investigations, clearly separated from the
  main confirmatory chain.
- [Reproducibility guide](docs/REPRODUCIBILITY.md) — environment, provenance,
  and rerun expectations.
- [Artifact registry](docs/ARTIFACT_REGISTRY.md) — what is public, what remains
  local, and why.
- [Public artifact paths](docs/PUBLIC_ARTIFACTS.md) — stable network roots and
  exact resolution of the paths cited in experiment logs.

## Evidence architecture

```text
2--4 page RP
    -> claim IDs and evidence index
        -> author research logs + objective agent reports
            -> frozen configs + source + tests + Git commits
                -> exported MLflow metadata + selected public artifacts
                    -> private/local full caches, weights, and raw traces
```

The public layer is intentionally selective. Model weights, upstream datasets,
rebuildable hidden-state caches, and hundreds of gigabytes of prefix-cache pages
are not redistributed. Public evidence packages contain reports, frozen
configuration, aggregate or audit-level results, hashes, and sufficient
provenance to locate or reproduce the full run.

## Reproducibility and provenance

The project uses Git, frozen configuration files, local MLflow tracking,
append-only or atomic run artifacts, objective agent reports, and SHA-256
manifests. The intended chain is:

```text
claim -> research log -> objective report -> config/run -> commit -> artifact
```

EXP-001/002 also have the historical evidence tag
`exp-001-002-evidence-v1`. A consolidated EXP-001--004 public-evidence bundle
is specified under [`release/`](release/README.md).

## AI-assistance disclosure

AI coding agents assisted with implementation, execution, auditing, and the
generation of explicitly labelled agent reports. Scientific framing,
experimental decisions, acceptance rules, and the author-written research logs
remain the researcher's responsibility. Reports disclose their AI-generated
status where applicable.

## License

No reuse license has yet been selected. Until a license is added, the repository
may be read for research review, but no permission to copy, modify, or
redistribute its contents is implied.
