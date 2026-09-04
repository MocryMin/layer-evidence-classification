# Public evidence release

This directory defines the curated, versioned evidence package supporting
EXP-001--004. It is designed for Hugging Face Dataset hosting because the
largest selected result file is unsuitable for the Git repository, while the
full 272 GB local artifact tree would be inappropriate and unnecessary.

## Release identity

- Bundle: `lec-exp-001-004-evidence-v1`
- Intended HF dataset: `MocryMin/lec-exp-001-004-evidence`
- Source revision: `exp-001-004-evidence-v1`
- Selection policy: [`public_bundle_spec.json`](public_bundle_spec.json)
- Dataset card template: [`HF_DATASET_CARD.md`](HF_DATASET_CARD.md)
- Tracked generated manifest: `release/manifests/`

## Build

From the repository root:

```bash
python scripts/build_public_evidence.py
```

The builder creates `dist/lec-exp-001-004-evidence-v1/`, exports selected
MLflow metadata, preserves selected repository `artifacts/...` paths, expands
only bounded file sets declared in the specification, checks for unapproved
tensor or database formats, scans the bundle for common credential patterns,
and writes SHA-256 manifests with HF URIs and web URLs.

Stable source/evidence/artifact roots are defined by `ARTIFACT_ROOTS.json`.
`ARTIFACT_POINTERS.json` normalizes every artifact reference in the four formal
author logs and is checked by the test suite.

The output directory must not already exist. This prevents an old or manually
modified file from being silently mixed into a new release.

The builder also requires a clean checkout whose `HEAD` resolves to the
specification's `source_revision`. To reconstruct this release after `main` has
moved forward, first check out `exp-001-004-evidence-v1` in a dedicated
worktree; the release must never be rebuilt from newer tracked files while
retaining the old version label.

## Publication gate

Do not label the release public until all of the following are true:

1. repository tests and release-builder tests pass;
2. manifest file count and size match the build summary;
3. Git source tree and history pass a credential scan;
4. Git tag and commit are pushed;
5. GitHub repository visibility is public;
6. the exact built directory is uploaded to HF;
7. GitHub, HF, the manifest, and a representative artifact are opened in an
   unsigned browser session.

The RP PDF has its own publication gate and should be anonymously verified when
it is added; it is not part of this evidence-only release.

The GitHub visibility change and HF upload are external publication actions.
They should be performed only after the local release candidate is complete.

## Publication record

The EXP-001--004 evidence release was published and anonymously verified on
2026-09-04:

- GitHub source tag: `exp-001-004-evidence-v1`, commit
  `6081e8ecf316400e3b69e820be0c557c7457a763`;
- HF revision: `exp-001-004-evidence-v1`, commit
  `7aa5ed568738ad006808830387b85e5a9eebb50e`;
- manifest entries: 4,132;
- physical files: 4,134;
- selected payload: 115,690,703 bytes.
