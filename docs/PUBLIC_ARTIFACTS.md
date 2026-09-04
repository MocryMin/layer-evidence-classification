# Public artifact paths

The public evidence dataset mirrors repository artifact paths instead of
placing them below an extra packaging directory. This makes the paths recorded
in experiment logs mechanically resolvable.

## Stable roots

```text
SOURCE_ROOT = https://github.com/MocryMin/layer-evidence-classification/tree/exp-001-004-evidence-v1

EVIDENCE_ROOT = hf://datasets/MocryMin/lec-exp-001-004-evidence@exp-001-004-evidence-v1/

ARTIFACTS_ROOT = hf://datasets/MocryMin/lec-exp-001-004-evidence@exp-001-004-evidence-v1/artifacts/

EXP_ROOT = ARTIFACTS_ROOT
```

The corresponding human-browsable artifact root is:

```text
https://huggingface.co/datasets/MocryMin/lec-exp-001-004-evidence/tree/exp-001-004-evidence-v1/artifacts
```

For an artifact recorded as `artifacts/EXP-X/result.json`, append the entire
repository-relative path to `EVIDENCE_ROOT`, or remove the leading
`artifacts/` and append the remainder to `ARTIFACTS_ROOT`. Both constructions
identify the same file. `EXP_ROOT` is the short alias used by experiment logs
and reports: for example, the EXP-003 root is
`EXP_ROOT/EXP-20260810-003/`.

## Historical relative paths

The author logs are byte-preserving historical snapshots, so their path syntax
is not silently rewritten. `release/ARTIFACT_POINTERS.json` records the base
root, normalized path, line number, and publication status for every artifact
pointer in EXP-001--004.

- `./file` is normally resolved against the experiment's `Artifact Path`.
- `../EXP-X/file` is resolved against the parent `artifacts/` directory.
- EXP-004 used `./EXP-X/...` relative to the global `artifacts/` directory,
  despite also naming `artifacts/EXP-20260824-004/` as an umbrella path. The
  public dataset supplies that umbrella as an index and keeps each actual H1/H2
  run at its recorded sibling path.
- A directory pointer can denote a deliberately selected public subset. Its
  `README.md`/manifest identifies excluded raw or rebuildable material.

No pointer may silently dangle. It must resolve to a public file/directory, a
public metadata page for deliberately private material, or an explicitly
registered pattern whose expansions are recorded in the release manifest.

## Scope boundary

Exact path preservation does not imply that every local byte is public. Model
weights, upstream data, hidden-state caches, prefix-cache pages, per-path head
tensors, and H2 raw simulation traces remain private or rebuildable. Their
directory-level public indexes explain the boundary without impersonating the
excluded file.
