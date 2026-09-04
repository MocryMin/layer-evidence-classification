# Public artifacts

This directory is the public, selected view of the repository-local
`artifacts/` tree. Paths are preserved exactly so references in EXP-001--004
can be resolved mechanically from the public artifact root.

The directory is intentionally incomplete relative to the approximately
272 GB local archive. It contains the files needed to audit reported claims,
all small files directly referenced by the author logs, complete H1 path-result
JSON inventories, and explicit index pages at logical or deliberately private
roots. It excludes model weights, raw hidden states, prefix caches, path-head
tensors, live databases, and H2 simulation traces.

See `../ARTIFACT_ROOTS.json`, `../ARTIFACT_POINTERS.json`, and
`../MANIFEST.json` for the resolution rule, historical pointer normalization,
and checksums.
