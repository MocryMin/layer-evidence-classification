# Artifact registry and publication boundary

Storage measurements below were taken on 2026-09-04 and are approximate.

## Storage tiers

| Tier | Location | Purpose | Public policy |
|---|---|---|---|
| A | GitHub repository | RP, navigation, logs, reports, protocols, configs, source, tests, small manifests | Public after review and anonymous-access verification. |
| B | HF dataset evidence bundle | Selected aggregate results, audit JSON, canonical MLflow export, hashes | Public; no upstream data, model weights, raw features, or full search traces. |
| C | Local/private archive | Full artifacts, feature caches, checkpoints, prefix cache, raw shards/traces, MLflow DB | Retained privately; publish only by an explicit later release decision. |

## Local inventory

| Local root | Approx. size | Scientific role | Initial public disposition |
|---|---:|---|---|
| `artifacts/EXP-20260729-001` | 398 MB | Original runs and hidden-state cache | Publish small config/result records; exclude tensor cache. |
| `artifacts/EXP-20260729-002` | 1.2 GB | Diagnostic controls | Publish selected JSON summaries/histories; exclude caches/checkpoint. |
| `artifacts/EXP-20260810-003` | 369 MB | Confirmatory result | Publish `results.json`, compact class/convergence summaries; exclude logits/predictions by default. |
| `artifacts/EXP-20260827-004-h1-discovery` | 30 GB | Legacy H1 train discovery | Publish aggregate, manifest, and selected witness; exclude per-path heads and event bulk. |
| `artifacts/EXP-20260828-004-h1-sourcewise-rerun` | 225 GB | Corrected H1 train discovery and cache study | Publish aggregate, manifest, policy, and selected witness; exclude GPU/SSD/HDD prefix pages, cache index, and per-path heads. |
| `artifacts/EXP-20260831-004-h2-mcts-v2` | 1.5 GB | H2 tuning/test search and audit | Publish final summary, post-run audit, tuning selection, gate, resolved configs, and manifest; retain 2.7M trace records and raw shards privately. |
| fragmented-experiment artifacts | about 13 GB plus readability preflights | Exploratory controls | Reports public; raw artifacts remain private unless promoted into a formal claim. |
| `models/` | 8.8 GB | Upstream and fine-tuned model files | Never include in the evidence bundle by default. |
| `mlruns/` + `mlruns.db` | about 46 MB plus DB | Local experiment tracking | Export selected metadata; do not publish the live database. |

The two H1 discovery directories dominate the 272 GB total because each cached
prefix node can contain the full sample state. Their size is an engineering
property of the cache, not an argument for uploading them.

## Selected public artifact principles

A file enters Tier B only if it is needed to reconstruct a headline number,
audit data access or run integrity, or identify the exact execution semantics.
Every entry must have:

- a repository-relative source path;
- a role and evidence category;
- a split-sensitivity label;
- byte size and SHA-256 hash;
- an explicit reason why it is safe and useful to publish.

The authoritative selection is
[`release/public_bundle_spec.json`](../release/public_bundle_spec.json). The
generated manifest, not directory size, defines the released evidence set.
The public bundle preserves every selected `artifacts/...` path exactly. Stable
network roots and the normalized historical log pointers are defined in
[`PUBLIC_ARTIFACTS.md`](PUBLIC_ARTIFACTS.md) and
[`release/ARTIFACT_POINTERS.json`](../release/ARTIFACT_POINTERS.json).

## Explicit exclusions

- credentials and local authentication state;
- upstream dataset rows or prompts beyond what is already necessary in reports;
- model weights and fine-tuned checkpoints;
- raw hidden states, logits, and per-sample predictions by default;
- H1 prefix-cache pages and per-path head tensors;
- H2 raw test shards and simulation traces in the initial outreach release;
- the live MLflow database and machine-specific cache indexes.

These exclusions do not weaken the provenance chain: the public manifest hashes
the selected evidence, while objective reports describe the retained private
records and their locations. A reviewer can request a controlled secondary
release if raw evidence becomes necessary.
