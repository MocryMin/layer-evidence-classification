# Agent Protocol

Stable working conventions for all coding-agent sessions in this repository, including resumed and compacted sessions.

Keep this file limited to durable rules. Research hypotheses, experiment sequences, and temporary implementation plans belong in task-specific documents.

---

## 1. Purpose and Priority

This is a research-oriented machine-learning repository intended to support reproducible experiments and later Research Proposal or paper preparation.

Instruction priority:

```text
Current user instruction
> task-specific plan or referenced document
> AGENT_PROTOCOL.md
> ~/projects/CLAUDE.md
> existing repository conventions
> agent preference
```

Do not infer or replace the research plan from this protocol.
Do not duplicate global environment facts here unless needed for project reproducibility.

---

## 2. Session Continuity

Before substantial work, read in this order:

1. `~/projects/CLAUDE.md` — global environment and machine constraints;
2. `AGENT_PROTOCOL.md` — stable project workflow;
3. `PROJECT_RESOURCES.md`, if present — reusable datasets, models, checkpoints, and caches;
4. `PROJECT_STATUS.md`, if present — current verified state and next action;
5. `step_plan.md`, if relevant — current executable plan;
6. relevant code, configs, tests, and documentation.

At session start:

1. run `git status`;
2. inspect existing changes before editing;
3. identify the task and completion criteria;
4. verify that repository state matches any existing plan;
5. reuse existing conventions where reasonable.

Do not overwrite, revert, or broadly reformat unrelated user changes.
If repository state invalidates a plan, record the discrepancy and update the plan instead of silently improvising.

After a verified milestone, update `PROJECT_STATUS.md` when cross-session continuity is needed. Keep it short:

- current verified state;
- latest valid result;
- unresolved blockers;
- one concrete next action;
- important paths.

Do not use it as a chronological diary.

---

## 3. Scope, Environment, and Safety

Implement only the requested task and the minimum supporting work required to make it correct and verifiable.

Preferred workflow:

```text
inspect → plan briefly → implement → test → record
```

Avoid speculative features, unrelated refactoring, premature generalisation, unrequested architecture changes, and unnecessary tools or dependencies.

Use the shared environment described in `~/projects/CLAUDE.md`.
Do not install, upgrade, downgrade, or remove packages from the shared environment without explicit approval.

When a new dependency is necessary:

1. explain why existing dependencies are insufficient;
2. choose a maintained and minimal package;
3. record the exact version;
4. update the tracked dependency specification;
5. verify that the existing environment still works.

Do not persist temporary proxy or environment changes globally unless requested.
Never store credentials, passwords, tokens, private keys, or other secrets in tracked files, scripts, logs, documentation, or Git history.

Ask before destructive, privileged, expensive, machine-wide, difficult-to-reverse, or out-of-scope actions.
Prefer a smoke test before a costly run.
Report assumptions, failures, and deviations honestly.

---

## 4. Repository and Git

Use Git as the source of truth for code and tracked documentation.

Use these directories when needed:

```text
configs/     tracked experiment and runtime configurations
src/         reusable implementation
scripts/     thin execution entrypoints
tests/       automated tests
data/        local datasets, normally ignored by Git
artifacts/   generated outputs, normally ignored by Git
reports/     tracked summaries, tables, and figures
docs/        stable documentation and templates
```

Create directories only when they serve an actual task.
Keep reusable logic under `src/`; keep entrypoints thin.
Do not place core logic only in notebooks.
Use configurable or repository-relative paths in reusable code.

Make small, coherent commits at verified checkpoints.
Use concise scoped messages such as `data:`, `train:`, `eval:`, `analysis:`, `fix:`, or `docs:`.

Do not commit datasets, model weights, checkpoints, large caches, generated tensors, local MLflow databases, secrets, or machine-specific temporary files unless explicitly approved.

Do not push, force-push, rewrite history, delete branches, or configure remotes unless requested.

Any result used in a report or RP must record its Git commit and whether the working tree was dirty.

---

## 5. Reproducible Runs and Data

Parameters that materially affect results must come from tracked configuration files or explicit command-line arguments. Do not hide them inside implementation code.

Use MLflow for machine-generated run records, with a project-local tracking configuration rather than another project's database or experiment name.

Each meaningful run should use a stable ID:

```text
EXP-YYYYMMDD-NNN-short-name
```

Use the same ID for the MLflow run, artifact directory, experiment report, and generated tables or figures when practical.

Record, where applicable:

- resolved configuration;
- model and dataset identifiers;
- data split or version;
- random seed;
- selected layers or components;
- software versions;
- Git commit and dirty-tree status;
- parameter counts and runtime-relevant settings;
- all metrics required by the research question.

Preserve enough evidence to reproduce and analyse the result, including as appropriate:

- aggregate metrics;
- per-sample predictions;
- evaluation tables and plots;
- environment summary;
- failure logs;
- pointers to large checkpoints or caches.

Store large generated files outside Git, normally under:

```text
artifacts/<experiment_id>/
```

Prefer structured formats such as JSON, CSV, Parquet, NPZ, or safetensors over console dumps.
Keep valid negative results. Mark engineering failures as failed or invalid rather than deleting them.
Do not use the test set to choose models, thresholds, hyperparameters, or architectures.

Data acquisition and preprocessing must be reproducible through scripts or documented commands. Record source, licence, version or download date, checksums when useful, split definitions, label mapping, and preprocessing steps.

Use `PROJECT_RESOURCES.md` only when reusable resources exist. Before downloading a model or dataset, check it, the shared resources in `~/projects/CLAUDE.md`, and relevant local directories. Reuse valid resources and verify paths before updating the registry.

`PROJECT_RESOURCES.md` records what is available; `PROJECT_STATUS.md` records what is currently being worked on.

---

## 6. Code Quality and Validation

Prefer clear, direct implementations over framework-heavy abstraction.

Use meaningful names, type hints for public interfaces, concise documentation for non-obvious behaviour, explicit tensor-shape notes at important boundaries, actionable exceptions, deterministic seeds where practical, and assertions for critical assumptions.

Separate training, evaluation, and analysis logic.
Do not generalise a component before a real second use case exists.
Comments should explain intent or non-obvious decisions rather than restate the code.

Before declaring work complete:

1. run relevant unit or integration tests;
2. run a small smoke test for executable workflows;
3. confirm expected outputs exist and are readable;
4. inspect important values for obvious corruption;
5. inspect `git diff` and `git status`;
6. confirm recorded results point to the correct config and commit.

Prioritise tests for data transformations, metrics, masking and indexing, tensor shapes, configuration resolution, save/load behaviour, and previously observed bugs.

Do not claim success from code inspection alone.

---

## 7. Public resource Register

Read PROJECT_RESOURCES.md if present.  
New models, training database, and other resources should be registered in `PROJECT_RESOURCES.md`, every registered reource item should contain at least:  
name | type | path | source | status | notes  
Record only resources likely to be reused across sessions. Do not record ordinary temporary files.  
Before downloading a model or resources from internet, check `PROJECT_RESOURCES.md` first. 

---

## 8. Research Records and Handoff

Use each record for one purpose:

- **MLflow:** machine-generated run facts;
- **repository reports:** concise, reproducible interpretations and decisions;
- **Obsidian:** broader reasoning, literature notes, and personal research reflection;
- **PROJECT_STATUS.md:** current implementation state and next action.

For experiments used in decisions, reports, or RP claims, create:

```text
agent-BuildReports/experiments/<experiment_id>.md
```

Use the repository experiment-report template when present. Keep reports concise and separate:

- AI reporting statement(report model version, for example: GLM-5.2/dsV4/GPT5.6-sol/...)
- question;
- setup;
- results;
- observations supported directly by outputs;
- interpretation, alternatives, and limitations;
- decision;
- next action.

Do not mix observations with interpretations.

Every RP claim should be traceable through:

```text
claim
→ experiment report
→ MLflow run
→ resolved configuration
→ Git commit
→ underlying evidence
```

The goal is not merely working code, but work that a future session, collaborator, supervisor, or reviewer can understand and reproduce.
