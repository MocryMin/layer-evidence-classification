"""Artifact path management and README generation for EXP-20260729-001."""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config


class ArtifactPaths:
    """Resolved paths under the artifact root."""

    def __init__(self, cfg: Config):
        self.root: Path = cfg.artifact_path
        self.cache: Path = self.root / "cache"
        self.checkpoints: Path = self.root / "checkpoints"
        self.predictions: Path = self.root / "predictions"
        self.metrics: Path = self.root / "metrics"
        self.confusion: Path = self.root / "confusion_matrices"
        self.plots: Path = self.root / "plots"

    def ensure(self) -> None:
        for p in (self.cache, self.checkpoints, self.predictions, self.metrics,
                  self.confusion, self.plots):
            p.mkdir(parents=True, exist_ok=True)

    def cache_file(self, split: str) -> Path:
        return self.cache / f"{split}_hidden.safetensors"

    def manifest(self) -> Path:
        return self.cache / "cache_manifest.json"

    def run_config(self) -> Path:
        return self.root / "run_config.yaml"

    def seeds_file(self) -> Path:
        return self.root / "seeds.json"

    def checkpoint_dir(self, seed: int, layer: int) -> Path:
        return self.checkpoints / f"seed_{seed}" / f"layer_{layer:02d}"

    def predictions_file(self, seed: int) -> Path:
        return self.predictions / f"seed_{seed}_test.parquet"

    def smoke_results(self) -> Path:
        return self.root / "smoke_lr_results.json"


def write_resolved_config(cfg: Config, lr: float, path: Path) -> None:
    """Write a resolved run_config.yaml with the chosen learning rate."""
    import yaml

    resolved = dict(cfg.raw)
    resolved["learning_rate"] = float(lr)
    resolved["lr_resolved_by"] = "validation-only smoke test on representative layers"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)


def write_artifact_readme(
    cfg: Config,
    paths: ArtifactPaths,
    git_commit: str,
    git_dirty: bool,
    mlflow_run_id: str,
    selected_lr: float,
    smoke_summary: dict | None = None,
) -> None:
    readme = paths.root / "README.md"
    lines = []
    ap = lines.append
    ap(f"# {cfg.experiment_id} - Artifacts\n")
    ap("Frozen-backbone intermediate-layer recoverability pilot on CLINC150 with "
       "DeBERTa-v3-base. See `AGENT_PROTOCOL.md` and the experiment plan for the full spec.\n")
    ap("## Provenance\n")
    ap(f"- Git commit: `{git_commit}` (dirty={git_dirty})")
    ap(f"- MLflow run ID: `{mlflow_run_id}`")
    ap(f"- Resolved learning rate: `{selected_lr}` (selected by validation-only smoke test)")
    ap(f"- Model: `{cfg.model_name}` (local: `{cfg.model_path}`)")
    ap(f"- Dataset: `{cfg.dataset}` config `{cfg.dataset_config}`, OOS label {cfg.drop_oos_label} dropped, "
       f"{cfg.n_classes} in-scope classes, splits 15000/3000/4500")
    ap(f"- Seeds: {cfg.seeds}")
    ap(f"- Backbone frozen; heads trained on cached CLS features of layers {cfg.probe_layers}.\n")

    ap("## Directory layout\n")
    ap("```")
    ap(f"{paths.root}/")
    ap("├── README.md                 # this file")
    ap("├── run_config.yaml           # resolved configuration (incl. selected lr)")
    ap("├── seeds.json                # seed list")
    ap("├── label2id.json / id2label.json")
    ap("├── smoke_lr_results.json     # lr smoke-test validation accuracies")
    ap("├── cache/                    # one-pass frozen-backbone CLS hidden states (float16)")
    ap("│   ├── train_hidden.safetensors  (15000, 12, 768)")
    ap("│   ├── validation_hidden.safetensors (3000, 12, 768)")
    ap("│   ├── test_hidden.safetensors (4500, 12, 768)")
    ap("│   └── cache_manifest.json   # shapes, dtypes, sha256")
    ap("├── checkpoints/seed_<seed>/layer_<01-12>/epoch_<001-100>.pt + best_checkpoint.json")
    ap("├── predictions/seed_<seed>_test.parquet")
    ap("├── metrics/                  # layer_metrics.csv, classwise_recovery.csv, ...")
    ap("├── confusion_matrices/       # <seed>_layer_<NN>.npz (150x150)")
    ap("└── plots/")
    ap("```\n")

    ap("## Layer indexing\n")
    ap("DeBERTa-v3-base has 12 Transformer layers. With `output_hidden_states=True`, "
       "`hidden_states[0]` is the embedding output and `hidden_states[l]` is the output of "
       f"Transformer block `l`. Probe layer `l` in {{1..12}} uses "
       f"`hidden_states[{cfg.hidden_state_offset} + l - 1][:, {cfg.cls_token_index}, :]` (CLS token). "
       f"Layer {cfg.final_layer} is the final layer L. Oracle uses intermediate layers 1..11.\n")

    ap("## Seed policy\n")
    ap("The same 10 seeds are used for every layer. Within a seed `s`, every layer shares the "
       "same head-initialisation seed and cached-data shuffle seed, so layer-wise comparisons are "
       "paired. RNGs reset at the start of each (seed, layer) run.\n")

    ap("## Checkpoint selection\n")
    ap(f"Every epoch end is saved. The selected checkpoint maximises validation accuracy; ties "
       f"are broken by lower validation NLL. The test split is evaluated exactly once with the "
       f"selected checkpoint. Test results are never used to choose layers, epochs, thresholds, "
       f"or hyperparameters.\n")

    ap("## Predictions schema (`predictions/seed_<seed>_test.parquet`)\n")
    ap("One row per (layer, sample). Columns: `sample_id, text, gold_label, layer, logits "
       "(list[150]), probabilities (list[150]), prediction, nll, probability_margin, "
       "logit_margin, gold_margin, entropy`.\n")

    ap("## Metric definitions\n")
    ap("- accuracy, macro-F1, NLL, probability margin (top1-top2 prob), logit margin "
       "(top1-top2 logit), gold margin (z_y - max_{c!=y} z_c), predictive entropy.")
    ap("- ECE: 10 equal-width confidence bins; ECE = sum_m (|I_m|/n) * |acc_m - conf_m|.")
    ap("- Recoverability R_l = P(y_l=y | y_L != y); harm H_l = P(y_l!=y | y_L=y).")
    ap("- Class-wise R_{l,c} = P(y_l=y | y=c, y_L!=c); H_{l,c} = P(y_l!=y | y=c, y_L=y). "
       "Class-wise ratios store (numerator, denominator); a zero denominator is recorded as NA.")
    ap("- Oracle: Acc_oracle = P(y_L=y OR exists l<L: y_l=y); "
       "R_oracle = P(exists l<L: y_l=y | y_L!=y). Identity asserted: "
       "Acc_oracle = Acc_L + (1-Acc_L)*R_oracle.")
    ap("- Class-wise JS divergence D_JS^class in [0,1]: comparing the class distribution of "
       "oracle-recoverable final-layer errors (r) against all final-layer errors (e), via the "
       "mixture m=(e+r)/2: D_JS = (0.5*KL(e||m) + 0.5*KL(r||m)) / log2, with 0*log0=0; NA if no "
       "final-layer errors or no recoverable errors.\n")

    ap("## Reproducing summary tables\n")
    ap("All summary tables are derived from `predictions/seed_<seed>_test.parquet` and the "
       "checkpoint `best_checkpoint.json` files. Re-run `scripts/run_analysis.py` to regenerate "
       "`metrics/` and the hypothesis judgement without retraining; the script only reads cached "
       "predictions and checkpoints.\n")

    if smoke_summary:
        ap("## LR smoke test\n")
        ap("Validation-only. Candidate LRs " + str(cfg.lr_smoke_test.get("candidates"))
           + " on layers " + str(cfg.lr_smoke_test.get("layers"))
           + f" seed {cfg.lr_smoke_test.get('seed')}. Selected lr = `{selected_lr}`.\n")
        ap("```json")
        ap(json.dumps(smoke_summary, indent=2, default=str))
        ap("```")

    with open(readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
