"""Configuration loading and resolution for EXP-20260729-001."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "exp_config.yaml"


@dataclass
class Config:
    """Resolved experiment configuration."""

    raw: dict
    experiment_id: str
    model_name: str
    model_path: str
    dataset_source: str
    dataset_config: str
    drop_oos_label: int
    n_classes: int
    prompt: str
    max_length: int
    truncation: str
    pooling: str
    cls_token_index: int
    n_transformer_layers: int
    final_layer: int
    probe_layers: list[int]
    hidden_state_offset: int
    cache_dtype: str
    training_dtype: str
    head_input_dim: int
    learning_rate: float | None
    optimizer: str
    weight_decay: float
    epochs: int
    batch_size: int
    gradient_clip_norm: float
    seeds: list[int]
    epsilon_1: float
    epsilon_2: float
    bootstrap_resamples: int
    bootstrap_ci: float
    artifact_root: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    deterministic: bool
    lr_smoke_test: dict = field(default_factory=dict)

    @property
    def artifact_path(self) -> Path:
        return PROJECT_ROOT / self.artifact_root

    @property
    def model_abs_path(self) -> Path:
        p = Path(self.model_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def dataset_abs_path(self) -> Path:
        p = Path(self.dataset_source)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        raw=raw,
        experiment_id=raw["experiment_id"],
        model_name=raw["model_name"],
        model_path=raw["model_path"],
        dataset_source=raw["dataset_source"],
        dataset_config=raw["dataset_config"],
        drop_oos_label=raw["drop_oos_label"],
        n_classes=raw["n_classes"],
        prompt=raw["prompt"],
        max_length=raw["max_length"],
        truncation=raw["truncation"],
        pooling=raw["pooling"],
        cls_token_index=raw["cls_token_index"],
        n_transformer_layers=raw["n_transformer_layers"],
        final_layer=raw["final_layer"],
        probe_layers=list(raw["probe_layers"]),
        hidden_state_offset=raw["hidden_state_offset"],
        cache_dtype=raw["cache_dtype"],
        training_dtype=raw["training_dtype"],
        head_input_dim=raw["head_input_dim"],
        learning_rate=raw.get("learning_rate"),
        optimizer=raw["optimizer"],
        weight_decay=raw["weight_decay"],
        epochs=raw["epochs"],
        batch_size=raw["batch_size_cached_head_training"],
        gradient_clip_norm=raw["gradient_clip_norm"],
        seeds=list(raw["seeds"]),
        epsilon_1=raw["epsilon_1"],
        epsilon_2=raw["epsilon_2"],
        bootstrap_resamples=raw["bootstrap_resamples"],
        bootstrap_ci=raw["bootstrap_ci"],
        artifact_root=raw["artifact_root"],
        mlflow_tracking_uri=raw["mlflow_tracking_uri"],
        mlflow_experiment_name=raw["mlflow_experiment_name"],
        deterministic=raw.get("deterministic", True),
        lr_smoke_test=raw.get("lr_smoke_test", {}),
    )


DIAG_CONFIG = PROJECT_ROOT / "configs" / "diag_config.yaml"


@dataclass
class DiagConfig:
    """Lightweight holder for the EXP-002 diagnostic config (raw dict + key fields)."""

    raw: dict
    experiment_id: str
    model_path: str
    dataset_abs_path: Path
    dataset_config: str
    drop_oos_label: int
    n_classes: int
    n_transformer_layers: int
    hidden_state_offset: int
    cls_token_index: int
    head_input_dim: int
    cache_dtype: str
    max_length: int
    truncation: str
    prompt_with_instruction: str
    prompt_pure: str
    lr_grid: list
    probe_layers: list
    representative_layers: list
    epochs: int
    batch_size: int
    weight_decay: float
    gradient_clip_norm: float
    seed: int
    optimizers: list
    lbfgs: dict
    finetune: dict
    artifact_root: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str

    @property
    def artifact_path(self) -> Path:
        return PROJECT_ROOT / self.artifact_root

    @property
    def model_abs_path(self) -> Path:
        p = Path(self.model_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_diag_config(path: str | Path = DIAG_CONFIG) -> DiagConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DiagConfig(
        raw=raw,
        experiment_id=raw["experiment_id"],
        model_path=raw["model_path"],
        dataset_abs_path=(PROJECT_ROOT / raw["dataset_source"]),
        dataset_config=raw["dataset_config"],
        drop_oos_label=raw["drop_oos_label"],
        n_classes=raw["n_classes"],
        n_transformer_layers=raw["n_transformer_layers"],
        hidden_state_offset=raw["hidden_state_offset"],
        cls_token_index=raw["cls_token_index"],
        head_input_dim=raw["head_input_dim"],
        cache_dtype=raw["cache_dtype"],
        max_length=raw["max_length"],
        truncation=raw["truncation"],
        prompt_with_instruction=raw["prompt_with_instruction"],
        prompt_pure=raw["prompt_pure"],
        lr_grid=list(raw["lr_grid"]),
        probe_layers=list(raw["probe_layers"]),
        representative_layers=list(raw["representative_layers"]),
        epochs=raw["epochs"],
        batch_size=raw["batch_size"],
        weight_decay=raw["weight_decay"],
        gradient_clip_norm=raw["gradient_clip_norm"],
        seed=raw["seed"],
        optimizers=list(raw["optimizers"]),
        lbfgs=raw.get("lbfgs", {}),
        finetune=raw.get("finetune", {}),
        artifact_root=raw["artifact_root"],
        mlflow_tracking_uri=raw["mlflow_tracking_uri"],
        mlflow_experiment_name=raw["mlflow_experiment_name"],
    )
