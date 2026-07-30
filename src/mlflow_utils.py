"""MLflow setup for EXP-20260729-001.

Project-local SQLite backend (not the file store, which MLflow 3.x rejects, and
not another project's database). Resolves a relative ``sqlite:///mlruns.db`` to
the project root so runs are location-independent.
"""
from __future__ import annotations

from pathlib import Path

import mlflow

from .config import PROJECT_ROOT


def resolve_tracking_uri(uri: str) -> str:
    """Make a relative sqlite URI absolute under the project root."""
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri.startswith("sqlite:////"):
        rel = uri[len(prefix):]
        if not Path(rel).is_absolute():
            return "sqlite:///" + str((PROJECT_ROOT / rel).resolve())
    return uri


def setup_mlflow(cfg) -> None:
    uri = resolve_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)


def log_env_and_config(cfg, git_commit: str, dirty: bool) -> None:
    """Log resolved config + environment to the active MLflow run."""
    import torch
    import transformers

    mlflow.log_param("experiment_id", cfg.experiment_id)
    mlflow.log_param("model_name", cfg.model_name)
    mlflow.log_param("dataset", cfg.dataset)
    mlflow.log_param("dataset_config", cfg.dataset_config)
    mlflow.log_param("n_classes", cfg.n_classes)
    mlflow.log_param("learning_rate", cfg.learning_rate)
    mlflow.log_param("optimizer", cfg.optimizer)
    mlflow.log_param("weight_decay", cfg.weight_decay)
    mlflow.log_param("epochs", cfg.epochs)
    mlflow.log_param("batch_size", cfg.batch_size)
    mlflow.log_param("gradient_clip_norm", cfg.gradient_clip_norm)
    mlflow.log_param("seeds", ",".join(map(str, cfg.seeds)))
    mlflow.log_param("pooling", cfg.pooling)
    mlflow.log_param("max_length", cfg.max_length)
    mlflow.log_param("truncation", cfg.truncation)
    mlflow.log_param("backbone_frozen", cfg.backbone_frozen)
    mlflow.log_param("git_commit", git_commit)
    mlflow.log_param("git_dirty", dirty)
    mlflow.log_param("torch_version", torch.__version__)
    mlflow.log_param("transformers_version", transformers.__version__)
    mlflow.log_param("cuda_available", torch.cuda.is_available())
    mlflow.log_param("sentencepiece_version", cfg.raw.get("sentencepiece_version"))
