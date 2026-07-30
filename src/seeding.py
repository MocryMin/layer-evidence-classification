"""Reproducible RNG seeding.

Within one seed `s`, every layer reuses the same head-initialisation seed and
cached-data shuffle seed, so layer-wise differences are paired (AgentProtocol §2).
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_determinism() -> None:
    """Enable deterministic CUDA algorithms.

    Used for the frozen-backbone forward (caching) and head training so that
    cached features and head fits are reproducible per seed. DeBERTa-v3 on CLINC150
    is small enough that the deterministic slowdown is acceptable.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # CUBLAS workspaces needed by some deterministic ops
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
