"""Stage 2: validation-only learning-rate smoke test on representative layers.

Picks the global lr used by all 12 layers x 10 seeds. Test split is NOT involved.
"""
from __future__ import annotations

import torch

from src.artifact import ArtifactPaths
from src.config import load_config
from src.pipeline import run_smoke_lr
from src.seeding import enable_determinism


def main():
    cfg = load_config()
    if cfg.deterministic:
        enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = ArtifactPaths(cfg)
    print(f"[smoke_lr] device={device} candidates={cfg.lr_smoke_test['candidates']} "
          f"layers={cfg.lr_smoke_test['layers']} seed={cfg.lr_smoke_test['seed']}")
    out = run_smoke_lr(cfg, device, paths)
    print(f"[smoke_lr] selected lr={out['selected_lr']} -> {paths.run_config()}")


if __name__ == "__main__":
    main()
