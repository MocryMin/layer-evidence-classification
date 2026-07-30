"""Stage 1: one-pass frozen-backbone CLS hidden-state caching."""
from __future__ import annotations

import torch

from src.artifact import ArtifactPaths
from src.config import load_config
from src.pipeline import cache_all_splits
from src.seeding import enable_determinism


def main():
    cfg = load_config()
    if cfg.deterministic:
        enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cache] device={device} model={cfg.model_path}")
    paths = ArtifactPaths(cfg)
    manifest = cache_all_splits(cfg, device, paths)
    print(f"[cache] done. manifest: {paths.manifest()}")
    for split, e in manifest["splits"].items():
        print(f"  {split}: {e['shapes']['hidden']} dtype={e['dtypes']['hidden']} sha256={e['sha256'][:12]}")


if __name__ == "__main__":
    main()
