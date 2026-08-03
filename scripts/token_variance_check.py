"""EXP-20260729-002 task 06: is the variance compression CLS-specific?

All collapse statistics so far (inter-sample std ~2e-4 at mid layers) are
computed on the CLS token only (the cached hidden states are the CLS of layers
1..12). This check re-forwards the frozen backbone (same config as EXP-001:
instruction prompt, left truncation, max_length 512) on a sample of 2000 train
utterances and measures, for each token position, the cross-sample per-dim std
mean - is the compression present on *all* positions or only on CLS?

Padding is excluded (attention mask); a position is reported only if it has at
least MIN_EFF effective samples.

Usage:
    python -u scripts/token_variance_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import load_diag_config  # noqa: E402
from src.data import build_label_maps, load_split, tokenise_split  # noqa: E402
from src.seeding import enable_determinism  # noqa: E402

N_SAMPLE = 2000          # sampling check (estimate error ~1.6% for per-dim std)
MIN_EFF = 300            # min effective (non-pad) samples to report a position
LAYERS = [1, 6, 12]      # shallow / mid / deep
SEED = 17


def _load_backbone(model_path, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path), dtype=torch.float32)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return tok, model


def _collate(batch_ids, pad_id):
    maxlen = max(len(ids) for ids in batch_ids)
    b = len(batch_ids)
    input_ids = torch.full((b, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((b, maxlen), dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        input_ids[i, : len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        attn[i, : len(ids)] = 1
    return input_ids, attn


@torch.no_grad()
def forward_sample(model, tok, tok_out, device, batch_size=64):
    """Streaming per-position stats across batches (dynamic padding: per-batch L varies).

    Returns {layer: {"count": (P,), "sum": (P, D), "sumsq": (P, D), "mask_sum": (P,)}}.
    """
    input_ids_list = tok_out["input_ids"][:N_SAMPLE]
    pad_id = tok.pad_token_id
    # pre-allocate to max_length (config bound)
    P, D = 512, 768
    acc = {l: {"count": torch.zeros(P, dtype=torch.long),
               "sum": torch.zeros(P, D, dtype=torch.float64),
               "sumsq": torch.zeros(P, D, dtype=torch.float64),
               "normsum": torch.zeros(P, dtype=torch.float64)} for l in LAYERS}
    max_seen = 0
    for s in range(0, len(input_ids_list), batch_size):
        batch_ids = input_ids_list[s : s + batch_size]
        ii, am = _collate(batch_ids, pad_id)
        outputs = model(ii.to(device), am.to(device), output_hidden_states=True)
        hs = outputs.hidden_states  # [0]=embedding, [l]=block l
        L = am.shape[1]
        max_seen = max(max_seen, L)
        for l in LAYERS:
            h = hs[l].detach().cpu().double()  # (B, L, D)
            for p in range(L):
                valid = am[:, p].bool()
                n = int(valid.sum())
                if n == 0:
                    continue
                x = h[valid, p, :]  # (n, D)
                acc[l]["count"][p] += n
                acc[l]["sum"][p] += x.sum(dim=0)
                acc[l]["sumsq"][p] += (x * x).sum(dim=0)
                acc[l]["normsum"][p] += x.norm(dim=1).sum()
    return acc, max_seen


def position_stats(acc: dict, max_seen: int, min_eff: int):
    """Per-position inter-sample std (per-dim std mean over effective samples)."""
    rows = []
    for p in range(max_seen):
        n_eff = int(acc["count"][p])
        if n_eff < min_eff:
            continue
        mean = acc["sum"][p] / n_eff
        var = acc["sumsq"][p] / n_eff - mean * mean
        var = var.clamp(min=0.0)
        inter_std = float(var.sqrt().mean())  # per-dim std across samples, mean over dims
        mean_norm = float(acc["normsum"][p] / n_eff)
        rows.append({"pos": p, "n_eff": n_eff, "inter_sample_std": inter_std, "mean_norm": mean_norm})
    return rows


def main():
    enable_determinism()
    cfg = load_diag_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[token_variance_check] n_sample={N_SAMPLE} layers={LAYERS} device={device}")
    print(f"  model={cfg.model_abs_path} prompt={cfg.prompt_with_instruction!r} "
          f"max_length={cfg.max_length} truncation={cfg.truncation}")

    tok, model = _load_backbone(cfg.model_abs_path, device)
    label2id, id2label, in_scope_ids = build_label_maps(
        cfg.dataset_abs_path, cfg.dataset_config, cfg.drop_oos_label)
    ds = load_split(cfg.dataset_abs_path, cfg.dataset_config, "train",
                    cfg.drop_oos_label, in_scope_ids)
    tok_out = tokenise_split(ds, tok, cfg.prompt_with_instruction, cfg.max_length, cfg.truncation)
    acc, max_seen = forward_sample(model, tok, tok_out, device)
    print(f"  sampled {N_SAMPLE} utterances; max seq len {max_seen}")

    results = {
        "description": "Per-token-position inter-sample std (variance compression check). "
                       "Same config as EXP-001 cache (frozen base, instruction prompt, left trunc, "
                       "max_length 512) on a 2000-sample train subset; padding excluded (mask).",
        "config": {"n_sample": N_SAMPLE, "min_eff": MIN_EFF, "layers": LAYERS, "seed": SEED,
                   "prompt": cfg.prompt_with_instruction, "max_length": cfg.max_length,
                   "truncation": cfg.truncation, "model": str(cfg.model_abs_path)},
        "layers": {},
    }
    print("\n=== per-position inter-sample std (n_eff >= %d) ===" % MIN_EFF)
    for l in LAYERS:
        rows = position_stats(acc[l], max_seen, MIN_EFF)
        results["layers"][str(l)] = rows
        inter = np.array([r["inter_sample_std"] for r in rows])
        cls_std = next(r["inter_sample_std"] for r in rows if r["pos"] == 0)
        n_pos = len(rows)
        print(f"  L{l:2d}: {n_pos} positions reported (n_eff>={MIN_EFF}) | "
              f"CLS std={cls_std:.6f} | non-CLS positions: min={inter[1:].min():.6f} "
              f"median={np.median(inter[1:]):.6f} max={inter[1:].max():.6f} | "
              f"fraction < 5e-4: {(inter[1:] < 5e-4).mean():.3f}, < 1e-3: {(inter[1:] < 1e-3).mean():.3f}")
        # first 12 positions + every 25th
        shown = list(range(min(12, n_pos))) + list(range(25, n_pos, 25))
        for i in shown:
            r = rows[i]
            print(f"    pos {r['pos']:3d} (n_eff={r['n_eff']:5d}): inter_std={r['inter_sample_std']:.6f} "
                  f"mean_norm={r['mean_norm']:.3f}")

    out_dir = ROOT / "artifacts/EXP-20260729-002/06_token_variance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "token_variance_check.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
