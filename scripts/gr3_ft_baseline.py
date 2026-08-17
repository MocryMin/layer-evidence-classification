"""gr3 step 3: fine-tuning baseline on DeBERTa-v3-base / WOS-46985.

Stages (resumable via --stage):
  probe    - LN-plain init probe on the frozen backbone's final layer (gr1
             protocol, reuses the 260812_04 baseline cache) -> probe_init.pt
  smoke    - short FT runs over lr candidates (full unfreeze) -> smoke.json
  ft       - 5-epoch fine-tune, variant full|attn (attention-only), LN-plain
             head initialised from probe_init and trained jointly; AdamW
             wd=0.01, bs=32 (microbs 8 x accum 4), val eval 4x/epoch, best
             checkpoint by final-layer val acc -> ft_<variant>/
  analyze  - on the best checkpoint: per-layer inter_std/PR/top1 (EXP-002
             collapse check) + all-layer ln_plain and ridge probes ->
             analysis/<variant>_results.json

Usage:
    python scripts/gr3_ft_baseline.py --stage probe
    python scripts/gr3_ft_baseline.py --stage smoke
    python scripts/gr3_ft_baseline.py --stage ft --variant full
    python scripts/gr3_ft_baseline.py --stage ft --variant attn
    python scripts/gr3_ft_baseline.py --stage analyze --variant full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fragmented import (  # noqa: E402
    RIDGE_ALPHAS, layer_variance_stats, load_cache_split, load_wos_46985,
    run_gradient_family_frag, run_ridge_family_frag,
)
from src.heads import build_head  # noqa: E402
from src.probe import train_probe_fullbatch_es  # noqa: E402
from src.seeding import enable_determinism, seed_all  # noqa: E402

ART = ROOT / "artifacts/fragmented-experiments/FT-BaselineDeBERTaV3BaseWOS46985_260818_03"
BASE = ROOT / "artifacts/fragmented-experiments/DeBERTaV3BaseWOS46985Baseline_260812_04"
MODEL_PATH = ROOT / "models/deberta-v3-base"

N_CLASSES, N_LAYERS, FINAL_LAYER = 134, 12, 12
MAX_LENGTH, SEED = 512, 17
GR1_TRAINING = dict(lr=0.01, weight_decay=0.0, grad_clip=0.0,
                    min_epochs=100, max_epochs=10000, patience=100,
                    min_delta=1e-4, seed=SEED)
FT = dict(lr=None, weight_decay=0.01, epochs=5, batch_size=32,
          micro_batch=8, evals_per_epoch=4, warmup_ratio=0.1, grad_clip=1.0)
SMOKE_LRS = [5e-6, 1e-5, 2e-5, 5e-5]
SMOKE_STEPS = 250


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #
def load_backbone():
    from transformers import AutoModel
    return AutoModel.from_pretrained(str(MODEL_PATH), dtype=torch.float32)


def tokenise(tok, texts):
    """Per-sample ids (no padding; padding is per-batch longest at use time)."""
    return tok(texts, truncation=True, max_length=MAX_LENGTH)["input_ids"]


def collate(batch_ids, pad_id):
    maxlen = max(len(ids) for ids in batch_ids)
    input_ids = torch.full((len(batch_ids), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch_ids), maxlen), dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        input_ids[i, : len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        attn[i, : len(ids)] = 1
    return input_ids, attn


class LNPlainClassifier(nn.Module):
    """Backbone + LNHead (LayerNorm + linear) on the final-layer CLS."""

    def __init__(self, backbone, in_dim: int, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = build_head("ln", in_dim=in_dim, n_classes=n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0])


def freeze_variant(model: LNPlainClassifier, variant: str) -> dict:
    """full: everything trainable. attn: only self-attn + attn output dense."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)
    if variant == "full":
        for p in model.backbone.parameters():
            p.requires_grad_(True)
    elif variant == "attn":
        n = 0
        for name, p in model.backbone.named_parameters():
            if ".attention.self." in name or ".attention.output.dense." in name:
                p.requires_grad_(True)
                n += p.numel()
        print(f"[freeze] attn-only: {n/1e6:.1f}M trainable backbone params")
    else:
        raise ValueError(variant)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable_params": n_tr, "variant": variant}


@torch.no_grad()
def eval_val(model, ids_val, y_val, pad_id, device, batch_size=64):
    model.eval()
    preds = []
    for s in range(0, len(ids_val), batch_size):
        ii, am = collate(ids_val[s : s + batch_size], pad_id)
        logits = model(ii.to(device), am.to(device))
        preds.append(logits.argmax(1).cpu())
    return (torch.cat(preds) == y_val).float().mean().item()


# --------------------------------------------------------------------------- #
# stage: probe
# --------------------------------------------------------------------------- #
def stage_probe(device):
    hidden, labels = {}, {}
    for split in ("train", "validation", "test"):
        h, y = load_cache_split(BASE / "cache", split)
        hidden[split], labels[split] = h, y
    tx = torch.as_tensor(hidden["train"][:, FINAL_LAYER - 1], dtype=torch.float32, device=device)
    vx = torch.as_tensor(hidden["validation"][:, FINAL_LAYER - 1], dtype=torch.float32, device=device)
    tr_y = torch.as_tensor(labels["train"].astype(np.int64), device=device)
    va_y = torch.as_tensor(labels["validation"].astype(np.int64), device=device)

    head = build_head("ln", in_dim=tx.shape[1], n_classes=N_CLASSES)
    res = train_probe_fullbatch_es(head=head, train_x=tx, train_y=tr_y,
                                   val_x=vx, val_y=va_y, device=device,
                                   **GR1_TRAINING)
    head.load_state_dict(res["best_state"])
    head.to(device).eval()
    tex = torch.as_tensor(hidden["test"][:, FINAL_LAYER - 1], dtype=torch.float32, device=device)
    te_y = torch.as_tensor(labels["test"].astype(np.int64), device=device)
    with torch.no_grad():
        te_pred = torch.cat([head(tex[s : s + 1024]) for s in range(0, len(tex), 1024)]).argmax(1)
    test_acc = (te_pred == te_y).float().mean().item()
    ART.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), ART / "probe_init.pt")
    out = dict(best_val_acc=res["best_val_acc"], best_epoch=res["best_epoch"],
               test_acc=test_acc, converged=res["converged"],
               crosscheck_baseline=dict(val=0.2954243, test=0.2851974),
               dev_val=abs(res["best_val_acc"] - 0.2954243),
               dev_test=abs(test_acc - 0.2851974))
    json.dump(out, open(ART / "probe_init.json", "w"), indent=1)
    print(f"[probe] val {res['best_val_acc']:.4f} (base 0.2954, dev {out['dev_val']:.4f}) "
          f"test {test_acc:.4f} (base 0.2852, dev {out['dev_test']:.4f})")


# --------------------------------------------------------------------------- #
# stages: smoke / ft
# --------------------------------------------------------------------------- #
def _tokenised_splits(tok):
    data = load_wos_46985()
    return ({k: tokenise(tok, v[0]) for k, v in data.items()},
            {k: torch.as_tensor(v[1], dtype=torch.long) for k, v in data.items()})


def _run_ft(model, ids_tr, y_tr, ids_va, y_va, pad_id, device, lr,
            n_epochs, eval_cb=None, max_steps=None, eval_every=None):
    """AdamW wd=0.01, bs=32 (microbs 8 x accum 4), linear warmup+decay."""
    from transformers import get_linear_schedule_with_warmup
    n_train = len(ids_tr)
    steps_per_epoch = n_train // FT["batch_size"] + 1
    n_steps = steps_per_epoch * n_epochs
    warmup = int(n_steps * FT["warmup_ratio"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=FT["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, warmup, n_steps)
    if eval_every is None:
        eval_every = max(1, steps_per_epoch // FT["evals_per_epoch"])
    n_steps = min(n_steps, max_steps) if max_steps else n_steps
    history = []
    step = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
        micro_losses = []
        opt.zero_grad()
        for s in range(0, n_train, FT["batch_size"]):
            idx = perm[s : s + FT["batch_size"]]
            y = y_tr[idx].to(device)
            for ms in range(0, len(idx), FT["micro_batch"]):
                midx = idx[ms : ms + FT["micro_batch"]]
                ii, am = collate([ids_tr[i] for i in midx], pad_id)
                logits = model(ii.to(device), am.to(device))
                loss = F.cross_entropy(logits, y[ms : ms + FT["micro_batch"]])
                (loss * len(midx) / len(idx)).backward()
                micro_losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], FT["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % eval_every == 0:
                va = eval_val(model, ids_va, y_va, pad_id, device)
                history.append({"step": step, "epoch": epoch,
                                "eval_in_epoch": step % steps_per_epoch or steps_per_epoch,
                                "train_loss": float(np.mean(micro_losses[-steps_per_epoch // 4:])),
                                "val_acc": va})
                print(f"  step {step}/{n_steps} (ep {epoch}): "
                      f"loss {history[-1]['train_loss']:.4f} val_acc {va:.4f}")
                if eval_cb:
                    eval_cb(model, history[-1])
                model.train()
            if max_steps and step >= max_steps:
                return history
    return history


def _init_model(device, variant=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    backbone = load_backbone()
    model = LNPlainClassifier(backbone, in_dim=backbone.config.hidden_size,
                              n_classes=N_CLASSES).to(device)
    model.head.load_state_dict(torch.load(ART / "probe_init.pt", map_location=device))
    if variant:
        freeze_variant(model, variant)
    return model, tok


def stage_smoke(device):
    _, tok = _init_model(device)
    ids, ys = _tokenised_splits(tok)
    pad_id = tok.pad_token_id
    out = {"lrs": SMOKE_LRS, "steps": SMOKE_STEPS, "batch_size": FT["batch_size"], "runs": []}
    for lr in SMOKE_LRS:
        print(f"[smoke] lr={lr:g}")
        model, _ = _init_model(device)          # fresh backbone + probe-init head
        freeze_variant(model, "full")
        hist = _run_ft(model, ids["train"], ys["train"], ids["validation"],
                       ys["validation"], pad_id, device, lr, n_epochs=1,
                       max_steps=SMOKE_STEPS, eval_every=50)
        out["runs"].append({"lr": lr, "final_val_acc": hist[-1]["val_acc"],
                            "best_val_acc": max(h["val_acc"] for h in hist),
                            "history": hist})
    best = max(out["runs"], key=lambda r: (r["best_val_acc"], -r["lr"]))
    out["chosen_lr"] = best["lr"]
    json.dump(out, open(ART / "smoke.json", "w"), indent=1)
    print(f"[smoke] chosen lr {best['lr']:g} (best val {best['best_val_acc']:.4f})")


def stage_ft(device, variant):
    model, tok = _init_model(device)
    freeze = freeze_variant(model, variant)
    ids, ys = _tokenised_splits(tok)
    pad_id = tok.pad_token_id
    lr = json.load(open(ART / "smoke.json"))["chosen_lr"]

    best = {"val_acc": -1.0, "state": None, "step": -1}

    def eval_cb(m, rec):
        nonlocal best
        if rec["val_acc"] > best["val_acc"]:
            best = {"val_acc": rec["val_acc"], "step": rec["step"],
                    "state": {k: v.detach().cpu().clone()
                              for k, v in m.state_dict().items()}}

    hist = _run_ft(model, ids["train"], ys["train"], ids["validation"],
                   ys["validation"], pad_id, device, lr, FT["epochs"], eval_cb)
    # final test acc at the best-val checkpoint
    model.load_state_dict(best["state"])
    model.to(device).eval()
    with torch.no_grad():
        preds = []
        for s in range(0, len(ids["test"]), 64):
            ii, am = collate(ids["test"][s : s + 64], pad_id)
            preds.append(model(ii.to(device), am.to(device)).argmax(1).cpu())
        test_acc = (torch.cat(preds) == ys["test"]).float().mean().item()

    d = ART / f"ft_{variant}"
    d.mkdir(parents=True, exist_ok=True)
    sd = best["state"]
    head_sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
    torch.save(head_sd, d / "ft_head.pt")
    bbd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    model.backbone.load_state_dict(bbd)
    model.backbone.save_pretrained(d)
    tok.save_pretrained(d)
    json.dump({"variant": variant, "freeze": freeze, "lr": lr, "ft": FT,
               "history": hist, "best_step": best["step"],
               "best_val_acc": best["val_acc"], "test_acc_at_best": test_acc},
              open(d / "ft_history.json", "w"), indent=1)
    print(f"[ft:{variant}] best val {best['val_acc']:.4f} @step {best['step']} "
          f"test {test_acc:.4f} -> {d}")


# --------------------------------------------------------------------------- #
# stage: analyze
# --------------------------------------------------------------------------- #
def stage_analyze(device, variant):
    from src.fragmented import extract_features
    d = ART / f"ft_{variant}"
    cache = d / "cache"
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(d))
    backbone = AutoModel.from_pretrained(str(d), dtype=torch.float16)
    data = load_wos_46985()
    hidden, labels = {}, {}
    for split in ("train", "validation", "test"):
        if (cache / f"{split}_hidden.npz").exists():
            h, y = load_cache_split(cache, split)
        else:
            h = extract_features(backbone, tok, data[split][0], pooling="cls",
                                 max_length=MAX_LENGTH, batch_size=32,
                                 device=device)
            y = data[split][1]
            cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache / f"{split}_hidden.npz",
                                hidden=h, labels=y.astype(np.int16))
        hidden[split], labels[split] = h, y
        print(f"[analyze] {split}: {h.shape}")

    union = np.concatenate([hidden[s] for s in ("train", "validation", "test")], axis=0)
    stats = layer_variance_stats(union)
    for l, s in enumerate(stats, 1):
        print(f"  L{l:2d}: inter_std {s['inter_std']:.4f} PR {s['participation_ratio']:.1f} "
              f"top1 {s['top1_var_frac']:.3f}")

    layers = list(range(1, N_LAYERS + 1))
    ln_fam = {"name": "ln_plain", "head_type": "ln", "centering": "none"}
    ln_res = run_gradient_family_frag(hidden, labels, ln_fam, GR1_TRAINING,
                                      layers, FINAL_LAYER, N_CLASSES, device)
    ridge_res = run_ridge_family_frag(hidden, labels, RIDGE_ALPHAS, layers,
                                      FINAL_LAYER, N_CLASSES)
    (ART / "analysis").mkdir(exist_ok=True)
    json.dump({"variant": variant,
               "variance": [dict(layer=i + 1, **s) for i, s in enumerate(stats)],
               "collapse_min_inter_std": min(s["inter_std"] for s in stats),
               "threshold": 1e-3,
               "ln_plain": {"per_layer": {k: {kk: vv for kk, vv in v.items()
                                              if not isinstance(vv, np.ndarray)}
                                          for k, v in ln_res["per_layer"].items()},
                            "recoverability": ln_res.get("recoverability")},
               "ridge": {"per_layer": ridge_res["per_layer"],
                         "recoverability": ridge_res["recoverability"]}},
              open(ART / f"analysis/{variant}_results.json", "w"), indent=1)
    np.save(ART / f"analysis/{variant}_ridge_test_pred.npy",
            np.asarray(ridge_res["test_pred"]))
    print(f"[analyze:{variant}] done -> {ART}/analysis/{variant}_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["probe", "smoke", "ft", "analyze"])
    ap.add_argument("--variant", choices=["full", "attn"], default="full")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enable_determinism()
    ART.mkdir(parents=True, exist_ok=True)
    dict(probe=stage_probe, smoke=stage_smoke, ft=stage_ft,
         analyze=stage_analyze)[args.stage](device, **({} if args.stage in ("probe", "smoke") else {"variant": args.variant}))


if __name__ == "__main__":
    main()
