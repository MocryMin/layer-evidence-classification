"""gr2 tasks 1-3 (modularized layer probe) on WOS-46985 for two encoders.

`user_exp_plans/fragmented_exp_gr2.md` tasks 1-3, config as gr2
(ridge alpha=1e-6 closed form, CLS tail, train fit / val acc, seed 17):
  1. every layer as 1st/only layer (modular single [i]) vs in-place acc
     (raw chain in trained order, pre-norm readouts);
  2. greedy layer queue: best single first, greedily append the best of
     all layers (repeats allowed) to max len 50, acc per len recorded;
  3. pairwise add-layer matrices: A_ij, G_ij = A_ij - A_i,
     S_ij = A_ij - max(A_i, A_j) for all i,j.

Models: deberta-v3-base (12 layers; rel_pos is position-only constant,
no conv, no final norm) and modernbert-base (22 layers; RoPE cos/sin and
global/sliding attention masks are position/mask-only constants; each
layer carries its own attention_type). Every per-application input to a
layer module is constant across applications, so arbitrary layer
sequences are well-defined and the raw chain equals a true model forward
(smoke-verified). Readouts are PRE-norm for every layer (final_norm is
excluded from the layer modules; a post-norm L22 cross-check is provided
against the gr1 baseline).

WOS-46985: HYDRA-count split 30070/7518/9397 (seed 17), 134 L2 classes,
max_length 512 (gr1 baseline config; median doc length 250 tokens).

VRAM strategy: a full-length fp16 state for the whole train split is
~23.6 GB and cannot be materialized, so all forwards stream doc-chunks
with fp16 states; per-branch cost is len(path) layer applications per
chunk (embeddings are recomputed per branch, they are cheap). Per-model
batch/chunk in MODELS: deberta 128/2048 (legacy disentangled attention
materializes fp32 scores per batch — 256/4096 peaked at 29.5 GiB reserved
and WDDM-paged), modernbert 256/2048 (mask tensors scale with chunk;
measured peak reserved 11.8 GiB). All states and layer compute are fp16
(weights loaded fp16); the ridge solve is fp64 closed form
(sklearn-equivalent), as in gr2.

Usage:
    python scripts/frag_modular_probe_wos.py --model deberta --smoke
    python scripts/frag_modular_probe_wos.py --model deberta --time-probe
    python scripts/frag_modular_probe_wos.py --model deberta [--resume]
    python scripts/frag_modular_probe_wos.py --model modernbert [--resume]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.modernbert.modeling_modernbert import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fragmented import exp_dir, git_state, load_cache_split, load_wos_46985  # noqa: E402
from src.metrics import confusion_matrix, macro_f1  # noqa: E402
from src.seeding import enable_determinism  # noqa: E402

N_CLASSES = 134
ALPHA = 1e-6
BATCH = 256
CHUNK = 4096
MAX_LEN = 512

MODELS = {
    # VRAM: DeBERTa-v3 uses legacy disentangled attention (fp32 scores per
    # batch) — batch 128 keeps transients ~3.2 GiB; modernbert mask tensors
    # scale with chunk — chunk 2048 keeps peak reserved ~14 GiB. Measured
    # 2026-08-14: batch 256/chunk 4096 hit reserved 29.5 GiB on deberta
    # (WDDM paging, ~7x slowdown); fixed by these settings.
    "deberta": dict(
        name="DeBERTaV3BaseWOS46985LayerProbe_260814_01",
        path="models/deberta-v3-base",
        baseline_art="DeBERTaV3BaseWOS46985Baseline_260812_04",
        batch=128, chunk=2048,
    ),
    "modernbert": dict(
        name="ModernBERTBaseWOS46985LayerProbe_260814_02",
        path="models/modernbert-base",
        baseline_art="ModernBERTBaseWOS46985Baseline_260812_05",
        batch=256, chunk=2048,
    ),
}


class Deadline(Exception):
    pass


def parse_deadline(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M"))


# --------------------------------------------------------------------------- #
# Modular layer stacks
# --------------------------------------------------------------------------- #
class StackDeberta:
    """DeBERTa-v3 layer modules; rel_pos/mask constants per application."""

    def __init__(self, model):
        self.enc = model.encoder
        self.layers = self.enc.layer
        self.rel_emb = self.enc.get_rel_embedding()

    @torch.no_grad()
    def apply(self, h, mask_c, li):
        am = self.enc.get_attention_mask(mask_c)          # [b, 1, L, L] additive
        rel_pos = self.enc.get_rel_pos(h)                 # position-only constant
        out, _ = self.layers[li](h, am, relative_pos=rel_pos,
                                 rel_embeddings=self.rel_emb)
        return out


class StackModernBert:
    """modernBERT layer modules; RoPE + attention masks are per-split constants."""

    def __init__(self, model):
        self.model = model
        self.layers = model.layers
        self.final_norm = model.final_norm
        self.rotary = model.rotary_emb
        self.am = self.pe = None

    def prepare(self, x0_c, mask_c):
        """Per-chunk constants (depend only on shape/dtype/mask, not values)."""
        kw = dict(config=self.model.config, inputs_embeds=x0_c,
                  attention_mask=mask_c)
        self.am = {
            "full_attention": create_bidirectional_mask(**kw),
            "sliding_attention": create_bidirectional_sliding_window_mask(**kw),
        }
        pos_ids = torch.arange(x0_c.shape[1], device=x0_c.device).unsqueeze(0)
        self.pe = {t: self.rotary(x0_c, pos_ids, t)
                   for t in set(self.model.config.layer_types)}

    @torch.no_grad()
    def apply(self, h, mask_c, li):
        b = h.shape[0]
        layer = self.layers[li]
        out = layer(h, attention_mask=self.am[layer.attention_type][:b],
                    position_embeddings=self.pe[layer.attention_type][:b])
        return out


@torch.no_grad()
def embeddings(model, ids_c, mask_c) -> torch.Tensor:
    if model.config.model_type == "modernbert":
        return model.embeddings(input_ids=ids_c)
    return model.embeddings(ids_c, mask=mask_c)  # deberta: pad positions zeroed


@torch.no_grad()
def apply_batched(stack, h, mask_c, li, batch) -> torch.Tensor:
    out = torch.empty_like(h)
    for s in range(0, h.shape[0], batch):
        out[s:s + batch] = stack.apply(h[s:s + batch], mask_c[s:s + batch], li)
    return out


@torch.no_grad()
def apply_path(stack, x0, mask_c, path, batch) -> torch.Tensor:
    h = x0
    for li in path:
        h = apply_batched(stack, h, mask_c, li, batch)
    return h


@torch.no_grad()
def chunk_pass(stack, model, tok_ids, mask, path, cands, batch, chunk, device):
    """Streamed multi-branch forward.

    For each doc-chunk: x0 = embeddings, h = apply(path), then every
    candidate layer applied on h. Returns (path_cls, cand_cls) — [N, D] fp16
    CLS tensors on device (path_cls = embeddings CLS when path == []).
    """
    n = tok_ids.shape[0]
    D = model.config.hidden_size
    path_cls = torch.empty((n, D), dtype=torch.float16, device=device)
    cand_cls = {li: torch.empty((n, D), dtype=torch.float16, device=device)
                for li in cands}
    for s in range(0, n, chunk):
        e = s + chunk
        ids_c = tok_ids[s:e].to(device)
        mask_c = mask[s:e].to(device)
        x0 = embeddings(model, ids_c, mask_c)
        if isinstance(stack, StackModernBert):
            stack.prepare(x0, mask_c)
        h = apply_path(stack, x0, mask_c, path, batch)
        path_cls[s:e] = h[:, 0]
        for li in cands:
            cand_cls[li][s:e] = apply_batched(stack, h, mask_c, li, batch)[:, 0]
    return path_cls, cand_cls


@torch.no_grad()
def collect_inplace(stack, model, tok_ids, mask, n_layers, batch, chunk, device):
    """Pre-norm CLS per layer of the raw chain (trained order) -> {1..L: [N,D]}."""
    n = tok_ids.shape[0]
    D = model.config.hidden_size
    out = {li: torch.empty((n, D), dtype=torch.float16, device=device)
           for li in range(1, n_layers + 1)}
    for s in range(0, n, chunk):
        e = s + chunk
        ids_c = tok_ids[s:e].to(device)
        mask_c = mask[s:e].to(device)
        h = embeddings(model, ids_c, mask_c)
        if isinstance(stack, StackModernBert):
            stack.prepare(h, mask_c)
        for li in range(n_layers):
            h = apply_batched(stack, h, mask_c, li, batch)
            out[li + 1][s:e] = h[:, 0]
    return out


# --------------------------------------------------------------------------- #
# Ridge (closed form, sklearn-equivalent)
# --------------------------------------------------------------------------- #
def fit_ridge_torch(x_tr, y_tr, x_va, y_va, n_classes=N_CLASSES):
    """Closed-form ridge alpha=ALPHA, fp64 inputs — equal to sklearn
    ``RidgeClassifier(alpha, fit_intercept=True, solver='svd')`` with
    {+1,-1} targets, X/Y centered, decision on uncentered val."""
    x_tr, x_va = x_tr.double(), x_va.double()
    n = x_tr.shape[0]
    mu = x_tr.mean(dim=0)
    xc = x_tr - mu
    Y = torch.full((n, n_classes), -1.0, dtype=torch.float64, device=x_tr.device)
    Y[torch.arange(n, device=x_tr.device), y_tr] = 1.0
    Yc = Y - Y.mean(dim=0)
    G = xc.t() @ xc
    h = xc.t() @ Yc
    try:
        lam, V = torch.linalg.eigh(G)
    except RuntimeError:
        lam, V = torch.linalg.eigh(G.cpu())
        lam, V = lam.to(x_tr.device), V.to(x_tr.device)
    lam = lam.clamp(min=0.0)
    W = V @ ((1.0 / (lam + ALPHA)).unsqueeze(1) * (V.t() @ h))
    b = Y.mean(dim=0) - mu @ W
    dec = x_va @ W + b
    pred = dec.argmax(dim=1)
    acc = float((pred == y_va).double().mean())
    f1 = macro_f1(confusion_matrix(y_va.cpu().numpy(), pred.cpu().numpy(),
                                   n_classes))
    return acc, f1, pred


# --------------------------------------------------------------------------- #
# Node recorder (resumable)
# --------------------------------------------------------------------------- #
class NodeRecorder:
    """Append-only node results (resumable); preds rows align with JSONL lines."""

    def __init__(self, d: Path, resume: bool):
        self.jsonl = d / "nodes.jsonl"
        self.preds_path = d / "nodes_pred.npy"
        self.completed: dict[str, tuple[float, float]] = {}
        self.preds: list[np.ndarray] = []
        if resume and self.jsonl.exists():
            for line in self.jsonl.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.completed[",".join(map(str, r["path"]))] = (
                    r["val_acc"], r["macro_f1"])
            if self.preds_path.exists():
                arr = np.load(self.preds_path)
                self.preds = [arr[i] for i in range(arr.shape[0])]
        self.f = self.jsonl.open("a", encoding="utf-8")

    def record(self, path, tasks, acc, f1, pred):
        rec = {"path": list(path), "len": len(path), "tail_layer": path[-1],
               "tasks": tasks, "val_acc": acc, "macro_f1": f1, "ts": time.time()}
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.completed[",".join(map(str, path))] = (acc, f1)
        self.preds.append(pred.cpu().numpy().astype(np.int16))

    def flush_preds(self):
        if self.preds:
            np.save(self.preds_path, np.stack(self.preds))

    def close(self):
        self.f.close()


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def run_singles_pairs(stack, model, ids_tr, mask_tr, ids_va, mask_va, y_tr, y_va,
                      rec, deadline_ts, n_layers, batch, chunk, device,
                      print_every=10):
    """Task 1 (singles) + task 3 (pairs): one chunk_pass per i covers [i] + [i,j]."""
    for i in range(1, n_layers + 1):
        if time.time() > deadline_ts:
            raise Deadline()
        if (str(i) in rec.completed and
                all(f"{i},{j}" in rec.completed
                    for j in range(1, n_layers + 1))):
            continue  # fully recorded in an earlier run (resume)
        cands = [j - 1 for j in range(1, n_layers + 1)]
        tr_path, tr_cands = chunk_pass(stack, model, ids_tr, mask_tr, [i - 1],
                                       cands, batch, chunk, device)
        va_path, va_cands = chunk_pass(stack, model, ids_va, mask_va, [i - 1],
                                       cands, batch, chunk, device)
        if str(i) not in rec.completed:
            acc, f1, pred = fit_ridge_torch(tr_path, y_tr, va_path, y_va)
            rec.record((i,), ["single"], acc, f1, pred)
        for j in range(1, n_layers + 1):
            key = f"{i},{j}"
            if key in rec.completed:
                continue
            acc, f1, pred = fit_ridge_torch(tr_cands[j - 1], y_tr,
                                            va_cands[j - 1], y_va)
            rec.record((i, j), ["pair"], acc, f1, pred)
        print(f"[singles/pairs] i={i:2d} done · {len(rec.completed)} nodes · "
              f"single={rec.completed[str(i)][0]:.4f} · t={time.time():.0f}")
        if i % print_every == 0:
            rec.flush_preds()


def run_greedy(stack, model, ids_tr, mask_tr, ids_va, mask_va, y_tr, y_va,
               rec, steps_path, deadline_ts, max_len, n_layers, batch, chunk,
               device):
    """Task 2: greedy queue growth from the best single layer; repeats allowed."""
    steps = []
    if steps_path.exists():
        steps = [json.loads(l) for l in steps_path.read_text(encoding="utf-8").splitlines()]
    f = steps_path.open("a", encoding="utf-8")

    if steps:
        queue = list(steps[-1]["queue"])
        prev_acc = steps[-1]["val_acc"]
    else:
        missing = [i for i in range(1, n_layers + 1)
                   if str(i) not in rec.completed]
        if missing:
            print(f"[greedy] singles missing for layers {missing} — skipping "
                  f"greedy; resume singles/pairs first")
            f.close()
            return []
        singles = {i: rec.completed[str(i)][0] for i in range(1, n_layers + 1)}
        best = max(singles, key=lambda i: (singles[i], -i))
        queue, prev_acc = [best], singles[best]

    cands = [li - 1 for li in range(1, n_layers + 1)]
    for step in range(len(steps) + 1, max_len):
        if time.time() > deadline_ts:
            raise Deadline()
        path = [li - 1 for li in queue]  # queue is 1-based; apply is 0-based
        tr_path, tr_cands = chunk_pass(stack, model, ids_tr, mask_tr, path,
                                       cands, batch, chunk, device)
        va_path, va_cands = chunk_pass(stack, model, ids_va, mask_va, path,
                                       cands, batch, chunk, device)
        cand_accs = {}
        for li in range(1, n_layers + 1):
            key = ",".join(map(str, queue + [li]))
            if key in rec.completed:
                cand_accs[li] = rec.completed[key][0]
                continue
            acc, f1, pred = fit_ridge_torch(tr_cands[li - 1], y_tr,
                                            va_cands[li - 1], y_va)
            rec.record(tuple(queue + [li]), ["greedy_cand"], acc, f1, pred)
            cand_accs[li] = acc
        chosen = max(cand_accs, key=lambda li: (cand_accs[li], -li))
        queue.append(chosen)
        step_rec = {"step": step, "queue": list(queue), "chosen": chosen,
                    "val_acc": cand_accs[chosen],
                    "gain": cand_accs[chosen] - prev_acc,
                    "candidates": cand_accs, "ts": time.time()}
        f.write(json.dumps(step_rec) + "\n")
        f.flush()
        steps.append(step_rec)
        prev_acc = cand_accs[chosen]
        rec.flush_preds()
        print(f"[greedy] step {step:2d}: +L{chosen:2d} -> acc={prev_acc:.4f} "
              f"(gain={step_rec['gain']:+.4f}) queue={queue}")
    f.close()
    return steps


# --------------------------------------------------------------------------- #
# Smoke checks
# --------------------------------------------------------------------------- #
def run_smoke_checks(stack, model, ids_tr, mask_tr, y_tr, y_va, inplace_tr,
                     inplace_va, n_layers, d, device):
    from sklearn.linear_model import RidgeClassifier

    checks = {}
    ids = ids_tr[:64].to(device)
    mask = mask_tr[:64].to(device)
    with torch.no_grad():
        o = model(ids, attention_mask=mask, output_hidden_states=True)
        x0 = embeddings(model, ids, mask)
        checks["embeddings_max_abs_diff"] = float(
            (o.hidden_states[0].float() - x0.float()).abs().max())
    # full chain pre-norm per layer, first 64 docs
    n = 64
    with torch.no_grad():
        h = embeddings(model, ids, mask)
        if isinstance(stack, StackModernBert):
            stack.prepare(h, mask)
        chain = {}
        for li in range(n_layers):
            h = stack.apply(h, mask, li)
            chain[li + 1] = h[:, 0]
    # raw model hidden_states: [0]=emb, [1..L-1]=pre-norm layers, [L]=post-norm
    # (capture_outputs tie_last_hidden_states overwrites the last entry)
    diffs = {}
    for li in range(1, n_layers + 1):
        if isinstance(stack, StackModernBert) and li == n_layers:
            raw = o.hidden_states[li][:, 0]          # already post-final_norm
            mod = stack.final_norm(chain[li])
        else:
            raw = o.hidden_states[li][:, 0]
            mod = chain[li]
        diffs[li] = float((raw.float() - mod.float()).abs().max())
    checks["chain_vs_raw_max_abs_diff_per_layer"] = diffs
    checks["chain_vs_raw_max_abs_diff_max"] = max(diffs.values())

    # torch vs sklearn ridge on in-place layers {1, mid, last} (smoke subset)
    y_tr_np = y_tr[:2000].cpu().numpy()
    y_va_np = y_va[:1000].cpu().numpy()
    for li in (1, n_layers // 2, n_layers):
        xt = inplace_tr[li][:2000].cpu().numpy().astype(np.float64)
        xv = inplace_va[li][:1000].cpu().numpy().astype(np.float64)
        acc_t = fit_ridge_torch(inplace_tr[li][:2000], y_tr[:2000],
                                inplace_va[li][:1000], y_va[:1000])[0]
        clf = RidgeClassifier(alpha=ALPHA, fit_intercept=True, solver="svd")
        clf.fit(xt, y_tr_np)
        acc_s = float((clf.predict(xv) == y_va_np).mean())
        checks[f"ridge_inplace_L{li}_torch"] = acc_t
        checks[f"ridge_inplace_L{li}_sklearn"] = acc_s
    (d / "smoke_checks.json").write_text(json.dumps(checks, indent=2),
                                         encoding="utf-8")
    return checks


def baseline_crosscheck(d: Path, baseline_art: str, n_layers: int) -> dict:
    """Fit alpha=1e-6 on the gr1 baseline caches; compare against the baseline
    results.json per_alpha val accs (validates cache pipeline + readout defs)."""
    out = {}
    bdir = ROOT / "artifacts/fragmented-experiments" / baseline_art
    results = json.loads((bdir / "results.json").read_text(encoding="utf-8"))
    ridge_pp = results["families"]["ridge"]["per_layer"]
    for split in ("train", "validation"):
        h, _ = load_cache_split(bdir / "cache", split)
        if split == "train":
            tr = torch.as_tensor(h, dtype=torch.float64)
            tr_y = torch.as_tensor(_, dtype=torch.long)
        else:
            va = torch.as_tensor(h, dtype=torch.float64)
            va_y = torch.as_tensor(_, dtype=torch.long)
    per_layer = {}
    for li in range(1, n_layers + 1):
        acc, _, _ = fit_ridge_torch(tr[:, li - 1].contiguous(), tr_y,
                                    va[:, li - 1].contiguous(), va_y)
        per_layer[li] = {"my_a1e6_val_acc": acc,
                         "baseline_a1e6_val_acc":
                             ridge_pp[str(li)]["per_alpha"]["1e-06"]["val_acc"],
                         "baseline_best_alpha": ridge_pp[str(li)]["best_alpha"],
                         "baseline_best_val_acc": ridge_pp[str(li)]["best_val_acc"]}
    (d / "baseline_crosscheck.json").write_text(
        json.dumps(per_layer, indent=2), encoding="utf-8")
    return per_layer


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--time-probe", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--deadline", default="2026-08-15 08:00")
    ap.add_argument("--greedy-max-len", type=int, default=50)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = MODELS[args.model]
    batch, chunk = mcfg["batch"], mcfg["chunk"]
    d = exp_dir(mcfg["name"])
    deadline_ts = parse_deadline(args.deadline)
    print(f"[{mcfg['name']}] device={device} deadline={args.deadline} "
          f"smoke={args.smoke}")

    # ---- data ------------------------------------------------------------ #
    data = load_wos_46985()
    texts_tr, y_np_tr = data["train"]
    texts_va, y_np_va = data["validation"]
    print(f"[data] train={len(texts_tr)} val={len(texts_va)} "
          f"classes={N_CLASSES}")
    if args.smoke:
        rng0 = np.random.default_rng(args.seed)
        idx_tr = rng0.choice(len(texts_tr), size=2000, replace=False)
        idx_va = rng0.choice(len(texts_va), size=1000, replace=False)
        texts_tr = [texts_tr[i] for i in idx_tr]
        y_np_tr = y_np_tr[idx_tr]
        texts_va = [texts_va[i] for i in idx_va]
        y_np_va = y_np_va[idx_va]

    tok = AutoTokenizer.from_pretrained(str(ROOT / mcfg["path"]))
    enc_tr = tok(texts_tr, padding="max_length", max_length=MAX_LEN,
                 truncation=True, return_tensors="pt")
    enc_va = tok(texts_va, padding="max_length", max_length=MAX_LEN,
                 truncation=True, return_tensors="pt")
    ids_tr, mask_tr = enc_tr["input_ids"], enc_tr["attention_mask"].long()
    ids_va, mask_va = enc_va["input_ids"], enc_va["attention_mask"].long()
    print(f"[tok] max_length={MAX_LEN} truncation=right")

    # weights fp16: direct layer-module calls compute in fp16 (states fp16);
    # ridge is fp64 closed form on the fp16 CLS (gr2 stack-mode precedent)
    model = AutoModel.from_pretrained(str(ROOT / mcfg["path"]),
                                      torch_dtype=torch.float16).to(device).eval()
    n_layers = model.config.num_hidden_layers
    stack = (StackModernBert(model) if model.config.model_type == "modernbert"
             else StackDeberta(model))

    y_tr = torch.as_tensor(y_np_tr, dtype=torch.long, device=device)
    y_va = torch.as_tensor(y_np_va, dtype=torch.long, device=device)

    # ---- time probe (full data only) ------------------------------------- #
    if args.time_probe:
        assert not args.smoke
        for rep in range(2):
            t0 = time.time()
            _, _ = chunk_pass(stack, model, ids_tr, mask_tr, [0], [1],
                              batch, chunk, device)
            _, _ = chunk_pass(stack, model, ids_va, mask_va, [0], [1],
                              batch, chunk, device)
            t1 = time.time()
            G = torch.randn(768, 768, dtype=torch.float64, device=device)
            G = G.t() @ G
            torch.linalg.eigh(G)
            t2 = time.time()
            print(f"[time-probe] rep{rep}: 1-app train+val = {t1 - t0:.2f}s "
                  f"(ridge eigh 768x768 = {t2 - t1:.2f}s)")
        print(f"[time-probe] rough: pairs ~{(t1 - t0) * n_layers * 2 * 2:.0f}s, "
              f"greedy sum_k (k+n) ~{(t1 - t0) * (50 * 49 / 2 + 49 * n_layers) * 2:.0f}s")
        return

    # ---- in-place reference (raw chain, trained order) ------------------- #
    inplace_path = d / "inplace.json"
    if args.smoke or not inplace_path.exists():
        print("[inplace] raw chain per layer...")
        t0 = time.time()
        inplace_tr = collect_inplace(stack, model, ids_tr, mask_tr, n_layers,
                                     batch, chunk, device)
        inplace_va = collect_inplace(stack, model, ids_va, mask_va, n_layers,
                                     batch, chunk, device)
        print(f"[inplace] forwards done in {time.time() - t0:.1f}s; fitting...")
        inplace = {}
        for li in range(1, n_layers + 1):
            acc, f1, _ = fit_ridge_torch(inplace_tr[li], y_tr,
                                         inplace_va[li], y_va)
            inplace[li] = {"val_acc": acc, "macro_f1": f1}
        if isinstance(stack, StackModernBert):
            # raw model final readout is post-final_norm (baseline L22 def)
            acc, f1, _ = fit_ridge_torch(stack.final_norm(inplace_tr[n_layers]),
                                         y_tr,
                                         stack.final_norm(inplace_va[n_layers]),
                                         y_va)
            inplace[f"L{n_layers}_postnorm"] = {"val_acc": acc, "macro_f1": f1}
        if not args.smoke:
            inplace_path.write_text(json.dumps(inplace, indent=2),
                                    encoding="utf-8")
        print(f"[inplace] L1..L{n_layers} accs: "
              f"{[round(inplace[li]['val_acc'], 4) for li in range(1, n_layers + 1)]}")
    else:
        inplace = {}
        for k, v in json.loads(inplace_path.read_text(encoding="utf-8")).items():
            try:
                k = int(k)
            except ValueError:
                pass  # e.g. "L22_postnorm"
            inplace[k] = v

    rec = NodeRecorder(d, args.resume)
    print(f"[resume] completed nodes loaded: {len(rec.completed)}")

    # ---- smoke mode ------------------------------------------------------ #
    if args.smoke:
        checks = run_smoke_checks(stack, model, ids_tr, mask_tr, y_tr, y_va,
                                  inplace_tr, inplace_va, n_layers, d, device)
        print(json.dumps(checks, indent=2))
        cc = baseline_crosscheck(d, mcfg["baseline_art"], n_layers)
        print("[baseline-crosscheck]")
        for li, v in cc.items():
            print(f"  L{li:2d}: mine={v['my_a1e6_val_acc']:.4f} "
                  f"base_a1e6={v['baseline_a1e6_val_acc']:.4f} "
                  f"base_best={v['baseline_best_val_acc']:.4f}")
        rec.close()
        return

    # ---- tasks 1 + 3 (singles + pairs) ----------------------------------- #
    print("[singles/pairs] ...")
    t0 = time.time()
    try:
        run_singles_pairs(stack, model, ids_tr, mask_tr, ids_va, mask_va,
                          y_tr, y_va, rec, deadline_ts, n_layers, batch, chunk,
                          device)
    except Deadline:
        print("[stop] deadline reached during singles/pairs")
    else:
        print(f"[singles/pairs] done in {time.time() - t0:.1f}s "
              f"({len(rec.completed)} nodes)")

    # ---- task 2 (greedy) ------------------------------------------------- #
    print("[greedy] ...")
    t0 = time.time()
    try:
        steps = run_greedy(stack, model, ids_tr, mask_tr, ids_va, mask_va,
                           y_tr, y_va, rec, d / "greedy_steps.jsonl",
                           deadline_ts, args.greedy_max_len, n_layers, batch,
                           chunk, device)
    except Deadline:
        steps_path = d / "greedy_steps.jsonl"
        steps = ([json.loads(l) for l in
                  steps_path.read_text(encoding="utf-8").splitlines()]
                 if steps_path.exists() else [])
        print("[stop] deadline reached during greedy — finalizing partial results")
    else:
        print(f"[greedy] done in {time.time() - t0:.1f}s ({len(steps)} steps)")

    # ---- finalize -------------------------------------------------------- #
    rec.flush_preds()
    accs = [s["val_acc"] for s in steps]
    gains = [s["gain"] for s in steps]
    neg_steps = [s["step"] for s in steps if s["gain"] < 0]
    max_step = accs.index(max(accs)) + 1 if accs else None
    greedy_analysis = {
        "n_steps": len(steps),
        "final_queue": steps[-1]["queue"] if steps else None,
        "final_val_acc": accs[-1] if accs else None,
        "max_val_acc": max(accs) if accs else None,
        "max_acc_step": max_step,
        "n_negative_gain_steps": len(neg_steps),
        "negative_gain_steps": neg_steps,
        "max_acc_after_negative_step": bool(neg_steps) and max_step > neg_steps[0],
    }

    A = np.full((n_layers, n_layers), np.nan)
    for i in range(1, n_layers + 1):
        for j in range(1, n_layers + 1):
            key = f"{i},{j}"
            if key in rec.completed:
                A[i - 1, j - 1] = rec.completed[key][0]
    a_diag = np.array([rec.completed.get(str(i), (np.nan, np.nan))[0]
                       for i in range(1, n_layers + 1)])
    Gmat = A - a_diag[:, None]
    Smat = A - np.maximum(a_diag[:, None], a_diag[None, :])
    np.savez(d / "task3_pairwise.npz", A=A, G=Gmat, S=Smat)
    (d / "task3_pairwise.json").write_text(json.dumps({
        "A": A.tolist(), "G": Gmat.tolist(), "S": Smat.tolist(),
        "definition": "A_ij = val acc of path [i,j]; G_ij = A_ij - A_i "
                      "(single-layer acc of i); S_ij = A_ij - max(A_i, A_j)",
    }, indent=2), encoding="utf-8")

    cc = baseline_crosscheck(d, mcfg["baseline_art"], n_layers)

    results = {
        "experiment": mcfg["name"],
        "date": "2026-08-14",
        "reporting_model": "deepseek-v4-flash",
        "git": git_state(),
        "config": {
            "plan": "user_exp_plans/fragmented_exp_gr2.md (tasks 1-3)",
            "model": f"{args.model} (frozen, fp16 weights + fp16 states)",
            "model_path": mcfg["path"], "dataset": "wos (WOS-46985)",
            "n_classes": N_CLASSES,
            "split_sizes": {"train": len(texts_tr), "validation": len(texts_va)},
            "max_length": MAX_LEN, "truncation": "right (baseline config; "
            "median doc 250 tokens)",
            "pooling": "cls (pre-norm for every layer; final_norm excluded "
            "from modernbert layer modules)", "seed": args.seed,
            "batch": batch, "chunk": chunk,
            "ridge": {"alpha": ALPHA, "method": "closed-form eigen-solve fp64, "
                      "sklearn-equivalent (RidgeClassifier solver='svd', "
                      "fit_intercept=True)", "fit": "train", "eval": "validation"},
            "modular_semantics": "layer module applied to its input states; "
            "all per-application inputs (attention masks, relative position, "
            "RoPE) are position/mask-only constants; raw chain == true model "
            "forward (smoke-verified)",
            "streaming": "fp16 doc-chunks (no full-sequence materialization; "
            "full train state would be ~23.6 GB fp16)",
            "greedy": {"max_queue_len": args.greedy_max_len, "allow_repeat": True,
                       "tie_break": "higher val acc, then lower layer index"},
            "deadline": args.deadline,
        },
        "task1": {
            "single": {str(i): ({"val_acc": rec.completed[str(i)][0],
                                 "macro_f1": rec.completed[str(i)][1]}
                                if str(i) in rec.completed else None)
                       for i in range(1, n_layers + 1)},
            "inplace": {str(k): v for k, v in inplace.items()},
        },
        "task2": {"steps": steps, "analysis": greedy_analysis},
        "task3": "task3_pairwise.npz (A/G/S) + task3_pairwise.json",
        "baseline_crosscheck": cc,
        "counts": {"nodes_recorded": len(rec.completed)},
    }
    (d / "results.json").write_text(json.dumps(results, indent=2),
                                    encoding="utf-8")
    rec.close()
    print(f"[finalize] nodes={len(rec.completed)} greedy_steps={len(steps)}")
    if steps:
        print(f"[finalize] greedy: max_acc={greedy_analysis['max_val_acc']:.4f} "
              f"at step {max_step}, final={greedy_analysis['final_val_acc']:.4f}, "
              f"negative steps={neg_steps}")
    print(f"[done] artifacts: {d}")


if __name__ == "__main__":
    main()
