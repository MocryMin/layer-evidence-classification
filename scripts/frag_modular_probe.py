"""Run gr2 fragmented experiment: modularized layer probe (mudularized_layer_probe).

`plans/fragmented_exp_gr2.md` — DeBERTa-v3-base on CLINC150, ridge classifier
alpha=1e-6 (EXP-003 config). Layer modules are composed into arbitrary
repeatable sequences ("paths"); CLS readout at the tail.

Tasks:
  1. single-layer probes [i] vs in-raw-place acc (true model forward);
  2. greedy layer queue construction up to 50 layers (negative-gain tracking);
  3. pairwise add-layer gain matrices A/G/S (12x12);
  4. n random 3..12-step paths (uniform len, uniform layers, repeats allowed),
     generated first, then processed through a shared prefix trie (branch-stack
     cache reuse; the set is fixed, only computation order changes).

Ridge: closed form ``(Xc^T Xc + alpha I) W = Xc^T Yc`` solved in fp64 via
eigendecomposition — mathematically equal to sklearn
``RidgeClassifier(alpha, fit_intercept=True, solver='svd')``; equivalence
verified in smoke mode. Train fit, val acc (gr2 plan; no test access).

Modular semantics: each application runs the layer module on its input
states. DeBERTa-v3-base has no encoder conv and rel_pos is position-only
(constant across applications), so layer composition equals the raw pipeline
and the raw chain equals a true model forward (verified in smoke). The
branch stack stores fp16 states (rounding between applications); smoke mode
compares fp16-stack vs exact fp32-from-scratch accs.

Usage:
    python scripts/frag_modular_probe.py --smoke
    python scripts/frag_modular_probe.py --time-probe
    python scripts/frag_modular_probe.py --n-paths 4000
    python scripts/frag_modular_probe.py --n-paths 4000 --resume
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fragmented import CLINC_PROMPT, exp_dir, git_state, load_clinc_plus  # noqa: E402
from src.metrics import confusion_matrix, macro_f1  # noqa: E402
from src.seeding import enable_determinism  # noqa: E402

N_CLASSES = 150
N_LAYERS = 12
ALPHA = 1e-6
BATCH = 512
MODEL_PATH = "models/deberta-v3-base"


class Deadline(Exception):
    pass


def parse_deadline(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M"))


# --------------------------------------------------------------------------- #
# Modular layer stack
# --------------------------------------------------------------------------- #
class ModularStack:
    """Arbitrary layer-sequence forward on frozen DeBERTa-v3 (fp32 compute).

    ``apply`` runs layer ``li`` (0-based) on input states. ``rel_pos`` and
    ``rel_embeddings`` are position-only constants, so arbitrary composition
    is well-defined and matches the raw encoder (no conv in deberta-v3-base).
    """

    def __init__(self, model: torch.nn.Module):
        self.enc = model.encoder
        self.layers = self.enc.layer
        self.rel_emb = self.enc.get_rel_embedding()

    @torch.no_grad()
    def apply(self, h: torch.Tensor, mask_b: torch.Tensor, li: int) -> torch.Tensor:
        """One application of layer ``li``; returns fp32 states [B, L, D]."""
        h = h.float()
        am = self.enc.get_attention_mask(mask_b)          # [B, L] -> [B, 1, L, L]
        rel_pos = self.enc.get_rel_pos(h)                 # position-only, constant
        out, _ = self.layers[li](h, am, relative_pos=rel_pos,
                                 rel_embeddings=self.rel_emb)
        return out


@torch.no_grad()
def forward_path(stack: ModularStack, x0: torch.Tensor, mask: torch.Tensor,
                 path: list[int], batch: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Forward ``path`` (0-based layers) from ``x0``; tail states in ``out_dtype``.

    ``x0`` may be fp32 embeddings (exact) or fp16 cached states (rounded input).
    """
    n, L, D = x0.shape
    out = torch.empty((n, L, D), dtype=out_dtype, device=x0.device)
    for s in range(0, n, batch):
        h = x0[s:s + batch]
        m = mask[s:s + batch]
        for li in path:
            h = stack.apply(h, m, li)
        out[s:s + batch] = h.to(out_dtype)
    return out


# --------------------------------------------------------------------------- #
# Ridge (closed form, sklearn-equivalent)
# --------------------------------------------------------------------------- #
def fit_ridge_torch(x_tr: torch.Tensor, y_tr: torch.Tensor,
                    x_va: torch.Tensor, y_va: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    """Closed-form ridge, alpha=ALPHA, fp64 — equal to sklearn
    ``RidgeClassifier(alpha, fit_intercept=True, solver='svd')``: target
    encoding is {+1,-1} (sklearn label binarizer neg_label=-1), X and Y
    centered, svd-style weights s/(s^2+a). Returns (val_acc, macro_f1, pred).
    """
    n = x_tr.shape[0]
    mu = x_tr.mean(dim=0)
    xc = x_tr - mu
    Y = torch.full((n, N_CLASSES), -1.0, dtype=torch.float64, device=x_tr.device)
    Y[torch.arange(n, device=x_tr.device), y_tr] = 1.0
    Yc = Y - Y.mean(dim=0)
    G = xc.t() @ xc
    h = xc.t() @ Yc
    try:
        lam, V = torch.linalg.eigh(G)
    except RuntimeError:
        lam, V = torch.linalg.eigh(G.cpu())
        lam, V = lam.to(x_tr.device), V.to(x_tr.device)
    # eigh-space ridge: W = V diag(1/(lam+a)) V^T h. Null-space directions are
    # safe: V_null^T h = 0 (h lives in Xc^T's range), so 1/a amplification
    # multiplies ~1e-16 noise. Equivalent to sklearn solver='svd' (s/(s^2+a)
    # in svd space with U^T y) via s^2 = lam.
    lam = lam.clamp(min=0.0)
    W = V @ ((1.0 / (lam + ALPHA)).unsqueeze(1) * (V.t() @ h))
    b = Y.mean(dim=0) - mu @ W
    # decision on UNcentered val features (b already absorbs the centering;
    # centering here again would subtract mu@W twice)
    dec = x_va @ W + b
    pred = dec.argmax(dim=1)
    acc = float((pred == y_va).double().mean())
    f1 = macro_f1(confusion_matrix(y_va.cpu().numpy(), pred.cpu().numpy(), N_CLASSES))
    return acc, f1, pred


# --------------------------------------------------------------------------- #
# Data / embeddings
# --------------------------------------------------------------------------- #
@torch.no_grad()
def make_x0(model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor,
            batch: int) -> torch.Tensor:
    # mask passed like DebertaV2Model.forward does (pad positions zeroed;
    # CLS unaffected but keeps the modular chain bit-equal to the raw model)
    out = torch.empty((ids.shape[0], ids.shape[1], model.config.hidden_size),
                      dtype=torch.float32, device=ids.device)
    for s in range(0, ids.shape[0], batch):
        out[s:s + batch] = model.embeddings(ids[s:s + batch], mask=mask[s:s + batch])
    return out


def tokenize_data(tok, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Tokenize with the EXP-003 prompt; pad to the observed max length
    (no truncation; CLINC texts are short). Returns (ids, mask, max_len)."""
    lens = [len(tok(t)["input_ids"]) for t in texts]
    max_len = max(lens)
    enc = tok(texts, padding="max_length", max_length=max_len, truncation=True,
              return_tensors="pt")
    mask = (enc["input_ids"] != tok.pad_token_id).long()
    return enc["input_ids"], mask, max_len


# --------------------------------------------------------------------------- #
# In-place reference (true raw model forward)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_raw_cls(model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor,
                    batch: int) -> list[torch.Tensor]:
    """Per-layer CLS of a true raw forward, fp64, [L] x [N, D]."""
    out = [torch.empty((ids.shape[0], model.config.hidden_size),
                       dtype=torch.float64, device=ids.device) for _ in range(N_LAYERS)]
    for s in range(0, ids.shape[0], batch):
        o = model(ids[s:s + batch], attention_mask=mask[s:s + batch],
                  output_hidden_states=True)
        for li, h in enumerate(o.hidden_states[1:]):
            out[li][s:s + batch] = h[:, 0].double()
    return out


def run_inplace(model, ids_tr, mask_tr, ids_va, mask_va, y_tr, y_va, batch):
    """True raw forward per layer + ridge alpha=1e-6. Returns {layer: {...}}."""
    cls_tr = collect_raw_cls(model, ids_tr, mask_tr, batch)
    cls_va = collect_raw_cls(model, ids_va, mask_va, batch)
    per_layer = {}
    for li in range(N_LAYERS):
        acc, f1, _ = fit_ridge_torch(cls_tr[li], y_tr, cls_va[li], y_va)
        per_layer[li + 1] = {"val_acc": acc, "macro_f1": f1}
    return per_layer


# --------------------------------------------------------------------------- #
# Trie / DFS
# --------------------------------------------------------------------------- #
class TrieNode:
    __slots__ = ("path", "tasks", "children")

    def __init__(self, path: tuple, tasks: list[str]):
        self.path = path
        self.tasks = tasks
        self.children: dict[int, TrieNode] = {}


def build_trie(path_sets: list[tuple[tuple[int, ...], str]]) -> TrieNode:
    root = TrieNode((), [])
    for path, task in path_sets:
        node = root
        for li in path:
            if li not in node.children:
                node.children[li] = TrieNode(node.path + (li,), [])
            node = node.children[li]
        node.tasks.append(task)
    return root


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
                key = ",".join(map(str, r["path"]))
                self.completed[key] = (r["val_acc"], r["macro_f1"])
            if self.preds_path.exists():
                arr = np.load(self.preds_path)
                self.preds = [arr[i] for i in range(arr.shape[0])]
        self.f = self.jsonl.open("a", encoding="utf-8")

    def record(self, path: tuple, tasks: list[str], acc: float, f1: float,
               pred: torch.Tensor) -> None:
        rec = {"path": list(path), "len": len(path), "tail_layer": path[-1],
               "tasks": tasks, "val_acc": acc, "macro_f1": f1, "ts": time.time()}
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.completed[",".join(map(str, path))] = (acc, f1)
        self.preds.append(pred.cpu().numpy().astype(np.int16))

    def flush_preds(self) -> None:
        if self.preds:
            np.save(self.preds_path, np.stack(self.preds))

    def close(self) -> None:
        self.f.close()


def run_dfs(root: TrieNode, stack: ModularStack, x0_tr: torch.Tensor,
            x0_va: torch.Tensor, mask_tr: torch.Tensor, mask_va: torch.Tensor,
            y_tr: torch.Tensor, y_va: torch.Tensor, rec: NodeRecorder,
            deadline_ts: float, out_dtype: torch.dtype, print_every: int = 200) -> None:
    """Prefix-sharing DFS: branch stack holds fp16 tail states per depth."""

    def rec_fn(node: TrieNode, st_tr: torch.Tensor, st_va: torch.Tensor) -> None:
        if node.path:
            key = ",".join(map(str, node.path))
            if key not in rec.completed:
                if time.time() > deadline_ts:
                    raise Deadline()
                acc, f1, pred = fit_ridge_torch(
                    st_tr[:, 0].double(), y_tr, st_va[:, 0].double(), y_va)
                rec.record(node.path, node.tasks, acc, f1, pred)
                if len(rec.preds) % print_every == 0:
                    print(f"[dfs] {len(rec.preds):6d} nodes · tail {key} · "
                          f"acc={acc:.4f} · t={time.time():.0f}")
                    rec.flush_preds()
        for li in sorted(node.children):
            child = node.children[li]
            h_tr = forward_path(stack, st_tr, mask_tr, [li - 1], BATCH, out_dtype)
            h_va = forward_path(stack, st_va, mask_va, [li - 1], BATCH, out_dtype)
            rec_fn(child, h_tr, h_va)

    rec_fn(root, x0_tr, x0_va)


# --------------------------------------------------------------------------- #
# Greedy (task 2)
# --------------------------------------------------------------------------- #
def run_greedy(stack: ModularStack, x0_tr: torch.Tensor, x0_va: torch.Tensor,
               mask_tr: torch.Tensor, mask_va: torch.Tensor, y_tr: torch.Tensor,
               y_va: torch.Tensor, rec: NodeRecorder, steps_path: Path,
               deadline_ts: float, max_len: int, out_dtype: torch.dtype) -> list[dict]:
    """Greedy queue growth from the best single layer; allow repeats."""
    steps: list[dict] = []
    if steps_path.exists():
        steps = [json.loads(l) for l in steps_path.read_text(encoding="utf-8").splitlines()]
    f = steps_path.open("a", encoding="utf-8")

    if steps:
        queue = list(steps[-1]["queue"])
        prev_acc = steps[-1]["val_acc"]
    else:
        singles = {i: rec.completed[str(i)][0] for i in range(1, N_LAYERS + 1)}
        best = max(singles, key=lambda i: (singles[i], -i))
        queue, prev_acc = [best], singles[best]
    # parent states = tail of the FULL current queue path (resume-safe)
    parent_tr = forward_path(stack, x0_tr, mask_tr, [li - 1 for li in queue],
                             BATCH, out_dtype)
    parent_va = forward_path(stack, x0_va, mask_va, [li - 1 for li in queue],
                             BATCH, out_dtype)

    for step in range(len(steps) + 1, max_len):
        if time.time() > deadline_ts:
            raise Deadline()
        cand_accs: dict[int, float] = {}
        for li in range(1, N_LAYERS + 1):
            key = ",".join(map(str, queue + [li]))
            if key in rec.completed:
                cand_accs[li] = rec.completed[key][0]
                continue
            h_tr = forward_path(stack, parent_tr, mask_tr, [li - 1], BATCH, out_dtype)
            h_va = forward_path(stack, parent_va, mask_va, [li - 1], BATCH, out_dtype)
            acc, f1, pred = fit_ridge_torch(
                h_tr[:, 0].double(), y_tr, h_va[:, 0].double(), y_va)
            rec.record(tuple(queue + [li]), ["greedy_cand"], acc, f1, pred)
            cand_accs[li] = acc
        chosen = max(cand_accs, key=lambda li: (cand_accs[li], -li))
        queue.append(chosen)
        parent_tr = forward_path(stack, parent_tr, mask_tr, [chosen - 1], BATCH, out_dtype)
        parent_va = forward_path(stack, parent_va, mask_va, [chosen - 1], BATCH, out_dtype)
        step_rec = {"step": step, "queue": list(queue), "chosen": chosen,
                    "val_acc": cand_accs[chosen],
                    "gain": cand_accs[chosen] - prev_acc,
                    "candidates": cand_accs, "ts": time.time()}
        f.write(json.dumps(step_rec) + "\n")
        f.flush()
        steps.append(step_rec)
        prev_acc = cand_accs[chosen]
        print(f"[greedy] step {step:2d}: +L{chosen:2d} -> acc={prev_acc:.4f} "
              f"(gain={step_rec['gain']:+.4f}) queue={queue}")
    f.close()
    return steps


# --------------------------------------------------------------------------- #
# Smoke checks
# --------------------------------------------------------------------------- #
def run_smoke_checks(stack: ModularStack, model: torch.nn.Module, ids_tr, ids_va,
                     x0_tr, x0_va, mask_tr, mask_va, y_tr, y_va,
                     rec: NodeRecorder, five_paths: list[tuple], d: Path) -> dict:
    from sklearn.linear_model import RidgeClassifier

    checks: dict = {}
    with torch.no_grad():
        o = model(ids_tr[:64], attention_mask=mask_tr[:64],
                  output_hidden_states=True)
    checks["embeddings_max_abs_diff"] = float((o.hidden_states[0] - x0_tr[:64]).abs().max())
    h_f32 = forward_path(stack, x0_tr[:64], mask_tr[:64], list(range(N_LAYERS)),
                         BATCH, torch.float32)
    checks["chain_scratch_max_abs_diff"] = float((h_f32 - o.hidden_states[-1]).abs().max())
    h_f16 = forward_path(stack, x0_tr[:64], mask_tr[:64], list(range(N_LAYERS)),
                         BATCH, torch.float16)
    checks["chain_stack_max_abs_diff"] = float((h_f16.float() - o.hidden_states[-1]).abs().max())

    cls_tr = collect_raw_cls(model, ids_tr, mask_tr, BATCH)
    cls_va = collect_raw_cls(model, ids_va, mask_va, BATCH)
    # sklearn vs torch on raw layers 1/6/12 (smoke subset)
    for li in (1, 6, 12):
        xt = cls_tr[li - 1].cpu().numpy()
        xv = cls_va[li - 1].cpu().numpy()
        acc_t = fit_ridge_torch(cls_tr[li - 1], y_tr, cls_va[li - 1], y_va)[0]
        clf = RidgeClassifier(alpha=ALPHA, fit_intercept=True, solver="svd")
        clf.fit(xt, y_tr.cpu().numpy())
        acc_s = float((clf.predict(xv) == y_va.cpu().numpy()).mean())
        checks[f"ridge_raw_L{li}_torch"] = acc_t
        checks[f"ridge_raw_L{li}_sklearn"] = acc_s
    # stack (fp16) vs scratch (fp32 exact) accs on 5 paths
    for path in five_paths:
        key = ",".join(map(str, path))
        acc_stack = rec.completed.get(key, (None, None))[0]
        h_tr = forward_path(stack, x0_tr, mask_tr, [li - 1 for li in path],
                            BATCH, torch.float32)
        h_va = forward_path(stack, x0_va, mask_va, [li - 1 for li in path],
                            BATCH, torch.float32)
        acc_scr, _, _ = fit_ridge_torch(h_tr[:, 0].double(), y_tr,
                                        h_va[:, 0].double(), y_va)
        checks[f"path_{key}_stack"] = acc_stack
        checks[f"path_{key}_scratch"] = acc_scr
    (d / "smoke_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mudularized_layer_probe_260813_01")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--time-probe", action="store_true")
    ap.add_argument("--n-paths", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mode", choices=["stack", "scratch"], default="stack")
    ap.add_argument("--deadline", default="2026-08-14 07:30")
    ap.add_argument("--greedy-max-len", type=int, default=50)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dtype = torch.float16 if args.mode == "stack" else torch.float32
    d = exp_dir(args.name)
    deadline_ts = parse_deadline(args.deadline)
    print(f"[{args.name}] device={device} mode={args.mode} deadline={args.deadline}")

    # ---- data ------------------------------------------------------------ #
    data = load_clinc_plus()
    texts_tr, y_np_tr = data["train"]
    texts_va, y_np_va = data["validation"]
    print(f"[data] train={len(texts_tr)} val={len(texts_va)}")
    if args.smoke:
        # random subset (not a prefix: CLINC parquet is class-grouped, a
        # prefix would miss most classes and break the sklearn comparison)
        rng0 = np.random.default_rng(args.seed)
        idx_tr = rng0.choice(len(texts_tr), size=2000, replace=False)
        idx_va = rng0.choice(len(texts_va), size=1000, replace=False)
        texts_tr = [texts_tr[i] for i in idx_tr]
        y_np_tr = y_np_tr[idx_tr]
        texts_va = [texts_va[i] for i in idx_va]
        y_np_va = y_np_va[idx_va]

    tok = AutoTokenizer.from_pretrained(str(ROOT / MODEL_PATH))
    ids_tr, mask_tr, max_len_tr = tokenize_data(
        tok, [CLINC_PROMPT.format(utterance=t) for t in texts_tr])
    ids_va, mask_va, max_len_va = tokenize_data(
        tok, [CLINC_PROMPT.format(utterance=t) for t in texts_va])
    max_len = max(max_len_tr, max_len_va)
    print(f"[tok] max_len={max_len} (no truncation)")

    # checkpoint weights are fp16; force fp32 so direct layer-module calls
    # (which bypass the model's autocast wrapper) match prior fp32 extractions
    model = AutoModel.from_pretrained(str(ROOT / MODEL_PATH),
                                      torch_dtype=torch.float32).to(device).eval()
    stack = ModularStack(model)

    ids_tr, mask_tr = ids_tr.to(device), mask_tr.to(device)
    ids_va, mask_va = ids_va.to(device), mask_va.to(device)
    y_tr = torch.as_tensor(y_np_tr, dtype=torch.long, device=device)
    y_va = torch.as_tensor(y_np_va, dtype=torch.long, device=device)

    print("[x0] embeddings...")
    x0_tr = make_x0(model, ids_tr, mask_tr, BATCH)
    x0_va = make_x0(model, ids_va, mask_va, BATCH)

    # ---- time probe (calibration, full data only) ------------------------ #
    if args.time_probe:
        assert not args.smoke, "--time-probe runs on full data"
        for rep in range(3):
            t0 = time.time()
            h_tr = forward_path(stack, x0_tr, mask_tr, [0], BATCH, out_dtype)
            h_va = forward_path(stack, x0_va, mask_va, [0], BATCH, out_dtype)
            acc, _, _ = fit_ridge_torch(h_tr[:, 0].double(), y_tr,
                                        h_va[:, 0].double(), y_va)
            dt = time.time() - t0
            print(f"[time-probe] rep{rep}: one-node (1 edge + fit) = {dt:.2f}s "
                  f"(acc={acc:.4f})")
        return

    # ---- path sets ------------------------------------------------------- #
    singles = [( (i,), "single") for i in range(1, N_LAYERS + 1)]
    pairs = [( (i, j), "pair") for i in range(1, N_LAYERS + 1) for j in range(1, N_LAYERS + 1)]
    rng = np.random.default_rng(args.seed)
    randoms: list[tuple] = []
    for _ in range(args.n_paths if not args.smoke else 20):
        L = int(rng.integers(3, 13))
        randoms.append(tuple(int(x) for x in rng.integers(1, N_LAYERS + 1, size=L)))
    (d / "random_paths.json").write_text(json.dumps([list(p) for p in randoms]), encoding="utf-8")
    print(f"[paths] singles={len(singles)} pairs={len(pairs)} randoms={len(randoms)}")

    path_sets = singles + pairs + [(p, "random") for p in randoms]
    root = build_trie(path_sets)
    rec = NodeRecorder(d, args.resume)
    print(f"[resume] completed nodes loaded: {len(rec.completed)}")

    # ---- in-place reference (task 1 baseline) ---------------------------- #
    inplace_path = d / "inplace.json"
    if args.smoke or not inplace_path.exists():
        print("[inplace] true raw forward per layer...")
        t0 = time.time()
        inplace = run_inplace(model, ids_tr, mask_tr, ids_va, mask_va, y_tr, y_va, BATCH)
        if not args.smoke:
            inplace_path.write_text(json.dumps(inplace, indent=2), encoding="utf-8")
        print(f"[inplace] done in {time.time() - t0:.1f}s")
    else:
        inplace = {int(k): v for k, v in json.loads(inplace_path.read_text()).items()}

    # ---- DFS over trie (tasks 1/3/4) ------------------------------------- #
    print("[dfs] trie traversal...")
    t0 = time.time()
    try:
        run_dfs(root, stack, x0_tr, x0_va, mask_tr, mask_va, y_tr, y_va, rec,
                deadline_ts if not args.smoke else float("inf"), out_dtype)
    except Deadline:
        print("[stop] deadline reached during DFS — finalizing partial results")
    else:
        print(f"[dfs] done in {time.time() - t0:.1f}s ({len(rec.completed)} nodes)")

    # ---- smoke checks ---------------------------------------------------- #
    if args.smoke:
        checks = run_smoke_checks(stack, model, ids_tr, ids_va, x0_tr, x0_va,
                                  mask_tr, mask_va, y_tr, y_va, rec, randoms[:5], d)
        print(json.dumps(checks, indent=2))
        rec.close()
        return

    # ---- greedy (task 2) ------------------------------------------------- #
    print("[greedy] ...")
    t0 = time.time()
    try:
        steps = run_greedy(stack, x0_tr, x0_va, mask_tr, mask_va, y_tr, y_va, rec,
                           d / "greedy_steps.jsonl", deadline_ts,
                           args.greedy_max_len, out_dtype)
    except Deadline:
        steps_path = d / "greedy_steps.jsonl"
        steps = ([json.loads(l) for l in steps_path.read_text(encoding="utf-8").splitlines()]
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
        "note": "gain_k = acc(queue_k) - acc(queue_{k-1}); max_acc_after_negative_step "
                "= the best acc occurs after at least one negative-gain step",
    }

    # task 3 matrices (NaN where a node did not complete)
    A = np.full((N_LAYERS, N_LAYERS), np.nan)
    for i in range(1, N_LAYERS + 1):
        for j in range(1, N_LAYERS + 1):
            key = f"{i},{j}"
            if key in rec.completed:
                A[i - 1, j - 1] = rec.completed[key][0]
    a_diag = np.array([rec.completed.get(str(i), (np.nan, np.nan))[0]
                       for i in range(1, N_LAYERS + 1)])
    Gmat = A - a_diag[:, None]
    Smat = A - np.maximum(a_diag[:, None], a_diag[None, :])
    np.savez(d / "task3_pairwise.npz", A=A, G=Gmat, S=Smat)
    (d / "task3_pairwise.json").write_text(json.dumps({
        "A": A.tolist(), "G": Gmat.tolist(), "S": Smat.tolist(),
        "definition": "A_ij = val acc of path [i,j]; G_ij = A_ij - A_i "
                      "(single-layer acc of i); S_ij = A_ij - max(A_i, A_j)",
    }, indent=2), encoding="utf-8")

    # task 1 / task 4 summaries
    n_random_done = sum(1 for p in randoms
                        if ",".join(map(str, p)) in rec.completed)
    results = {
        "experiment": args.name,
        "date": "2026-08-13",
        "reporting_model": "deepseek-v4-flash",
        "git": git_state(),
        "config": {
            "plan": "plans/fragmented_exp_gr2.md",
            "model": "deberta-v3-base (frozen)", "dataset": "clinc",
            "prompt": CLINC_PROMPT, "n_classes": N_CLASSES,
            "split_sizes": {"train": len(texts_tr), "validation": len(texts_va)},
            "max_length": max_len, "truncation": "none (pad to observed max)",
            "pooling": "cls", "seed": args.seed, "batch": BATCH,
            "ridge": {"alpha": ALPHA, "method": "closed-form eigen-solve fp64, "
                      "sklearn-equivalent (RidgeClassifier solver='svd', fit_intercept=True)",
                      "fit": "train", "eval": "validation"},
            "modular_semantics": "layer module applied to its input states; rel_pos "
                      "position-only constant; no conv in deberta-v3-base; raw chain "
                      "== true model forward (smoke-verified)",
            "stack": {"mode": args.mode, "dtype": "float16" if args.mode == "stack" else "float32",
                      "note": "fp16 rounding between applications (stack mode) or exact "
                              "fp32 (scratch mode); equivalence smoke-checked"},
            "greedy": {"max_queue_len": args.greedy_max_len, "allow_repeat": True,
                       "tie_break": "higher val acc, then lower layer index"},
            "random_paths": {"n_generated": len(randoms), "len_range": [3, 12],
                             "layer_range": [1, N_LAYERS], "rng": "numpy default_rng(17)"},
            "deadline": args.deadline,
        },
        "task1": {
            "single": {str(i): ({"val_acc": rec.completed[str(i)][0],
                                 "macro_f1": rec.completed[str(i)][1]}
                                if str(i) in rec.completed else None)
                       for i in range(1, N_LAYERS + 1)},
            "inplace": {str(k): v for k, v in inplace.items()},
        },
        "task2": {"steps": steps, "analysis": greedy_analysis},
        "task3": "task3_pairwise.npz (A/G/S) + task3_pairwise.json",
        "task4": {"n_generated": len(randoms), "n_processed": n_random_done,
                  "unprocessed": [p for p in randoms
                                  if ",".join(map(str, p)) not in rec.completed]},
        "counts": {"nodes_recorded": len(rec.completed)},
    }
    (d / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rec.close()
    print(f"[finalize] nodes={len(rec.completed)} random_done={n_random_done}/"
          f"{len(randoms)} greedy_steps={len(steps)}")
    print(f"[finalize] greedy: max_acc={greedy_analysis['max_val_acc']:.4f} at step "
          f"{max_step}, final={greedy_analysis['final_val_acc']:.4f}, "
          f"negative steps={neg_steps}")
    print(f"[done] artifacts: {d}")


if __name__ == "__main__":
    main()
