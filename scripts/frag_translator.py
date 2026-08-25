"""Run sig1-recovery experiment: per-path low-rank translators in front of
the canonical classification head.

`user_exp_plans/sig1_recovery-Translator_exp.md` - hypothesis: the uniform-
head collapse of uniform_head_260824_01 is a GEOMETRY mismatch between path
features; a tiny residual low-rank translator per top-10 path,
T_P(h) = h + B_P A_P h (A in R^{r x d}, B in R^{d x r}, bias-free),
should pull h_P back into a form the CANONICAL head can read out. Canonical
head (user-confirmed 2026-08-25) = ridge alpha=1e-6 on canonical path
[1..12] CLS train features of frozen deberta-v3-base (== uniform_head
random_1 head). Grid r in {2, 4, 8, 16} + full-rank reference; top-10
paths by stored own val acc (sig1 ranking) + canonical itself.

Two objectives (user choice 2026-08-25: both):
  ce  - cross-entropy through the FROZEN canonical head with a fixed
        temperature s = std of the canonical head's logits on canonical
        train features (the ridge +/-1 target scale, ~0.1; argmax-
        invariant, pure optimization reparametrisation - the raw CE has
        logits at scale ~100 and Adam diverges / LBFGS line search fails,
        diag-verified 2026-08-25). Full-batch Adam, A=0 init (T starts
        exactly at identity = direct baseline), B ~ N(0, 1/r); train-CE
        plateau early stop; lr selected per r on path [2,3,5] by train CE
        only - val never enters training;
  reg - closed-form paired feature regression, the literal "pull back to
        canonical form": min ||X_c - X_P - X_P W||^2 s.t. rank(W) <= r
        (reduced-rank regression: W_r = W_full V_r V_r^T, V_r = top-r
        right singular vectors of the fitted values X_P W_full; ridge
        alpha per path from {1e-6,1e-4,1e-2,1} by 90/10 train-split
        reconstruction MSE).

Effective-head identity: logits = h^T(W_c + A^T B^T W_c) + b_c - the
translated head is the canonical head plus a rank-<=r correction with the
bias frozen at b_c. Proximity metrics (cos / rel-L2 of T(h_P) vs
same-utterance canonical features) separate "geometry pulled back" from
"rank-r head in disguise".

Usage:
    python scripts/frag_translator.py --smoke
    python scripts/frag_translator.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.fragmented import CLINC_PROMPT, exp_dir, git_state, load_clinc_plus  # noqa: E402
from src.seeding import enable_determinism  # noqa: E402
from frag_modular_probe import (  # noqa: E402
    ALPHA, BATCH, MODEL_PATH, N_CLASSES, ModularStack, build_trie,
    fit_ridge_torch, make_x0, tokenize_data,
)
from frag_uniform_head import dfs_union, key, load_pool  # noqa: E402

CANONICAL = tuple(range(1, 13))
SRC_NAME = "mudularized_layer_probe_260813_01"
RS = [2, 4, 8, 16]
LR = 1e-3
WARMUP = 250
REG_ALPHAS = [1e-6, 1e-4, 1e-2, 1.0]
MAX_STEPS = 5000


# --------------------------------------------------------------------------- #
# Heads / translators (all closed-form pieces in fp64)
# --------------------------------------------------------------------------- #
def _eigh(G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return torch.linalg.eigh(G)
    except RuntimeError:
        lam, V = torch.linalg.eigh(G.cpu())
        return lam.to(G.device), V.to(G.device)


def _svd(F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return torch.linalg.svd(F, full_matrices=False)[1:]
    except RuntimeError:
        U, S, Vh = torch.linalg.svd(F.cpu(), full_matrices=False)
        return S.to(F.device), Vh.to(F.device)


def fit_head(x_tr: torch.Tensor, y_tr: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Ridge ALPHA head (W d x C, b C) on raw features - fit_ridge_torch's
    math factored to also return the weights."""
    n = x_tr.shape[0]
    mu = x_tr.mean(dim=0)
    xc = x_tr - mu
    Y = torch.full((n, N_CLASSES), -1.0, dtype=torch.float64, device=x_tr.device)
    Y[torch.arange(n, device=x_tr.device), y_tr] = 1.0
    Yc = Y - Y.mean(dim=0)
    G = xc.t() @ xc
    h = xc.t() @ Yc
    lam, V = _eigh(G)
    lam = lam.clamp(min=0.0)
    W = V @ ((1.0 / (lam + ALPHA)).unsqueeze(1) * (V.t() @ h))
    b = Y.mean(dim=0) - mu @ W
    return W, b


def train_ce(x_tr: torch.Tensor, y_tr: torch.Tensor, W_c, b_c, r, lr, seed,
             temp: float, max_steps=MAX_STEPS, full_rank=False):
    """Full-batch Adam on temperature-scaled CE through the frozen canonical
    head (fp32): CE((T(h)@W_c + b_c)/temp, y) - argmax-invariant; the raw
    loss has logits at scale ~100 where Adam diverges (diag 2026-08-25).

    T(h) = h + h A^T B^T, A (r x d) zeros, B (d x r) ~ N(0, 1/r): T starts
    exactly at identity. FIXED budget with linear warmup (WARMUP steps -
    Adam's first normalised steps kick T far off identity) + cosine decay
    to 0.1*lr; grad-norm clip 1.0. Returns state + metadata.
    """
    device = x_tr.device
    d = x_tr.shape[1]
    gen = torch.Generator().manual_seed(seed)
    h32 = x_tr.float()
    W32, b32 = W_c.float(), b_c.float()
    if full_rank:
        M = torch.zeros(d, d, device=device, requires_grad=True)
        params = [M]
    else:
        A = torch.zeros(r, d, device=device, requires_grad=True)
        B = (torch.randn(d, r, generator=gen).to(device) / math.sqrt(r)
             ).requires_grad_(True)
        params = [A, B]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda t: min((t + 1) / WARMUP, 1.0)
        * (0.1 + 0.45 * (1.0 + math.cos(math.pi * t / max_steps))))
    best, loss = float("inf"), float("nan")
    t0 = time.time()
    for step in range(1, max_steps + 1):
        if full_rank:
            z = h32 + h32 @ M
        else:
            z = h32 + (h32 @ A.t()) @ B.t()
        loss = F.cross_entropy((z @ W32 + b32) / temp, y_tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        loss = loss.item()
        best = min(best, loss)
    state = {"M": M.detach()} if full_rank else {"A": A.detach(), "B": B.detach()}
    meta = {"steps": step, "final_train_ce": loss, "best_train_ce": best,
            "lr": lr, "secs": round(time.time() - t0, 1)}
    return state, meta


def apply_T(state: dict, x: torch.Tensor, full_rank: bool) -> torch.Tensor:
    """T(h) = h + h M  |  h + (h A^T) B^T, evaluated in fp64."""
    if full_rank:
        return x + x @ state["M"].double()
    return x + (x @ state["A"].t().double()) @ state["B"].t().double()


def delta_head(state: dict, W_c: torch.Tensor, full_rank: bool) -> torch.Tensor:
    """Effective head correction A^T B^T W_c (rank <= r) / M W_c."""
    if full_rank:
        return state["M"].double() @ W_c
    return state["A"].t().double() @ (state["B"].t().double() @ W_c)


def fit_reg_full(x_p: torch.Tensor, x_c: torch.Tensor, alpha: float
                 ) -> torch.Tensor:
    """W = argmin ||X_c - X_P - X_P W||^2 + alpha ||W||^2 (uncentered,
    bias-free - the translator family has no intercept by plan), eigh solve."""
    X, D = x_p, x_c - x_p
    G = X.t() @ X
    lam, V = _eigh(G)
    lam = lam.clamp(min=0.0)
    return V @ ((1.0 / (lam + alpha)).unsqueeze(1) * (V.t() @ (X.t() @ D)))


def reduce_rank(W_full: torch.Tensor, x_p: torch.Tensor, r: int) -> torch.Tensor:
    """Reduced-rank regression: W_r = W_full V_r V_r^T with V_r the top-r
    right singular vectors of the fitted values F = X_P W_full (X_P W_r is
    the best rank-r approx of F; the LS residual is orthogonal to F, so
    this minimises ||D - X_P W||^2 over rank(W) <= r)."""
    if r >= W_full.shape[0]:
        return W_full
    F = x_p @ W_full
    _, Vh = _svd(F)
    Vr = Vh[:r]                              # (r, d) right singular vectors
    return W_full @ (Vr.t() @ Vr)


def select_reg_alpha(x_p: torch.Tensor, x_c: torch.Tensor, seed: int
                     ) -> tuple[float, dict]:
    """Pick alpha by 90/10 train-internal reconstruction MSE (train-only)."""
    n = x_p.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(0.9 * n)
    itr = torch.as_tensor(perm[:cut], device=x_p.device)
    iva = torch.as_tensor(perm[cut:], device=x_p.device)
    scores = {}
    for a in REG_ALPHAS:
        W = fit_reg_full(x_p[itr], x_c[itr], a)
        res = (x_c[iva] - x_p[iva]) - (x_p[iva] @ W)
        scores[a] = float(res.pow(2).mean())
    return min(scores, key=scores.get), scores


def proximity(x_new: torch.Tensor, x_orig: torch.Tensor, x_ref: torch.Tensor
              ) -> dict:
    """Cosine of T(h_P) to same-utterance canonical features + relative L2
    distance (1.0 = untranslated). rel_l2 is None when h_P == h_ref
    (canonical path: 0/0)."""
    cos = float(F.cosine_similarity(x_new, x_ref, dim=1).mean())
    num = (x_new - x_ref).norm(dim=1)
    den = (x_orig - x_ref).norm(dim=1)
    ok = den > 1e-9
    rel = float((num[ok] / den[ok]).mean()) if bool(ok.any()) else None
    return {"cos_to_canonical": cos, "rel_l2_to_canonical": rel}


def score_head(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor,
               y: torch.Tensor) -> float:
    return float(((x @ W + b).argmax(1) == y).double().mean())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--source", default=SRC_NAME)
    args = ap.parse_args()
    if args.name is None:
        args.name = ("canonical_translator_smoke_260825" if args.smoke
                     else "canonical_translator_260825_01")

    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dtype = torch.float16
    src = ROOT / "artifacts" / "fragmented-experiments" / args.source
    d = exp_dir(args.name)
    rs = [2, 4] if args.smoke else RS
    n_top = 3 if args.smoke else 10
    ck = key(CANONICAL)
    print(f"[{args.name}] device={device} smoke={args.smoke} src={src.name}")

    # ---- paths: sig1 top-k + canonical ----------------------------------- #
    pool = load_pool(src)
    top = pool[:n_top]
    paths = [tuple(n["path"]) for n in top] + [CANONICAL]
    own_stored = {key(n["path"]): n["val_acc"] for n in top}
    own_stored[ck] = json.load((src / "inplace.json").open())["12"]["val_acc"]
    print(f"[paths] top-{n_top} + canonical ({len(paths)} total)")

    # ---- data (full: 15000/3000; smoke: seeded subsets as sig1) ----------- #
    data = load_clinc_plus()
    texts_tr, y_np_tr = data["train"]
    texts_va, y_np_va = data["validation"]
    if args.smoke:
        rng0 = np.random.default_rng(args.seed)
        idx_tr = rng0.choice(len(texts_tr), size=2000, replace=False)
        idx_va = rng0.choice(len(texts_va), size=1000, replace=False)
        texts_tr = [texts_tr[i] for i in idx_tr]
        y_np_tr = y_np_tr[idx_tr]
        texts_va = [texts_va[i] for i in idx_va]
        y_np_va = y_np_va[idx_va]
    print(f"[data] train={len(texts_tr)} val={len(texts_va)}")

    tok = AutoTokenizer.from_pretrained(str(ROOT / MODEL_PATH))
    ids_tr, mask_tr, _ = tokenize_data(
        tok, [CLINC_PROMPT.format(utterance=t) for t in texts_tr])
    ids_va, mask_va, _ = tokenize_data(
        tok, [CLINC_PROMPT.format(utterance=t) for t in texts_va])
    model = AutoModel.from_pretrained(str(ROOT / MODEL_PATH),
                                      torch_dtype=torch.float32).to(device).eval()
    stack = ModularStack(model)
    ids_tr, mask_tr = ids_tr.to(device), mask_tr.to(device)
    ids_va, mask_va = ids_va.to(device), mask_va.to(device)
    y_tr = torch.as_tensor(y_np_tr, dtype=torch.long, device=device)
    y_va = torch.as_tensor(y_np_va, dtype=torch.long, device=device)

    # ---- features via union trie (fp16 branch stack, sig1-identical) ------ #
    trie = build_trie([(p, "") for p in paths])
    members = {key(p) for p in paths}
    feats: dict[str, dict[str, torch.Tensor]] = {"train": {}, "val": {}}

    def visit(split):
        def f(path, st):
            feats[split][key(path)] = st[:, 0].double()
        return f

    t0 = time.time()
    for split, ids, mask in (("train", ids_tr, mask_tr), ("val", ids_va, mask_va)):
        print(f"[feat] {split} pass over {len(members)} nodes...")
        x0 = make_x0(model, ids, mask, BATCH)
        dfs_union(stack, trie, x0, mask, members, visit(split), out_dtype)
        del x0
        torch.cuda.empty_cache()
    del model, stack, ids_tr, ids_va
    torch.cuda.empty_cache()
    print(f"[feat] done in {time.time() - t0:.0f}s")

    # ---- heads + baselines ------------------------------------------------ #
    W_c, b_c = fit_head(feats["train"][ck], y_tr)
    with torch.no_grad():
        TEMP = float((feats["train"][ck] @ W_c + b_c).std())
    print(f"[head] canonical head fitted; |W_c|={float(W_c.norm()):.1f} "
          f"CE temperature s={TEMP:.4f}")
    per_path: dict[str, dict] = {}
    for i, p in enumerate(paths):
        k_ = key(p)
        own_acc, own_f1, _ = fit_ridge_torch(feats["train"][k_], y_tr,
                                             feats["val"][k_], y_va)
        direct = score_head(feats["val"][k_], W_c, b_c, y_va)
        per_path[k_] = {
            "path": list(p), "rank": (i if p != CANONICAL else -1),
            "own_stored": own_stored[k_], "own_recomp": own_acc,
            "macro_f1": own_f1, "direct_acc": direct,
            "prox_before": proximity(feats["val"][k_], feats["val"][k_],
                                     feats["val"][ck]),
        }
        print(f"[base] {k_:>16} own={own_acc:.4f} (stored {own_stored[k_]:.4f}) "
              f"direct={direct:.4f} "
              f"cos_before={per_path[k_]['prox_before']['cos_to_canonical']:.4f}")

    crosschecks: dict = {}
    if not args.smoke:
        diffs = [abs(per_path[key(p)]["own_recomp"] - own_stored[key(p)])
                 for p in paths if p != CANONICAL]
        crosschecks["own_vs_stored"] = {
            "n": len(diffs), "max_abs_diff": max(diffs),
            "note": "expect 0.0 - fp16-stack forward is deterministic"}
        sig1_npz = ROOT / "artifacts/fragmented-experiments/uniform_head_260824_01/heads.npz"
        if sig1_npz.exists():
            hz = np.load(sig1_npz)
            crosschecks["canonical_head_vs_sig1_random1"] = {
                "max_abs_dW": float(np.abs(hz["W_random_1"] - W_c.cpu().numpy()).max()),
                "max_abs_db": float(np.abs(hz["b_random_1"] - b_c.cpu().numpy()).max()),
                "val_acc_ours": per_path[ck]["direct_acc"],
                "val_acc_sig1_report": 0.9003,
            }
        print(f"[crosscheck] {json.dumps(crosschecks)}")
        assert max(diffs) < 1e-9, "own-head refit diverged from stored accs"

    # ---- CE translators (fixed lr 1e-3 + warmup/cosine, diag-validated; ---- #
    # an 800-step lr selection ranks the post-init transient, not final
    # quality - 3e-4 underconverges full-rank, 3e-3 is unstable on some paths)
    print("[ce] training translators (full-batch Adam through frozen head)...")
    cfg_i = {"n": 0}
    d_head_norm = float(W_c.norm())
    trans_store: dict = {}
    for p in paths:
        k_ = key(p)
        entry: dict = {}
        for r in rs + [None]:
            full = r is None
            tag = "full" if full else f"r{r}"
            cfg_i["n"] += 1
            state, meta = train_ce(feats["train"][k_], y_tr, W_c, b_c,
                                   r or 0, LR, args.seed + 1000 * cfg_i["n"],
                                   temp=TEMP, full_rank=full)
            t_va = apply_T(state, feats["val"][k_], full)
            t_tr = apply_T(state, feats["train"][k_], full)
            dw = delta_head(state, W_c, full)
            meta.update({
                "val_acc": score_head(t_va, W_c, b_c, y_va),
                "train_acc": score_head(t_tr, W_c, b_c, y_tr),
                "prox_after": proximity(t_va, feats["val"][k_], feats["val"][ck]),
                "params": (768 * 768) if full else 2 * 768 * r,
                "delta_head_fro_over_head": float(dw.norm()) / d_head_norm,
            })
            entry[tag] = meta
            if full:
                trans_store[f"M_ce_{k_}_full"] = state["M"].float().cpu().numpy()
            else:
                trans_store[f"A_ce_{k_}_{tag}"] = state["A"].float().cpu().numpy()
                trans_store[f"B_ce_{k_}_{tag}"] = state["B"].float().cpu().numpy()
            trans_store[f"dWh_ce_{k_}_{tag}"] = dw.float().cpu().numpy()
            print(f"[ce] {k_:>16} {tag:>4} steps={meta['steps']:4d} "
                  f"ce={meta['final_train_ce']:.4f} val={meta['val_acc']:.4f} "
                  f"cos={meta['prox_after']['cos_to_canonical']:.4f} "
                  f"|dW|/|W|={meta['delta_head_fro_over_head']:.3f} "
                  f"({meta['secs']}s)")
        per_path[k_]["ce"] = entry

    # ---- reg translators (closed form) ------------------------------------ #
    print("[reg] closed-form reduced-rank regression translators...")
    for pi, p in enumerate(paths):
        k_ = key(p)
        alpha, alpha_scores = select_reg_alpha(feats["train"][k_],
                                               feats["train"][ck],
                                               args.seed + 100 * pi)
        W_full = fit_reg_full(feats["train"][k_], feats["train"][ck], alpha)
        entry: dict = {"alpha": alpha, "alpha_scores": alpha_scores}
        trans_store[f"Wfull_reg_{k_}"] = W_full.float().cpu().numpy()
        for r in rs + [None]:
            full = r is None
            tag = "full" if full else f"r{r}"
            W = W_full if full else reduce_rank(W_full, feats["train"][k_], r)
            t_va = feats["val"][k_] + feats["val"][k_] @ W
            t_tr = feats["train"][k_] + feats["train"][k_] @ W
            dw = W @ W_c
            entry[tag] = {
                "val_acc": score_head(t_va, W_c, b_c, y_va),
                "train_acc": score_head(t_tr, W_c, b_c, y_tr),
                "prox_after": proximity(t_va, feats["val"][k_], feats["val"][ck]),
                "params": (768 * 768) if full else 2 * 768 * r,
                "rank": r,
                "delta_head_fro_over_head": float(dw.norm()) / d_head_norm,
            }
            trans_store[f"dWh_reg_{k_}_{tag}"] = dw.float().cpu().numpy()
            print(f"[reg] {k_:>16} {tag:>4} a={alpha:g} "
                  f"val={entry[tag]['val_acc']:.4f} "
                  f"cos={entry[tag]['prox_after']['cos_to_canonical']:.4f} "
                  f"|dW|/|W|={entry[tag]['delta_head_fro_over_head']:.3f}")
        per_path[k_]["reg"] = entry

    # ---- smoke checks ------------------------------------------------------ #
    if args.smoke:
        checks: dict = {}
        # fit_head == fit_ridge_torch on the same features (canonical path)
        checks["fit_head_vs_fit_ridge_canonical"] = abs(
            per_path[ck]["direct_acc"] - per_path[ck]["own_recomp"])
        # CE identity is recoverable: canonical path, smallest r
        checks["ce_canonical_r2_val"] = per_path[ck]["ce"]["r2"]["val_acc"]
        checks["ce_canonical_r2_vs_own"] = (per_path[ck]["ce"]["r2"]["val_acc"]
                                            - per_path[ck]["own_recomp"])
        # CE full-rank must move the top-1 path far off the direct baseline
        top1 = key(pool[0]["path"])
        checks["ce_full_top1_gain"] = (per_path[top1]["ce"]["full"]["val_acc"]
                                       - per_path[top1]["direct_acc"])
        # reg closed form vs sklearn: weights on a well-conditioned
        # synthetic problem + train predictions on the real (ill-conditioned,
        # uncentered-Gram) problem, where near-null-space weights legitimately
        # differ between solvers but fitted values must agree
        from sklearn.linear_model import Ridge
        g = torch.Generator().manual_seed(0)
        Xs = torch.randn(500, 64, generator=g, dtype=torch.float64)
        Ds = (Xs @ torch.randn(64, 64, generator=g, dtype=torch.float64)
              + 0.01 * torch.randn(500, 64, generator=g, dtype=torch.float64))
        Xcs = Xs + Ds                       # paired features: D = X_c - X_p
        W_syn = fit_reg_full(Xs, Xcs, 1e-2)
        clf_syn = Ridge(alpha=1e-2, fit_intercept=False,
                        solver="svd").fit(Xs.numpy(), Ds.numpy())
        checks["reg_synthetic_max_dW"] = float(
            np.abs(clf_syn.coef_.T - W_syn.cpu().numpy()).max())  # coef_ is
        # (n_targets, n_features) in sklearn multi-target convention
        top1 = key(pool[0]["path"])
        x_p = feats["train"][top1]
        W_cf = fit_reg_full(x_p, feats["train"][ck], 1e-2)
        clf_real = Ridge(alpha=1e-2, fit_intercept=False,
                         solver="svd").fit(x_p.cpu().numpy(),
                                           (feats["train"][ck] - x_p).cpu().numpy())
        pred_mine = x_p @ W_cf
        pred_sk = torch.as_tensor(clf_real.predict(x_p.cpu().numpy()),
                                  device=pred_mine.device, dtype=pred_mine.dtype)
        checks["reg_real_train_pred_rel_l2"] = float(
            (pred_mine - pred_sk).norm() / pred_mine.norm())
        # reg rank reduction: W_r has exactly r nonzero singular values
        W_r2 = reduce_rank(W_cf, x_p, 2)
        s = torch.linalg.svdvals(W_r2)
        checks["reg_rank2_svd_ratio"] = float(s[2:].max() / s[0])
        # reg on canonical: D == 0 -> W == 0 -> T == identity exactly
        checks["reg_canonical_full_val"] = per_path[ck]["reg"]["full"]["val_acc"]
        checks["reg_canonical_full_vs_direct"] = (per_path[ck]["reg"]["full"]["val_acc"]
                                                  - per_path[ck]["direct_acc"])
        # translator param count matches the plan's formula
        checks["params_r4"] = 2 * 768 * 4
        checks["params_r4_expected"] = 768 * 8      # plan: 4/75 of the head
        print(f"[smoke-checks] {json.dumps(checks, indent=2)}")
        assert checks["fit_head_vs_fit_ridge_canonical"] < 1e-9
        assert checks["ce_canonical_r2_vs_own"] > -0.02, \
            "CE translator damaged the identity map on the canonical path"
        assert checks["ce_full_top1_gain"] > 0.2, \
            "full-rank CE failed to move off the direct baseline"
        assert checks["reg_synthetic_max_dW"] < 1e-8
        assert checks["reg_real_train_pred_rel_l2"] < 1e-6
        assert checks["reg_rank2_svd_ratio"] < 1e-8
        assert checks["reg_canonical_full_vs_direct"] == 0.0
        assert checks["params_r4"] == checks["params_r4_expected"]
        print("[smoke] all checks passed")

    # ---- summary + artifacts ----------------------------------------------- #
    head_params = 768 * N_CLASSES
    summary = {"per_r": {}, "identity_sanity": {
        "canonical_ce_r2_val": per_path[ck]["ce"]["r2"]["val_acc"],
        "canonical_reg_full_val": per_path[ck]["reg"]["full"]["val_acc"],
        "canonical_own": per_path[ck]["own_recomp"],
        "canonical_direct": per_path[ck]["direct_acc"],
    }}
    tops = [key(p) for p in paths if p != CANONICAL]
    for obj in ("ce", "reg"):
        for tag in ([f"r{r}" for r in rs] + ["full"]):
            accs = [per_path[k_][obj][tag]["val_acc"] for k_ in tops]
            rec = [(per_path[k_][obj][tag]["val_acc"] - per_path[k_]["direct_acc"])
                   / (per_path[k_]["own_recomp"] - per_path[k_]["direct_acc"])
                   for k_ in tops]
            cos = [per_path[k_][obj][tag]["prox_after"]["cos_to_canonical"]
                   for k_ in tops]
            summary["per_r"][f"{obj}_{tag}"] = {
                "mean_val_acc": float(np.mean(accs)),
                "mean_recovery": float(np.mean(rec)),
                "mean_cos_to_canonical": float(np.mean(cos)),
                "mean_direct": float(np.mean([per_path[k_]["direct_acc"] for k_ in tops])),
                "mean_own": float(np.mean([per_path[k_]["own_recomp"] for k_ in tops])),
            }

    results = {
        "experiment": args.name,
        "date": "2026-08-25",
        "git": git_state(),
        "config": {
            "plan": "user_exp_plans/sig1_recovery-Translator_exp.md",
            "source": args.source, "model": "deberta-v3-base (frozen)",
            "dataset": "clinc", "n_classes": N_CLASSES,
            "prompt": CLINC_PROMPT,
            "split_sizes": {"train": len(texts_tr), "validation": len(texts_va)},
            "pooling": "cls (tail of path)", "seed": args.seed, "batch": BATCH,
            "stack": "fp16 branch stack (as mudularized_layer_probe_260813_01)",
            "canonical_head": "ridge alpha=1e-6 fp64 on canonical [1..12] CLS "
                              "train features (== uniform_head_260824_01 "
                              "random_1 head)",
            "paths": f"sig1 top-{n_top} by stored own val acc + canonical",
            "translator": "T(h) = h + B A h, A in R^{r x 768}, B in R^{768 x r}, "
                          "bias-free (plan); r grid "
                          + str(rs) + " + full-rank reference",
            "effective_head": "W_eff = W_c + A^T (B^T W_c), rank(W_eff - W_c) "
                              "<= r, bias frozen at b_c",
            "param_counts": {**{f"r{r}": 2 * 768 * r for r in rs},
                             "full": 768 * 768, "head": head_params},
            "ce": {"objective": "cross-entropy through frozen canonical head",
                   "temperature": TEMP,
                   "temperature_def": "s = std of canonical-head logits on "
                                      "canonical train features (ridge +/-1 "
                                      "target scale); CE((T(h)@W_c+b_c)/s, y), "
                                      "argmax-invariant",
                   "optimizer": "Adam full-batch fp32, grad-norm clip 1.0",
                   "init": "A=0 (T starts at identity), B ~ N(0, 1/r)",
                   "lr": LR, "schedule": f"linear warmup {WARMUP} -> cosine "
                                          "decay to 0.1*lr",
                   "max_steps": MAX_STEPS,
                   "budget": "fixed step budget (plateau rule misfires on "
                             "the post-init excursion, diag 2026-08-25); "
                             "val never enters training"},
            "reg": {"objective": "min ||X_c - X_P - X_P W||^2 s.t. "
                                 "rank(W) <= r (reduced-rank regression)",
                    "form": "W_r = W_full V_r V_r^T, V_r = top-r right "
                            "singular vectors of X_P W_full",
                    "alpha_grid": REG_ALPHAS,
                    "alpha_rule": "per path, 90/10 train-split reconstruction MSE"},
        },
        "crosschecks": crosschecks,
        "per_path": per_path,
        "summary": summary,
    }
    (d / "results.json").write_text(json.dumps(results, indent=2),
                                    encoding="utf-8")
    np.savez(d / "translators.npz", **trans_store,
             **{"W_c": W_c.cpu().numpy(), "b_c": b_c.cpu().numpy()})

    print(f"[summary] mean over top-{len(tops)} paths "
          f"(direct={summary['per_r'][f'ce_r{rs[0]}']['mean_direct']:.4f}, "
          f"own={summary['per_r'][f'ce_r{rs[0]}']['mean_own']:.4f}):")
    for obj in ("ce", "reg"):
        for tag in ([f"r{r}" for r in rs] + ["full"]):
            s = summary["per_r"][f"{obj}_{tag}"]
            print(f"[sum] {obj:>3} {tag:>4} acc={s['mean_val_acc']:.4f} "
                  f"recovery={s['mean_recovery']:+.3f} "
                  f"cos={s['mean_cos_to_canonical']:.4f}")
    print(f"[done] artifacts: {d}")


if __name__ == "__main__":
    main()
