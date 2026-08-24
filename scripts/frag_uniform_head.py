"""Run sig1 fragmented experiment: uniform head over modularized paths.

`user_exp_plans/fragmented_exp_sig1.md` - one shared ridge head fitted on the
POOLED train features of k paths with EQUAL weights (the plan's original
acc-weighting was dropped 2026-08-24: path acc is a posteriori val information
and must not enter training). Grid: k in {1,10,50,100,200,1000} x strategy
  top_k     - best k paths by stored own-head val acc (selection leaky by
              design; that leak is what unleaky_k controls for),
  random_k  - seeded nested shuffle of the pool; k=1 -> canonical [1..12],
  unleaky_k - seeded nested shuffle of ranks 101.. (top-100 excluded).
Everything else as in mudularized_layer_probe_260813_01: CLINC150, CLS tail
readout, ridge alpha=1e-6 closed-form fp64 (sklearn-equivalent, fit_intercept
on the pooled mean), fp16 branch stack, seed 17, batch 512, train fit / val
eval, no test access.

Two passes over the union prefix-trie:
  A (train): per-node sufficient stats X^T X, sum(X), X^T Y -> per-config
     pooled stats (stacked-data ridge via G = S - N mu mu^T, H = C - N mu ybar^T);
     per-node OWN head refit from the same states (same-forward reference and
     crosscheck against the stored own val acc).
  B (val):   every union node scored under all config heads + its own head;
     gap = uniform_acc - own_acc(recomputed).

Smoke mode additionally verifies the pooled closed form against sklearn
RidgeClassifier fitted on the literally stacked features of one config.

Usage:
    python scripts/frag_uniform_head.py --smoke
    python scripts/frag_uniform_head.py
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
sys.path.insert(0, str(ROOT / "scripts"))

from src.fragmented import CLINC_PROMPT, exp_dir, git_state, load_clinc_plus  # noqa: E402
from src.seeding import enable_determinism  # noqa: E402
from frag_modular_probe import (  # noqa: E402
    ALPHA, BATCH, MODEL_PATH, N_CLASSES, ModularStack, build_trie,
    forward_path, make_x0, tokenize_data,
)

CANONICAL = tuple(range(1, 13))
KS = [1, 10, 50, 100, 200, 1000]
SRC_NAME = "mudularized_layer_probe_260813_01"


def key(path) -> str:
    return ",".join(map(str, path))


def load_pool(src: Path) -> list[dict]:
    """Unique random-tagged nodes ranked by stored val acc (desc, path asc)."""
    nodes = [json.loads(l) for l in (src / "nodes.jsonl").open(encoding="utf-8")]
    pool = [n for n in nodes if "random" in n["tasks"]]
    return sorted(pool, key=lambda n: (-n["val_acc"], tuple(n["path"])))


def batched_stats(x: torch.Tensor, y_1hot: torch.Tensor, batch: int
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp64 sufficient stats (X^T X, sum X, X^T Y) accumulated in batches."""
    d = x.shape[1]
    S = torch.zeros((d, d), dtype=torch.float64, device=x.device)
    s = torch.zeros(d, dtype=torch.float64, device=x.device)
    C = torch.zeros((d, y_1hot.shape[1]), dtype=torch.float64, device=x.device)
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch].double()
        S += xb.t() @ xb
        s += xb.sum(dim=0)
        C += xb.t() @ y_1hot[i:i + batch]
    return S, s, C


def ridge_from_stats(S: torch.Tensor, s: torch.Tensor, C: torch.Tensor, n: int,
                     ybar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge from sufficient stats: pooled mu centering,
    G = S - n mu mu^T, H = C - n mu ybar^T, W = V diag(1/(lam+a)) V^T H,
    b = ybar - mu^T W (== sklearn on the stacked data)."""
    mu = s / n
    G = S - n * torch.outer(mu, mu)
    H = C - n * torch.outer(mu, ybar)
    try:
        lam, V = torch.linalg.eigh(G)
    except RuntimeError:
        lam, V = torch.linalg.eigh(G.cpu())
        lam, V = lam.to(G.device), V.to(G.device)
    lam = lam.clamp(min=0.0)
    W = V @ ((1.0 / (lam + ALPHA)).unsqueeze(1) * (V.t() @ H))
    b = ybar - mu @ W
    return W, b


@torch.no_grad()
def dfs_union(stack: ModularStack, root, x0: torch.Tensor, mask: torch.Tensor,
              members: set[str], visit, out_dtype):
    """Prefix-sharing DFS over the union trie; ``visit`` called at members."""

    def rec_fn(node, st: torch.Tensor) -> None:
        if node.path and key(node.path) in members:
            visit(tuple(node.path), st)
        for li in sorted(node.children):
            child = node.children[li]
            h = forward_path(stack, st, mask, [li - 1], BATCH, out_dtype)
            rec_fn(child, h)

    rec_fn(root, x0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="uniform_head_260824_01")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--source", default=SRC_NAME)
    args = ap.parse_args()

    enable_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dtype = torch.float16
    src = ROOT / "artifacts" / "fragmented-experiments" / args.source
    d = exp_dir(args.name)
    print(f"[{args.name}] device={device} smoke={args.smoke} src={src.name}")

    # ---- source pool + configs ------------------------------------------- #
    pool = load_pool(src)
    ks = [1, 3, 10] if args.smoke else KS
    excl = 10 if args.smoke else 100        # unleaky exclusion depth
    if args.smoke:
        pool = pool[:60]
    ranked_paths = [tuple(n["path"]) for n in pool]
    rng_r = np.random.default_rng(args.seed)
    perm_r = rng_r.permutation(len(ranked_paths))
    rng_u = np.random.default_rng(args.seed + 1)          # distinct stream
    perm_u = rng_u.permutation(len(ranked_paths) - excl)   # ranks excl+1.. only
    configs: dict[str, list[tuple]] = {}
    for k in ks:
        configs[f"top_{k}"] = ranked_paths[:k]
        configs[f"random_{k}"] = ([CANONICAL] if k == 1
                                  else [ranked_paths[i] for i in perm_r[:k]])
        configs[f"unleaky_{k}"] = [ranked_paths[excl + i] for i in perm_u[:k]]
    own_stored = {key(n["path"]): n["val_acc"] for n in pool}
    own_stored[key(CANONICAL)] = json.load((src / "inplace.json").open())["12"]["val_acc"]
    rank_of = {key(p): i for i, p in enumerate(ranked_paths)}
    union = set().union(*configs.values())
    members = {key(p) for p in union}
    cfg_of = {key(p): [c for c, ps in configs.items() if p in ps] for p in union}
    print(f"[pool] unique_random={len(pool)} configs={len(configs)} "
          f"union={len(union)} (incl canonical)")

    # ---- data ------------------------------------------------------------ #
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
    n_tr, n_va = len(texts_tr), len(texts_va)

    Y = torch.full((n_tr, N_CLASSES), -1.0, dtype=torch.float64, device=device)
    Y[torch.arange(n_tr, device=device), y_tr] = 1.0
    ybar = Y.mean(dim=0)

    print("[x0] embeddings...")
    x0_tr = make_x0(model, ids_tr, mask_tr, BATCH)
    x0_va = make_x0(model, ids_va, mask_va, BATCH)

    trie = build_trie([(p, "") for p in union])
    D = model.config.hidden_size

    # ---- pass A: train sufficient stats ---------------------------------- #
    t0 = time.time()
    cfg_stats = {c: {"S": torch.zeros((D, D), dtype=torch.float64, device=device),
                     "s": torch.zeros(D, dtype=torch.float64, device=device),
                     "C": torch.zeros((D, N_CLASSES), dtype=torch.float64,
                                      device=device)}
                 for c in configs}
    own_heads: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    feat_cache: dict[str, torch.Tensor] = {} if args.smoke else None
    done_a = {"n": 0}

    def visit_train(path: tuple, st: torch.Tensor) -> None:
        x = st[:, 0].double()
        S, s, C = batched_stats(x, Y, BATCH)
        W, b = ridge_from_stats(S, s, C, n_tr, ybar)     # own head, same states
        own_heads[key(path)] = (W.cpu(), b.cpu())
        if feat_cache is not None:
            feat_cache[key(path)] = x.cpu()              # smoke: stacked check
        for c in cfg_of[key(path)]:
            stt = cfg_stats[c]
            stt["S"] += S
            stt["s"] += s
            stt["C"] += C
        done_a["n"] += 1
        if done_a["n"] % 100 == 0:
            print(f"[A] {done_a['n']:5d}/{len(union)} nodes · "
                  f"{time.time() - t0:6.0f}s · tail {key(path)}")

    print(f"[A] train pass over {len(union)} member nodes...")
    dfs_union(stack, trie, x0_tr, mask_tr, members, visit_train, out_dtype)
    del x0_tr
    torch.cuda.empty_cache()
    print(f"[A] done in {time.time() - t0:.0f}s")

    # ---- fit uniform heads ----------------------------------------------- #
    heads: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for c, ps in configs.items():
        stt = cfg_stats[c]
        heads[c] = ridge_from_stats(stt["S"], stt["s"], stt["C"],
                                    n_tr * len(ps), ybar)
    np.savez(d / "heads.npz",
             **{f"W_{c}": W.cpu().numpy() for c, (W, _) in heads.items()},
             **{f"b_{c}": b.cpu().numpy() for c, (_, b) in heads.items()})
    print(f"[heads] fitted {len(heads)} uniform heads")

    if args.smoke:
        # pooled closed form vs sklearn on literally stacked features
        from sklearn.linear_model import RidgeClassifier
        cfg = configs["top_3"]
        Xs = torch.cat([feat_cache[key(p)] for p in cfg]).numpy()
        ys = np.tile(y_np_tr, len(cfg))
        clf = RidgeClassifier(alpha=ALPHA, fit_intercept=True, solver="svd")
        clf.fit(Xs, ys)
        W, b = heads["top_3"]
        x_va_all = None                        # sklearn acc on stacked val feats
        # rebuild stacked val features via a third mini-pass
        va_cache: dict[str, np.ndarray] = {}
        dfs_union(stack, trie, x0_va, mask_va,
                  {key(p) for p in cfg},
                  lambda p, st: va_cache.__setitem__(
                      key(p), st[:, 0].double().cpu().numpy()), out_dtype)
        Xv = np.concatenate([va_cache[key(p)] for p in cfg])
        yv = np.tile(y_np_va, len(cfg))
        acc_sk = float((clf.predict(Xv) == yv).mean())
        acc_cf = float(((torch.as_tensor(Xv) @ W.cpu() + b.cpu().numpy()).argmax(1)
                        .numpy() == yv).mean())
        dW = float(np.abs(clf.coef_.T - W.cpu().numpy()).max())
        print(f"[smoke] top_3 stacked: sklearn acc={acc_sk:.4f} "
              f"closed-form acc={acc_cf:.4f} max|dW|={dW:.2e}")
        # acc exact, weights within fp solve-path noise (eigh vs svd ~1e-5)
        assert acc_sk == acc_cf, "pooled closed form != sklearn on stacked data"
        assert dW < 1e-4, "weight deviation beyond solve-path noise"

    # ---- pass B: val evaluation ------------------------------------------ #
    t1 = time.time()
    rows: list[dict] = []
    f_out = (d / "per_node.jsonl").open("w", encoding="utf-8")

    def visit_val(path: tuple, st: torch.Tensor) -> None:
        x = st[:, 0].double()
        k_ = key(path)
        Wo, bo = own_heads[k_]
        Wo, bo = Wo.to(device), bo.to(device)
        accs = {"own": float(((x @ Wo + bo).argmax(1) == y_va).double().mean())}
        for c, (W, b) in heads.items():
            accs[c] = float(((x @ W + b).argmax(1) == y_va).double().mean())
        rec = {"path": list(path), "len": len(path), "rank": rank_of.get(k_, -1),
               "own_stored": own_stored.get(k_), "own_recomp": accs["own"],
               "uniform": accs, "ts": time.time()}
        f_out.write(json.dumps(rec) + "\n")
        rows.append(rec)
        if len(rows) % 200 == 0:
            f_out.flush()
            print(f"[B] {len(rows):5d}/{len(union)} · "
                  f"{time.time() - t1:6.0f}s · tail {k_}")

    print(f"[B] val pass over {len(union)} member nodes...")
    dfs_union(stack, trie, x0_va, mask_va, members, visit_val, out_dtype)
    f_out.close()
    print(f"[B] done in {time.time() - t1:.0f}s")

    # ---- summary ---------------------------------------------------------- #
    cross = [abs(r["own_recomp"] - r["own_stored"]) for r in rows
             if r["rank"] >= 0]     # canonical excluded (different source)
    canon = next(r for r in rows if r["rank"] == -1)
    summary = {
        "experiment": args.name,
        "date": "2026-08-24",
        "git": git_state(),
        "config": {
            "plan": "user_exp_plans/fragmented_exp_sig1.md",
            "amendment": "uniform (equal) per-path weights; plan's acc-weighting "
                         "dropped 2026-08-24 (path acc is a posteriori val info)",
            "source": args.source, "model": "deberta-v3-base (frozen)",
            "dataset": "clinc", "prompt": CLINC_PROMPT, "n_classes": N_CLASSES,
            "split_sizes": {"train": n_tr, "validation": n_va},
            "pooling": "cls (tail of path)", "seed": args.seed, "batch": BATCH,
            "ridge": {"alpha": ALPHA, "weights": "uniform per path",
                      "method": "closed-form eigen-solve fp64 on pooled "
                                "sufficient stats, sklearn-equivalent "
                                "(RidgeClassifier solver='svd', fit_intercept "
                                "on pooled mean)"},
            "stack": "fp16 branch stack (as mudularized_layer_probe_260813_01)",
            "k_grid": ks, "strategies": ["top", "random (k=1 canonical)",
                                         f"unleaky (ranks {excl + 1}+)"],
            "sampling": "nested seeded shuffles default_rng(17) / "
                        "default_rng(18); ranking ties broken by path",
            "gap_def": "uniform_acc - own_recomp (same-forward own head); "
                       "own_stored crosscheck reported separately",
        },
        "crosscheck_own": {"n": len(cross),
                           "mean_abs_diff": float(np.mean(cross)) if cross else None,
                           "max_abs_diff": float(np.max(cross)) if cross else None},
        "crosscheck_canonical": {"path": list(CANONICAL),
                                 "own_recomp": canon["own_recomp"],
                                 "inplace12_stored": own_stored[key(CANONICAL)]},
        "per_config": {},
    }
    cfg_members = {c: {key(p) for p in ps} for c, ps in configs.items()}
    for c in configs:
        mem = [r for r in rows if key(r["path"]) in cfg_members[c]]
        gaps = np.array([r["uniform"][c] - r["own_recomp"] for r in mem])
        summary["per_config"][c] = {
            "k": len(cfg_members[c]),
            "member_mean_own_recomp": float(np.mean([r["own_recomp"] for r in mem])),
            "member_mean_uniform": float(np.mean([r["uniform"][c] for r in mem])),
            "member_gap_mean": float(gaps.mean()), "member_gap_std": float(gaps.std()),
        }
    # top-10 / top-100 reference sets evaluated under every head
    for tag, n_ref in (("top10", min(10, len(ranked_paths))),
                       ("top100", min(100, len(ranked_paths)))):
        ref = [r for r in rows if 0 <= r["rank"] < n_ref]
        summary[tag] = {
            "n": len(ref),
            "per_config": {c: {
                "gap_mean": float(np.mean([r["uniform"][c] - r["own_recomp"]
                                           for r in ref])),
                "gap_std": float(np.std([r["uniform"][c] - r["own_recomp"]
                                         for r in ref])),
                "uniform_mean": float(np.mean([r["uniform"][c] for r in ref])),
            } for c in configs},
            "per_path": [{"rank": r["rank"], "path": r["path"],
                          "own_recomp": r["own_recomp"],
                          "own_stored": r["own_stored"],
                          "uniform": r["uniform"]} for r in
                         sorted(ref, key=lambda r: r["rank"])[:10]],
        }
    (d / "results.json").write_text(json.dumps(summary, indent=2),
                                    encoding="utf-8")
    print(f"[crosscheck] own recomp vs stored: mean|d|="
          f"{summary['crosscheck_own']['mean_abs_diff']:.4f} max|d|="
          f"{summary['crosscheck_own']['max_abs_diff']:.4f} (n={len(cross)})")
    print(f"[crosscheck] canonical own_recomp={canon['own_recomp']:.4f} vs "
          f"inplace12={own_stored[key(CANONICAL)]:.4f}")
    for c in configs:
        s = summary["per_config"][c]
        print(f"[cfg {c:>11}] k={s['k']:4d} own={s['member_mean_own_recomp']:.4f} "
              f"uni={s['member_mean_uniform']:.4f} gap={s['member_gap_mean']:+.4f}"
              f"±{s['member_gap_std']:.4f}")
    print(f"[done] artifacts: {d}")


if __name__ == "__main__":
    main()
