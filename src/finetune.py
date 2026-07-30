"""Full fine-tune of DeBERTa-v3-base for CLINC150 intent classification (EXP-002, task 3c).

Optimises the last-layer CLS classification accuracy. The resulting backbone is
saved and later re-probed to see whether fine-tuning "uses up" the mid-layer
recoverability (last layer expected to exceed ~90%, leaving few errors).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import build_label_maps, load_split, save_label_maps, tokenise_split
from .seeding import seed_all


class FTClassifier(nn.Module):
    """Backbone + linear head on the final-layer CLS token."""

    def __init__(self, backbone, in_dim: int, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(in_dim, n_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)


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
def _eval_split(model, tok_out, labels, pad_id, device, batch_size=64):
    model.eval()
    preds, losses = [], []
    n = len(tok_out["input_ids"])
    for s in range(0, n, batch_size):
        ids = tok_out["input_ids"][s : s + batch_size]
        y = labels[s : s + batch_size].to(device)
        ii, am = _collate(ids, pad_id)
        logits = model(ii.to(device), am.to(device))
        losses.append(F.cross_entropy(logits, y, reduction="sum").item())
        preds.append(logits.argmax(1).cpu())
    pred = torch.cat(preds)
    acc = (pred == labels).float().mean().item()
    nll = sum(losses) / n
    return acc, nll


def finetune_backbone(cfg, device) -> dict:
    """Full fine-tune; save backbone+head+tokenizer; return history + final accs."""
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    ft = cfg.finetune
    seed_all(cfg.seed)
    save_path = Path(ft["save_backbone"])
    save_path.mkdir(parents=True, exist_ok=True)

    label2id, id2label, in_scope_ids = build_label_maps(
        cfg.dataset_abs_path, cfg.dataset_config, cfg.drop_oos_label)
    save_label_maps(label2id, id2label, save_path)

    tok = AutoTokenizer.from_pretrained(str(cfg.model_abs_path))
    backbone = AutoModel.from_pretrained(str(cfg.model_abs_path), dtype=torch.float32)
    model = FTClassifier(backbone, cfg.head_input_dim, cfg.n_classes).to(device)

    splits = {}
    for split in ["train", "validation", "test"]:
        ds = load_split(cfg.dataset_abs_path, cfg.dataset_config, split, cfg.drop_oos_label, in_scope_ids)
        splits[split] = tokenise_split(ds, tok, cfg.prompt_with_instruction, cfg.max_length, cfg.truncation)

    pad_id = tok.pad_token_id
    train_ids = splits["train"]["input_ids"]
    train_y = splits["train"]["labels"]
    n_train = len(train_ids)

    n_steps = (n_train // ft["batch_size"] + 1) * ft["epochs"]
    warmup = int(n_steps * ft["warmup_ratio"])
    opt = torch.optim.AdamW(model.parameters(), lr=ft["lr"], weight_decay=ft["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, warmup, n_steps)

    history = []
    for epoch in range(1, ft["epochs"] + 1):
        model.train()
        perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(cfg.seed + epoch)).tolist()
        total_loss = 0.0
        for s in range(0, n_train, ft["batch_size"]):
            idx = perm[s : s + ft["batch_size"]]
            ids = [train_ids[i] for i in idx]
            y = train_y[idx].to(device)
            ii, am = _collate(ids, pad_id)
            logits = model(ii.to(device), am.to(device))
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item() * len(idx)
        val_acc, val_nll = _eval_split(model, splits["validation"], splits["validation"]["labels"], pad_id, device)
        test_acc, test_nll = _eval_split(model, splits["test"], splits["test"]["labels"], pad_id, device)
        history.append({"epoch": epoch, "train_loss": total_loss / n_train,
                        "val_acc": val_acc, "val_nll": val_nll, "test_acc": test_acc, "test_nll": test_nll})
        print(f"  epoch {epoch}: train_loss={total_loss/n_train:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")

    # save backbone + tokenizer + head
    backbone.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    torch.save(model.head.state_dict(), save_path / "ft_head.pt")
    with open(save_path / "ft_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {"save_path": str(save_path), "history": history,
            "final_val_acc": history[-1]["val_acc"], "final_test_acc": history[-1]["test_acc"]}
