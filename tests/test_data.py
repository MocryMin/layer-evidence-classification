"""Tests for src/data.py: OOS drop, label remap, split sizes, tokenisation.

Uses stdlib ``unittest`` only (no pytest dependency in the shared env).
Run: ``python -m unittest tests.test_data -v``
"""
from __future__ import annotations

import collections
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import (  # noqa: E402
    build_label_maps,
    load_split,
    make_prompt,
    save_label_maps,
    tokenise_split,
)

cfg = load_config()
DS_ROOT = cfg.dataset_abs_path
CONFIG = cfg.dataset_config
OOS = cfg.drop_oos_label


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.label2id, cls.id2label, cls.in_scope_ids = build_label_maps(DS_ROOT, CONFIG, OOS)

    def test_label_maps(self):
        assert len(self.in_scope_ids) == 150
        assert OOS not in self.in_scope_ids
        assert self.in_scope_ids == sorted(self.in_scope_ids)
        assert set(self.label2id.values()) == set(range(150))
        assert set(self.id2label.keys()) == {str(i) for i in range(150)}
        assert "oos" not in self.label2id
        for name, idx in self.label2id.items():
            assert self.id2label[str(idx)] == name
        assert self.in_scope_ids[0] == 0
        assert self.in_scope_ids[42] == 43  # slot 42 holds original id 43 (oos=42 skipped)

    def test_split_sizes_and_label_ranges(self):
        train = load_split(DS_ROOT, CONFIG, "train", OOS, self.in_scope_ids)
        val = load_split(DS_ROOT, CONFIG, "validation", OOS, self.in_scope_ids)
        test = load_split(DS_ROOT, CONFIG, "test", OOS, self.in_scope_ids)
        assert len(train) == 15000
        assert len(val) == 3000
        assert len(test) == 4500
        for ds in (train, val, test):
            labs = ds["label"]
            assert min(labs) == 0 and max(labs) == 149
            assert len(set(labs)) == 150
        assert all(v == 100 for v in collections.Counter(train["label"]).values())
        assert set(collections.Counter(val["label"]).values()) == {20}
        assert set(collections.Counter(test["label"]).values()) == {30}
        assert train["sample_id"] == list(range(15000))
        assert test["sample_id"] == list(range(4500))

    def test_prompt_and_tokenisation_cls_position(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(cfg.model_abs_path))
        test = load_split(DS_ROOT, CONFIG, "test", OOS, self.in_scope_ids)
        out = tokenise_split(test.select(range(8)), tok, cfg.prompt, cfg.max_length, "left")
        assert make_prompt(test[0]["text"], cfg.prompt) == f"Classify the intent: {test[0]['text']}"
        cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
        for ids in out["input_ids"]:
            assert ids[0] == cls_id, "CLS must be at index 0 for CLS pooling"
            assert ids[-1] == sep_id
        assert all(len(ids) <= cfg.max_length for ids in out["input_ids"])
        for am in out["attention_mask"]:
            assert all(a == 1 for a in am)
        assert out["labels"].shape == (8,)
        assert out["sample_ids"].tolist() == list(range(8))

    def test_left_truncation_preserves_cls_and_sep(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(cfg.model_abs_path))
        long_text = " ".join(["banking"] * 600)
        tok.truncation_side = "left"
        enc = tok(long_text, truncation=True, max_length=cfg.max_length, add_special_tokens=True)
        ids = enc["input_ids"]
        assert len(ids) == cfg.max_length
        assert ids[0] == tok.cls_token_id, "left truncation must keep CLS at index 0"
        assert ids[-1] == tok.sep_token_id, "left truncation must keep SEP at the end"

    def test_save_label_maps(self):
        with tempfile.TemporaryDirectory() as td:
            save_label_maps(self.label2id, self.id2label, td)
            with open(Path(td) / "label2id.json") as f:
                assert json.load(f) == self.label2id
            with open(Path(td) / "id2label.json") as f:
                assert json.load(f) == self.id2label


if __name__ == "__main__":
    unittest.main(verbosity=2)
