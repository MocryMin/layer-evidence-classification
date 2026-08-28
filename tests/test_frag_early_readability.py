import numpy as np
import torch

from scripts.frag_llama_early_readability import wilson_interval
from src.frag_early_readability import (
    masked_numpy_accuracy,
    path_specs,
    select_smoke_value,
    variance_stats,
)


def test_path_specs_cover_two_families_and_56_paths():
    specs = path_specs(28)
    assert len(specs) == 56
    assert len({item["path_id"] for item in specs}) == 56
    assert specs[0]["path"] == [1]
    assert specs[27]["path"] == [28]
    assert specs[28]["path"] == [1]
    assert specs[-1]["path"] == list(range(1, 29))


def test_variance_stats_flags_only_exp002_scale_collapse():
    collapsed = variance_stats(torch.zeros(20, 7))
    varied = variance_stats(torch.arange(140, dtype=torch.float32).reshape(20, 7))
    assert collapsed["exp002_collapsed"] is True
    assert varied["exp002_collapsed"] is False
    assert varied["inter_sample_std_mean"] > 0


def test_masked_numpy_accuracy_excludes_absent_choice_columns():
    decision = np.array([[0.0, 1.0, 100.0], [2.0, 1.0, 0.0]])
    labels = np.array([1, 0])
    choice_counts = np.array([2, 3])
    assert masked_numpy_accuracy(decision, labels, choice_counts) == 1.0


def test_smoke_tie_breaks_toward_lower_value():
    rows = [
        {"learning_rate": 0.01, "best_accuracy": 0.5, "best_epoch": 80},
        {"learning_rate": 0.001, "best_accuracy": 0.5, "best_epoch": 90},
    ]
    selected, summaries = select_smoke_value(rows, "learning_rate")
    assert selected == 0.001
    assert summaries["0.001"]["median_best_epoch"] == 90


def test_smoke_treats_float32_ulps_as_ties():
    rows = [
        {"learning_rate": 0.1, "best_accuracy": 0.50000002, "best_epoch": 50},
        {"learning_rate": 0.03, "best_accuracy": 0.5, "best_epoch": 50},
    ]
    selected, _ = select_smoke_value(rows, "learning_rate")
    assert selected == 0.03


def test_wilson_interval_contains_observed_fraction():
    lower, upper = wilson_interval(137, 501)
    assert lower < 137 / 501 < upper
