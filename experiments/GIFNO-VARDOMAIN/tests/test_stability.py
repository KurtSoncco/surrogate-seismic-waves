"""Hyperparameter grid, padded columns, and seed-to-seed ranking."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screen_hifreq import linear_broadcast  # noqa: E402
from stability_ae import (  # noqa: E402
    DEPTH_METRICS,
    FREQ_METRICS,
    _equal_weight,
    depth_configs,
    depth_holdout,
    fit_depth,
    frequency_configs,
    summarize,
)


def test_grids_keep_a_single_copy_of_the_reference():
    freq = frequency_configs()
    keys = {(row["n_loc"], row["knn"], row["block"], row["lr"]) for row in freq}
    assert len(keys) == len(freq) == 13
    assert (3, 5, 4, 1e-3) in keys
    assert (2, 5, 4, 1e-3) in keys
    assert (3, 9, 8, 1e-3) in keys
    assert (3, 5, 4, 3e-4) in keys
    depth = depth_configs()
    assert len(depth) == 15
    assert {"code_dim": 32, "lr": 1e-3} in depth
    assert {"code_dim": 2, "lr": 3e-3} in depth


def test_padded_column_matches_the_short_column_and_the_linear_smear():
    torch.manual_seed(0)
    from model import ColumnDepthAutoencoder

    model = ColumnDepthAutoencoder(2, code_dim=8)
    short = torch.randn(4, 2, 7, 1)
    mask = torch.ones(4, 7, 1)
    depth = (torch.arange(7, dtype=torch.float32) / 150).view(1, 1, 7, 1).expand(4, 1, 7, 1)
    surface, recon = model(short, mask, depth)
    padded = torch.zeros(4, 2, 12, 1)
    padded[:, :, :7] = short
    mask_p = torch.zeros(4, 12, 1)
    mask_p[:, :7] = 1
    depth_p = (torch.arange(12, dtype=torch.float32) / 150).view(1, 1, 12, 1).expand(4, 1, 12, 1)
    surface_p, recon_p = model(padded, mask_p, depth_p)
    assert torch.allclose(surface, surface_p)
    assert torch.allclose(recon, recon_p[:, :, :7])
    smear = linear_broadcast(short, mask)
    assert torch.allclose(_equal_weight(short, mask), smear)


def test_holdout_ignores_padded_rows_and_a_short_fit_stays_finite():
    torch.manual_seed(0)
    from model import ColumnDepthAutoencoder

    values = torch.randn(6, 2, 10)
    n_active = torch.tensor([10, 10, 8, 8, 5, 5])
    model = ColumnDepthAutoencoder(2, code_dim=4)
    report = depth_holdout(model, values, n_active)
    assert set(DEPTH_METRICS).issubset(report)
    assert all(value == value for value in report["row_mae"])
    row = fit_depth(values[:4], n_active[:4], values[4:], n_active[4:], {"code_dim": 4, "lr": 1e-3}, 0)
    assert row["failed"] is False
    assert 1 <= row["best_epoch"] <= 40
    assert row["masked_l1"] >= 0.0


def test_seed_spread_marks_an_overlapping_winner():
    def row(seed, mae, high):
        base = {metric: mae for metric in FREQ_METRICS}
        base.update({
            "n_loc": 3, "knn": 5, "block": 4, "lr": 1e-3, "seed": seed, "failed": False,
            "best_epoch": 10, "mae": mae, "band_5-10": high, "gap_high": high,
        })
        return base

    def other(seed, mae):
        base = {metric: mae for metric in FREQ_METRICS}
        base.update({
            "n_loc": 2, "knn": 5, "block": 4, "lr": 1e-3, "seed": seed, "failed": False,
            "best_epoch": 12, "mae": mae,
        })
        return base

    runs = [row(0, 0.20, 0.05), row(1, 0.28, 0.07), other(0, 0.25), other(1, 0.27)]
    summary = summarize(runs, ("n_loc", "knn", "block", "lr"), FREQ_METRICS)
    winner = summary["winners"]["mae"]
    assert winner["factors"]["n_loc"] == 3
    assert winner["separated"] is False
    ranks = {item["factors"]["n_loc"]: item["mean_rank"] for item in summary["mean_rank"]["mae"]}
    assert ranks[3] == 1.5
    assert ranks[2] == 1.5
