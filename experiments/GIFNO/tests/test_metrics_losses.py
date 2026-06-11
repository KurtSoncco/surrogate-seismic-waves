# tests/test_metrics_losses.py
"""Unit tests for GIFNO metrics and composite loss."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

GIFNO_DIR = Path(__file__).resolve().parents[1]
if str(GIFNO_DIR) not in sys.path:
    sys.path.insert(0, str(GIFNO_DIR))

import config  # noqa: E402
from losses import MaskedCompositeLoss, MaskedLpLoss  # noqa: E402
from metrics import (  # noqa: E402
    aggregate_test_metrics,
    per_sample_h1_freq_numpy,
    per_sample_pearson_numpy,
    per_sample_rel_l2_numpy,
)


def _synthetic_batch(n: int = 4, n_freq: int = 100):
    nx = config.NX
    rec_x = config.recorder_x_indices()
    pred = np.zeros((n, nx, n_freq), dtype=np.float32)
    target = np.zeros((n, nx, n_freq), dtype=np.float32)
    mask = np.zeros((n, nx), dtype=np.float32)
    freq = np.logspace(-1, 1, n_freq)

    for i in range(n):
        for x in rec_x:
            mask[i, x] = 1.0
            target[i, x, :] = np.abs(
                1.0 + 0.5 * np.sin(np.linspace(0, 6, n_freq) + i * 0.2)
            )
            pred[i, x, :] = target[i, x, :] * (0.98 + 0.02 * i)

    return pred, target, mask, freq


def test_perfect_prediction_metrics():
    pred, target, mask, _freq = _synthetic_batch(n=3)
    pred = target.copy()
    per = per_sample_rel_l2_numpy(pred, target)
    pear = per_sample_pearson_numpy(pred, target)
    assert np.allclose(per, 0.0, atol=1e-5)
    assert np.allclose(pear, 1.0, atol=1e-5)


def test_scaled_prediction_high_pearson_nonzero_rel_l2():
    pred, target, _mask, _freq = _synthetic_batch(n=2)
    pred = target * 0.001 + 500.0
    pear = per_sample_pearson_numpy(pred, target)
    rel = per_sample_rel_l2_numpy(pred, target)
    assert np.all(pear > 0.99)
    assert np.all(rel > 0.5)


def test_h1_freq_detects_shape_mismatch():
    n, n_freq = 2, 128
    nx = config.NX
    rec_x = config.recorder_x_indices()
    freq = np.logspace(-1, 1, n_freq)
    pred = np.zeros((n, nx, n_freq), dtype=np.float32)
    target = np.zeros((n, nx, n_freq), dtype=np.float32)

    for i in range(n):
        for x in rec_x:
            t = 1.0 + 0.3 * np.sin(2 * np.pi * np.arange(n_freq) / 8.0)
            target[i, x, :] = t
            pred[i, x, :] = t + 0.05 * np.sin(2 * np.pi * np.arange(n_freq) / 7.0)

    h1 = per_sample_h1_freq_numpy(pred, target, freq)
    rel = per_sample_rel_l2_numpy(pred, target)
    assert np.all(h1 > rel)


def test_aggregate_test_metrics_keys():
    pred, target, mask, freq = _synthetic_batch()
    summary, per_sample = aggregate_test_metrics(pred, target, mask, freq)
    assert "test_rel_l2_p10" in summary
    assert "test_linf_max" in summary
    assert "test_pearson_p10" in summary
    assert "test_h1_freq_mean" in summary
    assert set(per_sample.keys()) == {"rel_l2", "linf", "pearson", "h1_freq"}


def test_masked_lp_loss_forward():
    pred, target, mask, _ = _synthetic_batch()
    t_pred = torch.from_numpy(pred)
    t_target = torch.from_numpy(target)
    t_mask = torch.from_numpy(mask)
    loss = MaskedLpLoss()(t_pred, t_target, t_mask)
    assert loss.ndim == 0
    assert float(loss) >= 0.0


def test_composite_loss_forward_and_hard_mining():
    pred, target, mask, freq = _synthetic_batch()
    t_pred = torch.from_numpy(pred)
    t_target = torch.from_numpy(target)
    t_mask = torch.from_numpy(mask)

    crit = MaskedCompositeLoss(
        rel_weight=1.0,
        h1_weight=0.1,
        freq_weight=0.05,
        hard_mining=False,
        freq=freq,
    )
    loss = crit(t_pred, t_target, t_mask)
    assert float(loss) >= 0.0

    crit_mine = MaskedCompositeLoss(
        rel_weight=1.0,
        h1_weight=0.0,
        freq_weight=0.0,
        hard_mining=True,
        hard_mining_power=1.0,
        freq=freq,
    )
    loss_m = crit_mine(t_pred, t_target, t_mask)
    assert float(loss_m) >= 0.0
