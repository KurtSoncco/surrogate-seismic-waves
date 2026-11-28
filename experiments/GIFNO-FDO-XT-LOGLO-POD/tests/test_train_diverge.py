"""Abort training when a batch or epoch loss is NaN or explodes."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import config
from train import (
    DivergenceError,
    _zero_init_pod_branch_last,
    check_loss_finite,
    masked_log_pearson,
    window_consistency_loss,
)


def test_finite_loss_passes():
    check_loss_finite(1.2, epoch=0, where="train")


def test_nan_loss_raises():
    with pytest.raises(DivergenceError, match="non-finite"):
        check_loss_finite(float("nan"), epoch=3, where="train batch")


def test_inf_loss_raises():
    with pytest.raises(DivergenceError, match="non-finite"):
        check_loss_finite(float("inf"), epoch=1, where="val")


def test_exploded_loss_raises(monkeypatch):
    monkeypatch.setattr(config, "DIVERGE_LOSS_MAX", 100.0)
    with pytest.raises(DivergenceError, match=r"8e\+06"):
        check_loss_finite(8e6, epoch=0, where="train")


def test_bound_zero_disables_magnitude_check(monkeypatch):
    monkeypatch.setattr(config, "DIVERGE_LOSS_MAX", 0.0)
    check_loss_finite(8e6, epoch=0, where="train")


def test_window_consistency_stays_finite_for_huge_outputs():
    rec = torch.arange(4)
    huge = torch.full((4, 32, 5), 1e30)
    huge[1] = 1e-30
    loss = window_consistency_loss(huge, rec, pair_offset=1, dense=True)
    assert torch.isfinite(loss)
    assert float(loss) <= (math.log(1e6) - math.log(1e-6)) ** 2 + 1e-6


def test_consistency_zero_when_masks_empty():
    rec = torch.arange(4)
    out = torch.full((4, 32, 5), 1e6)
    masks = torch.zeros(4, 32)
    loss = window_consistency_loss(out, rec, pair_offset=1, dense=True, masks=masks)
    assert float(loss) == 0.0


def test_consistency_ignores_unlabeled_spike():
    rec = torch.arange(4)
    out = torch.ones(4, 32, 5)
    out[:, 0, :] = 1e6
    masks = torch.ones(4, 32)
    masks[:, 0] = 0
    loss = window_consistency_loss(out, rec, pair_offset=1, dense=True, masks=masks)
    assert float(loss) < 1e-6


def test_zero_init_pod_branch_last():
    class _Head:
        def __init__(self):
            self.branch = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 3))

    class _Model:
        def __init__(self):
            self.head = _Head()

    m = _Model()
    torch.nn.init.ones_(m.head.branch[-1].weight)
    _zero_init_pod_branch_last(m)
    assert torch.count_nonzero(m.head.branch[-1].weight) == 0
    assert torch.count_nonzero(m.head.branch[-1].bias) == 0


def test_h1_stays_finite_for_flat_residual():
    from losses import MaskedCompositeLoss

    freq = np.logspace(-1, 1, 32)
    crit = MaskedCompositeLoss(
        rel_weight=1.0,
        h1_weight=0.25,
        radial_weight=0.0,
        p=1,
        log_tf_loss=False,
        freq=freq,
    )
    b, nx, f = 2, 16, 32
    pred = torch.full((b, nx, f), 5e-4)
    target = torch.ones(b, nx, f)
    mask = torch.zeros(b, nx)
    mask[:, :8] = 1.0
    target[:, 8:] = 0.0
    loss = crit(pred, target, mask)
    assert torch.isfinite(loss)
    assert float(loss) < 100.0


def test_masked_log_pearson_perfect_match():
    pred = torch.ones(2, 4, 8)
    pred[..., :3] = 2.0
    freq = torch.linspace(0.1, 4.0, 8)
    mask = torch.ones(2, 4)
    r = masked_log_pearson(pred, pred, mask, freq, (0.1, 2.5))
    assert torch.allclose(r, torch.ones(2), atol=1e-5)
