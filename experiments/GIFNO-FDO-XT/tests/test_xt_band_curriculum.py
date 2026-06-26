"""Tests for the frequency-band loss curriculum (GIFNO-FDO-XT)."""

from __future__ import annotations

import numpy as np
import torch

from losses import MaskedCompositeLoss, band_curriculum_weights


def _make_loss(n_freq: int = 128) -> MaskedCompositeLoss:
    freq = np.logspace(-1, 1, n_freq).astype(np.float32)
    return MaskedCompositeLoss(rel_weight=1.0, h1_weight=0.0, p=1, freq=freq)


def test_schedule_phase_ordering_and_bounds():
    floor = 0.25
    num_epochs = 100
    kw = dict(floor=floor, mid_start=0.33, high_start=0.66, ramp=0.15)

    # Early: low full, mid/high at the floor.
    w_low, w_mid, w_high = band_curriculum_weights(0, num_epochs, **kw)
    assert w_low == 1.0
    assert np.isclose(w_mid, floor)
    assert np.isclose(w_high, floor)

    # Final epoch: every band has ramped to full weight.
    w_low, w_mid, w_high = band_curriculum_weights(num_epochs - 1, num_epochs, **kw)
    assert np.isclose(w_high, 1.0)
    assert np.isclose(w_mid, 1.0)

    # Mid leads high through training, and both are monotonic non-decreasing.
    mids, highs = [], []
    for e in range(num_epochs):
        _, m, h = band_curriculum_weights(e, num_epochs, **kw)
        assert floor - 1e-6 <= m <= 1.0 + 1e-6
        assert floor - 1e-6 <= h <= 1.0 + 1e-6
        assert m >= h - 1e-6  # mid_start < high_start
        mids.append(m)
        highs.append(h)
    assert all(b >= a - 1e-6 for a, b in zip(mids, mids[1:]))
    assert all(b >= a - 1e-6 for a, b in zip(highs, highs[1:]))


def test_neutral_weights_reproduce_unweighted_loss():
    loss = _make_loss()
    torch.manual_seed(0)
    pred = torch.randn(3, 5, 128)
    target = torch.randn(3, 5, 128)

    base = loss._relative_lp_on_recorder(pred, target)

    # Uniform per-band weights normalize to all-ones -> identical loss.
    loss.set_band_weights(2.0, 2.0, 2.0)
    uniform = loss._relative_lp_on_recorder(pred, target)
    assert torch.allclose(base, uniform, atol=1e-6)

    # Clearing weights restores the exact neutral objective.
    loss.set_band_weights(None)
    assert loss._band_w is None
    cleared = loss._relative_lp_on_recorder(pred, target)
    assert torch.allclose(base, cleared, atol=1e-7)


def test_band_weights_are_mean_one_normalized():
    loss = _make_loss()
    loss.set_band_weights(1.0, 0.25, 0.25)
    assert loss._band_w is not None
    assert np.isclose(float(loss._band_w.mean()), 1.0, atol=1e-5)


def test_downweighting_high_band_reduces_high_band_error_loss():
    loss = _make_loss()
    freq = np.logspace(-1, 1, 128)
    high_idx = np.where(freq >= 2.0)[0]

    target = torch.ones(2, 5, 128)
    pred = target.clone()
    pred[..., high_idx] += 0.5  # error lives only in the high band

    loss.set_band_weights(None)
    neutral = loss._relative_lp_on_recorder(pred, target).mean()

    loss.set_band_weights(1.0, 1.0, 0.25)  # de-emphasize high band
    deweighted = loss._relative_lp_on_recorder(pred, target).mean()

    assert deweighted < neutral
