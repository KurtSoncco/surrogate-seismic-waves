"""Tests for the frequency-band loss curriculum (GIFNO-FDO-XT)."""

from __future__ import annotations

import numpy as np
import torch

import config
from losses import MaskedCompositeLoss, band_curriculum_weights
from train import band_balanced_val_metric


def _make_loss(n_freq: int = 128, band_balanced_weight: float = 0.0):
    freq = np.logspace(-1, 1, n_freq).astype(np.float32)
    return MaskedCompositeLoss(
        rel_weight=1.0,
        h1_weight=0.0,
        p=1,
        band_balanced_weight=band_balanced_weight,
        freq=freq,
    )


def _manual_band_balanced(loss, pred, target, weights):
    rels = []
    used = []
    for slc, w in zip(loss._bb_slices, weights):
        if w == 0.0:
            continue
        p = pred[..., slc].reshape(pred.shape[0], -1)
        t = target[..., slc].reshape(target.shape[0], -1)
        rels.append(
            w
            * torch.linalg.norm(p - t, dim=1)
            / torch.linalg.norm(t, dim=1).clamp_min(1e-8)
        )
        used.append(w)
    return sum(rels) / sum(used)


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


def test_band_balanced_equals_mean_of_band_rel_l2():
    loss = _make_loss(128)
    torch.manual_seed(1)
    pred = torch.randn(3, 5, 128)
    target = torch.randn(3, 5, 128)
    bb = loss._band_balanced_loss(pred, target)
    manual = _manual_band_balanced(loss, pred, target, (1.0, 1.0, 1.0))
    assert torch.allclose(bb, manual, atol=1e-6)


def test_band_balanced_zero_weight_excludes_band():
    loss = _make_loss(128)
    torch.manual_seed(2)
    pred = torch.randn(2, 5, 128)
    target = torch.randn(2, 5, 128)
    loss.set_band_balanced_weights(1.0, 1.0, 0.0)
    bb = loss._band_balanced_loss(pred, target)
    manual = _manual_band_balanced(loss, pred, target, (1.0, 1.0, 0.0))
    assert torch.allclose(bb, manual, atol=1e-6)


def test_band_balanced_weight_off_by_default():
    # Baseline config: term must be off so the loss is unchanged.
    assert config.LOSS_BAND_BALANCED_WEIGHT == 0.0
    assert config.BAND_CURRICULUM_MODE == "time"
    loss = _make_loss()  # default band_balanced_weight=0.0
    assert loss.band_balanced_weight == 0.0


def test_config_schedule_defaults_retuned():
    # Tier 1: mid/high ramps start earlier so they reach target before early-stop.
    assert config.BAND_CURRICULUM_MID_START == 0.20
    assert config.BAND_CURRICULUM_HIGH_START == 0.50
    assert config.BAND_CURRICULUM_RAMP == 0.20


def test_band_balanced_val_metric_means_bands():
    val_tail = {
        "val_rel_l2_band_low_mean": 0.10,
        "val_rel_l2_band_mid_mean": 0.20,
        "val_rel_l2_band_high_mean": 0.60,
        "val_rel_l2_mean": 0.12,  # ignored: energy-weighted, not band-balanced
    }
    assert np.isclose(band_balanced_val_metric(val_tail), 0.30)


def test_band_balanced_val_metric_nan_when_missing():
    assert np.isnan(band_balanced_val_metric({"val_rel_l2_mean": 0.1}))


def test_selection_metric_parse_and_validate():
    assert config._parse_env_value("SELECTION_METRIC", "band_balanced") == (
        "band_balanced"
    )
    assert config._parse_env_value("SELECTION_METRIC", "VAL_LOSS") == "val_loss"
    try:
        config._parse_env_value("SELECTION_METRIC", "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid SELECTION_METRIC")


def test_lr_sched_overrides_parse():
    assert config._parse_env_value("LR_SCHED_PATIENCE", "40") == 40
    assert config._parse_env_value("LR_SCHED_FACTOR", "0.5") == 0.5


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
