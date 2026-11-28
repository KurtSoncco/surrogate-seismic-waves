"""The high-pass used to score transfer-function wiggles."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pod_rank import oscillation_window, smooth_bins  # noqa: E402


def test_sqrt_weight_is_one_at_one_hertz_and_matches_an_unweighted_pod():
    from model import fit_log_pod, fit_weighted_pod, reconstruct_pod, spectral_weight

    freq = np.array([0.1, 1.0, 10.0])
    weight = spectral_weight(freq)
    assert weight[1] == 1.0
    assert np.isclose(weight[0], np.sqrt(0.1))
    assert np.isclose(weight[2], np.sqrt(10.0))
    rng = np.random.default_rng(0)
    curves = rng.normal(size=(40, 16))
    plain_modes, plain_mean = fit_log_pod(curves, 5)
    modes, mean = fit_weighted_pod(curves, 5, np.ones(16))
    assert np.allclose(plain_mean, mean)
    left = (curves - plain_mean) @ plain_modes.T @ plain_modes
    right = ((curves - mean) @ modes.T) @ modes
    assert np.allclose(left, right)
    rising = spectral_weight(np.geomspace(0.1, 10.0, 16))
    full, full_mean = fit_weighted_pod(curves, 16, rising)
    coeff = ((curves - full_mean) * rising) @ full.T
    restored = reconstruct_pod(
        torch.as_tensor(full_mean), torch.as_tensor(coeff[:, None, :], dtype=torch.float32),
        torch.as_tensor(full), torch.as_tensor(rising, dtype=torch.float32),
    )
    assert torch.allclose(restored[:, 0, :], torch.as_tensor(curves, dtype=torch.float32), atol=1e-4)


def test_training_loss_charges_more_for_the_same_miss_at_higher_frequency():
    from train import weighted_l1

    pred = torch.zeros(1, 1, 3)
    valid = torch.ones(1, 1, dtype=torch.bool)
    freq = torch.tensor([0.1, 1.0, 10.0])
    width = torch.tensor([500.0])
    weight = freq.sqrt()
    low = pred.clone()
    low[..., 0] = 1.0
    high = pred.clone()
    high[..., 2] = 1.0
    low_loss = float(weighted_l1(pred, low, valid, freq, width, weight))
    high_loss = float(weighted_l1(pred, high, valid, freq, width, weight))
    assert high_loss > low_loss


def test_moving_mean_drops_a_constant_and_keeps_a_bin_to_bin_wiggle():
    constant = np.ones((2, 40))
    assert np.allclose(smooth_bins(constant, 9), 1.0)
    wiggle = np.resize(np.array([1.0, -1.0]), (1, 40))
    smooth = smooth_bins(wiggle, 9)
    kept = wiggle - smooth
    assert np.mean(np.abs(kept)) > 0.7


def test_uniform_weight_matches_ordinary_pod_and_full_rank_is_exact():
    from pod_rank import rank_table, weighted_band_errors

    rng = np.random.default_rng(0)
    curves = rng.normal(size=(40, 24))
    freq = np.geomspace(0.1, 10.0, 24)
    plain = rank_table(curves[:30], curves[30:], freq, (4,))
    weighted = weighted_band_errors(curves[:30], curves[30:], freq, np.ones(24), (4,))
    assert abs(plain["ranks"][0]["mae"] - weighted[0]["mae"]) < 1e-8
    full = weighted_band_errors(curves[:30], curves[30:], freq, np.linspace(1.0, 8.0, 24), (24,))
    assert full[0]["mae"] < 1e-8


def test_a_rising_weight_spends_a_short_basis_on_the_shifting_peak():
    from pod_rank import weighted_band_errors

    rng = np.random.default_rng(1)
    curves = np.zeros((80, 30))
    curves[:, :5] = rng.normal(size=(80, 5)) * 5.0
    for i in range(80):
        curves[i, 20 + (i % 5)] = 1.0
    freq = np.linspace(0.1, 10.0, 30)
    plain = weighted_band_errors(curves[:60], curves[60:], freq, np.ones(30), (2,))
    heavy = weighted_band_errors(curves[:60], curves[60:], freq, np.linspace(1.0, 40.0, 30), (2,))
    assert heavy[0]["bands"]["5-10"] < plain[0]["bands"]["5-10"]


def test_one_sixth_octave_is_a_few_tens_of_bins_on_the_label_grid():
    freq = np.geomspace(0.1, 10.0, 1000)
    window = oscillation_window(freq, octave_fraction=6)
    assert window % 2 == 1
    assert 15 <= window <= 45
