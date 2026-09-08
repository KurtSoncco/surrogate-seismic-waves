"""Wave A spectral aux + band-balanced query tests (no checkpoint)."""

from __future__ import annotations

import numpy as np
import torch
from scipy.fft import dct as scipy_dct

from data import freq_band_balanced_indices, freq_screen_indices
from spectral_aux import (
    DCT_CUTOFF,
    dct_along_freq,
    dct_below_cutoff_loss,
    trough_keep_mask,
    trough_safe_logspec_loss,
)


def test_band_balanced_equal_counts():
    freq = np.logspace(-1, 1, 1000)
    idx = freq_band_balanced_indices(freq, 210)
    assert len(idx) == 210
    f = freq[idx]
    n_low = int(np.sum((f >= 0.1) & (f < 0.5)))
    n_mid = int(np.sum((f >= 0.5) & (f < 2.0)))
    n_high = int(np.sum((f >= 2.0) & (f <= 10.0)))
    assert n_low == 70
    assert n_mid == 70
    assert n_high == 70


def test_nf1000_has_more_high_hz_queries_than_train_200():
    freq = np.logspace(-1, 1, 1000)
    n200 = int(np.sum(freq[freq_screen_indices(freq, 200)] >= 2.0))
    n1000 = int(np.sum(freq[freq_screen_indices(freq, 1000)] >= 2.0))
    assert n1000 > n200
    assert n1000 == int(np.sum(freq >= 2.0))


def test_trough_safe_logspec_ignores_near_zero():
    tf_true = torch.ones(2, 8)
    tf_true[:, :3] = 1e-8
    tf_hat = tf_true.clone()
    tf_hat[:, :3] = 10.0
    loss = trough_safe_logspec_loss(tf_hat, tf_true, floor_frac=0.05)
    assert float(loss) < 1e-5
    keep = trough_keep_mask(tf_true, floor_frac=0.05)
    assert not bool(keep[0, 0])
    assert bool(keep[0, -1])


def test_dct_along_frequency_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 32)).astype(np.float64)
    got = dct_along_freq(torch.from_numpy(x)).numpy()
    want = scipy_dct(x, type=2, norm="ortho", axis=-1)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_dct_cutoff_is_frequency_axis():
    import config

    assert DCT_CUTOFF == 16
    assert DCT_CUTOFF == config.FNO_N_MODES[1]
    r = torch.zeros(2, 7 * 64)
    r_hat = r.clone()
    r_hat[:, :] = 0.1
    loss = dct_below_cutoff_loss(r_hat, r, n_rec=7, cutoff=16)
    assert float(loss) > 0.0
    zero = dct_below_cutoff_loss(r, r, n_rec=7, cutoff=16)
    assert float(zero) < 1e-6
