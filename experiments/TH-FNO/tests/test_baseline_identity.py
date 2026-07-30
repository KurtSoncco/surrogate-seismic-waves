"""Baseline must be H_1D(trend), not realization-column geomean (AGENTS §1.2)."""

from __future__ import annotations

import numpy as np

from haskell_baseline import (
    H_1D_trend,
    haskell_af_within,
    haskell_realization_geomean,
    haskell_trend_af_within,
)


def test_trend_alias_is_single_layer_params():
    freq = np.logspace(-1, 1, 128)
    a = H_1D_trend(freq, vs1=220.0, H=35.0, vs2=850.0, xi=0.04, freq_scale=1.0)
    b = haskell_trend_af_within(freq, vs1=220.0, H=35.0, vs2=850.0, xi=0.04)
    assert np.allclose(a, b)
    assert np.all(np.isfinite(a))


def test_freq_scale_shifts_resonance_down():
    """freq_scale < 1 thickens H_eff → lower f0 (OpenSees bias fix)."""
    freq = np.logspace(-1, 1, 512)
    raw = haskell_trend_af_within(freq, vs1=200.0, H=40.0, vs2=900.0, xi=0.05)
    cal = H_1D_trend(
        freq, vs1=200.0, H=40.0, vs2=900.0, xi=0.05, freq_scale=0.94
    )
    assert float(freq[int(np.argmax(cal))]) < float(freq[int(np.argmax(raw))])


def test_trend_differs_from_realization_geomean():
    """Training baseline must not silently become realization geomean."""
    rng = np.random.RandomState(0)
    freq = np.logspace(-1, 1, 64)
    vs1, H, vs2, xi = 200.0, 40.0, 900.0, 0.05
    nz, nx = int(H), 21
    vs = np.full((nz + 10, nx), vs2, dtype=float)
    vs[:nz] = vs1 * (1.0 + 0.3 * rng.randn(nz, nx))
    zeta = np.full_like(vs, xi)
    trend = H_1D_trend(freq, vs1=vs1, H=H, vs2=vs2, xi=xi, freq_scale=1.0)
    geo = haskell_realization_geomean(
        freq, vs, zeta, list(range(nx)), dz=1.0, vs_rock=vs2, soil_nz=nz
    )
    # Heterogeneous columns → geomean ≠ trend parameters
    assert not np.allclose(trend, geo, rtol=1e-2)


def test_local_column_is_not_training_baseline_alias():
    freq = np.logspace(-1, 1, 32)
    vs_col = np.concatenate([np.full(40, 200.0), np.full(10, 900.0)])
    zeta = np.full_like(vs_col, 0.05)
    local = haskell_af_within(freq, vs_col, zeta, dz=1.0, vs_rock=900.0, soil_nz=40)
    trend = H_1D_trend(
        freq, vs1=200.0, H=40.0, vs2=900.0, xi=0.05, freq_scale=1.0
    )
    # Flat column equals trend; still distinct API (geomean helper must not be default)
    assert np.allclose(local, trend, rtol=1e-5)
