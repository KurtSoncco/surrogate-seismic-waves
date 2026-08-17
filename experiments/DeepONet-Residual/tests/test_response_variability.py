"""Tests for Response_Variability-style IID metrics (no checkpoint)."""

from __future__ import annotations

import numpy as np
import pytest

from response_variability.metrics import (
    anderson_frequency_domain,
    band_rel_l2,
    log_residual_bias,
    peak_af,
    rel_l2,
    sigma_ln,
    spatial_sigma_ln,
    theoretical_f0,
)
from response_variability.names import (
    COMPARE_METHODS,
    GINO,
    HASKELL_COLUMN,
    HASKELL_NOMINAL,
    OPENSEES,
    PASSERI,
    PRETELL,
    TORO,
)
from response_variability.plot_iid import (
    _panel_title,
    compare_methods_in,
    select_diverse_indices,
    select_f0_quantile_indices,
    select_impedance_indices,
)
from response_variability.seiskit_arms import hallal_config, pretell_strip_columns


def test_peak_af_finds_resonance():
    freq = np.logspace(-1, 1, 400)
    f_true = 1.7
    af = 1.0 + 8.0 * np.exp(-((np.log(freq) - np.log(f_true)) ** 2) / 0.02)
    f_hat, a_hat = peak_af(freq, af)
    assert abs(f_hat - f_true) < 0.05
    assert a_hat == pytest.approx(af.max(), rel=1e-6)


def test_gof_zero_on_identical_curves():
    freq = np.logspace(-1, 1, 100)
    af = 2.0 + np.sin(np.log(freq))
    assert anderson_frequency_domain(freq, af, af) == pytest.approx(0.0, abs=1e-12)


def test_log_residual_bias_sign():
    ref = np.array([1.0, 2.0, 4.0])
    hi = 2.0 * ref
    lo = 0.5 * ref
    assert log_residual_bias(ref, hi) == pytest.approx(np.log(2.0))
    assert log_residual_bias(ref, lo) == pytest.approx(np.log(0.5))


def test_spatial_sigma_ln_zero_when_recorders_match():
    freq_n = 32
    rec = np.tile(np.linspace(1.0, 3.0, freq_n), (21, 1))
    sig = spatial_sigma_ln(rec)
    assert sig.shape == (freq_n,)
    assert np.allclose(sig, 0.0)


def test_sigma_ln_increases_with_spread():
    tight = np.array([1.0, 1.05, 0.95, 1.02])
    wide = np.array([0.4, 1.0, 2.5, 0.7])
    assert sigma_ln(wide) > sigma_ln(tight)


def test_band_rel_l2_masks_frequency():
    freq = np.array([0.2, 0.3, 1.0, 5.0, 8.0])
    true = np.ones(5)
    pred = np.array([2.0, 2.0, 1.0, 1.0, 1.0])
    low = band_rel_l2(pred, true, freq, lo=0.1, hi=0.5)
    high = band_rel_l2(pred, true, freq, lo=2.0, hi=10.0)
    assert low > 0.5
    assert high == pytest.approx(0.0)


def test_rel_l2_identical_is_zero():
    x = np.linspace(1.0, 4.0, 20)
    assert rel_l2(x, x) == pytest.approx(0.0)


def test_theoretical_f0():
    assert theoretical_f0(200.0, 50.0) == pytest.approx(1.0)
    assert np.isnan(theoretical_f0(200.0, 0.0))


def test_select_diverse_indices_spreads_and_sorts_by_f0():
    n = 20
    pack = {
        "vs1": np.linspace(100.0, 400.0, n),
        "H": np.linspace(20.0, 80.0, n)[::-1],
        "cov": np.linspace(0.1, 0.4, n),
        "f0": np.linspace(0.5, 2.0, n),
    }
    idx = select_diverse_indices(pack, n=5, seed=0)
    assert len(idx) == 5
    assert len(set(idx.tolist())) == 5
    assert np.all(np.diff(pack["f0"][idx]) >= 0)


def test_select_f0_quantiles_are_unique_and_sorted():
    n = 20
    pack = {
        "f0": np.linspace(0.3, 4.0, n),
        "vs1": np.linspace(100.0, 400.0, n),
        "H": np.full(n, 40.0),
        "cov": np.linspace(0.1, 0.3, n),
    }
    idx = select_f0_quantile_indices(pack, n=5)
    assert len(idx) == 5
    assert len(set(idx.tolist())) == 5
    assert np.all(np.diff(pack["f0"][idx]) >= 0)
    assert idx[0] == 0
    assert idx[-1] == n - 1


def test_select_impedance_uses_rh_ahv_cov():
    n = 30
    pack = {
        "rH": np.linspace(10.0, 100.0, n),
        "aHV": np.linspace(10.0, 48.0, n)[::-1],
        "cov": np.linspace(0.1, 0.3, n),
        "vs1": np.full(n, 200.0),
        "H": np.full(n, 40.0),
        "f0": np.linspace(0.5, 2.0, n),
    }
    idx = select_impedance_indices(pack, n=5)
    assert len(idx) == 5
    assert len(set(idx.tolist())) == 5
    rh = pack["rH"][idx]
    ahv = pack["aHV"][idx]
    assert rh.max() - rh.min() > 40.0
    assert ahv.max() - ahv.min() > 15.0


def test_panel_title_includes_rh_ahv_cov():
    pack = {
        "vs1": np.array([188.0]),
        "H": np.array([97.0]),
        "cov": np.array([0.21]),
        "rH": np.array([62.4]),
        "aHV": np.array([27.2]),
    }
    title = _panel_title(pack, 0)
    assert "CoV=0.21" in title
    assert r"$r_H$=62" in title
    assert r"$a_{HV}$=27" in title


def test_readable_method_names():
    assert OPENSEES == "OpenSees 2-D"
    assert GINO == "GINO"
    assert TORO == "Toro Vs"
    assert PASSERI == "Passeri tts"
    assert PRETELL == "Pretell"
    assert HASKELL_NOMINAL == "1D Base Case"
    assert HASKELL_COLUMN == "Pretell's approach"
    assert HASKELL_NOMINAL in COMPARE_METHODS
    assert HASKELL_COLUMN in COMPARE_METHODS


def test_pretell_strip_columns_span_cropped_domain():
    cols = pretell_strip_columns(200, n_strip=500)
    assert len(cols) == 200
    assert cols[0] == 0
    assert cols[-1] == 499
    assert np.all(np.diff(cols) >= 0)


def test_hallal_config_matches_rv_simplified_flags():
    from response_variability.seiskit_arms import ensure_seiskit

    try:
        ensure_seiskit()
    except ImportError:
        pytest.skip("seiskit not installed")
    cfg = hallal_config(vs1=200.0, H=40.0, cov=0.2, vs2=800.0, dz=0.5)
    assert cfg.use_full_model is False
    assert cfg.randomize_layer_thickness is False
    assert cfg.randomize_bedrock_depth is False
    assert cfg.vary_bedrock_vs is False
    assert cfg.dz == pytest.approx(0.5)


def test_compare_methods_in_follows_pack_keys():
    pack = {"tf_gino": 1, "tf_toro": 1, "tf_opensees": 1}
    assert compare_methods_in(pack) == [GINO, TORO]
