"""Tests for Response_Variability-style IID metrics (no checkpoint)."""

from __future__ import annotations

import numpy as np
import pytest

from response_variability.metrics import (
    anderson_frequency_domain,
    band_anderson,
    band_pearson,
    band_rel_l2,
    log_residual_bias,
    odd_quarter_wave_peaks,
    peak_af,
    rel_l2,
    sigma_ln,
    spatial_sigma_ln,
    theoretical_f0,
)
from response_variability.names import (
    COMPARE_METHODS,
    DMULT,
    GINO,
    HASKELL_COLUMN,
    HASKELL_NOMINAL,
    OPENSEES,
    PASSERI,
    PRETELL,
    PRETELL_P84,
    TORO,
)
from response_variability.plot_iid import (
    _panel_title,
    compare_methods_in,
    select_diverse_indices,
    select_f0_quantile_indices,
    select_impedance_indices,
)
from response_variability.seiskit_arms import hallal_config, lognormal_upper, pretell_strip_columns


def test_peak_af_finds_resonance():
    freq = np.logspace(-1, 1, 400)
    f_true = 1.7
    af = 1.0 + 8.0 * np.exp(-((np.log(freq) - np.log(f_true)) ** 2) / 0.02)
    f_hat, a_hat = peak_af(freq, af)
    assert abs(f_hat - f_true) < 0.05
    assert a_hat == pytest.approx(af.max(), rel=1e-6)


def test_odd_quarter_wave_peaks_use_trough_windows():
    freq = np.logspace(-1, 1, 800)
    f0 = 1.0
    af = np.ones_like(freq)
    for k, amp in enumerate((10.0, 6.0, 3.0), start=1):
        fc = (2 * k - 1) * f0
        af = af + amp * np.exp(-((np.log(freq) - np.log(fc)) ** 2) / 0.015)
    peaks = odd_quarter_wave_peaks(freq, af, f0=f0, n_modes=3)
    for k, (f_hat, a_hat) in enumerate(peaks, start=1):
        assert abs(f_hat - (2 * k - 1) * f0) < 0.08
        assert a_hat > 1.5
    # Pooled argmax would take mode 1; mode windows must still return mode 3.
    f_all, _ = peak_af(freq, af)
    assert abs(f_all - f0) < 0.08
    assert abs(peaks[2][0] - 5.0) < 0.15


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


def test_band_pearson_masks_frequency():
    freq = np.array([0.2, 0.3, 1.0, 5.0, 8.0])
    true = np.linspace(1.0, 5.0, 5)
    pred = true.copy()
    pred[:2] = true[:2][::-1]
    low = band_pearson(pred, true, freq, lo=0.1, hi=0.5)
    high = band_pearson(pred, true, freq, lo=2.0, hi=10.0)
    assert high == pytest.approx(1.0)
    assert low < 0.5


def test_band_anderson_masks_frequency():
    freq = np.array([0.2, 0.3, 1.0, 5.0, 8.0])
    true = np.ones(5)
    pred = np.array([np.e, np.e, 1.0, 1.0, 1.0])
    low = band_anderson(pred, true, freq, lo=0.1, hi=0.5)
    high = band_anderson(pred, true, freq, lo=2.0, hi=10.0)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(0.0)


def test_dmult_multipliers_match_seiskit():
    from response_variability.seiskit_arms import DMULT_MULTIPLIERS

    np.testing.assert_allclose(DMULT_MULTIPLIERS, np.linspace(3.0, 6.0, 10))
    assert len(DMULT_MULTIPLIERS) == 10


def test_leftover_panel_is_r_not_prediction_error():
    from response_variability.plot_presentation import leftover_central, make_synthetic_pack

    pack = make_synthetic_pack(n=4, seed=1)
    i = 0
    rec = pack["tf_opensees"].shape[1] // 2
    r_true, r_hat = leftover_central(pack, i)
    np.testing.assert_allclose(
        r_true, pack["tf_opensees"][i, rec] - pack["tf_haskell_nominal"][i, rec]
    )
    np.testing.assert_allclose(
        r_hat, pack["tf_gino"][i, rec] - pack["tf_haskell_nominal"][i, rec]
    )
    assert not np.allclose(r_hat, pack["tf_gino"][i, rec] - pack["tf_opensees"][i, rec])


def test_extracted_f0_is_not_quarter_wave():
    from response_variability.covariates import attach_extracted_f0, f0_window_center
    from unified_metrics import max_peak_near_f0_calc

    freq = np.logspace(-1, 1, 400)
    f0_calc = 1.0
    f_true = 1.12
    f_harm = 3.0
    af = 2.0 * np.exp(-0.5 * ((np.log(freq) - np.log(f_true)) / 0.03) ** 2)
    af += 6.0 * np.exp(-0.5 * ((np.log(freq) - np.log(f_harm)) / 0.03) ** 2)
    f_pk, _ = max_peak_near_f0_calc(freq, af, f0_calc)
    assert abs(f_pk - f_true) < 0.08
    assert abs(f_pk - f_harm) > 1.0
    pack = {
        "freq": freq,
        "tf_opensees": np.broadcast_to(af, (8, 3, freq.size)).copy(),
        "vs1": np.full(8, 200.0),
        "H": np.full(8, 50.0),
        "vs2": np.full(8, 900.0),
    }
    out = attach_extracted_f0(pack)
    assert np.allclose(out["f0_calc"], 1.0)
    assert abs(float(np.nanmean(out["f0"])) - f_true) < 0.08
    assert f0_window_center(pack, 0) == pytest.approx(1.0)


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
    assert PRETELL == "Pretell median"
    assert PRETELL_P84 == "Pretell percentile"
    assert DMULT == "Dmult"
    assert HASKELL_NOMINAL == "1D Base Case"
    assert HASKELL_COLUMN == "1D column"
    assert HASKELL_NOMINAL in COMPARE_METHODS
    assert HASKELL_COLUMN not in COMPARE_METHODS
    from response_variability.names import SEISKIT_METHODS

    assert PRETELL in SEISKIT_METHODS
    assert PRETELL_P84 in SEISKIT_METHODS
    assert DMULT in SEISKIT_METHODS


def test_only_two_pretell_methods_when_geomean_present():
    pack = {
        "tf_gino": 1,
        "tf_haskell_column": 1,
        "tf_pretell": 1,
        "tf_pretell_p84": 1,
        "tf_opensees": 1,
    }
    pretell = [m for m in compare_methods_in(pack) if "Pretell" in m]
    assert pretell == [PRETELL, PRETELL_P84]


def test_lognormal_upper_is_geomean_times_exp_sigma():
    geo = np.array([np.e, 2.0])
    sig = np.array([1.0, 0.0])
    out = lognormal_upper(geo, sig, z=1.0)
    np.testing.assert_allclose(out, [np.e**2, 2.0])
    stacked = np.full((3, 4), 2.0)
    scalar = lognormal_upper(stacked, np.array([0.0, 0.0, np.log(2.0)]))
    np.testing.assert_allclose(scalar[0], 2.0)
    np.testing.assert_allclose(scalar[2], 4.0)


def test_attach_pretell_p84_scores_as_method():
    from response_variability.seiskit_arms import attach_pretell_p84

    pack = {
        "tf_gino": 1,
        "tf_pretell": np.array([[1.0, 2.0]]),
        "sigma_ln_pretell": np.array([[0.0, np.log(2.0)]]),
        "tf_opensees": 1,
    }
    pack = attach_pretell_p84(pack)
    np.testing.assert_allclose(pack["tf_pretell_p84"], [[1.0, 4.0]])
    assert compare_methods_in(pack)[-1] == PRETELL_P84


def test_upgrade_sigma_ln_from_presentation_requires_matching_idx(tmp_path):
    from response_variability.seiskit_arms import (
        attach_pretell_p84,
        upgrade_sigma_ln_from_presentation,
    )

    collapsed = {
        "sample_idx": np.array([10, 20]),
        "tf_pretell": np.ones((2, 3)),
        "sigma_ln_pretell": np.array([0.0, 0.0]),
    }
    src = tmp_path / "iid_pack.npz"
    np.savez(
        src,
        sample_idx=np.array([10, 20]),
        sigma_ln_pretell=np.array([[0.0, np.log(2.0), 0.0], [0.0, 0.0, 0.0]]),
    )
    upgraded = upgrade_sigma_ln_from_presentation(collapsed, src)
    assert upgraded["sigma_ln_pretell"].shape == (2, 3)
    p84 = attach_pretell_p84(upgraded)["tf_pretell_p84"]
    np.testing.assert_allclose(p84[0], [1.0, 2.0, 1.0])

    mismatch = dict(collapsed)
    mismatch["sample_idx"] = np.array([99, 20])
    skipped = upgrade_sigma_ln_from_presentation(mismatch, src)
    np.testing.assert_array_equal(skipped["sigma_ln_pretell"], [0.0, 0.0])


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


def test_dmult_zeta_uses_seiskit_elemental_varying():
    from response_variability.seiskit_arms import dmult_zeta, ensure_seiskit

    try:
        ensure_seiskit()
    except ImportError:
        pytest.skip("seiskit not installed")
    vs = np.full(8, 200.0)
    z3 = dmult_zeta(vs, 3.0)
    z6 = dmult_zeta(vs, 6.0)
    assert z3.shape == vs.shape
    assert np.all(z3 > 0)
    np.testing.assert_allclose(z6, 2.0 * z3)


def test_compare_methods_in_follows_pack_keys():
    pack = {"tf_gino": 1, "tf_toro": 1, "tf_opensees": 1}
    assert compare_methods_in(pack) == [GINO, TORO]


def test_default_checkpoint_is_rebal_ft():
    import config

    assert config.DEFAULT_CHECKPOINT.name == "M7680_gino_rebal_ft.pt"


def test_pearson_tf_freq_perfect_is_one():
    from response_variability.plot_presentation import pearson_tf_freq_per_sample

    tf = np.linspace(1.0, 3.0, 40).reshape(1, 1, 40)
    tf = np.broadcast_to(tf, (4, 21, 40)).copy()
    out = pearson_tf_freq_per_sample(tf, tf)
    assert out.shape == (4,)
    assert np.allclose(out, 1.0)


def test_pick_pearson_quantile_indices_unique_and_spread():
    from response_variability.plot_presentation import (
        PEARSON_QUANTILES,
        pick_pearson_quantile_indices,
    )

    p = np.linspace(0.40, 0.98, 40)
    idx = pick_pearson_quantile_indices(p)
    assert len(idx) == len(PEARSON_QUANTILES)
    assert len(set(idx.tolist())) == len(idx)
    assert p[idx[0]] < p[idx[-1]]


def test_nominal_vs_profile_two_and_three_layer():
    from response_variability.plot_presentation import nominal_vs_profile

    z, vs = nominal_vs_profile(vs1=200.0, H=30.0, vs2=800.0, nz=60, dz=1.0)
    assert z[0] < z[-1]
    assert vs[0] == pytest.approx(200.0)
    assert vs[-1] == pytest.approx(800.0)
    z3, vs3 = nominal_vs_profile(
        vs1=180.0,
        H=50.0,
        vs2=900.0,
        nz=80,
        dz=1.0,
        h1=20.0,
        h2=30.0,
        vs_mid=350.0,
    )
    assert vs3[5] == pytest.approx(180.0)
    assert vs3[25] == pytest.approx(350.0)
    assert vs3[70] == pytest.approx(900.0)
    assert z3.shape == vs3.shape
    from response_variability.plot_presentation import nominal_vs_stairs

    vs_s, z_s = nominal_vs_stairs(vs1=200.0, H=30.0, vs2=800.0, z_max=50.0)
    assert set(np.unique(vs_s).tolist()) == {200.0, 800.0}
    assert z_s[0] == pytest.approx(0.0)
    assert z_s[-1] == pytest.approx(50.0)
    vs3s, z3s = nominal_vs_stairs(
        vs1=180.0, H=50.0, vs2=900.0, z_max=70.0, h1=20.0, h2=30.0, vs_mid=350.0
    )
    assert set(np.unique(vs3s).tolist()) == {180.0, 350.0, 900.0}
    assert z3s[-1] == pytest.approx(70.0)


def test_case_title_includes_rh_ahv_cov():
    from response_variability.plot_presentation import _case_title, make_synthetic_pack

    pack = make_synthetic_pack(n=4, seed=0)
    title = _case_title(pack, 0, 0.10, "iid")
    assert r"$r_H$" in title
    assert r"$a_{HV}$" in title
    assert "CoV=" in title
    assert r"$V_{s2}$" in title


def test_stored_nz_includes_bedrock_below_soil():
    from response_variability.plot_presentation import _stored_nz, make_synthetic_pack

    pack = make_synthetic_pack(n=4, nz=40)
    i = 0
    n_plot = _stored_nz(pack["vs_2d"][i])
    assert n_plot > int(pack["soil_nz"][i])
    assert float(np.nanmax(pack["vs_2d"][i, :n_plot])) == pytest.approx(
        float(pack["vs2"][i])
    )


def test_presentation_plots_write_eleven_files(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from response_variability.plot_presentation import (
        DOMAIN_SPECS,
        make_synthetic_pack,
        plot_all_from_packs,
    )

    packs = {
        domain: make_synthetic_pack(domain=domain, n=12, seed=i)
        for i, domain in enumerate(DOMAIN_SPECS)
    }
    paths = plot_all_from_packs(packs, tmp_path)
    assert len(paths) == 11
    names = {p.name for p in paths}
    for domain in DOMAIN_SPECS:
        for page in (1, 2, 3):
            assert f"compare_{domain}_page{page}.png" in names
    assert "pearson_histograms.png" in names
    assert "vs_mosaic.png" in names
    for p in paths:
        assert p.is_file()
        assert p.stat().st_size > 0


def test_presentation_live_skipped_without_caches():
    from response_variability.plot_presentation import caches_ready

    if caches_ready():
        pytest.skip("mix caches present; live GINO/Pretell scoring is not a unit test")
    assert caches_ready() is False
