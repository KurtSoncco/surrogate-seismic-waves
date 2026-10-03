"""Uniform dip-depth formulas. Depth does not move bedrock Vs."""

from __future__ import annotations

import numpy as np
import pytest

from response_variability.dip_depth import (
    along_line_length,
    depth_range,
    empirical_sigma_y,
    interface_depths,
    sigma_L,
    sigma_y,
    uniform_dip_depths,
    uniform_spectrum,
)
from response_variability.seiskit_arms import ensure_seiskit, hallal_config


def _require_seiskit() -> None:
    try:
        ensure_seiskit()
    except ImportError:
        pytest.skip("seiskit not installed")


def test_sigma_L_continuous_and_discrete_match_uniform_samples():
    L = 500.0
    assert sigma_L(L) == pytest.approx(L / np.sqrt(12.0))
    n = 500
    samples = np.linspace(0.0, L, n)
    assert sigma_L(L, n) == pytest.approx(float(np.std(samples, ddof=0)))
    fine = np.linspace(0.0, L, 200_001)
    assert float(np.std(fine, ddof=0)) == pytest.approx(sigma_L(L), rel=1e-5)


def test_sigma_y_is_sigma_L_times_abs_sin():
    L_h = 500.0
    theta = -2.79
    L = along_line_length(L_h, theta)
    s = np.linspace(0.0, L, 500)
    y = s * np.sin(np.deg2rad(theta))
    assert sigma_y(L_h, theta, n=500) == pytest.approx(float(np.std(y, ddof=0)))
    assert sigma_y(L_h, theta) == pytest.approx(
        sigma_L(L) * abs(np.sin(np.deg2rad(theta)))
    )
    assert sigma_y(L_h, theta) == sigma_y(L_h, -theta)
    assert depth_range(L_h, theta) == pytest.approx(L * abs(np.sin(np.deg2rad(theta))))


def test_straight_interface_std_tracks_discrete_sigma_y():
    """Integer stair-step of a straight dip stays within one dz of the formula."""
    L_h = 500.0
    theta = 2.79
    H = 40.0
    x = np.arange(L_h)
    y = H + (x - (L_h - 1) / 2.0) * np.tan(np.deg2rad(theta))
    y_i = np.clip(np.rint(y), 1, 79).astype(int)
    field = np.full((80, int(L_h)), 200.0)
    for col, iface in enumerate(y_i):
        field[iface:, col] = 1000.0
    emp = empirical_sigma_y(field, 1000.0, dz=1.0)
    assert emp == pytest.approx(sigma_y(L_h, theta, n=int(L_h)), abs=1.0)
    assert interface_depths(field, 1000.0).size == int(L_h)


def test_uniform_depths_are_centered_equal_weight_nodes():
    H = 40.0
    theta = 1.5
    L_h = 500.0
    depths = uniform_dip_depths(H, theta, L_h=L_h, n=64)
    half = 0.5 * depth_range(L_h, theta)
    assert depths.shape == (64,)
    assert depths[0] == pytest.approx(H - half)
    assert depths[-1] == pytest.approx(H + half)
    assert np.all(np.diff(depths) > 0)
    assert float(np.mean(depths)) == pytest.approx(H)


def test_dipping_toro_calls_seiskit_frozen_h():
    """NHPP thicknesses stay off. Vs comes from seiskit's frozen-H generator."""
    _require_seiskit()
    from seiskit.profile_randomization import generate_vs_randomized_profile

    cfg = hallal_config(vs1=195.0, H=53.0, cov=0.16, vs2=1081.0, dz=0.5)
    assert cfg.use_full_model is False
    assert cfg.randomize_layer_thickness is False
    assert cfg.randomize_bedrock_depth is False
    vs = generate_vs_randomized_profile(cfg, np.random.default_rng(1))
    soil_nz = int(round(cfg.thickness / cfg.dz))
    assert vs[:soil_nz].std() > 1.0
    assert np.all(vs[soil_nz:] == pytest.approx(cfg.vs_bedrock))


def test_depth_draw_does_not_change_vs2_for_toro_or_passeri():
    _require_seiskit()
    from seiskit.profile_randomization.passeri import _passeri_joint_bedrock_draw

    vs2 = 1234.5
    rng = np.random.default_rng(0)
    for H in uniform_dip_depths(36.0, 2.2, n=5):
        cfg = hallal_config(vs1=180.0, H=float(H), cov=0.22, vs2=vs2, dz=0.5)
        assert cfg.vs_bedrock == vs2
        assert cfg.vary_bedrock_vs is False
        assert cfg.randomize_bedrock_depth is False
        depth, bed = _passeri_joint_bedrock_draw(cfg, rng)
        assert depth == pytest.approx(float(H))
        assert bed == pytest.approx(vs2)


def test_uniform_spectrum_is_equal_weight_geomean_and_envelope():
    stack = np.array([[1.0, 4.0], [1.0, 1.0], [1.0, 16.0]])
    geo, lo, hi = uniform_spectrum(stack)
    np.testing.assert_allclose(geo, [1.0, 4.0])
    np.testing.assert_allclose(lo, [1.0, 1.0])
    np.testing.assert_allclose(hi, [1.0, 16.0])
