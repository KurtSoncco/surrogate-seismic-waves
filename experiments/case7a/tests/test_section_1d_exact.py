"""Exact complex 1D outcrop transfer function under viscous Rayleigh damping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_1d import RHO, outcrop_tf_rayleigh, rayleigh_coefficients  # noqa: E402

FREQ = np.geomspace(0.1, 10.0, 60)


def test_rayleigh_pair_hits_zeta_at_both_anchors():
    a0, a1 = rayleigh_coefficients(0.02, 0.3, 0.9)
    for f in (0.3, 0.9):
        w = 2 * np.pi * f
        assert np.isclose(a0 / (2 * w) + a1 * w / 2, 0.02)


def test_undamped_homogeneous_column_is_one():
    h = outcrop_tf_rayleigh(FREQ, np.full(120, 400.0), 0.0, 0.0, dz=1.0)
    np.testing.assert_allclose(np.abs(h), 1.0, atol=1e-9)


def test_one_layer_matches_closed_form():
    thick, v1, v2, a1 = 40, 200.0, 800.0, 1e-3
    vs = np.r_[np.full(thick, v1), np.full(10, v2)]
    a1_cells = np.r_[np.full(thick, a1), np.zeros(10)]
    h = outcrop_tf_rayleigh(FREQ, vs, 0.0, a1_cells, dz=1.0)
    w = 2 * np.pi * FREQ
    g1, g2 = RHO * v1**2 * (1 + 1j * w * a1), RHO * v2**2
    k1 = w * np.sqrt(RHO / g1)
    alpha = np.sqrt(g1 * RHO) / np.sqrt(g2 * RHO)
    expected = 1.0 / np.abs(np.cos(k1 * thick) + 1j * alpha * np.sin(k1 * thick))
    np.testing.assert_allclose(np.abs(h), expected, rtol=1e-8)


def test_complex_modulus_engines_match_the_exact_one():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "GIFNO"))
    from absolute_inputs import column_tf1d_outcrop
    from section_1d import surface_over_outcrop

    xi = 0.02
    for vs in (np.full(100, 300.0), np.r_[np.full(50, 150.0), np.full(50, 600.0)], np.linspace(120.0, 700.0, 200)):
        exact = np.array([abs(outcrop_tf_rayleigh(np.array([f]), vs, 0.0, 2 * xi / (2 * np.pi * f), dz=1.0)[0])
                          for f in FREQ])
        old = surface_over_outcrop(FREQ, vs, dz=1.0, xi=np.full(FREQ.size, xi))
        gifno = column_tf1d_outcrop(vs[:, None], np.full((vs.size, 1), xi), 1.0, FREQ)[:, 0]
        np.testing.assert_allclose(old, exact, rtol=1e-9)
        np.testing.assert_allclose(gifno, exact, rtol=1e-9)


def test_column_tf1d_rayleigh_matches_the_exact_one_at_each_frequency():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "GIFNO"))
    from absolute_inputs import column_groups, column_tf1d_rayleigh

    vs = np.r_[np.linspace(150.0, 400.0, 60), np.full(20, 900.0)]
    zeta = np.full(vs.size, 0.03)
    got = column_tf1d_rayleigh(vs[:, None], zeta[:, None], 1.0, FREQ)[:, 0]
    f0 = float(np.asarray(column_groups(vs[:, None], 1.0)[0]).ravel()[0])
    w1, w2, w = 2 * np.pi * f0, 2 * np.pi * 10.0, 2 * np.pi * FREQ
    xi_f = 0.5 * (2 * 0.03 * w1 * w2 / (w1 + w2) / w + 2 * 0.03 / (w1 + w2) * w)
    exact = np.array([abs(outcrop_tf_rayleigh(np.array([f]), vs, 0.0, 2 * x / (2 * np.pi * f), dz=1.0)[0])
                      for f, x in zip(FREQ, xi_f)])
    np.testing.assert_allclose(got, exact, rtol=1e-9)


def test_impulse_response_is_causal():
    n, dt = 2**15, 0.01
    f = np.fft.rfftfreq(n, dt)
    vs = np.linspace(150.0, 700.0, 120)
    a0, a1 = rayleigh_coefficients(0.02, 0.4, 1.2)
    h = np.ones(f.size, dtype=complex)
    h[1:] = outcrop_tf_rayleigh(f[1:], vs, a0, a1, dz=1.0)
    imp = np.fft.irfft(h, n)
    assert np.sum(imp[n // 2 :] ** 2) / np.sum(imp**2) < 1e-6
