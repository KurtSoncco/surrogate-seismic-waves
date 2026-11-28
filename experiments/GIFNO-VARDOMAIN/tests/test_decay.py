"""Rayleigh decay bound, TRBDF2 numerics, and the applied zeta a row is solved with."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "case7a"))

from decay import applied_zeta, decay_rate, max_dt, t_needed, trbdf2_numerics  # noqa: E402
from section_1d import rayleigh_coefficients  # noqa: E402


def test_decay_rate_is_zeta_omega_at_the_anchors():
    zeta, f1, f2 = 0.005, 0.2, 0.6
    a0, a1 = rayleigh_coefficients(zeta, f1, f2)
    r = decay_rate(np.array([f1, f2]), a0, a1)
    np.testing.assert_allclose(r, zeta * 2 * np.pi * np.array([f1, f2]))
    assert np.isclose(2.0 / a0, 2.0 / (2 * zeta * (2 * np.pi * f1) * (2 * np.pi * f2) / (2 * np.pi * (f1 + f2))))


def test_damped_sdof_decays_at_the_bound():
    a0, a1 = rayleigh_coefficients(0.01, 0.5, 1.5)
    f = 0.8
    w = 2 * np.pi * f
    c = a0 + a1 * w**2
    roots = np.roots([1.0, c, w**2])
    assert np.isclose(-roots.real.max(), decay_rate(np.array([f]), a0, a1)[0])


def test_t_needed_is_shift_plus_crossing_plus_log_ratio_over_rate():
    a0, a1 = rayleigh_coefficients(0.02, 0.4, 1.2)
    f = np.array([0.4])
    r = decay_rate(f, a0, a1)[0]
    got = t_needed(f, a0, a1, eps=0.01, t_cross=12.0, amplitude=2.0)[0]
    assert np.isclose(got, 0.5 + 12.0 + np.log(200.0) / r)


def test_trbdf2_is_second_order_and_dissipative():
    z, err = trbdf2_numerics(np.array([0.1, 1.0, 5.0]), 0.01)
    assert np.all(z > 0) and np.all(np.diff(z) > 0)
    assert z[0] < 1e-8 and abs(err[0]) < 1e-5
    z2, _ = trbdf2_numerics(np.array([5.0]), 0.02)
    assert 4.0 < z2[0] / z[2] < 10.0


def test_max_dt_meets_the_damping_budget():
    dt = max_dt(2.0, 0.005, frac=0.1)
    assert trbdf2_numerics(np.array([2.0]), dt)[0][0] <= 0.1 * 0.005 * (1 + 1e-9)
    assert trbdf2_numerics(np.array([2.0]), dt * 1.05)[0][0] > 0.1 * 0.005


def test_applied_zeta_explicit_and_qvs():
    import section_generator as sg
    from case7_crops import qvs_xi
    from run_sections import PAD_M, pad_free_field

    row = dict(sg.sample_rows(1, seed=2)[0], lx_m=500.0)
    assert applied_zeta(dict(row, damping_family="explicit")) == (row["zeta_soil"], row["zeta_rock"])
    vs, _, rock = sg.section_field(row)
    expected = qvs_xi(pad_free_field(vs, PAD_M, 1.0), pad_free_field(rock, PAD_M, 1.0))
    assert applied_zeta(dict(row, damping_family="qvs")) == expected
