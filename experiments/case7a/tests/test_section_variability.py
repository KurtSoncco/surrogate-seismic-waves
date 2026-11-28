"""``vs_cov`` soil variability hook in ``section_field``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import section_generator as sg  # noqa: E402


@pytest.fixture(scope="module")
def row():
    return dict(sg.sample_rows(1, seed=3)[0], lx_m=500.0)


def test_zero_cov_is_bit_identical(row):
    base = sg.section_field(row)
    for extra in ({"vs_cov": 0.0, "field_seed": 7}, {"vs_cov": ""}, {}):
        out = sg.section_field(dict(row, **extra))
        for a, b in zip(base, out):
            assert a.dtype == b.dtype and np.array_equal(a, b)


def test_cov_changes_soil_only_and_follows_the_seed(row):
    vs0, zeta0, rock0 = sg.section_field(row)
    vs1, zeta1, rock1 = sg.section_field(dict(row, vs_cov=0.2, field_seed=11))
    np.testing.assert_array_equal(rock0, rock1)
    np.testing.assert_array_equal(zeta0, zeta1)
    np.testing.assert_array_equal(vs0[rock0], vs1[rock0])
    assert not np.allclose(vs0[~rock0], vs1[~rock0])
    again = sg.section_field(dict(row, vs_cov=0.2, field_seed=11))[0]
    other = sg.section_field(dict(row, vs_cov=0.2, field_seed=12))[0]
    np.testing.assert_array_equal(vs1, again)
    assert not np.array_equal(vs1, other)


def test_factor_is_mean_one_with_the_requested_log_std():
    f = sg.soil_variability({"vs_cov": 0.2, "field_seed": 5, "vs_corr_h_m": 20.0, "vs_corr_aniso": 4.0}, 200, 2000)
    s = np.sqrt(np.log1p(0.2**2))
    assert abs(f.mean() - 1.0) < 0.02
    assert abs(np.log(f).std() - s) < 0.1 * s


def test_matches_seiskit_generator():
    seiskit_root = Path("/home/kurt-/seiskit")
    if str(seiskit_root) not in sys.path:
        sys.path.insert(0, str(seiskit_root))
    try:
        import seiskit.gaussian_field as gf
    except Exception as exc:  # absent, or its numba cache directory is read-only
        pytest.skip(f"seiskit.gaussian_field unavailable: {exc}")
    nz, nx, cov = 60, 300, 0.25
    g = gf.generate_gaussian_field_fft(nx, nz, 1.0, 1.0, 30.0, 6.0, np.random.default_rng(9))
    s = np.sqrt(np.log1p(cov**2))
    expected = np.exp(-0.5 * s**2 + s * np.clip(g, -2.0, 2.0))
    got = sg.soil_variability({"vs_cov": cov, "field_seed": 9, "vs_corr_h_m": 30.0, "vs_corr_aniso": 6.0}, nz, nx)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
