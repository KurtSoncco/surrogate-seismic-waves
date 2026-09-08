"""Sobol covering / frequency-probe helpers (no checkpoint, no packs)."""

from __future__ import annotations

import numpy as np
import pytest

from data import freq_screen_indices
from response_variability.sobol_design import (
    AHV_FIXED,
    BOUNDS_H,
    RH_FIXED,
    aabb_inside,
    cases_matrix,
    fill_distance,
    freq_train_mask,
    generate_rv_base_cases,
    knn_distance,
    nested_permutation_prefixes,
    physical_to_unit_6d,
    standardize,
    unique_rows,
)


def test_rv_cases_nested_and_fixed_geostats():
    small = generate_rv_base_cases(8)
    large = generate_rv_base_cases(32)
    assert len(small) == 8
    assert len(large) == 32
    for a, b in zip(small, large[:8]):
        assert a.vs1 == pytest.approx(b.vs1)
        assert a.H == pytest.approx(b.H)
        assert a.cov == pytest.approx(b.cov)
        assert a.vs2 == pytest.approx(b.vs2)
    for case in large:
        assert case.rH == RH_FIXED
        assert case.aHV == AHV_FIXED
        assert BOUNDS_H[0] <= case.H <= BOUNDS_H[1]


def test_rv_unit_cube_is_geostat_corner():
    rv6 = cases_matrix(generate_rv_base_cases(16), kind="6d")
    unit = physical_to_unit_6d(rv6)
    np.testing.assert_allclose(unit[:, 3], 0.0, atol=1e-12)
    assert np.all(unit[:, 4] >= 0.99)
    assert np.all(unit[:, 4] <= 1.0 + 1e-12)
    assert np.all((unit[:, :3] >= 0.0) & (unit[:, :3] <= 1.0))
    assert np.all((unit[:, 5] >= 0.0) & (unit[:, 5] <= 1.0))


def test_unique_rows_and_knn_identity():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 4))
    dup = np.vstack([x, x[:5]])
    assert len(unique_rows(dup, decimals=12)) == 20
    d = knn_distance(x, x, k=1)
    np.testing.assert_allclose(d, 0.0, atol=1e-12)
    assert fill_distance(x, x) == pytest.approx(0.0, abs=1e-12)


def test_aabb_flags_outside_points():
    train = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    query = np.array([[0.5, 0.5], [1.5, 0.5], [-0.1, 0.2]])
    inside = aabb_inside(train, query)
    np.testing.assert_array_equal(inside, [True, False, False])


def test_nested_prefixes_are_nested():
    prefixes = nested_permutation_prefixes(20, (4, 8, 16), seed=42)
    assert set(prefixes[4]).issubset(set(prefixes[8]))
    assert set(prefixes[8]).issubset(set(prefixes[16]))


def test_standardize_zero_mean_unit_std():
    rng = np.random.default_rng(1)
    train = rng.normal(loc=3.0, scale=2.0, size=(200, 3))
    z, q = standardize(train, train[:10])
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-12)
    assert q.shape == (10, 3)


def test_freq_train_mask_is_log_screen_complement():
    freq = np.logspace(-1, 1, 1000)
    mask = freq_train_mask(freq, 200)
    idx = freq_screen_indices(freq, 200)
    assert mask.sum() == 200
    np.testing.assert_array_equal(np.where(mask)[0], idx)
    assert (~mask).sum() == 800


def test_aabb_rv_outside_narrow_rH_aHV():
    """RV (rH=10, aHV=50) is outside a train cloud that never reaches the corner."""
    train = np.column_stack(
        [
            np.full(12, 200.0),
            np.full(12, 40.0),
            np.full(12, 0.2),
            np.linspace(12.0, 40.0, 12),
            np.linspace(12.0, 45.0, 12),
            np.full(12, 900.0),
        ]
    )
    rv = cases_matrix(generate_rv_base_cases(4), kind="6d")
    inside = aabb_inside(train, rv)
    assert not np.any(inside)
