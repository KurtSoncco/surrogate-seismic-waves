"""Free-field pads on section meshes match seiskit ``_extend_profile``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from run_sections import PAD_M, coarsen_fields, pad_free_field  # noqa: E402


def test_pad_repeats_edge_columns_500_m_each_side():
    vs = np.arange(12, dtype=np.float64).reshape(3, 4)
    out = pad_free_field(vs, 2.0, 1.0)
    assert out.shape == (3, 8)
    np.testing.assert_allclose(out[:, :2], np.repeat(vs[:, :1], 2, axis=1))
    np.testing.assert_allclose(out[:, 2:6], vs)
    np.testing.assert_allclose(out[:, 6:], np.repeat(vs[:, -1:], 2, axis=1))


def test_default_pad_on_2000_m_interior_is_3000_m():
    vs = np.ones((250, 2000))
    out = pad_free_field(vs, PAD_M, 1.0)
    assert out.shape == (250, 3000)
    assert PAD_M == 500.0


def test_zero_pad_is_a_no_op():
    vs = np.ones((4, 6))
    np.testing.assert_array_equal(pad_free_field(vs, 0.0, 1.0), vs)


def _fields(vs):
    return vs, np.full_like(vs, 0.02), np.zeros(vs.shape, dtype=bool)


def test_horizontal_layers_average_in_series():
    vs = np.repeat(np.array([100.0, 400.0, 100.0, 400.0])[:, None], 8, axis=1)
    out, _, _ = coarsen_fields(*_fields(vs), 4)
    expected = np.sqrt(1.0 / np.mean(1.0 / np.array([100.0, 400.0, 100.0, 400.0]) ** 2))
    np.testing.assert_allclose(out, expected)


def test_side_by_side_columns_average_in_parallel():
    vs = np.tile(np.array([100.0, 400.0]), (4, 4))
    out, _, _ = coarsen_fields(*_fields(vs), 2)
    np.testing.assert_allclose(out, np.sqrt((100.0**2 + 400.0**2) / 2))


def test_rock_majority_and_zeta_mean():
    vs = np.full((4, 4), 300.0)
    zeta = np.array([[0.01] * 4, [0.01] * 4, [0.01] * 4, [0.002] * 4])
    rock = np.zeros((4, 4), dtype=bool)
    rock[3:] = True
    _, z, r = coarsen_fields(vs, zeta, rock, 4)
    np.testing.assert_allclose(z, 0.008)
    assert not r.any()


def test_sample_mapping_keeps_the_centre_cell_and_k1_is_identity():
    vs = np.arange(36, dtype=np.float64).reshape(6, 6)
    out, _, _ = coarsen_fields(*_fields(vs), 3, "sample")
    np.testing.assert_array_equal(out, vs[1::3, 1::3])
    same = coarsen_fields(*_fields(vs), 1)
    assert same[0] is vs


def test_blocks_must_tile():
    with pytest.raises(ValueError):
        coarsen_fields(*_fields(np.ones((5, 6))), 2)
