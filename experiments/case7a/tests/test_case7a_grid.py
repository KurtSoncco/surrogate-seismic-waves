"""Crop, station, and LOGLO-input checks. Does not read the surface archives."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from case7a import (  # noqa: E402
    N_COL,
    N_ROW,
    NX,
    NZ_MAX,
    ROCK_VS,
    ROCK_ZETA_NORM,
    build_model_input,
    build_window_input,
    crop_column_range,
    depth_bin_weights,
    element_center_x,
    resample_vs_full_depth,
    station_node_indices,
    station_x_m,
    window_center_x,
    window_column_start,
    window_node_index,
)


def test_crop_centers_lie_in_center_strip():
    j0, j1 = crop_column_range()
    assert (j0, j1) == (350, 450)
    centers = element_center_x(np.arange(j0, j1))
    assert centers.min() >= -250.0
    assert centers.max() < 250.0
    assert element_center_x(np.array([j0 - 1]))[0] < -250.0
    assert element_center_x(np.array([j1]))[0] >= 250.0


def test_stations_land_on_five_meter_nodes():
    x = station_x_m()
    nodes = station_node_indices()
    assert x.shape == (21,)
    assert nodes.shape == (21,)
    np.testing.assert_allclose(x, -2000.0 + 5.0 * nodes)
    assert nodes[10] == 400
    assert x[0] == -150.0
    assert x[-1] == 150.0
    assert np.all(np.diff(nodes) == 3)
    assert np.all((nodes >= 0) & (nodes < N_COL + 1))


def test_model_input_shape_and_normalization():
    vs = 140.0 + np.arange(N_COL, dtype=np.float64)[None, :]
    vs = vs + np.arange(N_ROW, dtype=np.float64)[:, None]
    model_input = build_model_input(vs)
    assert model_input.shape == (4, NZ_MAX, NX)
    assert model_input.dtype == np.float32
    np.testing.assert_allclose(model_input[0, 0], 1.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(model_input[1], 1.0)
    # One 5 m cell becomes five identical 1 m columns.
    np.testing.assert_allclose(model_input[0, :, 0], model_input[0, :, 4])
    assert not np.allclose(model_input[0, :, 4], model_input[0, :, 5])
    # x increases to the east; z increases with depth. Both are normalized.
    assert model_input[2, 0, 0] == 0.0
    np.testing.assert_allclose(model_input[2, 0, -1], 499.0 / 500.0)
    assert model_input[3, 0, 0] == 0.0
    np.testing.assert_allclose(model_input[3, -1, 0], 127.0 / 128.0)


def test_full_depth_bins_keep_the_bottom_and_windows_fit():
    weights = depth_bin_weights()
    np.testing.assert_allclose(weights.sum(axis=1), 150.0 / 128.0)
    np.testing.assert_allclose(weights.sum(), 150.0)

    vs = np.full((N_ROW, N_COL), 140.0)
    vs[-1, :] = 3000.0
    binned = resample_vs_full_depth(vs)
    assert binned.shape == (NZ_MAX, N_COL * 5)
    np.testing.assert_allclose(binned[0], 140.0)
    assert np.all(binned[-1] > ROCK_VS)

    centers = window_center_x()
    assert centers[0] == -1740.0
    assert centers[-1] == 1740.0
    assert np.any(np.isclose(centers, 0.0))
    np.testing.assert_allclose(np.diff(centers), 15.0)
    assert window_column_start(0.0) == 1750
    assert window_node_index(0.0) == 400
    assert window_column_start(centers[0]) >= 0
    assert window_column_start(centers[-1]) + NX <= N_COL * 5

    flat = build_window_input(binned, 0.0, "flat")
    rock = build_window_input(binned, 0.0, "soil_rock")
    assert flat.shape == (4, NZ_MAX, NX)
    np.testing.assert_allclose(flat[1], 1.0)
    np.testing.assert_allclose(rock[1, 0], 1.0)
    np.testing.assert_allclose(rock[1, -1], ROCK_ZETA_NORM)
    np.testing.assert_allclose(flat[3, -1, 0], 127.0 / 128.0)
