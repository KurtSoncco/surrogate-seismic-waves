"""1D soilgen columns stay inside 2D contacts; they are not a 2D field."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_generator import DEPTH_M, contact_depth, sample_rows, section_field  # noqa: E402
from soilgen_columns import layered_column, paint_soil  # noqa: E402


def test_paint_stretches_one_column_to_h():
    rng = np.random.default_rng(0)
    bounds, vs = layered_column(80.0, 140.0, 400.0, 5, rng)
    z = (np.arange(150) + 0.5) * 1.0
    h = np.array([60.0, 90.0, 110.0])
    grid = paint_soil(z, h, bounds, vs)
    assert grid.shape == (150, 3)
    np.testing.assert_allclose(grid[0], vs[0])
    # Below the contact the caller masks to rock; here z=0.5 m is always soil.
    assert grid[0, 0] == grid[0, 1]


def test_soilgen_section_keeps_2d_contact_and_layered_soil():
    row = sample_rows(1, seed=1)[0]
    row["feature"] = 2
    row["feature_name"] = "step"
    row["soil_fill"] = "soilgen"
    row["n_soil_layers"] = 6
    row["soilgen_seed"] = 7
    vs_g, _, rock_g = section_field({**row, "soil_fill": "gradient"})
    vs_s, zeta, rock_s = section_field(row)
    assert vs_s.shape == vs_g.shape == (int(DEPTH_M), int(row["lx_m"]))
    np.testing.assert_array_equal(rock_s, rock_g)
    x = (np.arange(vs_s.shape[1]) + 0.5)
    h = contact_depth(row, x)
    soil_j = int(np.argmax(h))
    # Layered: more than two unique Vs values in a soil column.
    col = vs_s[np.arange(int(np.floor(h[soil_j]))), soil_j]
    assert np.unique(np.round(col, 1)).size >= 3
    # Lateral change comes from H(x), not independent columns: same layer at surface.
    assert np.allclose(vs_s[0, ~rock_s[0]], vs_s[0, ~rock_s[0]][0])


def test_gradient_default_unchanged():
    row = sample_rows(1, seed=2)[0]
    vs_a, _, _ = section_field(row)
    vs_b, _, _ = section_field({**row, "soil_fill": "gradient"})
    np.testing.assert_allclose(vs_a, vs_b)
