from __future__ import annotations

import numpy as np
from haskell_baseline import (
    haskell_at_columns,
    haskell_nominal_af_within,
    haskell_nominal_layered_af_within,
)


def test_uniform_column_matches_nominal():
    freq = np.logspace(-1, 1, 64)
    vs1, H, vs2, xi, dz = 180.0, 40.0, 900.0, 0.05, 1.0
    soil_nz = int(H / dz)
    vs = np.full((soil_nz + 10, 20), vs2)
    vs[:soil_nz, :] = vs1
    zeta = np.full_like(vs, xi)
    rec = np.array([5, 10, 15])
    col = haskell_at_columns(
        freq, vs, zeta, rec, dz=dz, vs_rock=vs2, soil_nz=soil_nz, rho=2000.0
    )
    nom = haskell_nominal_af_within(freq, vs1=vs1, H=H, vs2=vs2, xi=xi, rho=2000.0)
    assert col.shape == (3, 64)
    np.testing.assert_allclose(col[0], nom, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(col[1], col[0], rtol=1e-8)


def test_layered_nom_matches_two_layer_column():
    freq = np.logspace(-1, 1, 64)
    h1, h2, vs1, vs_mid, vs_rock, xi, dz = 10.0, 12.0, 180.0, 420.0, 900.0, 0.05, 1.0
    n1, n2 = int(h1 / dz), int(h2 / dz)
    soil_nz = n1 + n2
    vs = np.full((soil_nz + 8, 5), vs_rock)
    vs[:n1, :] = vs1
    vs[n1:soil_nz, :] = vs_mid
    zeta = np.full_like(vs, xi)
    rec = np.array([1, 3])
    col = haskell_at_columns(
        freq, vs, zeta, rec, dz=dz, vs_rock=vs_rock, soil_nz=soil_nz, rho=2000.0
    )
    nom3 = haskell_nominal_layered_af_within(
        freq, H=[h1, h2], Vs=[vs1, vs_mid], vs_rock=vs_rock, xi=xi, rho=2000.0
    )
    np.testing.assert_allclose(col[0], nom3, rtol=1e-5, atol=1e-6)
    one = haskell_nominal_af_within(
        freq, vs1=vs1, H=h1 + h2, vs2=vs_rock, xi=xi, rho=2000.0
    )
    assert float(np.linalg.norm(nom3 - one)) > 0.1
