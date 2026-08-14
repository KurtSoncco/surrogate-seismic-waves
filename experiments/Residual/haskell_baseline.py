"""Thomson–Haskell |TF| baselines: local-column and nominal (Vs1, H, Vs2)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

_SEISKIT_ROOT = Path.home() / "seiskit"
if _SEISKIT_ROOT.is_dir() and str(_SEISKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEISKIT_ROOT))

# ruff: noqa: E402
try:
    from seiskit.theory import Layer, RockHalfspace
except ImportError:  # pragma: no cover - optional; AF_within is self-contained
    Layer = None  # type: ignore[misc, assignment]
    RockHalfspace = None  # type: ignore[misc, assignment]

_EPS = 1e-12
DEFAULT_RHO = 2000.0


def column_to_layers(
    vs_col: np.ndarray,
    zeta_col: np.ndarray,
    *,
    dz: float,
    vs_rock: float,
    soil_nz: int | None = None,
    rho: float = DEFAULT_RHO,
) -> tuple[list[Layer], RockHalfspace]:
    """Discretize a vertical Vs/zeta column into soil layers + rock halfspace."""
    if Layer is None or RockHalfspace is None:
        raise ImportError("seiskit is required for column_to_layers")
    vs_col = np.asarray(vs_col, dtype=float).ravel()
    zeta_col = np.asarray(zeta_col, dtype=float).ravel()
    if vs_col.shape != zeta_col.shape:
        raise ValueError("vs_col and zeta_col must match")
    if soil_nz is None:
        thresh = 0.5 * (float(vs_col[0]) + float(vs_rock))
        soil_mask = vs_col < thresh
        soil_nz = int(np.argmax(~soil_mask)) if not soil_mask.all() else len(vs_col)
        soil_nz = max(1, min(soil_nz, len(vs_col) - 1))
    else:
        soil_nz = max(1, min(int(soil_nz), len(vs_col)))
    layers: list[Layer] = []
    for i in range(soil_nz):
        layers.append(
            Layer(
                H=float(dz),
                Vs=max(float(vs_col[i]), _EPS),
                rho=float(rho),
                xi=max(float(zeta_col[i]), 0.0),
            )
        )
    if soil_nz < len(vs_col):
        rock_vs = max(float(vs_rock), float(vs_col[soil_nz:].mean()))
    else:
        rock_vs = float(vs_rock)
    rock = RockHalfspace(Vs=rock_vs, rho=float(rho), xi=0.0)
    return layers, rock


def _af_within_vectorized(
    freq: np.ndarray,
    H: np.ndarray,
    Vs: np.ndarray,
    xi: np.ndarray,
    *,
    vs_rock: float,
    rho: float,
) -> np.ndarray:
    """Vectorized AF_within over frequency (same matrices as seiskit theory)."""
    f = np.atleast_1d(np.asarray(freq, dtype=np.float64))
    H = np.asarray(H, dtype=np.float64).ravel()
    Vs = np.maximum(np.asarray(Vs, dtype=np.float64).ravel(), _EPS)
    xi = np.asarray(xi, dtype=np.float64).ravel()
    if not (H.size == Vs.size == xi.size):
        raise ValueError("H, Vs, xi must match")
    omega = 2.0 * np.pi * f
    af = np.ones(f.shape, dtype=np.float64)
    active = omega > 0.0
    if not np.any(active):
        return af

    u = np.ones(np.count_nonzero(active), dtype=np.complex128)
    tau = np.zeros_like(u)
    om = omega[active]
    for h, vs, x in zip(H, Vs, xi):
        vs_c = vs * np.sqrt(1.0 + 2.0j * x)
        kh = (om / vs_c) * h
        c = np.cos(kh)
        s = np.sin(kh)
        gk = 1j * om * rho * vs_c
        u_new = u * c + tau * (s / gk)
        tau_new = -u * gk * s + tau * c
        u, tau = u_new, tau_new
    af[active] = np.abs(1.0 / u)
    return af


def haskell_af_within(
    freq: np.ndarray,
    vs_col: np.ndarray,
    zeta_col: np.ndarray,
    *,
    dz: float,
    vs_rock: float,
    soil_nz: int | None = None,
    rho: float = DEFAULT_RHO,
) -> np.ndarray:
    """Return AF_within(|TF|) on ``freq`` for one column (vectorized in freq)."""
    vs_col = np.asarray(vs_col, dtype=float).ravel()
    zeta_col = np.asarray(zeta_col, dtype=float).ravel()
    if soil_nz is None:
        thresh = 0.5 * (float(vs_col[0]) + float(vs_rock))
        soil_mask = vs_col < thresh
        soil_nz = int(np.argmax(~soil_mask)) if not soil_mask.all() else len(vs_col)
        soil_nz = max(1, min(soil_nz, len(vs_col)))
    else:
        soil_nz = max(1, min(int(soil_nz), len(vs_col)))
    return _af_within_vectorized(
        freq,
        np.full(soil_nz, float(dz)),
        np.maximum(vs_col[:soil_nz], _EPS),
        np.maximum(zeta_col[:soil_nz], 0.0),
        vs_rock=vs_rock,
        rho=rho,
    )


def haskell_nominal_af_within(
    freq: np.ndarray,
    *,
    vs1: float,
    H: float,
    vs2: float,
    xi: float = 0.05,
    rho: float = DEFAULT_RHO,
) -> np.ndarray:
    """Single-layer nominal H_1D(Vs1, H, Vs2) — no spatial variability."""
    return _af_within_vectorized(
        freq,
        np.array([float(H)]),
        np.array([float(vs1)]),
        np.array([float(xi)]),
        vs_rock=float(vs2),
        rho=float(rho),
    )


def haskell_at_columns(
    freq: np.ndarray,
    vs_field: np.ndarray,
    zeta_field: np.ndarray,
    col_indices: Sequence[int],
    *,
    dz: float,
    vs_rock: float,
    soil_nz: int | None = None,
    rho: float = DEFAULT_RHO,
) -> np.ndarray:
    """AF_within for each column. Returns (n_cols, n_freq)."""
    out = np.empty((len(col_indices), len(freq)), dtype=np.float64)
    for i, c in enumerate(col_indices):
        out[i] = haskell_af_within(
            freq,
            vs_field[:, int(c)],
            zeta_field[:, int(c)],
            dz=dz,
            vs_rock=vs_rock,
            soil_nz=soil_nz,
            rho=rho,
        )
    return out
