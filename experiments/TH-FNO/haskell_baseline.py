"""Thomson–Haskell local-column |TF| baseline (seiskit theory)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

_SEISKIT_ROOT = Path.home() / "seiskit"
if _SEISKIT_ROOT.is_dir() and str(_SEISKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEISKIT_ROOT))

from seiskit.theory import Layer, RockHalfspace, layered_transfer_function  # noqa: E402

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
    """Discretize a vertical Vs/zeta column into soil layers + rock halfspace.

    Soil rows are ``0 .. soil_nz-1`` (surface at row 0). If ``soil_nz`` is None,
    rows with Vs < 0.5*(median_soil_guess + vs_rock) are treated as soil via a
    simple threshold at ``0.5 * (vs_col[0] + vs_rock)``.
    """
    vs_col = np.asarray(vs_col, dtype=float).ravel()
    zeta_col = np.asarray(zeta_col, dtype=float).ravel()
    if vs_col.shape != zeta_col.shape:
        raise ValueError("vs_col and zeta_col must match")
    if soil_nz is None:
        thresh = 0.5 * (float(vs_col[0]) + float(vs_rock))
        soil_mask = vs_col < thresh
        # Contiguous from surface
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


def haskell_trend_af_within(
    freq: np.ndarray,
    *,
    vs1: float,
    H: float,
    vs2: float,
    xi: float = 0.05,
    rho: float = DEFAULT_RHO,
) -> np.ndarray:
    """Single-layer H_1D(trend) — uncalibrated analytic (attrs)."""
    return _af_within_vectorized(
        freq,
        np.array([float(H)]),
        np.array([float(vs1)]),
        np.array([float(xi)]),
        vs_rock=float(vs2),
        rho=float(rho),
    )


def _resolve_trend_freq_scale(freq_scale: float | None) -> float:
    if freq_scale is not None:
        return float(freq_scale)
    try:
        import config as _cfg  # noqa: WPS433

        return float(getattr(_cfg, "TREND_FREQ_SCALE", 1.0))
    except Exception:
        return 1.0


def H_1D_trend(
    freq: np.ndarray,
    *,
    vs1: float,
    H: float,
    vs2: float,
    xi: float = 0.05,
    rho: float = DEFAULT_RHO,
    freq_scale: float | None = None,
) -> np.ndarray:
    """Training baseline: 2-layer analytic with optional resonance calibration.

    ``freq_scale`` ≈ median(f_truth / f_uncalibrated). Resonance f₀ ∝ 1/H, so we
    use ``H_eff = H / freq_scale`` to shift peaks onto OpenSees (fixes ~6% bias
    seen on RV pancake). Scale 1.0 = uncalibrated attrs baseline.
    """
    scale = _resolve_trend_freq_scale(freq_scale)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    H_eff = float(H) / scale
    return haskell_trend_af_within(
        freq, vs1=vs1, H=H_eff, vs2=vs2, xi=xi, rho=rho
    )


def mean_trend_column(
    vs_field: np.ndarray,
    zeta_field: np.ndarray,
    *,
    x_slice: slice | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth-averaged (lateral mean) trend columns — true mean profile.

    Prefer this over {Vs1, H, Vs2} attrs when the field has depth structure
    beyond a single soil layer (see diagnostics baseline-fix).
    """
    vs = np.asarray(vs_field, dtype=float)
    zeta = np.asarray(zeta_field, dtype=float)
    if x_slice is not None:
        vs = vs[:, x_slice]
        zeta = zeta[:, x_slice]
    return vs.mean(axis=1), zeta.mean(axis=1)


def haskell_trend_from_mean_profile(
    freq: np.ndarray,
    vs_field: np.ndarray,
    zeta_field: np.ndarray,
    *,
    dz: float,
    vs_rock: float,
    soil_nz: int | None = None,
    x_slice: slice | None = None,
    rho: float = DEFAULT_RHO,
    xi_override: float | None = None,
) -> np.ndarray:
    """H_1D from lateral-mean Vs/ζ profile (fitted ζ = mean soil ζ unless overridden)."""
    vs_col, zeta_col = mean_trend_column(vs_field, zeta_field, x_slice=x_slice)
    if xi_override is not None:
        zeta_col = np.full_like(zeta_col, float(xi_override))
    return haskell_af_within(
        freq,
        vs_col,
        zeta_col,
        dz=dz,
        vs_rock=vs_rock,
        soil_nz=soil_nz,
        rho=rho,
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
    """AF_within for each column. Returns (n_cols, n_freq). D3 / diagnostics only."""
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


def haskell_realization_geomean(
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
    """Geometric mean of column-wise realization Haskell — D3 opponent only."""
    stack = haskell_at_columns(
        freq,
        vs_field,
        zeta_field,
        col_indices,
        dz=dz,
        vs_rock=vs_rock,
        soil_nz=soil_nz,
        rho=rho,
    )
    return np.exp(np.mean(np.log(np.maximum(stack, _EPS)), axis=0))


def scatter_recorder_tf(
    tf_rec: np.ndarray,
    recorder_x: np.ndarray,
    nx: int,
) -> np.ndarray:
    """(R, F) or (F,) -> (Nx, F) with zeros off-recorder."""
    tf_rec = np.asarray(tf_rec, dtype=np.float32)
    out = np.zeros((nx, tf_rec.shape[-1]), dtype=np.float32)
    if tf_rec.ndim == 1:
        c = int(recorder_x[len(recorder_x) // 2])
        out[c] = tf_rec
        return out
    for r, x in enumerate(recorder_x):
        if 0 <= int(x) < nx:
            out[int(x)] = tf_rec[r]
    return out
