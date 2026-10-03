"""Uniform depth along a dipping soil–bedrock interface.

The interface is a straight line of horizontal span ``L_h`` at dip angle
``theta``. Depth is uniform along that line, so

    L = L_h / cos(theta)
    sigma_L = L / sqrt(12)                         continuous
    sigma_L = L * sqrt((N+1) / (12*(N-1)))         N equally spaced points
    sigma_y = sigma_L * |sin(theta)|

Bedrock velocity is not a function of this depth.
"""

from __future__ import annotations

import numpy as np

DIP_SPAN_M = 500.0
N_STRIP = 500
N_DEPTHS = 64
H_MIN_M = 1.0


def along_line_length(L_h: float, theta_deg: float) -> float:
    """Arc length of a straight dip spanning horizontal distance ``L_h``."""
    theta = np.deg2rad(float(theta_deg))
    c = float(np.cos(theta))
    if abs(c) < 1e-8:
        raise ValueError(f"dip angle {theta_deg} is too close to vertical")
    return float(L_h) / c


def sigma_L(L: float, n: int | None = None) -> float:
    """Std of a uniform coordinate along a segment of length ``L``.

    ``n is None`` is the continuous uniform. ``n >= 2`` is the population std
    of ``n`` equally spaced points that include both endpoints.
    """
    length = float(L)
    if n is None:
        return length / np.sqrt(12.0)
    n = int(n)
    if n < 2:
        raise ValueError(f"discrete sigma_L needs n >= 2, got {n}")
    return length * np.sqrt((n + 1) / (12.0 * (n - 1)))


def sigma_y(L_h: float, theta_deg: float, n: int | None = None) -> float:
    """Vertical std of depth along the dipping line."""
    theta = np.deg2rad(float(theta_deg))
    return sigma_L(along_line_length(L_h, theta_deg), n) * abs(float(np.sin(theta)))


def depth_range(L_h: float, theta_deg: float) -> float:
    """Vertical span ``L |sin theta|`` of the dipping line."""
    theta = np.deg2rad(float(theta_deg))
    return along_line_length(L_h, theta_deg) * abs(float(np.sin(theta)))


def uniform_dip_depths(
    H: float,
    theta_deg: float,
    L_h: float = DIP_SPAN_M,
    n: int = N_DEPTHS,
    h_min: float = H_MIN_M,
) -> np.ndarray:
    """``n`` depths with equal spacing in ``y``, centered on ``H``.

    The endpoints sit at ``H ± (1/2) L |sin theta|``. Values below ``h_min``
    are clipped; ``V_s2`` is not involved.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"need at least 2 depths, got {n}")
    half = 0.5 * depth_range(L_h, theta_deg)
    depths = np.linspace(float(H) - half, float(H) + half, n, dtype=np.float64)
    return np.maximum(depths, float(h_min))


def interface_depths(
    vs_field: np.ndarray, vs2: float, *, dz: float = 1.0
) -> np.ndarray:
    """Soil–bedrock interface depth (m) on each finite column of ``vs_field``.

    ``vs_field`` is ``(nz, nx)``. The interface is the first depth where
    ``Vs`` exceeds half the bedrock velocity.
    """
    field = np.asarray(vs_field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"vs_field must be (nz, nx), got {field.shape}")
    finite = np.isfinite(field).all(axis=1)
    field = field[finite]
    if field.size == 0:
        return np.array([], dtype=np.float64)
    thr = 0.5 * float(vs2)
    hit = field > thr
    idx = np.argmax(hit, axis=0)
    valid = hit.any(axis=0) & (idx > 0)
    return idx[valid].astype(np.float64) * float(dz)


def empirical_sigma_y(vs_field: np.ndarray, vs2: float, *, dz: float = 1.0) -> float:
    """Population std of interface depth across the strip."""
    depths = interface_depths(vs_field, vs2, dz=dz)
    if depths.size < 2:
        return float("nan")
    return float(np.std(depths, ddof=0))


def uniform_spectrum(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-weight geomean and pointwise min/max of ``(n_depth, n_freq)`` |TF|."""
    rows = np.asarray(stack, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"stack must be (n_depth, n_freq), got {rows.shape}")
    geo = np.exp(np.mean(np.log(np.clip(rows, 1e-12, None)), axis=0))
    return geo, np.min(rows, axis=0), np.max(rows, axis=0)
