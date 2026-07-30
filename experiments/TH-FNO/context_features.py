"""Geometric / RF context features for gated delta residual (no K/M, no KL)."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def bedrock_interface_depth(
    vs_field: np.ndarray,
    *,
    vs_rock: float,
    dz: float = 1.0,
) -> np.ndarray:
    """Per-column soil thickness Z_bedrock(x) [m] from a Vs threshold."""
    thresh = 0.5 * (float(np.median(vs_field[0])) + float(vs_rock))
    nz, nx = vs_field.shape
    z_bed = np.empty(nx, dtype=np.float64)
    for j in range(nx):
        col = vs_field[:, j]
        below = np.where(col >= thresh)[0]
        i0 = int(below[0]) if below.size else nz - 1
        z_bed[j] = i0 * dz
    return z_bed


def interface_dip(z_bedrock: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """dZ/dx at each column (central differences)."""
    z = np.asarray(z_bedrock, dtype=np.float64)
    dip = np.gradient(z, dx)
    return dip.astype(np.float32)


def impedance_horizontal_gradient(
    vs_field: np.ndarray,
    *,
    rho: float = 2000.0,
    dx: float = 1.0,
    row: int | None = None,
) -> np.ndarray:
    """Horizontal gradient of impedance rho*Vs.

    If ``row`` is None, use depth-mean impedance per column; else that row.
    Returns (nx,) float32.
    """
    vs = np.asarray(vs_field, dtype=np.float64)
    if row is None:
        imp = rho * vs.mean(axis=0)
    else:
        imp = rho * vs[int(row)]
    return np.gradient(imp, dx).astype(np.float32)


def impedance_gradient_field(
    vs_field: np.ndarray,
    *,
    rho: float = 2000.0,
    dx: float = 1.0,
) -> np.ndarray:
    """Full (nz, nx) field of d(rho Vs)/dx along x."""
    vs = np.asarray(vs_field, dtype=np.float64)
    imp = rho * vs
    return np.gradient(imp, dx, axis=1).astype(np.float32)


def dip_field_broadcast(
    dip_1d: np.ndarray,
    nz: int,
) -> np.ndarray:
    """Broadcast (nx,) dip to (nz, nx)."""
    return np.broadcast_to(dip_1d.astype(np.float32), (nz, dip_1d.shape[0])).copy()


def residual_gate_scalar(
    cov: float,
    dip_rms: float,
    *,
    cov_ref: float = 0.1,
    dip_ref: float = 0.05,
) -> float:
    """Exact zero when cov=0 and dip_rms=0; →1 for large variability/dip."""
    c = float(cov) / max(cov_ref, _EPS)
    d = float(dip_rms) / max(dip_ref, _EPS)
    return float(1.0 - np.exp(-(c * c + d * d)))


def residual_gate_torch(
    cov: "torch.Tensor",
    dip_rms: "torch.Tensor",
    *,
    cov_ref: float = 0.1,
    dip_ref: float = 0.05,
) -> "torch.Tensor":
    import torch

    c = cov / cov_ref
    d = dip_rms / dip_ref
    return 1.0 - torch.exp(-(c * c + d * d))


def stack_delta_input_channels(
    vs: np.ndarray,
    zeta: np.ndarray,
    x_coord: np.ndarray,
    z_coord: np.ndarray,
    dip_2d: np.ndarray,
    imp_grad_2d: np.ndarray,
) -> np.ndarray:
    """(C=6, Nz, Nx): Vs, zeta, x, z, dip, d(rho Vs)/dx."""
    return np.stack(
        [
            vs.astype(np.float32),
            zeta.astype(np.float32),
            x_coord.astype(np.float32),
            z_coord.astype(np.float32),
            dip_2d.astype(np.float32),
            imp_grad_2d.astype(np.float32),
        ],
        axis=0,
    )
