"""Vertical-SH 1D transfer functions for Case 7 columns.

Surface/within ratios only depend on the layers above the receiver, so a
column truncated at the downhole depth gives the exact 1D counterpart of
surface / downhole. Surface/outcrop needs the half-space below the mesh,
which is the bottom-row Vs (the compliant Lysmer base).
"""

from __future__ import annotations

import numpy as np

from case7a import ELEM_M, N_ROW

RHO = 2000.0
RAYLEIGH_A0 = 0.053014376
RAYLEIGH_A1 = 0.000795775
DOWNHOLE_DEPTH_M = 120.0
_EPS = 1e-12


def rayleigh_xi(freq: np.ndarray, a0: float = RAYLEIGH_A0, a1: float = RAYLEIGH_A1) -> np.ndarray:
    """Damping ratio of Rayleigh damping C = a0 M + a1 K at each frequency."""
    omega = 2.0 * np.pi * np.maximum(np.asarray(freq, dtype=np.float64), _EPS)
    return 0.5 * (a0 / omega + a1 * omega)


def rayleigh_xi_target(freq: np.ndarray, zeta: float, f1: float, f2: float) -> np.ndarray:
    """Rayleigh damping fitted to ``zeta`` at ``f1`` and ``f2`` (seiskit ``global_avg``)."""
    w1, w2 = 2.0 * np.pi * float(f1), 2.0 * np.pi * float(f2)
    a0 = 2.0 * zeta * w1 * w2 / (w1 + w2)
    a1 = 2.0 * zeta / (w1 + w2)
    return rayleigh_xi(freq, a0, a1)


def _propagate(
    freq: np.ndarray, thickness: np.ndarray, vs: np.ndarray, xi: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """Displacement and stress at the bottom of the stack for unit surface motion.

    ``xi`` is (n_freq,) for frequency-dependent damping shared by all layers,
    or (n_layer, n_freq).
    """
    f = np.asarray(freq, dtype=np.float64)
    omega = 2.0 * np.pi * f
    xi2 = np.broadcast_to(np.atleast_2d(xi), (len(vs), f.size))
    u = np.ones(f.size, dtype=np.complex128)
    tau = np.zeros(f.size, dtype=np.complex128)
    for h, v, x in zip(thickness, vs, xi2):
        vs_c = v * np.sqrt(1.0 + 2.0j * x)
        kh = omega / vs_c * h
        c, s = np.cos(kh), np.sin(kh)
        gk = 1j * omega * rho * vs_c
        u, tau = u * c + tau * (s / gk), -u * gk * s + tau * c
    return u, tau


def surface_over_within(
    freq: np.ndarray,
    vs_col: np.ndarray,
    depth_m: float = DOWNHOLE_DEPTH_M,
    *,
    dz: float = ELEM_M,
    xi: np.ndarray | None = None,
    rho: float = RHO,
) -> np.ndarray:
    """|u(surface) / u(depth)| for a column of ``dz`` cells, surface first."""
    n = int(round(depth_m / dz))
    vs = np.asarray(vs_col, dtype=np.float64)[:n]
    xi = rayleigh_xi(freq) if xi is None else xi
    u, _ = _propagate(freq, np.full(vs.size, dz), vs, xi, rho)
    return 1.0 / np.maximum(np.abs(u), _EPS)


def surface_over_outcrop(
    freq: np.ndarray,
    vs_col: np.ndarray,
    *,
    dz: float = ELEM_M,
    vs_halfspace: float | None = None,
    xi: np.ndarray | None = None,
    rho: float = RHO,
) -> np.ndarray:
    """|u(surface) / (2 A_up)| with A_up the up-going amplitude in the half-space.

    ``_propagate``'s ``tau`` is i times the shear stress, so with gk = i w rho vs*
    the half-space amplitudes are (u -/+ i tau / gk) / 2, up-going first.
    """
    vs = np.asarray(vs_col, dtype=np.float64)
    vs_hs = float(vs[-1]) if vs_halfspace is None else float(vs_halfspace)
    xi = rayleigh_xi(freq) if xi is None else xi
    u, tau = _propagate(freq, np.full(vs.size, dz), vs, xi, rho)
    xi_hs = np.atleast_2d(xi)[-1]
    omega = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)
    gk = 1j * omega * rho * vs_hs * np.sqrt(1.0 + 2.0j * xi_hs)
    return 1.0 / np.maximum(np.abs(u - 1j * tau / gk), _EPS)


def rayleigh_coefficients(zeta: float, f1: float, f2: float) -> tuple[float, float]:
    """(a0, a1) of C = a0 M + a1 K with ``zeta`` at ``f1`` and ``f2`` (seiskit ``compute_rayleigh_coefficients``)."""
    w1, w2 = 2.0 * np.pi * float(f1), 2.0 * np.pi * float(f2)
    return 2.0 * zeta * w1 * w2 / (w1 + w2), 2.0 * zeta / (w1 + w2)


def outcrop_tf_rayleigh(
    freq: np.ndarray,
    vs_col: np.ndarray,
    a0: np.ndarray | float,
    a1: np.ndarray | float,
    *,
    dz: float = ELEM_M,
    rho: float = RHO,
) -> np.ndarray:
    """Complex u(surface) / (2 A_up) for viscous Rayleigh damping, e^{+i w t} convention.

    ``a0`` and ``a1`` are scalars or one value per cell; the last cell is also the
    half-space. Mass-proportional damping enters as a complex density and
    stiffness-proportional damping as a complex modulus, which is what
    C = a0 M + a1 K is in the frequency domain.
    """
    f = np.asarray(freq, dtype=np.float64)
    omega = 2.0 * np.pi * np.maximum(f, _EPS)
    vs = np.asarray(vs_col, dtype=np.float64)
    a0 = np.broadcast_to(np.asarray(a0, dtype=np.float64), vs.shape)
    a1 = np.broadcast_to(np.asarray(a1, dtype=np.float64), vs.shape)

    def impedance_and_k(i: int) -> tuple[np.ndarray, np.ndarray]:
        rho_c = rho * (1.0 - 1j * a0[i] / omega)
        g_c = rho * vs[i] ** 2 * (1.0 + 1j * omega * a1[i])
        return np.sqrt(g_c * rho_c), omega * np.sqrt(rho_c / g_c)

    u = np.ones(f.size, dtype=np.complex128)
    tau = np.zeros(f.size, dtype=np.complex128)
    for i in range(vs.size):
        z, k = impedance_and_k(i)
        c, s = np.cos(k * dz), np.sin(k * dz)
        u, tau = u * c + tau * s / (omega * z), -u * omega * z * s + tau * c
    z_hs, _ = impedance_and_k(vs.size - 1)
    return 1.0 / (u - 1j * tau / (omega * z_hs))


def element_column(vs_elem: np.ndarray, x_m: float) -> np.ndarray:
    """(30,) element Vs under surface node x (mean of the two adjacent columns)."""
    from case7a import N_COL, X_LEFT_M

    k = int(round((float(x_m) - X_LEFT_M) / ELEM_M))
    left = int(np.clip(k - 1, 0, N_COL - 1))
    right = int(np.clip(k, 0, N_COL - 1))
    col = 0.5 * (vs_elem[:, left] + vs_elem[:, right])
    if col.shape != (N_ROW,):
        raise ValueError(f"column shape {col.shape}")
    return col


def section_tf_1d(
    freq: np.ndarray, vs_elem: np.ndarray, x_m: np.ndarray, kind: str
) -> np.ndarray:
    """1D |TF| under each x. ``kind`` is 'within120' or 'outcrop'. Shape (n_x, n_freq)."""
    out = np.empty((len(x_m), len(freq)), dtype=np.float64)
    for i, x in enumerate(x_m):
        col = element_column(vs_elem, float(x))
        if kind == "within120":
            out[i] = surface_over_within(freq, col)
        elif kind == "outcrop":
            out[i] = surface_over_outcrop(freq, col)
        else:
            raise ValueError(f"Unknown kind {kind!r}")
    return out
