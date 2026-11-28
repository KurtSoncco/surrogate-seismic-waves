"""Absolute, dimension-aware input channels (DimINO / DimOL style).

The legacy encoding divides Vs by the surface value, zeta by its maximum and
depth by the mesh thickness, so two sites with different stiffness, damping or
thickness can map to the same input. Here every channel keeps physical units
(log of a fixed reference) and two per-column dimensionless groups are added:

0. log(Vs / VS_REF)
1. log(zeta / ZETA_REF)
2. mesh mask (1 inside the mesh, 0 in the half-space extension)
3. x / X_REF (physical metres, not x / Lx — a 250 m and a 500 m window
   must not look the same)
4. depth / DEPTH_REF (metres, not normalized by the column height)
5. log(f0 / F0_REF), f0 = 1 / (4 * sum(dz / Vs)) over the mesh column, the
   travel-time form of Vs1 / (4 H) that stays defined for gradational soil
6. log(Vs_halfspace / Vs_time_avg) of the column (impedance contrast)
7. log(Lx / X_REF) of this window (broadcast)
8. log(H / DEPTH_REF) of the active mesh (broadcast)

Channels 5-8 carry the dimensionless groups for DimNorm conditioning; with
DIMNORM_FILM the model reads them there and does not lift them as features.

Rows below the mesh repeat the bottom row, which is the half-space.

With ``n_prior`` > 0, channels 7.. hold log |TF_1D| (vertical-SH surface /
outcrop of each column, half-space = bottom row) at ``n_prior`` log-spaced
frequencies over PRIOR_BAND_HZ, broadcast over depth. Damping there is the
frequency-independent zeta field, so the channels need no Rayleigh metadata.
"""

from __future__ import annotations

import numpy as np

VS_REF_MPS = 400.0
ZETA_REF = 0.02
DEPTH_REF_M = 150.0
X_REF_M = 500.0
F0_REF_HZ = 1.0
N_ABS_CHANNELS = 9
CH_MASK = 2
CH_X = 3
CH_LOG_ZETA = 1
CH_LOG_F0 = 5
CH_LOG_IMPEDANCE = 6
CH_LOG_LX = 7
CH_LOG_H = 8
PRIOR_BAND_HZ = (0.1, 10.0)
RHO = 2000.0
_EPS = 1e-12


def prior_frequencies(n_prior: int) -> np.ndarray:
    return np.geomspace(PRIOR_BAND_HZ[0], PRIOR_BAND_HZ[1], int(n_prior))


def column_tf1d_rayleigh(
    vs_mesh: np.ndarray,
    zeta_mesh: np.ndarray,
    dz: float,
    freq: np.ndarray,
    f2_hz: float = 10.0,
) -> np.ndarray:
    """|surface / outcrop| with seiskit Rayleigh damping. Returns (n_freq, nx).

    Each layer is fit to its own zeta at the column quarter-wavelength frequency
    and at ``f2_hz``, matching ``relabel_outcrop.py`` (second target is 10 Hz).
    """
    vs = np.maximum(np.asarray(vs_mesh, dtype=np.float64), _EPS)
    zeta = np.maximum(np.asarray(zeta_mesh, dtype=np.float64), 0.0)
    freq = np.asarray(freq, dtype=np.float64)
    f0, _ = column_groups(vs, dz)
    w1 = 2.0 * np.pi * f0[None, :]
    w2 = 2.0 * np.pi * float(f2_hz)
    a0 = 2.0 * zeta * w1 * w2 / (w1 + w2)
    a1 = 2.0 * zeta / (w1 + w2)
    omega = 2.0 * np.pi * np.maximum(freq, _EPS)
    xi = 0.5 * (
        a0[None, :, :] / omega[:, None, None] + a1[None, :, :] * omega[:, None, None]
    )
    u = np.ones((freq.size, vs.shape[1]), dtype=np.complex128)
    tau = np.zeros_like(u)
    for i in range(vs.shape[0]):
        vs_c = vs[i][None, :] * np.sqrt(1.0 + 2.0j * xi[:, i, :])
        kh = omega[:, None] / vs_c * dz
        c, s = np.cos(kh), np.sin(kh)
        gk = 1j * omega[:, None] * RHO * vs_c
        u, tau = u * c + tau * (s / gk), -u * gk * s + tau * c
    gk_hs = 1j * omega[:, None] * RHO * vs[-1][None, :] * np.sqrt(1.0 + 2.0j * xi[:, -1, :])
    return 1.0 / np.maximum(np.abs(u - 1j * tau / gk_hs), _EPS)


def column_tf1d_outcrop(vs_mesh: np.ndarray, zeta_mesh: np.ndarray, dz: float, freq: np.ndarray) -> np.ndarray:
    """|surface / outcrop| for every column at once. Returns (n_freq, nx)."""
    vs = np.maximum(np.asarray(vs_mesh, dtype=np.float64), _EPS)
    xi = np.maximum(np.asarray(zeta_mesh, dtype=np.float64), 0.0)
    omega = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)[:, None]
    u = np.ones((omega.shape[0], vs.shape[1]), dtype=np.complex128)
    tau = np.zeros_like(u)
    for v, x in zip(vs, xi):
        vs_c = v[None, :] * np.sqrt(1.0 + 2.0j * x[None, :])
        kh = omega / vs_c * dz
        c, s = np.cos(kh), np.sin(kh)
        gk = 1j * omega * RHO * vs_c
        u, tau = u * c + tau * (s / gk), -u * gk * s + tau * c
    gk_hs = 1j * omega * RHO * vs[-1][None, :] * np.sqrt(1.0 + 2.0j * xi[-1][None, :])
    return 1.0 / np.maximum(np.abs(u - 1j * tau / gk_hs), _EPS)


def pad_depth_halfspace(arr: np.ndarray, nz_max: int) -> np.ndarray:
    """Truncate or extend to ``nz_max`` rows by repeating the bottom row."""
    a = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    nz = a.shape[0]
    if nz >= nz_max:
        return a[:nz_max]
    return np.concatenate([a, np.repeat(a[-1:], nz_max - nz, axis=0)], axis=0)


def column_groups(vs_mesh: np.ndarray, dz: float) -> tuple[np.ndarray, np.ndarray]:
    """(f0, impedance contrast) per column of an (n_active, nx) mesh Vs field."""
    vs = np.maximum(np.asarray(vs_mesh, dtype=np.float64), _EPS)
    travel = np.sum(dz / vs, axis=0)
    f0 = 1.0 / (4.0 * np.maximum(travel, _EPS))
    vs_avg = vs.shape[0] * dz / np.maximum(travel, _EPS)
    return f0, vs[-1] / vs_avg


def encode_absolute(
    vs: np.ndarray,
    zeta: np.ndarray,
    n_active: int,
    nz_max: int,
    *,
    dz: float = 1.0,
    dx: float = 1.0,
    lx: float | None = None,
    n_prior: int = 0,
    prior_mode: str = "constant",
    prior_log: np.ndarray | None = None,
) -> np.ndarray:
    """(N_ABS_CHANNELS + n_prior, nz_max, nx) from mesh fields, surface at row 0.

    ``vs`` / ``zeta`` hold at least ``n_active`` mesh rows; anything after is
    replaced by the half-space extension.
    """
    vs = np.asarray(vs, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    n_active = int(min(n_active, vs.shape[0], nz_max))
    nx = vs.shape[1]
    vs_p = pad_depth_halfspace(vs[:n_active], nz_max).astype(np.float64)
    zeta_p = pad_depth_halfspace(zeta[:n_active], nz_max).astype(np.float64)
    f0, imp = column_groups(vs[:n_active], dz)

    mask = np.zeros((nz_max, nx))
    mask[:n_active] = 1.0
    lx = float(nx * dx) if lx is None else float(lx)
    h_m = float(n_active * dz)
    x = np.broadcast_to((np.arange(nx) * dx / X_REF_M)[None, :], (nz_max, nx))
    z = np.broadcast_to((np.arange(nz_max) * dz / DEPTH_REF_M)[:, None], (nz_max, nx))
    chans = [
        np.log(np.maximum(vs_p, _EPS) / VS_REF_MPS),
        np.log(np.maximum(zeta_p, 1e-6) / ZETA_REF),
        mask,
        x,
        z,
        np.broadcast_to(np.log(f0 / F0_REF_HZ)[None, :], (nz_max, nx)),
        np.broadcast_to(np.log(np.maximum(imp, _EPS))[None, :], (nz_max, nx)),
        np.full((nz_max, nx), np.log(max(lx, _EPS) / X_REF_M)),
        np.full((nz_max, nx), np.log(max(h_m, _EPS) / DEPTH_REF_M)),
    ]
    if n_prior > 0:
        if prior_log is None:
            if prior_mode == "constant":
                tf1d = column_tf1d_outcrop(vs[:n_active], zeta[:n_active], dz, prior_frequencies(n_prior))
            elif prior_mode == "rayleigh":
                tf1d = column_tf1d_rayleigh(vs[:n_active], zeta[:n_active], dz, prior_frequencies(n_prior))
            else:
                raise ValueError(f"prior_mode must be 'constant' or 'rayleigh', got {prior_mode!r}")
            prior_log = np.log(np.maximum(tf1d, _EPS))
        else:
            prior_log = np.asarray(prior_log, dtype=np.float64)
            if prior_log.shape != (n_prior, nx):
                raise ValueError(f"prior_log {prior_log.shape} != {(n_prior, nx)}")
        chans.extend(np.broadcast_to(row[None, :], (nz_max, nx)) for row in prior_log)
    return np.stack(chans, axis=0).astype(np.float32)


def pad_lateral(encoded: np.ndarray, nx_max: int, mask_channel: int = CH_MASK) -> np.ndarray:
    """Right-pad or crop a (C, nz, nx) window onto ``nx_max`` columns.

    Padded columns are zeros (mask=0) so LOGLO can batch different Lx. Crop
    keeps the leading columns (window origin stays at x=0).
    """
    arr = np.asarray(encoded, dtype=np.float32)
    nx = arr.shape[-1]
    if nx == nx_max:
        return arr
    if nx > nx_max:
        return np.ascontiguousarray(arr[..., :nx_max])
    out = np.zeros(arr.shape[:-1] + (nx_max,), dtype=np.float32)
    out[..., :nx] = arr
    if 0 <= mask_channel < out.shape[0]:
        out[mask_channel, :, nx:] = 0.0
    return out
