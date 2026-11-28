"""Variable-width windows onto a fixed network grid, two ways.

``metric``: a canvas of NZ_METRIC x NX_METRIC cells at 1 m. The interior starts
at column CONTEXT_M. Columns outside it repeat the edge column (seiskit
``_extend_profile`` rule), 1 km on the left and the rest of the canvas on the
right, and a smoothed lateral mask marks the interior (DAFNO). A column is the
same metre for every width, so Fourier mode k is always the same wavelength.
log(f1) and log(f2) are constant channels for the Rayleigh pair.

``reference``: the window is resampled onto NZ_REF x NX_REF cells in x / W and
z / Lz (Geo-FNO for rectangles). Every width fills the grid; the network only
tells widths apart through log(W) and log(Lz), which it reads through FiLM.

Both start from ``GIFNO/absolute_inputs.encode_absolute`` (physical units) and
append one channel, CH_LATERAL, the lateral window mask.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GIFNO_DIR = Path(__file__).resolve().parents[1] / "GIFNO"
if str(GIFNO_DIR) not in sys.path:
    sys.path.append(str(GIFNO_DIR))

from absolute_inputs import (  # noqa: E402
    CH_LOG_F0,
    CH_LOG_H,
    CH_LOG_IMPEDANCE,
    CH_LOG_LX,
    CH_LOG_ZETA,
    CH_MASK,
    CH_X,
    DEPTH_REF_M,
    F0_REF_HZ,
    N_ABS_CHANNELS,
    VS_REF_MPS,
    X_REF_M,
    ZETA_REF,
    column_groups,
    encode_absolute,
)

CH_Z = 4
CH_LATERAL = N_ABS_CHANNELS
CH_LOG_F1 = N_ABS_CHANNELS + 1
CH_LOG_F2 = N_ABS_CHANNELS + 2
N_CHANNELS = N_ABS_CHANNELS + 3
# Site-level groups read through FiLM. f1 and f2 are constant, so they are not lifted.
# log(W) is not a FiLM input: width is the support of the lateral mask. A scalar
# code for W would let the network memorize the four solved lengths.
FILM_CHANNELS = (CH_LOG_F0, CH_LOG_IMPEDANCE, CH_LOG_ZETA, CH_LOG_H, CH_LOG_F1, CH_LOG_F2)
FILM_ONLY = (CH_LOG_F0, CH_LOG_IMPEDANCE, CH_LOG_LX, CH_LOG_H, CH_LOG_F1, CH_LOG_F2)

W_MIN_M = 500
W_MAX_M = 2000
CONTEXT_M = 1000  # repeated-edge context on each side of the interior
NZ_METRIC = 256
NX_METRIC = W_MAX_M + 2 * CONTEXT_M
F_REF_HZ = 1.0
NZ_REF = 128
NX_REF = 500
MASK_BETA = 0.1  # 1/m: 0.5 +- 0.025 in the cells at the edge, within 0.005 of 0 or 1 by 30 m
ENCODINGS = ("metric", "reference")


def grid_shape(encoding: str) -> tuple[int, int]:
    if encoding == "metric":
        return NZ_METRIC, NX_METRIC
    if encoding == "reference":
        return NZ_REF, NX_REF
    raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")


def smoothed_lateral_mask(
    width: int, nx: int, beta: float = MASK_BETA, dx: float = 1.0, origin: int = 0
) -> np.ndarray:
    """0.5 + (chi - 0.5) tanh(beta * d), d = distance of each cell center to a window edge (m)."""
    x = (np.arange(nx) + 0.5) * dx
    left, right = origin * dx, (origin + width) * dx
    chi = ((x >= left) & (x < right)).astype(np.float64)
    d = np.minimum(np.abs(x - left), np.abs(x - right))
    return 0.5 + (chi - 0.5) * np.tanh(beta * d)


def _freq_pair(nz: int, nx: int, f1_hz: float, f2_hz: float) -> np.ndarray:
    return np.stack(
        [
            np.full((nz, nx), np.log(max(float(f1_hz), 1e-6) / F_REF_HZ), np.float32),
            np.full((nz, nx), np.log(max(float(f2_hz), 1e-6) / F_REF_HZ), np.float32),
        ]
    )


_LATERAL_ROW: dict[tuple[int, float], np.ndarray] = {}


def _lateral_row(width: int, dx: float) -> np.ndarray:
    key = (int(width), float(dx))
    row = _LATERAL_ROW.get(key)
    if row is None:
        row = smoothed_lateral_mask(int(width), NX_METRIC, dx=dx, origin=CONTEXT_M).astype(np.float32)
        _LATERAL_ROW[key] = np.ascontiguousarray(row)
    return row


def _edge_pad_rows(mesh: np.ndarray, left: int, right: int) -> np.ndarray:
    """Repeat the edge column. Same rule as ``np.pad(..., mode='edge')`` on axis 1."""
    out = np.empty((mesh.shape[0], left + mesh.shape[1] + right), dtype=mesh.dtype)
    out[:, :left] = mesh[:, :1]
    out[:, left : left + mesh.shape[1]] = mesh
    out[:, left + mesh.shape[1] :] = mesh[:, -1:]
    return out


def encode_metric(
    vs: np.ndarray, zeta: np.ndarray, n_active: int, *, dz: float = 1.0, dx: float = 1.0, f1_hz: float = 1.0, f2_hz: float = 3.0
) -> np.ndarray:
    """(N_CHANNELS, NZ_METRIC, NX_METRIC). The interior starts at column CONTEXT_M.

    Columns outside the interior repeat the edge column, the same extension the
    OpenSees pad uses, out to 1 km on the left and whatever remains on the right.
    Logs are written once per mesh row into one float32 canvas. Half-space rows
    repeat the bottom mesh row, which is what ``encode_absolute`` does by padding
    first and then taking the log.
    """
    width = int(vs.shape[1])
    if width + 2 * CONTEXT_M > NX_METRIC:
        raise ValueError(f"window {width} m plus {CONTEXT_M} m context exceeds canvas {NX_METRIC}")
    n_active = int(min(n_active, vs.shape[0], NZ_METRIC))
    if n_active < 1:
        raise ValueError(f"n_active must be positive, got {n_active}")
    left = CONTEXT_M
    right = NX_METRIC - CONTEXT_M - width
    vs_m = np.asarray(vs[:n_active], dtype=np.float64)
    ze_m = np.asarray(zeta[:n_active], dtype=np.float64)
    f0, imp = column_groups(vs_m, dz)
    vs32 = np.nan_to_num(vs_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ze32 = np.nan_to_num(ze_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    log_vs = np.log(np.maximum(vs32.astype(np.float64), 1e-12) / VS_REF_MPS).astype(np.float32)
    log_ze = np.log(np.maximum(ze32.astype(np.float64), 1e-6) / ZETA_REF).astype(np.float32)
    log_vs = _edge_pad_rows(log_vs, left, right)
    log_ze = _edge_pad_rows(log_ze, left, right)
    f0 = np.pad(f0, (left, right), mode="edge")
    imp = np.pad(imp, (left, right), mode="edge")

    out = np.empty((N_CHANNELS, NZ_METRIC, NX_METRIC), dtype=np.float32)
    out[0, :n_active] = log_vs
    out[1, :n_active] = log_ze
    if n_active < NZ_METRIC:
        out[0, n_active:] = log_vs[-1]
        out[1, n_active:] = log_ze[-1]
    out[CH_MASK] = 0.0
    out[CH_MASK, :n_active] = 1.0
    out[CH_X] = (np.arange(NX_METRIC, dtype=np.float64) * dx / X_REF_M).astype(np.float32)
    out[CH_Z] = (np.arange(NZ_METRIC, dtype=np.float64) * dz / DEPTH_REF_M).astype(np.float32)[:, None]
    out[CH_LOG_F0] = np.log(f0 / F0_REF_HZ).astype(np.float32)
    out[CH_LOG_IMPEDANCE] = np.log(np.maximum(imp, 1e-12)).astype(np.float32)
    out[CH_LOG_LX] = np.float32(np.log(max(width * dx, 1e-12) / X_REF_M))
    out[CH_LOG_H] = np.float32(np.log(max(n_active * dz, 1e-12) / DEPTH_REF_M))
    out[CH_LATERAL] = _lateral_row(width, dx)
    out[CH_LOG_F1] = np.float32(np.log(max(float(f1_hz), 1e-6) / F_REF_HZ))
    out[CH_LOG_F2] = np.float32(np.log(max(float(f2_hz), 1e-6) / F_REF_HZ))
    return out


def _resample(arr: np.ndarray, nz: int, nx: int) -> np.ndarray:
    """Bilinear (cell-centred) resample of (C, h, w) to (C, nz, nx)."""
    c, h, w = arr.shape
    zi = np.clip((np.arange(nz) + 0.5) * h / nz - 0.5, 0, h - 1)
    xi = np.clip((np.arange(nx) + 0.5) * w / nx - 0.5, 0, w - 1)
    z0 = np.floor(zi).astype(int)
    x0 = np.floor(xi).astype(int)
    z1 = np.minimum(z0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    tz = (zi - z0)[None, :, None]
    tx = (xi - x0)[None, None, :]
    top = arr[:, z0][:, :, x0] * (1 - tx) + arr[:, z0][:, :, x1] * tx
    bot = arr[:, z1][:, :, x0] * (1 - tx) + arr[:, z1][:, :, x1] * tx
    return top * (1 - tz) + bot * tz


def encode_reference(
    vs: np.ndarray, zeta: np.ndarray, n_active: int, *, dz: float = 1.0, dx: float = 1.0, f1_hz: float = 1.0, f2_hz: float = 3.0
) -> np.ndarray:
    """(N_CHANNELS, NZ_REF, NX_REF): the active mesh of the window stretched onto the grid."""
    width = vs.shape[1]
    n_active = int(min(n_active, vs.shape[0]))
    enc = encode_absolute(vs[:n_active], zeta[:n_active], n_active, n_active, dz=dz, dx=dx, lx=width * dx)
    out = _resample(enc.astype(np.float64), NZ_REF, NX_REF)
    out[CH_X] = ((np.arange(NX_REF) + 0.5) / NX_REF)[None, :]
    out[CH_Z] = ((np.arange(NZ_REF) + 0.5) / NZ_REF)[:, None]
    out[CH_MASK] = 1.0
    lateral = np.ones((1, NZ_REF, NX_REF), dtype=np.float32)
    freq = _freq_pair(NZ_REF, NX_REF, f1_hz, f2_hz)
    return np.concatenate([out, lateral, freq], axis=0).astype(np.float32)


def encode(
    encoding: str,
    vs: np.ndarray,
    zeta: np.ndarray,
    n_active: int,
    *,
    dz: float = 1.0,
    dx: float = 1.0,
    f1_hz: float = 1.0,
    f2_hz: float = 3.0,
) -> np.ndarray:
    if encoding == "metric":
        return encode_metric(vs, zeta, n_active, dz=dz, dx=dx, f1_hz=f1_hz, f2_hz=f2_hz)
    if encoding == "reference":
        return encode_reference(vs, zeta, n_active, dz=dz, dx=dx, f1_hz=f1_hz, f2_hz=f2_hz)
    raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")


def query_columns(encoding: str, x_local_m: np.ndarray, width: int, dx: float = 1.0) -> np.ndarray:
    """Float canvas coordinate of each station. Metric x is metres from the canvas origin.

    The interior starts at column ``CONTEXT_M``. Reference x is the normalized
    column on the stretched grid; that encoding is not the trained model.
    """
    x = np.asarray(x_local_m, dtype=np.float64)
    if encoding == "metric":
        return x / dx + CONTEXT_M
    if encoding == "reference":
        return x / (width * dx) * NX_REF
    raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")


def station_columns(encoding: str, x_local_m: np.ndarray, width: int, dx: float = 1.0) -> np.ndarray:
    """Nearest grid column of each station. Prefer ``query_columns`` for the readout."""
    q = query_columns(encoding, x_local_m, width, dx)
    if encoding == "metric":
        return np.clip(np.round(q), 0, NX_METRIC - 1).astype(np.int64)
    return np.clip(np.floor(q), 0, NX_REF - 1).astype(np.int64)
