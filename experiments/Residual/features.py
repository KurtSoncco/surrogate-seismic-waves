"""OrbitAll-style geometric + physics features (no raw pixel coordinates)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

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
    return np.gradient(np.asarray(z_bedrock, dtype=np.float64), dx).astype(np.float32)


def impedance_horizontal_gradient(
    vs_field: np.ndarray,
    *,
    rho: float = 2000.0,
    dx: float = 1.0,
    soil_nz: int | None = None,
) -> np.ndarray:
    """Horizontal gradient of depth-mean soil impedance rho*Vs. Returns (nx,)."""
    vs = np.asarray(vs_field, dtype=np.float64)
    if soil_nz is None:
        imp = rho * vs.mean(axis=0)
    else:
        soil_nz = max(1, min(int(soil_nz), vs.shape[0]))
        imp = rho * vs[:soil_nz].mean(axis=0)
    return np.gradient(imp, dx).astype(np.float32)


def distance_to_edge(
    imp_grad: np.ndarray,
    *,
    x_coords: np.ndarray,
    percentile: float = 90.0,
) -> np.ndarray:
    """Lateral distance [m] to nearest major impedance contrast or domain edge."""
    g = np.asarray(imp_grad, dtype=np.float64)
    x = np.asarray(x_coords, dtype=np.float64)
    thr = np.percentile(np.abs(g), percentile)
    contrast = np.where(np.abs(g) >= thr)[0]
    edges = np.array([0, len(g) - 1], dtype=int)
    anchors = np.unique(np.concatenate([contrast, edges])) if contrast.size else edges
    anchor_x = x[anchors]
    dist = np.empty(len(x), dtype=np.float32)
    for i, xi in enumerate(x):
        dist[i] = float(np.min(np.abs(anchor_x - xi)))
    return dist


def exponential_psd(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    rH: float,
    aHV: float,
) -> np.ndarray:
    """PSD of the exponential covariance used in generate_gaussian_field_fft."""
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=dz)
    Kx, Kz = np.meshgrid(kx, kz)
    rV = rH / max(aHV, _EPS)
    return (2 * np.pi * rH * rV) / (1 + (rH * Kx) ** 2 + (rV * Kz) ** 2) ** 1.5


def spectral_kl_coefficients(
    *,
    rf_seed: int,
    rH: float,
    aHV: float,
    nx: int,
    nz: int,
    dx: float = 1.0,
    dz: float = 1.0,
    k: int = 8,
) -> Tuple[np.ndarray, List[str]]:
    """Replay FFT GRF white-noise and return top-K PSD modes as real/imag pairs.

    Matches ``seiskit.gaussian_field.generate_gaussian_field_fft``: the colored
    Fourier noise are the discrete KL coefficients for the stationary exponential
    covariance. Returns flat array of length 2K and feature names
    ``xi_{i}_re``, ``xi_{i}_im``.
    """
    rng = np.random.default_rng(int(rf_seed))
    noise_freq = rng.standard_normal((nz, nx)) + 1j * rng.standard_normal((nz, nx))
    psd = exponential_psd(nx, nz, dx, dz, float(rH), float(aHV))
    # Rank modes by PSD eigenvalue (unique half-plane to avoid conjugate dupes)
    flat_psd = psd.ravel()
    order = np.argsort(flat_psd)[::-1]
    # Skip DC (index 0) if present; take next k unique high-energy modes
    picked: list[int] = []
    for idx in order:
        if idx == 0:
            continue
        picked.append(int(idx))
        if len(picked) >= k:
            break
    while len(picked) < k:
        picked.append(0)

    values = np.empty(2 * k, dtype=np.float64)
    names: List[str] = []
    noise_flat = noise_freq.ravel()
    for i, idx in enumerate(picked):
        c = noise_flat[idx]
        values[2 * i] = float(np.real(c))
        values[2 * i + 1] = float(np.imag(c))
        names.append(f"xi_{i + 1}_re")
        names.append(f"xi_{i + 1}_im")
    return values.astype(np.float32), names


def log_freq_hat(
    freq: np.ndarray,
    *,
    f_min: float = 0.1,
    f_max: float = 10.0,
) -> np.ndarray:
    """Map frequency into [0, 1] on a log scale."""
    f = np.asarray(freq, dtype=np.float64)
    lo = np.log(max(f_min, _EPS))
    hi = np.log(max(f_max, f_min * (1 + _EPS)))
    return np.clip((np.log(np.maximum(f, _EPS)) - lo) / max(hi - lo, _EPS), 0.0, 1.0)


def fourier_freq_features(freq: np.ndarray, *, f_min: float = 0.1, f_max: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """sin(2π f̂), cos(2π f̂) with log-scaled f̂."""
    fhat = log_freq_hat(freq, f_min=f_min, f_max=f_max)
    return np.sin(2 * np.pi * fhat).astype(np.float32), np.cos(2 * np.pi * fhat).astype(np.float32)


def column_mean_soil_vs(
    vs_field: np.ndarray,
    *,
    vs_rock: float,
    soil_nz: int | None = None,
) -> np.ndarray:
    """Mean soil Vs per column (nx,)."""
    vs = np.asarray(vs_field, dtype=np.float64)
    nz, nx = vs.shape
    out = np.empty(nx, dtype=np.float64)
    for j in range(nx):
        col = vs[:, j]
        if soil_nz is not None:
            n = max(1, min(int(soil_nz), nz))
        else:
            thresh = 0.5 * (float(col[0]) + float(vs_rock))
            soil_mask = col < thresh
            n = int(np.argmax(~soil_mask)) if not soil_mask.all() else nz
            n = max(1, min(n, nz))
        out[j] = float(np.mean(col[:n]))
    return out


def geometric_features_at_recorders(
    vs_crop: np.ndarray,
    *,
    recorder_x: Sequence[int],
    vs_rock: float,
    dx: float = 1.0,
    dz: float = 1.0,
    rho: float = 2000.0,
    soil_nz: int | None = None,
    L: float = 500.0,
    edge_percentile: float = 90.0,
) -> Dict[str, np.ndarray]:
    """Per-recorder geometric features. Each value has shape (n_recorders,)."""
    z_bed = bedrock_interface_depth(vs_crop, vs_rock=vs_rock, dz=dz)
    dip = interface_dip(z_bed, dx=dx)
    imp_grad = impedance_horizontal_gradient(
        vs_crop, rho=rho, dx=dx, soil_nz=soil_nz
    )
    nx = vs_crop.shape[1]
    x_all = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    dist = distance_to_edge(imp_grad, x_coords=x_all, percentile=edge_percentile)
    vs_col_mean = column_mean_soil_vs(vs_crop, vs_rock=vs_rock, soil_nz=soil_nz)

    rec = np.asarray(recorder_x, dtype=int)
    x_m = (rec.astype(np.float64) + 0.5) * dx
    return {
        "dip_slope": dip[rec].astype(np.float32),
        "imp_grad": imp_grad[rec].astype(np.float32),
        "dist_edge": dist[rec].astype(np.float32),
        "x_over_L": (x_m / max(L, _EPS)).astype(np.float32),
        "x_m": x_m.astype(np.float32),
        "vs_col_mean": vs_col_mean[rec].astype(np.float32),
        "Z": z_bed[rec].astype(np.float32),
    }


FEATURE_BASE_NAMES = [
    "dip_slope",
    "imp_grad",
    "dist_edge",
    "x_over_L",
    "x_over_lambda",
    "r_H",
    "f_star",
    "sin_f",
    "cos_f",
]


def xi_feature_names(k: int) -> List[str]:
    names: List[str] = []
    for i in range(1, k + 1):
        names.append(f"xi_{i}_re")
        names.append(f"xi_{i}_im")
    return names


def all_feature_names(k_xi: int) -> List[str]:
    return FEATURE_BASE_NAMES + xi_feature_names(k_xi)
