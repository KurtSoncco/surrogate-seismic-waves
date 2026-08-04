"""GIFNO-XT-faithful tabular features for DeepONet screening.

Mirrors experiments/GIFNO data_loader normalizations and FDO-XT trunk coords:
  branch-like: Vs/Vs_surface, zeta/max(zeta), x_grid, z summaries
  trunk-like:  log(f), x_trunk (center-normalized ±1)
plus optional SCALE_SPLIT macro/RF column stats and depth bins.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

_EPS = 1e-12


def normalize_vs_by_surface(vs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    surface = np.maximum(vs[0:1, :], eps)
    return (vs / surface).astype(np.float64)


def normalize_zeta_by_max(zeta: np.ndarray, nz: int, eps: float = 1e-12) -> np.ndarray:
    if nz <= 0:
        return zeta.astype(np.float64)
    zmax = float(np.max(zeta[:nz, :]))
    if zmax < eps:
        return zeta.astype(np.float64)
    return (zeta / zmax).astype(np.float64)


def box_blur_2d(arr: np.ndarray, kernel: int = 15) -> np.ndarray:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    if k == 1:
        return arr.astype(np.float64, copy=True)
    pad = k // 2
    x = np.pad(arr.astype(np.float64), ((0, 0), (pad, pad)), mode="edge")
    c = np.cumsum(x, axis=1)
    c = np.pad(c, ((0, 0), (1, 0)), mode="constant")
    horiz = (c[:, k:] - c[:, :-k]) / k
    y = np.pad(horiz, ((pad, pad), (0, 0)), mode="edge")
    c2 = np.cumsum(y, axis=0)
    c2 = np.pad(c2, ((1, 0), (0, 0)), mode="constant")
    return (c2[k:, :] - c2[:-k, :]) / k


def recorder_x_trunk(
    recorder_x: Sequence[int],
    *,
    nx: int = 500,
    dx: float = 1.0,
) -> np.ndarray:
    """GIFNO-XT default trunk x: offset / half-width → edges ±1."""
    center = nx // 2
    x_m = (np.asarray(recorder_x, dtype=np.float64) - center) * dx
    half_w = float(nx // 2) * dx
    return (x_m / max(half_w, _EPS)).astype(np.float32)


def recorder_x_grid(
    recorder_x: Sequence[int],
    *,
    nx: int = 500,
    dx: float = 1.0,
) -> np.ndarray:
    """Branch channel x: i*dx / Lx with Lx = nx*dx conceptually L=500."""
    L = float(nx) * dx
    return ((np.asarray(recorder_x, dtype=np.float64) * dx) / max(L, _EPS)).astype(
        np.float32
    )


def column_stats(col: np.ndarray, soil_nz: int) -> Dict[str, float]:
    """Mean/std and depth-tercile means over soil rows."""
    n = max(1, min(int(soil_nz), len(col)))
    c = np.asarray(col[:n], dtype=np.float64)
    t = max(1, n // 3)
    return {
        "mean": float(np.mean(c)),
        "std": float(np.std(c)),
        "surf": float(np.mean(c[:t])),
        "mid": float(np.mean(c[t : 2 * t])),
        "deep": float(np.mean(c[2 * t : n])),
    }


def gifno_xt_features_at_recorders(
    vs_crop: np.ndarray,
    zeta_crop: np.ndarray,
    *,
    recorder_x: Sequence[int],
    soil_nz: int,
    nz: int,
    dx: float = 1.0,
    nx: int = 500,
    macro_kernel: int = 15,
) -> Dict[str, np.ndarray]:
    """Per-recorder GIFNO-XT-style summaries. Values shape (n_recorders,)."""
    vs_n = normalize_vs_by_surface(vs_crop)
    zeta_n = normalize_zeta_by_max(zeta_crop, nz=nz)
    macro = box_blur_2d(vs_n, macro_kernel)
    rf = vs_n - macro

    rec = np.asarray(recorder_x, dtype=int)
    n_r = len(rec)
    out: Dict[str, np.ndarray] = {
        "vs_norm_mean": np.empty(n_r, np.float32),
        "vs_norm_std": np.empty(n_r, np.float32),
        "vs_norm_surf": np.empty(n_r, np.float32),
        "vs_norm_mid": np.empty(n_r, np.float32),
        "vs_norm_deep": np.empty(n_r, np.float32),
        "vs_surface": np.empty(n_r, np.float32),
        "zeta_norm_mean": np.empty(n_r, np.float32),
        "zeta_norm_std": np.empty(n_r, np.float32),
        "vs_macro_mean": np.empty(n_r, np.float32),
        "vs_rf_rms": np.empty(n_r, np.float32),
        "vs_lat_grad": np.empty(n_r, np.float32),  # proposed: neighbor contrast
        "z_mean": np.empty(n_r, np.float32),
    }
    # lateral gradient of surface-normalized column mean
    col_means = np.array(
        [np.mean(vs_n[:soil_nz, j]) for j in range(vs_n.shape[1])], dtype=np.float64
    )
    lat_g = np.gradient(col_means, dx)

    Lz = float(max(nz, 1))
    for i, j in enumerate(rec):
        j = int(j)
        n_soil = max(1, min(int(soil_nz), vs_crop.shape[0]))
        st = column_stats(vs_n[:, j], n_soil)
        zt = column_stats(zeta_n[:, j], n_soil)
        out["vs_norm_mean"][i] = st["mean"]
        out["vs_norm_std"][i] = st["std"]
        out["vs_norm_surf"][i] = st["surf"]
        out["vs_norm_mid"][i] = st["mid"]
        out["vs_norm_deep"][i] = st["deep"]
        out["vs_surface"][i] = float(vs_crop[0, j])
        out["zeta_norm_mean"][i] = zt["mean"]
        out["zeta_norm_std"][i] = zt["std"]
        out["vs_macro_mean"][i] = float(np.mean(macro[:n_soil, j]))
        out["vs_rf_rms"][i] = float(np.sqrt(np.mean(rf[:n_soil, j] ** 2)))
        out["vs_lat_grad"][i] = float(lat_g[j])
        # mean normalized depth of soil column (branch z channel summary)
        z_rows = (np.arange(n_soil, dtype=np.float64) * dx) / Lz
        out["z_mean"][i] = float(np.mean(z_rows))

    out["x_grid"] = recorder_x_grid(rec, nx=nx, dx=dx)
    out["x_trunk"] = recorder_x_trunk(rec, nx=nx, dx=dx)
    return out


# Core XT analogs (what DeepONet branch+trunk effectively see, summarized)
GIFNO_XT_CORE = [
    "vs_norm_mean",
    "vs_norm_std",
    "vs_surface",
    "zeta_norm_mean",
    "zeta_norm_std",
    "x_grid",
    "x_trunk",
    "z_mean",
    "log_f",
]

# Proposed extras beyond vanilla XT channels
GIFNO_XT_PROPOSED = [
    "vs_norm_surf",
    "vs_norm_mid",
    "vs_norm_deep",
    "vs_macro_mean",
    "vs_rf_rms",
    "vs_lat_grad",
    "H",
    "CoV",
]

GIFNO_XT_ALL = GIFNO_XT_CORE + GIFNO_XT_PROPOSED


def gifno_xt_feature_names(*, include_proposed: bool = True) -> List[str]:
    return list(GIFNO_XT_ALL if include_proposed else GIFNO_XT_CORE)
