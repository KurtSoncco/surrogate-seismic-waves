"""Rebuild feature table with OrbitAll + GIFNO-XT columns."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import h5py

import config
from features import (
    all_feature_names,
    fourier_freq_features,
    geometric_features_at_recorders,
    spectral_kl_coefficients,
)
from gifno_xt_features import gifno_xt_feature_names, gifno_xt_features_at_recorders


def _freq_screen_indices(freq: np.ndarray, n: int) -> np.ndarray:
    if n >= len(freq):
        return np.arange(len(freq))
    targets = np.logspace(np.log10(freq[0]), np.log10(freq[-1]), n)
    idx = np.unique([int(np.argmin(np.abs(freq - t))) for t in targets])
    if len(idx) < n:
        extra = [i for i in range(len(freq)) if i not in set(idx)]
        idx = np.concatenate([idx, extra[: n - len(idx)]])
    return np.sort(idx[:n])


def build_feature_table(
    cache_dir: Path,
    *,
    n_freq_screen: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """One row per (sample, recorder, freq) with OrbitAll + GIFNO-XT features."""
    n_freq_screen = n_freq_screen or config.N_FREQ_SCREEN
    out = cache_dir / "feature_table_xt.parquet"
    legacy = cache_dir / "feature_table.parquet"
    if not force and out.exists():
        return pd.read_parquet(out)

    r_col = np.load(cache_dir / "r_col.npy")
    r_nom = np.load(cache_dir / "r_nom.npy")
    meta = dict(np.load(cache_dir / "meta.npz", allow_pickle=True))
    sample_indices = np.load(cache_dir / "sample_indices.npy")
    freq = np.load(config.TF_FREQ_PATH)
    recorder_x = np.load(config.RECORDER_X_IDX_PATH)
    f_idx = _freq_screen_indices(freq, n_freq_screen)
    freq_s = freq[f_idx]
    sin_f, cos_f = fourier_freq_features(
        freq_s, f_min=config.FREQ_START_HZ, f_max=config.FREQ_END_HZ
    )
    log_f = np.log(np.maximum(freq_s, 1e-12)).astype(np.float32)

    n_s, n_r, _ = r_col.shape
    rows: list[dict] = []
    orbit_names = all_feature_names(config.K_XI)
    xi_names = [n for n in orbit_names if n.startswith("xi_")]

    for i in range(n_s):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[table] {i + 1}/{n_s}", flush=True)
        h5_path = Path(str(meta["h5_path"][i]))
        with h5py.File(h5_path, "r") as f:
            vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
            zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
        vs_crop = vs[:, config.X_SLICE_START : config.X_SLICE_END]
        zeta_crop = zeta[:, config.X_SLICE_START : config.X_SLICE_END]
        vs_rock = float(meta["Vs2"][i])
        soil_nz = int(meta["soil_nz"][i])
        nz = int(meta["nz"][i])
        rH = float(meta["rH"][i])
        aHV = float(meta["aHV"][i])
        rf_seed = int(meta["rf_seed"][i])
        H = float(meta["H"][i])
        CoV = float(meta["CoV"][i])

        geom = geometric_features_at_recorders(
            vs_crop,
            recorder_x=recorder_x,
            vs_rock=vs_rock,
            dx=config.DX,
            dz=config.DZ,
            rho=config.RHO,
            soil_nz=soil_nz,
            L=float(config.LX_VARIABILITY),
            edge_percentile=config.IMP_GRAD_EDGE_PERCENTILE,
        )
        xt = gifno_xt_features_at_recorders(
            vs_crop,
            zeta_crop,
            recorder_x=recorder_x,
            soil_nz=soil_nz,
            nz=nz,
            dx=config.DX,
            nx=config.NX,
        )
        xi_vals, _ = spectral_kl_coefficients(
            rf_seed=rf_seed,
            rH=rH,
            aHV=aHV,
            nx=config.NX,
            nz=nz,
            dx=config.DX,
            dz=config.DZ,
            k=config.K_XI,
        )

        for r in range(n_r):
            vs_c = float(geom["vs_col_mean"][r])
            x_m = float(geom["x_m"][r])
            for j, fi in enumerate(f_idx):
                f = float(freq_s[j])
                lam = vs_c / max(f, 1e-12)
                f_star = f * H / max(vs_c, 1e-12)
                row = {
                    "sample_idx": int(sample_indices[i]),
                    "recorder": int(r),
                    "freq_hz": f,
                    "freq_idx": int(fi),
                    "R_col": float(r_col[i, r, fi]),
                    "R_nom": float(r_nom[i, r, fi]),
                    # OrbitAll
                    "dip_slope": float(geom["dip_slope"][r]),
                    "imp_grad": float(geom["imp_grad"][r]),
                    "dist_edge": float(geom["dist_edge"][r]),
                    "x_over_L": float(geom["x_over_L"][r]),
                    "x_over_lambda": float(x_m / max(lam, 1e-12)),
                    "r_H": rH,
                    "f_star": float(f_star),
                    "sin_f": float(sin_f[j]),
                    "cos_f": float(cos_f[j]),
                    # GIFNO-XT
                    "vs_norm_mean": float(xt["vs_norm_mean"][r]),
                    "vs_norm_std": float(xt["vs_norm_std"][r]),
                    "vs_norm_surf": float(xt["vs_norm_surf"][r]),
                    "vs_norm_mid": float(xt["vs_norm_mid"][r]),
                    "vs_norm_deep": float(xt["vs_norm_deep"][r]),
                    "vs_surface": float(xt["vs_surface"][r]),
                    "zeta_norm_mean": float(xt["zeta_norm_mean"][r]),
                    "zeta_norm_std": float(xt["zeta_norm_std"][r]),
                    "vs_macro_mean": float(xt["vs_macro_mean"][r]),
                    "vs_rf_rms": float(xt["vs_rf_rms"][r]),
                    "vs_lat_grad": float(xt["vs_lat_grad"][r]),
                    "z_mean": float(xt["z_mean"][r]),
                    "x_grid": float(xt["x_grid"][r]),
                    "x_trunk": float(xt["x_trunk"][r]),
                    "log_f": float(log_f[j]),
                    "H": H,
                    "CoV": CoV,
                }
                for name, val in zip(xi_names, xi_vals):
                    row[name] = float(val)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    # keep legacy name as alias for older scripts
    if not legacy.exists() or force:
        df.to_parquet(legacy, index=False)
    return df


def feature_set(name: str, k_xi: int | None = None) -> List[str]:
    """Named feature bundles for DeepONet ablation screening."""
    k_xi = k_xi or config.K_XI
    orbit = list(all_feature_names(k_xi))
    xt_core = gifno_xt_feature_names(include_proposed=False)
    xt_all = gifno_xt_feature_names(include_proposed=True)
    # Minimal physics add-ons on top of XT (OrbitAll winners)
    xt_plus = xt_all + ["f_star", "sin_f", "cos_f", "x_over_lambda"]
    combined = list(dict.fromkeys(orbit + xt_all))
    sets = {
        "orbitall": orbit,
        "gifno_xt": xt_core,
        "gifno_xt_full": xt_all,
        "gifno_xt_plus": xt_plus,
        "combined": combined,
    }
    if name not in sets:
        raise ValueError(f"Unknown feature set {name!r}; choose from {list(sets)}")
    return sets[name]


def feature_columns(k_xi: int | None = None) -> Sequence[str]:
    """Backward-compatible: OrbitAll-only names."""
    return all_feature_names(k_xi or config.K_XI)
