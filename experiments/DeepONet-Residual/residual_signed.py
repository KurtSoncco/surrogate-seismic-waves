"""Build signed residual cache R = TF_2D - TF_1D (col / nom)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import h5py

import config

_RES = config.RESIDUAL_DIR
if str(_RES) not in sys.path:
    sys.path.insert(0, str(_RES))


def _haskell():
    from haskell_baseline import haskell_at_columns, haskell_nominal_af_within

    return haskell_at_columns, haskell_nominal_af_within


def resolve_h5_path(stored_path: str) -> Path:
    return config.H5_DIR / Path(stored_path).name


def load_manifest(path: Path | None = None) -> List[Dict[str, str]]:
    path = path or config.MANIFEST_PATH
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_sample(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    with h5py.File(h5_path, "r") as f:
        vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
        zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
        params = {k: f["params"].attrs[k] for k in f["params"].attrs}
    return vs, zeta, params


def compute_signed_for_index(
    sample_idx: int,
    manifest_row: Dict[str, str],
    *,
    tf_2d: np.ndarray,
    freq: np.ndarray,
    recorder_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Return signed R_col, R_nom, TF1D_col, TF1D_nom and meta."""
    haskell_at_columns, haskell_nominal_af_within = _haskell()
    h5_path = resolve_h5_path(manifest_row["h5_path"])
    vs, zeta, params = _read_sample(h5_path)

    vs_crop = vs[:, config.X_SLICE_START : config.X_SLICE_END]
    zeta_crop = zeta[:, config.X_SLICE_START : config.X_SLICE_END]
    vs2 = float(params["Vs2"])
    soil_nz = int(
        params.get("soil_layer_count", params.get("H_discretized", vs_crop.shape[0]))
    )

    tf1d_col = haskell_at_columns(
        freq,
        vs_crop,
        zeta_crop,
        recorder_x,
        dz=config.DZ,
        vs_rock=vs2,
        soil_nz=soil_nz,
        rho=config.RHO,
    ).astype(np.float32)

    vs1 = float(params["Vs1"])
    H = float(params.get("H_discretized", params.get("H")))
    xi = float(config.DEFAULT_XI_TREND)
    tf1d_nom_1d = haskell_nominal_af_within(
        freq, vs1=vs1, H=H, vs2=vs2, xi=xi, rho=config.RHO
    ).astype(np.float32)
    tf1d_nom = np.broadcast_to(tf1d_nom_1d[None, :], tf1d_col.shape).copy()

    tf = tf_2d.astype(np.float64)
    r_col = (tf - tf1d_col.astype(np.float64)).astype(np.float32)
    r_nom = (tf - tf1d_nom.astype(np.float64)).astype(np.float32)

    meta = {
        "sample_idx": int(sample_idx),
        "run_index": int(manifest_row.get("run_index", sample_idx)),
        "h5_path": str(h5_path),
        "rf_seed": int(params["rf_seed"]),
        "rH": float(params["rH"]),
        "aHV": float(params["aHV"]),
        "CoV": float(params["CoV"]),
        "Vs1": vs1,
        "Vs2": vs2,
        "H": H,
        "soil_nz": soil_nz,
        "nz": int(vs_crop.shape[0]),
        "xi_damp": xi,
    }
    return r_col, r_nom, tf1d_col, tf1d_nom, meta


def sample_indices_from_residual(cache_tag: str) -> np.ndarray:
    """Reuse Residual indices when present; else a nested stratified draw."""
    from select_indices import resolve_sample_indices

    return resolve_sample_indices(cache_tag, write=True)


def build_signed_cache(
    cache_tag: str = "n1000_seed42",
    *,
    force: bool = False,
    max_samples: int | None = None,
) -> Path:
    """Write signed residuals + TF1D baselines under this experiment's cache/."""
    out_dir = config.CACHE_DIR / cache_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "r_col": out_dir / "r_col_signed.npy",
        "r_nom": out_dir / "r_nom_signed.npy",
        "tf1d_col": out_dir / "tf1d_col.npy",
        "tf1d_nom": out_dir / "tf1d_nom.npy",
        "meta": out_dir / "meta.npz",
        "idx": out_dir / "sample_indices.npy",
    }
    if not force and all(p.exists() for p in paths.values()):
        return out_dir

    sample_indices = sample_indices_from_residual(cache_tag)
    if max_samples is not None:
        sample_indices = sample_indices[: int(max_samples)]

    manifest = load_manifest()
    tf_all = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
    freq = np.load(config.TF_FREQ_PATH)
    recorder_x = np.load(config.RECORDER_X_IDX_PATH)

    n = len(sample_indices)
    n_rec = int(recorder_x.shape[0])
    n_freq = int(freq.shape[0])
    r_col = np.empty((n, n_rec, n_freq), dtype=np.float32)
    r_nom = np.empty((n, n_rec, n_freq), dtype=np.float32)
    tf1d_col = np.empty((n, n_rec, n_freq), dtype=np.float32)
    tf1d_nom = np.empty((n, n_rec, n_freq), dtype=np.float32)
    metas: list[Dict] = []

    for i, sidx in enumerate(sample_indices):
        sidx = int(sidx)
        print(f"[signed] {i + 1}/{n} sample_idx={sidx}", flush=True)
        rc, rn, tc, tn, meta = compute_signed_for_index(
            sidx,
            manifest[sidx],
            tf_2d=np.asarray(tf_all[sidx]),
            freq=freq,
            recorder_x=recorder_x,
        )
        r_col[i], r_nom[i] = rc, rn
        tf1d_col[i], tf1d_nom[i] = tc, tn
        metas.append(meta)

    np.save(paths["r_col"], r_col)
    np.save(paths["r_nom"], r_nom)
    np.save(paths["tf1d_col"], tf1d_col)
    np.save(paths["tf1d_nom"], tf1d_nom)
    np.save(paths["idx"], np.asarray(sample_indices, dtype=int))
    keys = list(metas[0].keys())
    packed = {k: np.array([m[k] for m in metas]) for k in keys}
    np.savez(paths["meta"], **packed)
    print(f"Wrote signed cache → {out_dir}", flush=True)
    return out_dir


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--cache-tag", default="n100_seed42")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()
    build_signed_cache(args.cache_tag, force=args.force, max_samples=args.max_samples)
