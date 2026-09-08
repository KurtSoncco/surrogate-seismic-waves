"""Build and cache |R_col| and |R_nom| residuals for a sample subset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import h5py

import config
from haskell_baseline import haskell_at_columns, haskell_nominal_af_within


def resolve_h5_path(stored_path: str) -> Path:
    """Map manifest H5 paths to local H5_DIR by basename."""
    return config.H5_DIR / Path(stored_path).name


def load_manifest(path: Path | None = None) -> List[Dict[str, str]]:
    path = path or config.MANIFEST_PATH
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def stratified_sample_indices(
    manifest: Sequence[Dict[str, str]],
    n: int,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Stratify by CoV / H / rH quantile bins when columns exist; else random."""
    rng = np.random.default_rng(seed)
    n_total = len(manifest)
    if n >= n_total:
        return np.arange(n_total)

    def _col(key: str) -> np.ndarray | None:
        if key not in manifest[0]:
            # try H5-backed later; CoV/H may be in manifest
            alt = {"H": "H_discretized", "rH": "rH"}.get(key, key)
            if alt not in manifest[0]:
                return None
            key = alt
        try:
            return np.array([float(row[key]) for row in manifest], dtype=np.float64)
        except (KeyError, ValueError):
            return None

    cov = _col("CoV")
    H = _col("H_discretized") if _col("H_discretized") is not None else _col("H")
    # rH may be missing from manifest — load later; for stratification use CoV+H only
    parts = [c for c in (cov, H) if c is not None]
    if not parts:
        return rng.choice(n_total, size=n, replace=False)

    # 2D quantile bins on available axes
    def qbin(a: np.ndarray, nbin: int = 4) -> np.ndarray:
        qs = np.quantile(a, np.linspace(0, 1, nbin + 1)[1:-1])
        return np.digitize(a, qs)

    labels = qbin(parts[0])
    for p in parts[1:]:
        labels = labels * 4 + qbin(p)

    chosen: list[int] = []
    groups = {int(g): np.where(labels == g)[0] for g in np.unique(labels)}
    # round-robin per group
    while len(chosen) < n:
        progressed = False
        for g in sorted(groups):
            pool = groups[g]
            remaining = [i for i in pool if i not in chosen]
            if not remaining:
                continue
            chosen.append(int(rng.choice(remaining)))
            progressed = True
            if len(chosen) >= n:
                break
        if not progressed:
            break
    if len(chosen) < n:
        leftover = [i for i in range(n_total) if i not in chosen]
        extra = rng.choice(leftover, size=n - len(chosen), replace=False)
        chosen.extend(int(i) for i in extra)
    return np.array(sorted(chosen), dtype=int)


def _read_sample(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    with h5py.File(h5_path, "r") as f:
        vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
        zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
        params = {k: f["params"].attrs[k] for k in f["params"].attrs}
    return vs, zeta, params


def compute_residuals_for_index(
    sample_idx: int,
    manifest_row: Dict[str, str],
    *,
    tf_2d: np.ndarray,
    freq: np.ndarray,
    recorder_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return |R_col|, |R_nom| each (n_recorders, n_freq) and params dict."""
    h5_path = resolve_h5_path(manifest_row["h5_path"])
    vs, zeta, params = _read_sample(h5_path)

    vs_crop = vs[:, config.X_SLICE_START : config.X_SLICE_END]
    zeta_crop = zeta[:, config.X_SLICE_START : config.X_SLICE_END]
    vs2 = float(params["Vs2"])
    soil_nz = int(
        params.get("soil_layer_count", params.get("H_discretized", vs_crop.shape[0]))
    )

    # Recorders are indexed on the cropped NX=500 strip
    tf1d_col = haskell_at_columns(
        freq,
        vs_crop,
        zeta_crop,
        recorder_x,
        dz=config.DZ,
        vs_rock=vs2,
        soil_nz=soil_nz,
        rho=config.RHO,
    )
    r_col = np.abs(tf_2d.astype(np.float64) - tf1d_col).astype(np.float32)

    vs1 = float(params["Vs1"])
    H = float(params.get("H_discretized", params.get("H")))
    xi = float(config.DEFAULT_XI_TREND)
    tf1d_nom = haskell_nominal_af_within(
        freq, vs1=vs1, H=H, vs2=vs2, xi=xi, rho=config.RHO
    )
    # Broadcast nominal (F,) across recorders
    r_nom = np.abs(tf_2d.astype(np.float64) - tf1d_nom[None, :]).astype(np.float32)

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
    }
    return r_col, r_nom, meta


def build_residual_cache(
    sample_indices: Sequence[int],
    *,
    force: bool = False,
) -> Path:
    """Compute residuals for indices; write cache/*.npz + meta.npz. Returns cache dir."""
    cache_tag = f"n{len(sample_indices)}_seed{config.SEED}"
    out_dir = config.CACHE_DIR / cache_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    r_col_path = out_dir / "r_col.npy"
    r_nom_path = out_dir / "r_nom.npy"
    meta_path = out_dir / "meta.npz"
    idx_path = out_dir / "sample_indices.npy"

    if (
        not force
        and r_col_path.exists()
        and r_nom_path.exists()
        and meta_path.exists()
        and idx_path.exists()
    ):
        return out_dir

    manifest = load_manifest()
    tf_all = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
    freq = np.load(config.TF_FREQ_PATH)
    recorder_x = np.load(config.RECORDER_X_IDX_PATH)

    n = len(sample_indices)
    n_rec = int(recorder_x.shape[0])
    n_freq = int(freq.shape[0])
    r_col = np.empty((n, n_rec, n_freq), dtype=np.float32)
    r_nom = np.empty((n, n_rec, n_freq), dtype=np.float32)
    metas: list[Dict] = []

    for i, sidx in enumerate(sample_indices):
        sidx = int(sidx)
        print(f"[residual] {i + 1}/{n} sample_idx={sidx}", flush=True)
        rc, rn, meta = compute_residuals_for_index(
            sidx,
            manifest[sidx],
            tf_2d=np.asarray(tf_all[sidx]),
            freq=freq,
            recorder_x=recorder_x,
        )
        r_col[i] = rc
        r_nom[i] = rn
        metas.append(meta)

    np.save(r_col_path, r_col)
    np.save(r_nom_path, r_nom)
    np.save(idx_path, np.asarray(sample_indices, dtype=int))
    # store meta as object arrays of keys
    keys = list(metas[0].keys())
    packed = {k: np.array([m[k] for m in metas]) for k in keys}
    np.savez(meta_path, **packed)
    return out_dir
