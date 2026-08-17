"""Build signed residual cache R = TF_2D - TF_1D (col / nom)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import config
import h5py

_RES = config.RESIDUAL_DIR
if str(_RES) not in sys.path:
    sys.path.insert(0, str(_RES))


def _haskell():
    from haskell_baseline import haskell_at_columns, haskell_nominal_af_within

    return haskell_at_columns, haskell_nominal_af_within


def resolve_h5_path(stored_path: str) -> Path:
    return config.H5_DIR / Path(stored_path).name


def load_manifest(path: Path | None = None) -> list[dict[str, str]]:
    path = path or config.MANIFEST_PATH
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_sample(h5_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(h5_path, "r") as f:
        vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
        zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
        params = {k: f["params"].attrs[k] for k in f["params"].attrs}
    return vs, zeta, params


def compute_signed_for_index(
    sample_idx: int,
    manifest_row: dict[str, str],
    *,
    tf_2d: np.ndarray,
    freq: np.ndarray,
    recorder_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
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
    """Reuse Residual experiment sample indices for apples-to-apples splits."""
    path = config.RESIDUAL_CACHE_DIR / cache_tag / "sample_indices.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Residual cache indices: {path}. "
            "Use resolve_sample_indices() (stratified, no RF screen)."
        )
    return np.load(path)


def _nested_parent_tag(cache_tag: str) -> str | None:
    """n2000_seed42 → n1000_seed42; n3000 → n2000; n7680 → n1000 (test-slice parent)."""
    from ood_io import parse_cache_tag

    n, seed = parse_cache_tag(cache_tag)
    parent_n = {2000: 1000, 3000: 2000, 7680: 1000}.get(n)
    if parent_n is None:
        return None
    return f"n{parent_n}_seed{seed}"


def _load_existing_indices(cache_tag: str) -> np.ndarray | None:
    for base in (config.CACHE_DIR, config.RESIDUAL_CACHE_DIR):
        path = base / cache_tag / "sample_indices.npy"
        if path.is_file():
            return np.load(path)
    return None


def resolve_sample_indices(
    cache_tag: str,
    *,
    allow_stratified: bool = True,
    nest_smaller: bool = True,
) -> np.ndarray:
    """Stratified CoV×H indices (seed from tag). Residual RF screen is optional.

    Nested: n=1000 ⊂ n=2000 ⊂ n=3000 when a smaller cache exists, or when both
    are generated with the same round-robin + seed.
    """
    existing = _load_existing_indices(cache_tag)
    if existing is not None:
        return np.asarray(existing, dtype=int)

    if not allow_stratified:
        return sample_indices_from_residual(cache_tag)

    from ood_io import parse_cache_tag
    from residual_target import load_manifest as _load_man
    from residual_target import stratified_sample_indices

    n, seed = parse_cache_tag(cache_tag)
    manifest = _load_man()
    chosen = stratified_sample_indices(manifest, n, seed=seed)

    if nest_smaller:
        parent = _nested_parent_tag(cache_tag)
        if parent is not None:
            smaller = _load_existing_indices(parent)
            if smaller is None:
                try:
                    smaller = resolve_sample_indices(
                        parent, allow_stratified=True, nest_smaller=True
                    )
                    write_sample_indices(parent, smaller)
                except (ValueError, FileNotFoundError):
                    smaller = None
            if smaller is not None and len(smaller) <= n:
                must = {int(i) for i in smaller}
                extra = [int(i) for i in chosen if int(i) not in must]
                need = n - len(must)
                if need < 0:
                    raise RuntimeError(
                        f"{parent} has {len(must)} indices; cannot nest into {cache_tag}"
                    )
                if len(extra) < need:
                    leftover = [
                        i
                        for i in range(len(manifest))
                        if i not in must and i not in extra
                    ]
                    rng = np.random.default_rng(seed)
                    extra.extend(
                        int(i)
                        for i in rng.choice(
                            leftover, size=need - len(extra), replace=False
                        )
                    )
                chosen = np.array(sorted(list(must) + extra[:need]), dtype=int)

    return np.asarray(chosen, dtype=int).reshape(-1)


def write_sample_indices(cache_tag: str, indices: np.ndarray) -> Path:
    out_dir = config.CACHE_DIR / cache_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sample_indices.npy"
    np.save(path, np.asarray(indices, dtype=int))
    return path


def _fields_from_h5(
    h5_path: Path, recorder_x: np.ndarray, soil_nz: int, nz: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fields [3, Nz_max, n_rec], vs_col [n_rec]) on the cropped strip."""
    from data import normalize_vs_surface, normalize_zeta_max, pad_depth

    vs, zeta, _ = _read_sample(h5_path)
    vs = vs[:, config.X_SLICE_START : config.X_SLICE_END]
    zeta = zeta[:, config.X_SLICE_START : config.X_SLICE_END]
    vs_pad = pad_depth(vs, config.NZ_MAX)
    zeta_pad = pad_depth(zeta, config.NZ_MAX)
    vs_n = normalize_vs_surface(vs_pad)
    zeta_n = normalize_zeta_max(zeta_pad, nz)
    z_imp = (config.RHO * vs_pad).astype(np.float32)
    z_imp = z_imp / max(float(z_imp.max()), 1e-12)
    cols = recorder_x.astype(int)
    fields = np.stack([vs_n[:, cols], zeta_n[:, cols], z_imp[:, cols]], axis=0).astype(
        np.float32
    )
    n = max(1, min(int(soil_nz), vs.shape[0]))
    vs_col = vs[:n, cols].mean(axis=0).astype(np.float32)
    return fields, vs_col


def build_signed_cache(
    cache_tag: str = "n1000_seed42",
    *,
    force: bool = False,
    max_samples: int | None = None,
    indices_only: bool = False,
    allow_stratified: bool = True,
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
        "fields": out_dir / "fields.npy",
        "vs_col": out_dir / "vs_col.npy",
    }
    sample_indices = resolve_sample_indices(
        cache_tag, allow_stratified=allow_stratified
    )
    if max_samples is not None:
        sample_indices = sample_indices[: int(max_samples)]
    np.save(paths["idx"], np.asarray(sample_indices, dtype=int))
    if indices_only:
        print(f"Wrote indices only → {paths['idx']}  n={len(sample_indices)}", flush=True)
        return out_dir

    core = ("r_col", "r_nom", "tf1d_col", "tf1d_nom", "meta", "idx")
    if not force and all(paths[k].exists() for k in core):
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
    tf1d_col = np.empty((n, n_rec, n_freq), dtype=np.float32)
    tf1d_nom = np.empty((n, n_rec, n_freq), dtype=np.float32)
    fields = np.empty((n, 3, config.NZ_MAX, n_rec), dtype=np.float32)
    vs_col = np.empty((n, n_rec), dtype=np.float32)
    metas: list[dict] = []

    for i, sidx in enumerate(tqdm(sample_indices, desc=f"signed {cache_tag}")):
        sidx = int(sidx)
        rc, rn, tc, tn, meta = compute_signed_for_index(
            sidx,
            manifest[sidx],
            tf_2d=np.asarray(tf_all[sidx]),
            freq=freq,
            recorder_x=recorder_x,
        )
        r_col[i], r_nom[i] = rc, rn
        tf1d_col[i], tf1d_nom[i] = tc, tn
        fld, vc = _fields_from_h5(
            Path(meta["h5_path"]),
            recorder_x,
            int(meta["soil_nz"]),
            int(meta["nz"]),
        )
        fields[i], vs_col[i] = fld, vc
        metas.append(meta)

    np.save(paths["r_col"], r_col)
    np.save(paths["r_nom"], r_nom)
    np.save(paths["tf1d_col"], tf1d_col)
    np.save(paths["tf1d_nom"], tf1d_nom)
    np.save(paths["idx"], np.asarray(sample_indices, dtype=int))
    np.save(paths["fields"], fields)
    np.save(paths["vs_col"], vs_col)
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
    p.add_argument(
        "--indices-only",
        action="store_true",
        help="Write sample_indices.npy via stratified CoV×H (no Residual RF, no Haskell).",
    )
    p.add_argument(
        "--require-residual-indices",
        action="store_true",
        help="Fail if Residual cache indices are missing (old behavior).",
    )
    args = p.parse_args()
    build_signed_cache(
        args.cache_tag,
        force=args.force,
        max_samples=args.max_samples,
        indices_only=args.indices_only,
        allow_stratified=not args.require_residual_indices,
    )
