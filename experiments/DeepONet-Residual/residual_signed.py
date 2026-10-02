"""Build signed residual cache R = TF_2D - TF_1D (col / nom)."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import config
import h5py


def _haskell():
    from haskell_baseline import (
        haskell_at_columns,
        haskell_nominal_af_within,
        haskell_nominal_layered_af_within,
    )

    return haskell_at_columns, haskell_nominal_af_within, haskell_nominal_layered_af_within


def soil_mean_xi(zeta: np.ndarray, soil_nz: int) -> float:
    """Mean damping over the soil column (x-averaged if 2D)."""
    z = np.asarray(zeta, dtype=np.float64)
    if z.ndim == 2:
        z = z.mean(axis=1)
    n = max(1, min(int(soil_nz), int(z.shape[0])))
    return float(np.mean(z[:n]))


def xi_per_layer(
    zeta: np.ndarray,
    layer_H: Sequence[float] | np.ndarray,
    *,
    dz: float,
) -> np.ndarray:
    """Mean ζ in each nominal layer (x-averaged if 2D)."""
    z = np.asarray(zeta, dtype=np.float64)
    if z.ndim == 2:
        z = z.mean(axis=1)
    out: list[float] = []
    i0 = 0
    for h in np.asarray(layer_H, dtype=np.float64).ravel():
        n = max(1, int(round(float(h) / max(float(dz), 1e-12))))
        i1 = min(int(z.shape[0]), i0 + n)
        if i1 <= i0:
            out.append(float(z[min(i0, int(z.shape[0]) - 1)]))
        else:
            out.append(float(np.mean(z[i0:i1])))
        i0 = i1
    return np.asarray(out, dtype=np.float64)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    """Return signed R_col, R_nom, TF1D_col, TF1D_nom, meta, plus sample-ξ nom extras."""
    haskell_at_columns, haskell_nominal_af_within, _layered = _haskell()
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
    xi_sample = soil_mean_xi(zeta_crop, soil_nz)
    tf1d_nom_1d = haskell_nominal_af_within(
        freq, vs1=vs1, H=H, vs2=vs2, xi=xi, rho=config.RHO
    ).astype(np.float32)
    tf1d_nom = np.broadcast_to(tf1d_nom_1d[None, :], tf1d_col.shape).copy()
    tf1d_nom_xi_1d = haskell_nominal_af_within(
        freq, vs1=vs1, H=H, vs2=vs2, xi=xi_sample, rho=config.RHO
    ).astype(np.float32)
    tf1d_nom_xi = np.broadcast_to(tf1d_nom_xi_1d[None, :], tf1d_col.shape).copy()

    tf = tf_2d.astype(np.float64)
    r_col = (tf - tf1d_col.astype(np.float64)).astype(np.float32)
    r_nom = (tf - tf1d_nom.astype(np.float64)).astype(np.float32)
    r_nom_xi = (tf - tf1d_nom_xi.astype(np.float64)).astype(np.float32)

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
        "xi_sample": xi_sample,
        "layer_H": np.array([H], dtype=np.float64),
        "layer_Vs": np.array([vs1], dtype=np.float64),
    }
    return r_col, r_nom, tf1d_col, tf1d_nom, meta, tf1d_nom_xi, r_nom_xi


def sample_indices_from_residual(cache_tag: str) -> np.ndarray:
    """Load previously written sample indices for this cache tag."""
    path = config.CACHE_DIR / cache_tag / "sample_indices.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing sample indices: {path}. "
            "Use resolve_sample_indices() (stratified CoV×H)."
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
    path = config.CACHE_DIR / cache_tag / "sample_indices.npy"
    if path.is_file():
        return np.load(path)
    return None


def resolve_sample_indices(
    cache_tag: str,
    *,
    allow_stratified: bool = True,
    nest_smaller: bool = True,
) -> np.ndarray:
    """Stratified CoV×H indices (seed from tag).

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


def support_column_indices(
    stride: int = config.SUPPORT_STRIDE,
    nx: int = config.NX,
) -> np.ndarray:
    """Column indices on the cropped 500 m strip (1 m grid)."""
    s = max(int(stride), 1)
    return np.arange(0, int(nx), s, dtype=int)


def column_x_m(cols: np.ndarray | Sequence[float]) -> np.ndarray:
    """Physical x (m) on the cropped strip: cell centers."""
    c = np.asarray(cols, dtype=np.float64).ravel()
    return ((c + 0.5) * float(config.DX)).astype(np.float32)


def stack_field_columns(
    vs_crop: np.ndarray,
    zeta_crop: np.ndarray,
    cols: np.ndarray,
    *,
    soil_nz: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fields [3, Nz_max, n_col], vs_col [n_col]) from a cropped strip."""
    from data import normalize_vs_surface, normalize_zeta_max, pad_depth

    vs_pad = pad_depth(vs_crop, config.NZ_MAX)
    zeta_pad = pad_depth(zeta_crop, config.NZ_MAX)
    vs_n = normalize_vs_surface(vs_pad)
    zeta_n = normalize_zeta_max(zeta_pad, nz)
    z_imp = (config.RHO * vs_pad).astype(np.float32)
    z_imp = z_imp / max(float(z_imp.max()), 1e-12)
    col = np.asarray(cols, dtype=int).ravel()
    col = np.clip(col, 0, vs_crop.shape[1] - 1)
    fields = np.stack(
        [vs_n[:, col], zeta_n[:, col], z_imp[:, col]], axis=0
    ).astype(np.float32)
    n = max(1, min(int(soil_nz), vs_crop.shape[0]))
    vs_col = vs_crop[:n, col].mean(axis=0).astype(np.float32)
    return fields, vs_col


def _fields_from_h5(
    h5_path: Path, recorder_x: np.ndarray, soil_nz: int, nz: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fields [3, Nz_max, n_rec], vs_col [n_rec]) on the cropped strip."""
    vs, zeta, _ = _read_sample(h5_path)
    vs = vs[:, config.X_SLICE_START : config.X_SLICE_END]
    zeta = zeta[:, config.X_SLICE_START : config.X_SLICE_END]
    return stack_field_columns(
        vs, zeta, recorder_x, soil_nz=soil_nz, nz=nz
    )


def build_support_fields_cache(
    cache_tag: str | Path = "n7680_seed42",
    *,
    stride: int = config.SUPPORT_STRIDE,
    force: bool = False,
) -> Path:
    """Dense Vs/ζ/Z columns for kernel GNO support. Does not touch R_nom labels.

    Reads existing signed-cache ``meta.npz`` / H5 paths. Stride ``s=5`` on the
    500 m strip → 100 support columns. No new OpenSees.
    """
    out_dir = Path(cache_tag)
    if not out_dir.is_dir():
        out_dir = config.CACHE_DIR / str(cache_tag)
    fields_path = out_dir / "fields_support.npy"
    x_path = out_dir / "support_x.npy"
    cols_path = out_dir / "support_cols.npy"
    meta_path = out_dir / "support_meta.npz"
    if (
        not force
        and fields_path.is_file()
        and x_path.is_file()
        and cols_path.is_file()
        and meta_path.is_file()
    ):
        prev = dict(np.load(meta_path, allow_pickle=True))
        if int(prev.get("stride", stride)) == int(stride):
            print(f"[support] reuse {fields_path}", flush=True)
            return out_dir
    idx_path = out_dir / "sample_indices.npy"
    signed_meta = out_dir / "meta.npz"
    if not idx_path.is_file() or not signed_meta.is_file():
        raise FileNotFoundError(
            f"need signed cache meta at {out_dir} before support fields"
        )
    sample_indices = np.load(idx_path)
    meta = dict(np.load(signed_meta, allow_pickle=True))
    cols = support_column_indices(stride)
    n = len(sample_indices)
    n_s = int(cols.size)
    fields = np.empty((n, 3, config.NZ_MAX, n_s), dtype=np.float32)
    for i in tqdm(range(n), desc=f"support s={stride} {out_dir.name}"):
        stored = str(meta["h5_path"][i])
        path = Path(stored)
        if not path.is_file():
            path = resolve_h5_path(stored)
        vs, zeta, _ = _read_sample(path)
        vs = vs[:, config.X_SLICE_START : config.X_SLICE_END]
        zeta = zeta[:, config.X_SLICE_START : config.X_SLICE_END]
        soil_nz = int(meta["soil_nz"][i]) if "soil_nz" in meta else int(meta["nz"][i])
        nz = int(meta["nz"][i])
        fld, _ = stack_field_columns(vs, zeta, cols, soil_nz=soil_nz, nz=nz)
        fields[i] = fld
    np.save(fields_path, fields)
    np.save(x_path, column_x_m(cols))
    np.save(cols_path, cols.astype(np.int64))
    np.savez(
        meta_path,
        stride=np.int64(stride),
        n_support=np.int64(n_s),
        nx=np.int64(config.NX),
    )
    print(f"Wrote support fields → {fields_path}  n={n} n_support={n_s}", flush=True)
    return out_dir


def slice_support_fields_from_parent(
    child_tag: str,
    parent_tag: str,
    *,
    force: bool = False,
) -> Path:
    """Copy stride-s support fields onto a nested child cache (no H5 reread)."""
    src = config.CACHE_DIR / parent_tag
    dst = config.CACHE_DIR / child_tag
    dst_fields = dst / "fields_support.npy"
    if (
        not force
        and dst_fields.is_file()
        and (dst / "support_x.npy").is_file()
        and (dst / "support_cols.npy").is_file()
    ):
        print(f"[support] reuse {dst_fields}", flush=True)
        return dst
    src_fields = src / "fields_support.npy"
    if not src_fields.is_file():
        raise FileNotFoundError(f"need {src_fields} before slicing {child_tag}")
    child_idx_path = dst / "sample_indices.npy"
    parent_idx_path = src / "sample_indices.npy"
    if not child_idx_path.is_file() or not parent_idx_path.is_file():
        raise FileNotFoundError(f"need sample_indices.npy in {src} and {dst}")
    parent = np.load(parent_idx_path)
    child = np.load(child_idx_path)
    loc = {int(s): i for i, s in enumerate(parent)}
    missing = [int(s) for s in child if int(s) not in loc]
    if missing:
        raise KeyError(
            f"{len(missing)} {child_tag} samples are not in {parent_tag} "
            f"(e.g. {missing[:5]})"
        )
    rows = np.array([loc[int(s)] for s in child], dtype=int)
    dst.mkdir(parents=True, exist_ok=True)
    np.save(dst_fields, np.load(src_fields, mmap_mode="r")[rows])
    for name in ("support_x.npy", "support_cols.npy"):
        sp = src / name
        if sp.is_file():
            np.save(dst / name, np.load(sp))
    src_meta = src / "support_meta.npz"
    if src_meta.is_file():
        import shutil

        shutil.copy2(src_meta, dst / "support_meta.npz")
    print(
        f"[support] sliced {child_tag} from {parent_tag} rows={len(rows)} "
        f"n_support={int(np.load(dst / 'support_cols.npy').size)}",
        flush=True,
    )
    return dst


def build_signed_cache(
    cache_tag: str = "n1000_seed42",
    *,
    force: bool = False,
    max_samples: int | None = None,
    indices_only: bool = False,
    allow_stratified: bool = True,
    sample_xi: bool = False,
) -> Path:
    """Write signed residuals + TF1D baselines under this experiment's cache/.

    Default TF_1D_nom still uses ``DEFAULT_XI_TREND``. Sample-ζ extras
    ``tf1d_nom_xi.npy`` / ``r_nom_xi_signed.npy`` are a separate tag so the
    shipped control stays comparable.
    """
    out_dir = config.CACHE_DIR / cache_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "r_col": out_dir / "r_col_signed.npy",
        "r_nom": out_dir / "r_nom_signed.npy",
        "tf1d_col": out_dir / "tf1d_col.npy",
        "tf1d_nom": out_dir / "tf1d_nom.npy",
        "tf1d_nom_xi": out_dir / "tf1d_nom_xi.npy",
        "r_nom_xi": out_dir / "r_nom_xi_signed.npy",
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
        print(
            f"Wrote indices only → {paths['idx']}  n={len(sample_indices)}", flush=True
        )
        return out_dir

    core = ("r_col", "r_nom", "tf1d_col", "tf1d_nom", "meta", "idx")
    if not force and all(paths[k].exists() for k in core):
        if sample_xi and (
            not paths["tf1d_nom_xi"].exists() or not paths["r_nom_xi"].exists()
        ):
            append_sample_xi_nom(cache_tag)
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
    r_nom_xi = np.empty((n, n_rec, n_freq), dtype=np.float32)
    tf1d_col = np.empty((n, n_rec, n_freq), dtype=np.float32)
    tf1d_nom = np.empty((n, n_rec, n_freq), dtype=np.float32)
    tf1d_nom_xi = np.empty((n, n_rec, n_freq), dtype=np.float32)
    fields = np.empty((n, 3, config.NZ_MAX, n_rec), dtype=np.float32)
    vs_col = np.empty((n, n_rec), dtype=np.float32)
    metas: list[dict] = []

    for i, sidx in enumerate(tqdm(sample_indices, desc=f"signed {cache_tag}")):
        sidx = int(sidx)
        rc, rn, tc, tn, meta, tn_xi, rn_xi = compute_signed_for_index(
            sidx,
            manifest[sidx],
            tf_2d=np.asarray(tf_all[sidx]),
            freq=freq,
            recorder_x=recorder_x,
        )
        r_col[i], r_nom[i] = rc, rn
        tf1d_col[i], tf1d_nom[i] = tc, tn
        tf1d_nom_xi[i], r_nom_xi[i] = tn_xi, rn_xi
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
    np.save(paths["tf1d_nom_xi"], tf1d_nom_xi)
    np.save(paths["r_nom_xi"], r_nom_xi)
    np.save(paths["idx"], np.asarray(sample_indices, dtype=int))
    np.save(paths["fields"], fields)
    np.save(paths["vs_col"], vs_col)
    keys = list(metas[0].keys())
    packed = {k: np.array([m[k] for m in metas]) for k in keys}
    np.savez(paths["meta"], **packed)
    print(f"Wrote signed cache → {out_dir}", flush=True)
    return out_dir


def append_sample_xi_nom(cache_tag: str, *, force: bool = False) -> Path:
    """Write ``tf1d_nom_xi`` extras onto an existing default-ξ cache (no R_nom rewrite)."""
    out_dir = config.CACHE_DIR / cache_tag
    tf_path = out_dir / "tf1d_nom_xi.npy"
    r_path = out_dir / "r_nom_xi_signed.npy"
    if not force and tf_path.is_file() and r_path.is_file():
        return out_dir
    idx_path = out_dir / "sample_indices.npy"
    if not idx_path.is_file():
        raise FileNotFoundError(f"need {idx_path} before sample-ξ extras")
    sample_indices = np.load(idx_path)
    manifest = load_manifest()
    tf_all = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
    freq = np.load(config.TF_FREQ_PATH)
    recorder_x = np.load(config.RECORDER_X_IDX_PATH)
    n = len(sample_indices)
    n_rec = int(recorder_x.shape[0])
    n_freq = int(freq.shape[0])
    tf1d_nom_xi = np.empty((n, n_rec, n_freq), dtype=np.float32)
    r_nom_xi = np.empty((n, n_rec, n_freq), dtype=np.float32)
    xi_sample = np.empty(n, dtype=np.float64)
    layer_H: list[np.ndarray] = []
    layer_Vs: list[np.ndarray] = []
    for i, sidx in enumerate(tqdm(sample_indices, desc=f"sample-xi {cache_tag}")):
        sidx = int(sidx)
        *_, meta, tn_xi, rn_xi = compute_signed_for_index(
            sidx,
            manifest[sidx],
            tf_2d=np.asarray(tf_all[sidx]),
            freq=freq,
            recorder_x=recorder_x,
        )
        tf1d_nom_xi[i] = tn_xi
        r_nom_xi[i] = rn_xi
        xi_sample[i] = float(meta["xi_sample"])
        layer_H.append(np.asarray(meta["layer_H"], dtype=np.float64))
        layer_Vs.append(np.asarray(meta["layer_Vs"], dtype=np.float64))
    np.save(tf_path, tf1d_nom_xi)
    np.save(r_path, r_nom_xi)
    meta_path = out_dir / "meta.npz"
    if meta_path.is_file():
        packed = dict(np.load(meta_path, allow_pickle=True))
        packed["xi_sample"] = xi_sample
        packed["layer_H"] = np.array(layer_H, dtype=object)
        packed["layer_Vs"] = np.array(layer_Vs, dtype=object)
        np.savez(meta_path, **packed)
    print(f"Wrote sample-ξ nom extras → {out_dir}", flush=True)
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
    p.add_argument(
        "--sample-xi",
        action="store_true",
        help="Write tf1d_nom_xi.npy extras using H5 soil-mean ζ (does not rewrite R_nom).",
    )
    p.add_argument(
        "--support-fields",
        action="store_true",
        help="Write stride-s kernel support fields from existing H5 (no new OpenSees).",
    )
    p.add_argument(
        "--support-stride",
        type=int,
        default=config.SUPPORT_STRIDE,
        help="Column stride on the 500 m strip (default 5 → 100 support).",
    )
    p.add_argument(
        "--slice-support-from",
        default=None,
        help="Slice fields_support.npy from this parent cache tag (no H5 reread).",
    )
    args = p.parse_args()
    if args.slice_support_from:
        slice_support_fields_from_parent(
            args.cache_tag, args.slice_support_from, force=args.force
        )
    elif args.support_fields:
        build_support_fields_cache(
            args.cache_tag, stride=args.support_stride, force=args.force
        )
    else:
        build_signed_cache(
            args.cache_tag,
            force=args.force,
            max_samples=args.max_samples,
            indices_only=args.indices_only,
            allow_stratified=not args.require_residual_indices,
            sample_xi=args.sample_xi,
        )
