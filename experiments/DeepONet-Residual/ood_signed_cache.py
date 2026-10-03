"""Geometry-aware signed R caches for Box ood_dipping / ood_three_layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

import config
from ood_io import (
    crop_variability,
    default_ood_roots,
    discover_h5_files,
    load_or_compute_tf,
    nominal_layer_params,
    read_h5_sample,
    recorder_x_indices,
    soil_nz_from_params,
)
from haskell_baseline import (
    haskell_at_columns,
    haskell_nominal_af_within,
    haskell_nominal_layered_af_within,
)


def materialize_signed_from_parent(child_tag: str, parent_tag: str) -> Path:
    """Slice a larger signed cache onto ``child_tag`` indices (no Haskell)."""
    src = config.CACHE_DIR / parent_tag
    dst = config.CACHE_DIR / child_tag
    need = ["r_nom_signed.npy", "tf1d_nom.npy", "fields.npy", "meta.npz"]
    if all((dst / k).is_file() for k in need):
        return dst
    if not all((src / k).is_file() for k in ("r_nom_signed.npy", "sample_indices.npy")):
        raise FileNotFoundError(f"need full {parent_tag} cache at {src}")
    from residual_signed import resolve_sample_indices, write_sample_indices

    child_idx = resolve_sample_indices(child_tag)
    write_sample_indices(child_tag, child_idx)
    parent = np.load(src / "sample_indices.npy")
    loc = {int(s): i for i, s in enumerate(parent)}
    missing = [int(s) for s in child_idx if int(s) not in loc]
    if missing:
        raise KeyError(
            f"{len(missing)} {child_tag} samples are not in {parent_tag} "
            f"(e.g. {missing[:5]})"
        )
    rows = np.array([loc[int(s)] for s in child_idx], dtype=int)
    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "r_col_signed.npy",
        "r_nom_signed.npy",
        "tf1d_col.npy",
        "tf1d_nom.npy",
        "tf1d_nom_xi.npy",
        "r_nom_xi_signed.npy",
        "fields.npy",
        "vs_col.npy",
        "fields_support.npy",
        "support_x.npy",
        "support_cols.npy",
        "sample_indices.npy",
    ):
        sp = src / name
        if not sp.is_file() and name != "sample_indices.npy":
            continue
        if name == "sample_indices.npy":
            np.save(dst / name, np.asarray(child_idx, dtype=int))
            continue
        if name in ("support_x.npy", "support_cols.npy"):
            np.save(dst / name, np.load(sp))
            continue
        np.save(dst / name, np.load(sp, mmap_mode="r")[rows])
    meta = dict(np.load(src / "meta.npz", allow_pickle=True))
    packed = {k: np.asarray(v)[rows] for k, v in meta.items()}
    np.savez(dst / "meta.npz", **packed)
    src_smeta = src / "support_meta.npz"
    if src_smeta.is_file():
        import shutil

        shutil.copy2(src_smeta, dst / "support_meta.npz")
    print(
        f"[cache] materialized {child_tag} from {parent_tag} rows={len(rows)}",
        flush=True,
    )
    return dst


def materialize_n1000_from_n2000() -> Path:
    """Slice the n2000 signed cache onto nested n1000 indices (no Haskell)."""
    return materialize_signed_from_parent("n1000_seed42", "n2000_seed42")
    src = config.CACHE_DIR / "n2000_seed42"
    dst = config.CACHE_DIR / "n1000_seed42"
    need = ["r_nom_signed.npy", "tf1d_nom.npy", "fields.npy", "meta.npz"]
    if all((dst / k).is_file() for k in need):
        return dst
    if not all((src / k).is_file() for k in ("r_nom_signed.npy", "sample_indices.npy")):
        raise FileNotFoundError(f"need full n2000 cache at {src}")
    parent = np.load(src / "sample_indices.npy")
    child = np.load(dst / "sample_indices.npy")
    loc = {int(s): i for i, s in enumerate(parent)}
    rows = np.array([loc[int(s)] for s in child], dtype=int)
    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "r_col_signed.npy",
        "r_nom_signed.npy",
        "tf1d_col.npy",
        "tf1d_nom.npy",
        "fields.npy",
        "vs_col.npy",
    ):
        sp = src / name
        if sp.is_file():
            np.save(dst / name, np.load(sp, mmap_mode="r")[rows])
    meta = dict(np.load(src / "meta.npz", allow_pickle=True))
    packed = {k: np.asarray(v)[rows] for k, v in meta.items()}
    np.savez(dst / "meta.npz", **packed)
    print(f"[cache] materialized n1000 from n2000 rows={len(rows)}", flush=True)
    return dst


def cache_dir_for(name: str) -> Path:
    return config.CACHE_DIR / f"{name}_signed"


def _stoch_meta(params: dict[str, Any], nz: int) -> dict[str, Any]:
    if "rf_seed" in params:
        return {
            "rf_seed": int(params["rf_seed"]),
            "rH": float(params["rH"]),
            "aHV": float(params["aHV"]),
            "CoV": float(params["CoV"]),
            "stoch_note": "iid_like_rf_seed",
        }
    return {
        "rf_seed": int(params.get("seed1", params.get("seed", 0))),
        "rH": float(params.get("rH1", params.get("rH", 1.0))),
        "aHV": float(params.get("aHV1", params.get("aHV", 1.0))),
        "CoV": float(params.get("CoV1", params.get("CoV", 0.0))),
        "stoch_note": "three_layer_layer1_standin",
    }


def _fields_vs_col(
    vs: np.ndarray, zeta: np.ndarray, recorder_x: np.ndarray, nz: int, soil_nz: int
) -> tuple[np.ndarray, np.ndarray]:
    from data import normalize_vs_surface, normalize_zeta_max, pad_depth

    vs_c = crop_variability(vs)
    zeta_c = crop_variability(zeta)
    vs_pad = pad_depth(vs_c, config.NZ_MAX)
    zeta_pad = pad_depth(zeta_c, config.NZ_MAX)
    vs_n = normalize_vs_surface(vs_pad)
    zeta_n = normalize_zeta_max(zeta_pad, nz)
    z_imp = (config.RHO * vs_pad).astype(np.float32)
    z_imp = z_imp / max(float(z_imp.max()), 1e-12)
    cols = recorder_x.astype(int)
    cols = np.clip(cols, 0, vs_c.shape[1] - 1)
    fields = np.stack([vs_n[:, cols], zeta_n[:, cols], z_imp[:, cols]], axis=0).astype(
        np.float32
    )
    n = max(1, min(int(soil_nz), vs_c.shape[0]))
    vs_col = vs_c[:n, cols].mean(axis=0).astype(np.float32)
    return fields, vs_col


def build_ood_signed_cache(name: str, *, force: bool = False) -> Path:
    roots = default_ood_roots()
    if name not in roots:
        raise KeyError(f"unknown corpus {name!r}")
    root = roots[name]
    out_dir = cache_dir_for(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "r_col": out_dir / "r_col_signed.npy",
        "r_nom": out_dir / "r_nom_signed.npy",
        "r_nom1": out_dir / "r_nom1_signed.npy",
        "tf1d_col": out_dir / "tf1d_col.npy",
        "tf1d_nom": out_dir / "tf1d_nom.npy",
        "tf1d_nom1": out_dir / "tf1d_nom1.npy",
        "tf1d_nom_xi": out_dir / "tf1d_nom_xi.npy",
        "r_nom_xi": out_dir / "r_nom_xi_signed.npy",
        "tf2d": out_dir / "tf2d.npy",
        "fields": out_dir / "fields.npy",
        "vs_col": out_dir / "vs_col.npy",
        "meta": out_dir / "meta.npz",
        "idx": out_dir / "sample_indices.npy",
        "freq": out_dir / "freq.npy",
        "recorder_x": out_dir / "recorder_x.npy",
    }
    core = ("r_col", "r_nom", "tf1d_col", "tf1d_nom", "tf2d", "fields", "meta", "idx")
    if not force and all(paths[k].exists() for k in core):
        print(f"[ood-cache] reuse {out_dir}", flush=True)
        return out_dir

    h5s = discover_h5_files(root)
    rec = recorder_x_indices(root)
    tf_cache = config.CACHE_DIR / f"{name}_tf"
    n = len(h5s)
    if n == 0:
        raise FileNotFoundError(f"no H5 under {root}")

    n_rec = int(rec.shape[0])
    n_freq = config.N_FREQ
    r_col = np.empty((n, n_rec, n_freq), dtype=np.float32)
    r_nom = np.empty((n, n_rec, n_freq), dtype=np.float32)
    r_nom1 = np.empty((n, n_rec, n_freq), dtype=np.float32)
    r_nom_xi = np.empty((n, n_rec, n_freq), dtype=np.float32)
    tf1d_col = np.empty((n, n_rec, n_freq), dtype=np.float32)
    tf1d_nom = np.empty((n, n_rec, n_freq), dtype=np.float32)
    tf1d_nom1 = np.empty((n, n_rec, n_freq), dtype=np.float32)
    tf1d_nom_xi = np.empty((n, n_rec, n_freq), dtype=np.float32)
    tf2d_all = np.empty((n, n_rec, n_freq), dtype=np.float32)
    fields = np.empty((n, 3, config.NZ_MAX, n_rec), dtype=np.float32)
    vs_col = np.empty((n, n_rec), dtype=np.float32)
    metas: list[dict[str, Any]] = []
    freq_ref: np.ndarray | None = None

    from residual_signed import soil_mean_xi, xi_per_layer

    for i, h5_path in enumerate(tqdm(h5s, desc=f"ood-cache {name}")):
        vs, zeta, params, extra = read_h5_sample(h5_path)
        tf, freq = load_or_compute_tf(h5_path, tf_cache)
        if freq_ref is None:
            freq_ref = np.asarray(freq, dtype=np.float64)
        vs_c = crop_variability(vs)
        zeta_c = crop_variability(zeta)
        nom = nominal_layer_params(params)
        vs2 = float(nom["vs2"])
        soil_nz = soil_nz_from_params(params, vs_c.shape[0])
        rec_i = rec.copy()
        if rec_i.max() >= vs_c.shape[1]:
            rec_i = np.clip(rec_i, 0, vs_c.shape[1] - 1)
        col = haskell_at_columns(
            freq_ref,
            vs_c,
            zeta_c,
            rec_i,
            dz=config.DZ,
            vs_rock=vs2,
            soil_nz=soil_nz,
            rho=config.RHO,
        ).astype(np.float32)
        nom1_1d = haskell_nominal_af_within(
            freq_ref,
            vs1=float(nom["vs1"]),
            H=float(nom["H"]),
            vs2=vs2,
            xi=float(config.DEFAULT_XI_TREND),
            rho=config.RHO,
        ).astype(np.float32)
        nom1 = np.broadcast_to(nom1_1d[None, :], col.shape).copy()
        true_layers = nom.get("true_layers")
        layer_H = np.array([float(nom["H"])], dtype=np.float64)
        layer_Vs = np.array([float(nom["vs1"])], dtype=np.float64)
        if true_layers is not None:
            layer_H = np.asarray(true_layers["H"], dtype=np.float64)
            layer_Vs = np.asarray(true_layers["Vs"], dtype=np.float64)
            nom_g_1d = haskell_nominal_layered_af_within(
                freq_ref,
                H=true_layers["H"],
                Vs=true_layers["Vs"],
                vs_rock=float(true_layers["vs_rock"]),
                xi=float(config.DEFAULT_XI_TREND),
                rho=config.RHO,
            ).astype(np.float32)
            nom_g = np.broadcast_to(nom_g_1d[None, :], col.shape).copy()
            geo_source = true_layers["source"]
        else:
            nom_g = nom1
            geo_source = nom["source"]

        xi_sample_layers = xi_per_layer(zeta_c, layer_H, dz=config.DZ)
        if true_layers is not None:
            nom_xi_1d = haskell_nominal_layered_af_within(
                freq_ref,
                H=true_layers["H"],
                Vs=true_layers["Vs"],
                vs_rock=float(true_layers["vs_rock"]),
                xi=xi_sample_layers,
                rho=config.RHO,
            ).astype(np.float32)
        else:
            nom_xi_1d = haskell_nominal_af_within(
                freq_ref,
                vs1=float(nom["vs1"]),
                H=float(nom["H"]),
                vs2=vs2,
                xi=float(soil_mean_xi(zeta_c, soil_nz)),
                rho=config.RHO,
            ).astype(np.float32)
        nom_xi = np.broadcast_to(nom_xi_1d[None, :], col.shape).copy()
        xi_sample = float(np.mean(xi_sample_layers)) if xi_sample_layers.size else 0.0

        tf = np.asarray(tf, dtype=np.float32)
        if tf.shape != col.shape:
            raise ValueError(f"{h5_path}: tf {tf.shape} vs haskell {col.shape}")
        tf1d_col[i], tf1d_nom[i], tf1d_nom1[i] = col, nom_g, nom1
        tf1d_nom_xi[i] = nom_xi
        tf2d_all[i] = tf
        r_col[i] = tf - col
        r_nom[i] = tf - nom_g
        r_nom1[i] = tf - nom1
        r_nom_xi[i] = tf - nom_xi
        fld, vc = _fields_vs_col(vs, zeta, rec_i, int(vs_c.shape[0]), soil_nz)
        fields[i], vs_col[i] = fld, vc
        sm = _stoch_meta(params, int(vs_c.shape[0]))
        metas.append(
            {
                "sample_idx": i,
                "run_index": i,
                "h5_path": str(h5_path),
                "rf_seed": sm["rf_seed"],
                "rH": sm["rH"],
                "aHV": sm["aHV"],
                "CoV": sm["CoV"],
                "Vs1": float(nom["vs1"]),
                "Vs2": vs2,
                "H": float(nom["H"]),
                "soil_nz": int(soil_nz),
                "nz": int(vs_c.shape[0]),
                "xi_damp": float(config.DEFAULT_XI_TREND),
                "xi_sample": xi_sample,
                "layer_H": layer_H,
                "layer_Vs": layer_Vs,
                "geo_source": geo_source,
                "nom_misspecified": bool(nom["misspecified"]),
                "stoch_note": sm["stoch_note"],
                "domain": name,
            }
        )

    assert freq_ref is not None
    np.save(paths["r_col"], r_col)
    np.save(paths["r_nom"], r_nom)
    np.save(paths["r_nom1"], r_nom1)
    np.save(paths["tf1d_col"], tf1d_col)
    np.save(paths["tf1d_nom"], tf1d_nom)
    np.save(paths["tf1d_nom1"], tf1d_nom1)
    np.save(paths["tf1d_nom_xi"], tf1d_nom_xi)
    np.save(paths["r_nom_xi"], r_nom_xi)
    np.save(paths["tf2d"], tf2d_all)
    np.save(paths["fields"], fields)
    np.save(paths["vs_col"], vs_col)
    np.save(paths["idx"], np.arange(n, dtype=int))
    np.save(paths["freq"], freq_ref.astype(np.float64))
    np.save(paths["recorder_x"], rec.astype(np.int64))
    packed = {}
    for k in metas[0]:
        if k in ("layer_H", "layer_Vs"):
            packed[k] = np.array([m[k] for m in metas], dtype=object)
        else:
            packed[k] = np.array([m[k] for m in metas])
    np.savez(paths["meta"], **packed)
    print(f"Wrote OOD signed cache → {out_dir}", flush=True)
    return out_dir


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--corpus", action="append", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--support-fields",
        action="store_true",
        help="Write stride-s kernel support fields onto an existing OOD signed cache.",
    )
    p.add_argument("--support-stride", type=int, default=config.SUPPORT_STRIDE)
    args = p.parse_args()
    names = args.corpus or ["ood_dipping", "ood_three_layer"]
    if args.support_fields:
        from residual_signed import build_support_fields_cache

        for name in names:
            build_support_fields_cache(
                cache_dir_for(name), stride=args.support_stride, force=args.force
            )
    else:
        for name in names:
            build_ood_signed_cache(name, force=args.force)
