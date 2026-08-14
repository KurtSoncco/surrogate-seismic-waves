"""Haskell floor on Box OOD corpora (E0).

Default roots (override with env or flags):
  $GIFNO_DATA_ROOT/ood_dipping
  $GIFNO_DATA_ROOT/ood_three_layer

Scores TF_1D_nom and TF_1D_col vs OpenSees TF (cached tf_true.npy,
transfer_function/tf_per_sample.npy, or seiskit TTF from accel).
Checkpoint residual inference (E4) is not wired yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import config
import numpy as np

_RES = config.RESIDUAL_DIR
if str(_RES) not in sys.path:
    sys.path.insert(0, str(_RES))
if str(config.EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(config.EXPERIMENT_DIR))

# ruff: noqa: E402
from haskell_baseline import (
    haskell_at_columns,
    haskell_nominal_af_within,
)

_EPS = 1e-12


def discover_h5(root: Path) -> list[Path]:
    """Find OpenSees H5 files under an OOD folder (GIFNO or seiskit layout)."""
    root = Path(root)
    if not root.is_dir():
        return []
    candidates: list[Path] = []
    h5_dir = root / "h5"
    search = h5_dir if h5_dir.is_dir() else root
    candidates.extend(sorted(search.glob("*.h5")))
    candidates.extend(sorted(search.glob("**/*.h5")))
    # unique, skip tiny placeholders
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp in seen or not p.is_file() or p.stat().st_size < 256:
            continue
        seen.add(rp)
        out.append(p)
    return out


def recorder_x_indices(
    nx: int = config.NX,
    n_lateral: int = config.N_LATERAL,
    spacing_m: float = 15.0,
    dx: float = config.DX,
) -> np.ndarray:
    cached = config.RECORDER_X_IDX_PATH
    if cached.is_file():
        idx = np.load(cached).astype(int)
        return np.clip(idx, 0, nx - 1)
    half = (n_lateral - 1) / 2.0
    x_m = (np.arange(n_lateral) - half) * spacing_m + 0.5 * nx * dx
    return np.clip(np.round(x_m / dx).astype(int), 0, nx - 1)


def _freq_grid(n_freq: int) -> np.ndarray:
    path = config.TF_FREQ_PATH
    if path.is_file():
        freq = np.load(path).astype(np.float64)
        if len(freq) == n_freq:
            return freq
        # resample log grid to requested length
    return np.logspace(
        np.log10(config.FREQ_START_HZ),
        np.log10(config.FREQ_END_HZ),
        n_freq,
        dtype=np.float64,
    )


def _read_h5(path: Path) -> dict[str, Any]:
    import h5py

    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    with h5py.File(path, "r") as f:
        vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
        zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
        params = {}
        if "params" in f:
            params = {k: f["params"].attrs[k] for k in f["params"].attrs}
        grid = {}
        if "grid" in f:
            grid = {k: f["grid"].attrs[k] for k in f["grid"].attrs}
        accel = None
        dt = float(grid.get("dt", 0.01))
        if "recorders" in f and "accel" in f["recorders"]:
            acc = f["recorders/accel"]
            if "data" in acc:
                accel = np.asarray(acc["data"][:], dtype=np.float32)
            if "time" in acc and acc["time"].shape[0] > 1:
                t = np.asarray(acc["time"][:], dtype=np.float64)
                dt = float(t[1] - t[0])
    return {
        "vs": vs,
        "zeta": zeta,
        "params": params,
        "grid": grid,
        "accel": accel,
        "dt": dt,
    }


def crop_variability(vs: np.ndarray, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx = vs.shape[1]
    if nx >= config.NX_FULL:
        sl = slice(config.X_SLICE_START, config.X_SLICE_END)
        return vs[:, sl], zeta[:, sl]
    if nx >= config.NX:
        start = max(0, (nx - config.NX) // 2)
        sl = slice(start, start + config.NX)
        return vs[:, sl], zeta[:, sl]
    return vs, zeta


def nominal_from_params(
    params: dict[str, Any],
    vs_crop: np.ndarray,
    *,
    dz: float = config.DZ,
) -> tuple[float, float, float]:
    """(Vs1, H, Vs2) for single-layer Haskell-nom.

    Three-layer OOD often still stores a trend (Vs1, H, Vs2); if missing, fall
    back to surface Vs / soil depth / deep-column Vs so E0 never silently skips.
    """
    vs1 = params.get("Vs1")
    vs2 = params.get("Vs2")
    H = params.get("H_discretized", params.get("H"))
    soil_nz = params.get("soil_layer_count", params.get("H_discretized"))
    if vs1 is None:
        vs1 = float(np.median(vs_crop[0]))
    if vs2 is None:
        vs2 = float(np.median(vs_crop[-1]))
    if H is None:
        if soil_nz is not None:
            H = float(soil_nz) * dz
        else:
            H = float(vs_crop.shape[0]) * dz
    return float(vs1), float(H), float(vs2)


def compute_tf_from_accel(
    accel: np.ndarray,
    dt: float,
    n_freq: int,
) -> tuple[np.ndarray, np.ndarray]:
    """OpenSees accel (time, 2*n_lat) -> (n_lat, n_freq) via seiskit TTF."""
    n_lat = config.N_LATERAL
    if accel.ndim != 2 or accel.shape[1] != 2 * n_lat:
        raise ValueError(
            f"Expected accel (T, {2 * n_lat}), got {accel.shape}"
        )
    try:
        seiskit_root = Path.home() / "seiskit"
        if seiskit_root.is_dir() and str(seiskit_root) not in sys.path:
            sys.path.insert(0, str(seiskit_root))
        from seiskit.ttf.TTF import TTF_batch_fast
    except ImportError as exc:
        raise ImportError(
            "seiskit is required to compute OOD TFs from accel; "
            "provide tf_true.npy or transfer_function/tf_per_sample.npy instead"
        ) from exc
    base = accel[:, :n_lat].T
    surf = accel[:, n_lat:].T
    freq, mags = TTF_batch_fast(
        base,
        surf,
        dt=dt,
        dz=config.DZ,
        smooth_coeff=500,
        Vsmin=None,
        n_points=n_freq,
    )
    return np.asarray(mags, dtype=np.float32), np.asarray(freq, dtype=np.float64)


def load_or_compute_tf(
    h5_path: Path,
    corpus_root: Path,
    *,
    n_freq: int,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Prefer cached TF next to the H5 or under corpus transfer_function/."""
    case_dir = h5_path.parent
    cached = case_dir / "tf_true.npy"
    freq_cached = case_dir / "freq.npy"
    if not force and cached.is_file():
        tf = np.load(cached)
        freq = np.load(freq_cached) if freq_cached.is_file() else _freq_grid(tf.shape[1])
        return tf.astype(np.float32), freq.astype(np.float64)

    tf_dir = corpus_root / "transfer_function"
    tf_all_path = tf_dir / "tf_per_sample.npy"
    manifest_path = tf_dir / "manifest.csv"
    if tf_all_path.is_file() and manifest_path.is_file():
        import csv

        with open(manifest_path, newline="") as f:
            rows = list(csv.DictReader(f))
        names = {Path(r.get("h5_path", r.get("path", ""))).name: i for i, r in enumerate(rows)}
        if h5_path.name in names:
            tf_all = np.load(tf_all_path, mmap_mode="r")
            tf = np.asarray(tf_all[names[h5_path.name]], dtype=np.float32)
            freq_path = tf_dir / "freq.npy"
            freq = np.load(freq_path) if freq_path.is_file() else _freq_grid(tf.shape[1])
            return tf, freq.astype(np.float64)

    rec = _read_h5(h5_path)
    if rec["accel"] is None:
        raise FileNotFoundError(
            f"No TF cache and no recorders/accel in {h5_path}"
        )
    tf, freq = compute_tf_from_accel(rec["accel"], rec["dt"], n_freq)
    np.save(cached, tf)
    np.save(freq_cached, freq)
    return tf, freq


def rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), _EPS))


def pearson_freq(pred: np.ndarray, true: np.ndarray) -> float:
    cors: list[float] = []
    for i in range(true.shape[0]):
        a = true[i].astype(np.float64)
        b = pred[i].astype(np.float64)
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        cors.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(cors)) if cors else 0.0


def haskell_floors(
    vs_crop: np.ndarray,
    zeta_crop: np.ndarray,
    params: dict[str, Any],
    freq: np.ndarray,
    recorder_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    vs2 = float(params["Vs2"]) if "Vs2" in params else float(np.median(vs_crop[-1]))
    soil_nz = params.get("soil_layer_count", params.get("H_discretized"))
    if soil_nz is not None:
        soil_nz = int(soil_nz)
    tf_col = haskell_at_columns(
        freq,
        vs_crop,
        zeta_crop,
        recorder_x,
        dz=config.DZ,
        vs_rock=vs2,
        soil_nz=soil_nz,
        rho=config.RHO,
    ).astype(np.float32)
    vs1, H, vs2_n = nominal_from_params(params, vs_crop)
    tf_nom_1d = haskell_nominal_af_within(
        freq, vs1=vs1, H=H, vs2=vs2_n, xi=config.DEFAULT_XI_TREND, rho=config.RHO
    ).astype(np.float32)
    tf_nom = np.broadcast_to(tf_nom_1d[None, :], tf_col.shape).copy()
    meta = {"vs1": vs1, "H": H, "vs2": vs2_n}
    return tf_nom, tf_col, meta


def score_pair(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "rel_l2": rel_l2(pred, true),
        "pearson_freq": pearson_freq(pred, true),
    }


def evaluate_h5(
    h5_path: Path,
    corpus_root: Path,
    *,
    n_freq: int | None = None,
    force_tf: bool = False,
) -> dict[str, Any]:
    rec = _read_h5(h5_path)
    vs_crop, zeta_crop = crop_variability(rec["vs"], rec["zeta"])
    n_freq = int(n_freq or config.N_FREQ)
    tf_true, freq = load_or_compute_tf(
        h5_path, corpus_root, n_freq=n_freq, force=force_tf
    )
    if tf_true.shape[1] != len(freq):
        freq = freq[: tf_true.shape[1]]
    rec_x = recorder_x_indices(nx=vs_crop.shape[1])
    n_lat = min(tf_true.shape[0], len(rec_x))
    rec_x = rec_x[:n_lat]
    tf_true = tf_true[:n_lat]
    tf_nom, tf_col, nom_meta = haskell_floors(
        vs_crop, zeta_crop, rec["params"], freq, rec_x
    )
    if tf_nom.shape[1] != tf_true.shape[1]:
        # interpolate Haskell onto TF freq if needed
        from numpy import interp

        def _interp(arr: np.ndarray) -> np.ndarray:
            out = np.empty_like(tf_true)
            src_f = np.linspace(freq[0], freq[-1], arr.shape[1])
            for i in range(arr.shape[0]):
                out[i] = interp(freq[: tf_true.shape[1]], src_f, arr[i])
            return out

        tf_nom = _interp(tf_nom)
        tf_col = _interp(tf_col)
    row = {
        "h5": str(h5_path),
        "corpus": corpus_root.name,
        "shape_vs": list(vs_crop.shape),
        "n_rec": int(n_lat),
        "n_freq": int(tf_true.shape[1]),
        "nom_attrs": nom_meta,
        "haskell_nom": score_pair(tf_nom, tf_true),
        "haskell_col": score_pair(tf_col, tf_true),
        "r_nom_rel_l2": rel_l2(tf_nom, tf_true),
        "r_col_rel_l2": rel_l2(tf_col, tf_true),
    }
    return row


def evaluate_corpus(root: Path, *, n_freq: int | None = None) -> dict[str, Any]:
    h5_paths = discover_h5(root)
    cases = [evaluate_h5(p, root, n_freq=n_freq) for p in h5_paths]
    summary: dict[str, Any] = {
        "root": str(root),
        "n_h5": len(h5_paths),
        "cases": cases,
    }
    if cases:
        for key in ("haskell_nom", "haskell_col"):
            summary[f"mean_{key}_rel_l2"] = float(
                np.mean([c[key]["rel_l2"] for c in cases])
            )
            summary[f"mean_{key}_pearson_freq"] = float(
                np.mean([c[key]["pearson_freq"] for c in cases])
            )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="OOD Haskell floor on Box ood_*")
    p.add_argument(
        "--dipping",
        type=Path,
        default=config.OOD_DIPPING_DIR,
    )
    p.add_argument(
        "--three-layer",
        type=Path,
        default=config.OOD_THREE_LAYER_DIR,
    )
    p.add_argument("--n-freq", type=int, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=config.RESULTS_DIR / "ood_haskell_floor.json",
    )
    args = p.parse_args()
    report = {
        "ood_dipping": evaluate_corpus(args.dipping, n_freq=args.n_freq),
        "ood_three_layer": evaluate_corpus(args.three_layer, n_freq=args.n_freq),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
