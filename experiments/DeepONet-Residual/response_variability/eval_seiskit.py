#!/usr/bin/env python3
"""Add seiskit Hallal / Pretell arms to the nested IID Response_Variability check.

Reuses GINO predictions from eval_iid.py. Loads Vs/zeta from seiskit neural-operator
H5s (Box GIFNO corpus). Hallal 1-D TFs use Thomson–Haskell, not OpenSees 1-D.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_EXP = Path(__file__).resolve().parents[1]
_RES = _EXP.parent / "Residual"
# DeepONet-Residual must win over Residual/config.py
for p in (_RES, _EXP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config  # noqa: E402
from residual_signed import resolve_h5_path  # noqa: E402

from response_variability.eval_iid import (  # noqa: E402
    OUT_DIR,
    IID_CACHE,
    aggregate_json,
    band_misfit_table,
    summarize_methods,
)
from response_variability.seiskit_arms import (  # noqa: E402
    ensure_seiskit,
    hallal_geomean_tf,
    pretell_haskell_tf,
)

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import h5py  # noqa: E402

SEISKIT_OUT = OUT_DIR / "seiskit"


def _load_h5(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(path, "r") as f:
        vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
        zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
        params = {k: f["params"].attrs[k] for k in f["params"].attrs}
    return vs, zeta, params


def add_seiskit_arms(
    pack: dict[str, np.ndarray],
    *,
    cache_dir: Path,
    n_hallal_seeds: int,
    n_pretell: int,
) -> dict[str, np.ndarray]:
    ensure_seiskit()
    meta = dict(np.load(cache_dir / "meta.npz", allow_pickle=True))
    local = np.asarray(pack["local_idx"], dtype=int)
    freq = pack["freq"]
    n = len(local)
    n_freq = len(freq)
    tf_toro = np.empty((n, n_freq), dtype=np.float64)
    tf_passeri = np.empty((n, n_freq), dtype=np.float64)
    tf_pretell = np.empty((n, n_freq), dtype=np.float64)
    sig_toro = np.empty(n, dtype=np.float64)
    sig_passeri = np.empty(n, dtype=np.float64)
    sig_pretell = np.empty(n, dtype=np.float64)

    for i, loc in enumerate(tqdm(local, desc="seiskit Hallal+Pretell")):
        h5_path = resolve_h5_path(str(meta["h5_path"][int(loc)]))
        vs, zeta, params = _load_h5(h5_path)
        vs_strip = vs[:, config.X_SLICE_START : config.X_SLICE_END]
        zeta_strip = zeta[:, config.X_SLICE_START : config.X_SLICE_END]
        vs1 = float(pack["vs1"][i])
        H = float(pack["H"][i])
        cov = float(pack["cov"][i])
        vs2 = float(pack["vs2"][i])
        xi = float(meta["xi_damp"][int(loc)]) if "xi_damp" in meta else config.DEFAULT_XI_TREND
        soil_nz = int(meta["soil_nz"][int(loc)]) if "soil_nz" in meta else int(
            params.get("soil_layer_count", params.get("H_discretized", vs_strip.shape[0]))
        )
        geo_t, sig_t = hallal_geomean_tf(
            freq=freq,
            vs1=vs1,
            H=H,
            cov=cov,
            vs2=vs2,
            xi=xi,
            n_seeds=n_hallal_seeds,
            kind="toro",
        )
        geo_p, sig_p = hallal_geomean_tf(
            freq=freq,
            vs1=vs1,
            H=H,
            cov=cov,
            vs2=vs2,
            xi=xi,
            n_seeds=n_hallal_seeds,
            kind="passeri",
        )
        geo_pr, sig_pr = pretell_haskell_tf(
            freq=freq,
            vs_strip=vs_strip,
            zeta_strip=zeta_strip,
            vs2=vs2,
            soil_nz=soil_nz,
            dz=config.DZ,
            n_samples=n_pretell,
        )
        tf_toro[i], sig_toro[i] = geo_t, float(np.mean(sig_t))
        tf_passeri[i], sig_passeri[i] = geo_p, float(np.mean(sig_p))
        tf_pretell[i], sig_pretell[i] = geo_pr, float(np.mean(sig_pr))

    pack = dict(pack)
    pack["tf_toro"] = tf_toro
    pack["tf_passeri"] = tf_passeri
    pack["tf_pretell"] = tf_pretell
    pack["sigma_ln_toro"] = sig_toro
    pack["sigma_ln_passeri"] = sig_passeri
    pack["sigma_ln_pretell"] = sig_pretell
    pack["n_hallal_seeds"] = np.array(n_hallal_seeds)
    pack["n_pretell"] = np.array(n_pretell)
    return pack


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, default=OUT_DIR / "predictions.npz")
    p.add_argument("--cache-dir", type=Path, default=IID_CACHE)
    p.add_argument("--out-dir", type=Path, default=SEISKIT_OUT)
    p.add_argument("--n-hallal-seeds", type=int, default=40)
    p.add_argument("--n-pretell", type=int, default=200)
    p.add_argument(
        "--skip-arms",
        action="store_true",
        help="Reuse seiskit_predictions.npz if present.",
    )
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if not args.predictions.is_file():
        raise SystemExit(f"Run eval_iid.py first; missing {args.predictions}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_out = args.out_dir / "predictions.npz"
    if args.skip_arms and pred_out.is_file():
        blob = np.load(pred_out, allow_pickle=True)
        pack = {k: blob[k] for k in blob.files}
    else:
        blob = np.load(args.predictions, allow_pickle=True)
        pack = {k: blob[k] for k in blob.files}
        pack = add_seiskit_arms(
            pack,
            cache_dir=args.cache_dir,
            n_hallal_seeds=args.n_hallal_seeds,
            n_pretell=args.n_pretell,
        )
        np.savez_compressed(pred_out, **pack)

    summary, peaks = summarize_methods(pack)
    misfit = band_misfit_table(pack)
    summary.to_csv(args.out_dir / "method_comparison_summary.csv", index=False)
    peaks.to_csv(args.out_dir / "per_sample_peaks.csv", index=False)
    misfit.to_csv(args.out_dir / "tf_band_misfit.csv", index=False)
    agg = aggregate_json(summary, misfit)
    (args.out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2), flush=True)
    if args.plot:
        from response_variability.plot_iid import plot_all

        plot_all(args.out_dir)


if __name__ == "__main__":
    main()
