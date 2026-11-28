#!/usr/bin/env python3
"""Relabel seiskit training runs as surface / (c * input) and add 1D outcrop priors.

Writes, next to each campaign manifest (row order preserved, NaN where the H5
is not available locally):

* ``tf_outcrop_per_sample.npy``   (N, 21, 1000) surface / (c * input)
* ``tf1d_outcrop_per_sample.npy`` (N, 21, 1000) vertical-SH surface / outcrop
  of the column under each recorder (the multifidelity prior)
* ``outcrop_valid.npy``           (N,) bool
* ``outcrop_freq.npy``            (1000,)
* ``outcrop_calibration.json``    the constant ``c`` and its spread

Campaigns differ in how the Ricker enters the run (see
``labels.seiskit_input_acceleration``). ``c`` absorbs the rest of the
injection convention and is fitted once per convention: median of ratio / 1D
over 0.4-2.0 Hz at the center recorder, over all runs sharing it.

    python relabel_outcrop.py --workers 20
    OUTCROP_DATA_ROOT=/scratch/.../gifno_data python relabel_outcrop.py   # full 7680 set

The data root needs ``h5/`` and ``transfer_function/manifest.csv`` and,
optionally, ``ood_dipping/``, ``ood_three_layer/``, and ``corner_is/``
with their own ``h5/`` and ``manifest.csv``. Missing campaigns are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from labels import CALIBRATION_BAND_HZ, band_mask, seiskit_input_acceleration, tf_over_input
from section_1d import RHO, rayleigh_xi_target, surface_over_outcrop

DATA_ROOT = Path(os.environ.get("OUTCROP_DATA_ROOT", Path(__file__).resolve().parents[2] / "data" / "gifno_screen"))
N_LATERAL = 21
RECORDER_COLS = 500 + 250 + 15 * np.arange(-10, 11)
DAMPING_FREQ_SECOND = 10.0
MOTION_T_SHIFT = 0.5
DEFAULT_MOTION_FREQ = 3.0


def _campaign(sub: str | None, tf_sub: str | None, ricker_is: str) -> dict:
    base = DATA_ROOT / sub if sub else DATA_ROOT
    out = base / tf_sub if tf_sub else base
    return {"manifest": out / "manifest.csv", "h5_dir": base / "h5", "out_dir": out, "ricker_is": ricker_is}


CAMPAIGNS = {
    "iid": _campaign(None, "transfer_function", "velocity"),
    "ood_dipping": _campaign("ood_dipping", None, "acceleration"),
    "ood_three_layer": _campaign("ood_three_layer", None, "acceleration"),
    "corner_is": _campaign("corner_is", None, "acceleration"),
}

_TTF = None


def _ttf():
    global _TTF
    if _TTF is None:
        from run_case7a import load_ttf_batch_fast

        _TTF = load_ttf_batch_fast()
    return _TTF


def _h5_name(row: dict) -> str:
    idx = row.get("run_index") or row.get("index") or row.get("sample_idx")
    return f"run_{int(idx)}.h5"


def _row_float(row: dict, key: str, default: float) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float(default)


def relabel_one(job: tuple[str, dict, str, float | None]):
    import hdf5plugin  # noqa: F401
    import h5py

    path, row, ricker_is, window_s = job
    if not os.path.isfile(path):
        return None
    with h5py.File(path, "r") as f:
        data = f["recorders/accel/data"][:].astype(np.float64)
        time = f["recorders/accel/time"][:].astype(np.float64)
        vs = f["Vs_realization_2D"][:].astype(np.float64)
        zeta = f["Damping_zeta"][:].astype(np.float64)
        grid = dict(f["grid"].attrs)
        params = dict(f["params"].attrs)
    if data.shape[1] != 2 * N_LATERAL:
        raise ValueError(f"{path}: expected {2 * N_LATERAL} channels, got {data.shape[1]}")

    motion_freq = float(grid.get("motion_freq", _row_float(row, "motion_freq", DEFAULT_MOTION_FREQ)))
    f1 = float(params.get("damping_freq_first", _row_float(row, "damping_freq_first", _row_float(row, "f0_effective", 1.0))))
    dz = float(grid.get("dz", 1.0))
    dt = float(time[1] - time[0])

    input_acc = seiskit_input_acceleration(motion_freq, MOTION_T_SHIFT, time, ricker_is)
    freq, tf_raw = tf_over_input(data[:, N_LATERAL:], input_acc, dt, _ttf(), window_s=window_s)

    tf1d = np.empty_like(tf_raw)
    for j, col in enumerate(RECORDER_COLS):
        xi = np.stack([rayleigh_xi_target(freq, z, f1, DAMPING_FREQ_SECOND) for z in zeta[:, col]])
        tf1d[j] = surface_over_outcrop(freq, vs[:, col], dz=dz, xi=xi, rho=RHO)
    return freq, tf_raw.astype(np.float32), tf1d.astype(np.float32)


def run_campaign(name: str, workers: int, limit: int | None, window_s: float | None = None) -> dict:
    spec = CAMPAIGNS[name]
    rows = list(csv.DictReader(open(spec["manifest"])))
    if limit:
        rows = rows[:limit]
    jobs = [(str(spec["h5_dir"] / _h5_name(r)), r, spec["ricker_is"], window_s) for r in rows]
    n = len(rows)
    tf = np.full((n, N_LATERAL, 1000), np.nan, dtype=np.float32)
    tf1d = np.full_like(tf, np.nan)
    valid = np.zeros(n, dtype=bool)
    freq = None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(relabel_one, jobs, chunksize=8)):
            if res is None:
                continue
            f, tf[i], tf1d[i] = res
            if freq is None:
                freq = f
            elif not np.allclose(f, freq, rtol=1e-6):
                raise RuntimeError(f"{name}: frequency grid differs at row {i}")
            valid[i] = True
    print(f"{name}: {valid.sum()}/{n} runs relabeled")
    return {"tf": tf, "tf1d": tf1d, "valid": valid, "freq": freq, **spec}


def calibrate(results: dict, convention: str, m: np.ndarray) -> dict:
    center = N_LATERAL // 2
    per_run = []
    for res in results.values():
        if res["ricker_is"] != convention:
            continue
        v = res["valid"]
        per_run.append(np.median(res["tf"][v, center][:, m] / res["tf1d"][v, center][:, m], axis=1))
    per_run = np.concatenate(per_run) if per_run else np.empty(0)
    if per_run.size == 0:
        raise RuntimeError(f"no runs to calibrate convention {convention!r}")
    return {
        "scale": float(np.median(per_run)),
        "per_run_p05": float(np.percentile(per_run, 5)),
        "per_run_p95": float(np.percentile(per_run, 95)),
        "n_runs": int(per_run.size),
        "ricker_is": convention,
        "band_hz": list(CALIBRATION_BAND_HZ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--campaigns", nargs="+", default=list(CAMPAIGNS))
    ap.add_argument("--limit", type=int, default=None, help="first N manifest rows (smoke test)")
    ap.add_argument("--window-s", type=float, default=None, help="fixed FFT window in seconds (labels.LABEL_WINDOW_S)")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="write <out-root>/<campaign>/ instead of next to the manifest; required with --window-s")
    args = ap.parse_args()
    if args.window_s is not None and args.out_root is None:
        raise SystemExit("--window-s relabels must not overwrite the shipped labels; pass --out-root")

    names = [n for n in args.campaigns if CAMPAIGNS[n]["manifest"].is_file()]
    results = {name: run_campaign(name, args.workers, args.limit, args.window_s) for name in names}
    grids = [r["freq"] for r in results.values() if r["freq"] is not None]
    freq = grids[0]
    for g in grids[1:]:
        if not np.allclose(g, freq, rtol=1e-6):
            raise RuntimeError("campaigns use different frequency grids")
    m = band_mask(freq, CALIBRATION_BAND_HZ)
    calib = {c: calibrate(results, c, m) for c in {r["ricker_is"] for r in results.values()}}
    print(json.dumps(calib, indent=2))

    for name, res in results.items():
        cal = calib[res["ricker_is"]]
        scale = np.float32(cal["scale"])
        out = Path(res["out_dir"]) if args.out_root is None else args.out_root / name
        out.mkdir(parents=True, exist_ok=True)
        tf = res["tf"] / scale
        np.save(out / "tf_outcrop_per_sample.npy", tf)
        np.save(out / "tf1d_outcrop_per_sample.npy", res["tf1d"])
        np.save(out / "outcrop_valid.npy", res["valid"])
        np.save(out / "outcrop_freq.npy", freq.astype(np.float32))
        v = res["valid"]
        resid = np.log(np.maximum(tf[v], 1e-6)) - np.log(np.maximum(res["tf1d"][v], 1e-6))
        stats = {
            **cal,
            "campaign": name,
            "window_s": args.window_s,
            "n_valid": int(v.sum()),
            "log_residual_rms_calib_band": float(np.sqrt(np.mean(resid[..., m] ** 2))),
            "log_residual_rms_0.1_2.5": float(np.sqrt(np.mean(resid[..., band_mask(freq, (0.1, 2.5))] ** 2))),
        }
        (out / "outcrop_calibration.json").write_text(json.dumps(stats, indent=2))
        print(name, {k: stats[k] for k in ("n_valid", "log_residual_rms_calib_band", "log_residual_rms_0.1_2.5")})


if __name__ == "__main__":
    main()
