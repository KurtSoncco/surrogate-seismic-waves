#!/usr/bin/env python3
"""Crop whole-section runs into 500 m training windows with outcrop labels.

For every section H5 written by run_sections.py, windows start every 15 m
from ``margin`` to ``Lx - 500 - margin`` (margin = shift augmentation reach,
90 m), about 90 windows per 2000 m run. Each window carries the 21 stations
at window columns 250 + 15k, the same slots as the 1500 m training runs, so
``recorder_x_idx.npy`` is unchanged and the loader crops at ``x_start``.

Labels match relabel_outcrop.py: surface / (c * input) with the acceleration
Ricker convention, and the vertical-SH surface/outcrop prior of the column
under each station with per-layer Rayleigh xi at the run's own
(damping_freq_first, damping_freq_second). ``c`` is fitted on these runs at
stations at least 300 m from every feature (0.4-2.0 Hz median of ratio/1D)
and written to outcrop_calibration.json next to the windows.

Output DIR is a ``--source`` for build_outcrop_dataset.py (rows carry
h5_path, x_start, group = run, split).

    python section_windows.py --h5-dir $SCRATCH/case7_sections/round0/h5 --out DIR [--workers 20]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from labels import CALIBRATION_BAND_HZ, band_mask, seiskit_input_acceleration, tf_over_input
from section_1d import RHO, rayleigh_xi_target, surface_over_outcrop

WINDOW_NX = 500
STATION_SPACING = 15
WINDOW_SLOTS = 250 + STATION_SPACING * np.arange(-10, 11)
MARGIN_M = 90


def collocation_slots(stride_m: float = STATION_SPACING, nx: int = WINDOW_NX) -> np.ndarray:
    """Window-local columns used as TF collocation points.

    The 21 training stations (15 m, centered) are always included. A finer
    ``stride_m`` adds extra surface columns so the shared-POD head can fit
    TF(x) between those stations. ``stride_m == 15`` is the 21-station set.
    """
    stations = np.asarray(WINDOW_SLOTS, dtype=int)
    stride = max(1, int(round(float(stride_m))))
    if stride == STATION_SPACING:
        return stations
    extra = np.arange(0, nx, stride, dtype=int)
    return np.unique(np.concatenate([stations, extra]))
FLAT_DISTANCE_M = 300.0
MOTION_T_SHIFT = 0.5
MAX_FREQ_HZ = 10.0

_TTF = None


def _ttf():
    global _TTF
    if _TTF is None:
        from run_case7a import load_ttf_batch_fast

        _TTF = load_ttf_batch_fast()
    return _TTF


def window_starts(lx: int, margin: int = MARGIN_M) -> np.ndarray:
    return np.arange(margin, lx - WINDOW_NX - margin + 1, STATION_SPACING)


def _feature_spans(params: dict) -> list[tuple[float, float]]:
    lx = float(params["lx_m"])
    feats = [dict(feature=params["feature"], feature_x_frac=params["feature_x_frac"], feature_width_m=params["feature_width_m"])]
    feats += json.loads(str(params.get("extra_features") or "[]"))
    spans = []
    for f in feats:
        if int(float(f["feature"])) == 0:
            continue
        xc, w = float(f["feature_x_frac"]) * lx, float(f["feature_width_m"])
        spans.append((xc - w / 2, xc + w / 2))
    return spans


def label_run(path: str, stride_m: float = STATION_SPACING):
    import hdf5plugin  # noqa: F401
    import h5py

    with h5py.File(path, "r") as f:
        surface = f["recorders/accel/surface"]
        x_nodes = surface.attrs["x_m"]
        vs = f["Vs_realization_2D"][:].astype(np.float64)
        from applied_damping import applied_zeta

        zeta = applied_zeta(f)
        time = f["recorders/accel/time"][:].astype(np.float64)
        grid = dict(f["grid"].attrs)
        params = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["params"].attrs.items()}
        dx = float(grid["dx"])
        starts = (window_starts(int(round(float(grid["Lx"]))), MARGIN_M) / dx).astype(int)
        slots = (collocation_slots(stride_m, WINDOW_NX) / dx).astype(int)
        nodes = np.unique((starts[:, None] + slots[None, :]).ravel())
        acc = surface[:, nodes].astype(np.float64)
    if not np.allclose(x_nodes[nodes], nodes * dx):
        raise ValueError(f"{path}: surface node x does not match index * dx")
    dt = float(time[1] - time[0])
    input_acc = seiskit_input_acceleration(float(grid["motion_freq"]), MOTION_T_SHIFT, time, str(params.get("ricker_is", "acceleration")))
    freq, tf_raw = tf_over_input(acc, input_acc, dt, _ttf())
    f1, f2 = float(params["damping_freq_first"]), float(params["damping_freq_second"])
    tf1d = np.empty_like(tf_raw)
    for j, col in enumerate(np.minimum(nodes, vs.shape[1] - 1)):
        xi = np.stack([rayleigh_xi_target(freq, z, f1, f2) for z in zeta[:, col]])
        tf1d[j] = surface_over_outcrop(freq, vs[:, col], dz=float(grid["dz"]), xi=xi, rho=RHO)
    spans = _feature_spans(params)
    x_m = nodes * dx
    flat = np.array([all(x < a - FLAT_DISTANCE_M or x > b + FLAT_DISTANCE_M for a, b in spans) for x in x_m])
    flat &= (x_m > FLAT_DISTANCE_M) & (x_m < float(grid["Lx"]) - FLAT_DISTANCE_M)
    return {
        "path": path,
        "sample_id": str(params.get("sample_id", Path(path).stem)),
        "split": str(params.get("split", "train")),
        "freq": freq,
        "nodes": nodes,
        "starts": starts,
        "slots": slots,
        "dx": dx,
        "tf": tf_raw.astype(np.float32),
        "tf1d": tf1d.astype(np.float32),
        "flat": flat,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5-dir", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--scale", type=float, default=None, help="fixed c instead of fitting it on flat stations")
    ap.add_argument(
        "--collocation-stride",
        type=float,
        default=5.0,
        help="meters between extra surface collocation points (15 = the 21 stations only)",
    )
    args = ap.parse_args()

    paths = sorted(str(p) for d in args.h5_dir for p in d.glob("*.h5") if not p.name.startswith(("smoke", "timing")))
    if not paths:
        raise SystemExit("no section H5 files")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(paths))) as pool:
        runs = list(pool.map(partial(label_run, stride_m=args.collocation_stride), paths))
    freq = runs[0]["freq"]
    m = band_mask(freq, CALIBRATION_BAND_HZ)
    ratios = [np.median(r["tf"][r["flat"]][:, m] / r["tf1d"][r["flat"]][:, m], axis=1) for r in runs if r["flat"].any()]
    ratios = np.concatenate(ratios) if ratios else np.full(1, np.nan)
    if not args.scale and not np.isfinite(ratios).any():
        raise SystemExit("no station is 300 m from every feature and edge; pass --scale")
    scale = float(args.scale) if args.scale else float(np.median(ratios))
    print(f"c = {scale:.4f} (flat stations: n={ratios.size}, p05-p95 {np.percentile(ratios, 5):.3f}-{np.percentile(ratios, 95):.3f})")

    rows, tf, tf1d = [], [], []
    for r in runs:
        col = {n: i for i, n in enumerate(r["nodes"])}
        for s in r["starts"]:
            idx = [col[s + k] for k in r["slots"]]
            rows.append({
                "h5_path": r["path"], "x_start": int(s), "group": r["sample_id"], "split": r["split"],
                "sample_id": r["sample_id"], "x_center_m": float(s * r["dx"] + 250),
            })
            tf.append(r["tf"][idx] / scale)
            tf1d.append(r["tf1d"][idx])
    tf, tf1d = np.stack(tf), np.stack(tf1d)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.save(out / "tf_outcrop_per_sample.npy", tf.astype(np.float32))
    np.save(out / "tf1d_outcrop_per_sample.npy", tf1d.astype(np.float32))
    np.save(out / "outcrop_valid.npy", np.isfinite(tf).all(axis=(1, 2)) & np.isfinite(tf1d).all(axis=(1, 2)))
    np.save(out / "outcrop_freq.npy", freq.astype(np.float32))
    np.save(out / "recorder_x_idx.npy", collocation_slots(args.collocation_stride, WINDOW_NX).astype(np.int64))
    resid = np.log(np.maximum(tf, 1e-6)) - np.log(np.maximum(tf1d, 1e-6))
    stats = {
        "scale": scale,
        "ricker_is": "acceleration",
        "n_runs": len(runs),
        "n_windows": len(rows),
        "n_collocation": int(tf.shape[1]),
        "collocation_stride_m": float(args.collocation_stride),
        "n_flat_stations": int(ratios.size),
        "band_hz": list(CALIBRATION_BAND_HZ),
        "log_residual_rms_0.1_2.5": float(np.sqrt(np.mean(resid[..., band_mask(freq, (0.1, 2.5))] ** 2))),
    }
    (out / "outcrop_calibration.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
