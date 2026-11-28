#!/usr/bin/env python3
"""Flat-west OpenSees versus the column, by record length.

The 15 s Q–Vs crop is local. The 30 s and 60 s pilots (Q–Vs and SVLAB) are
the H5s from Stampede job 3669168, copied to
``data/case7a/opensees_crops/record_pilot/h5``. The FFT window is the same
1310.72 s used for the training labels, so a longer record changes the
low-frequency label rather than the bin spacing.

    python score_record_length.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parents[1]
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from applied_damping import applied_zeta  # noqa: E402
from labels import LABEL_WINDOW_S, seiskit_input_acceleration, tf_over_input  # noqa: E402
from section_1d import RHO, rayleigh_xi_target, surface_over_outcrop  # noqa: E402

# Acceleration-Ricker outcrop scale, the same constant as GIFNO-VARDOMAIN/plot_results.py.
SCALE = 1.9776999950408936
BANDS = {
    "0.1-0.4": (0.1, 0.4),
    "0.4-2.5": (0.4, 2.5),
    "0.1-2.5": (0.1, 2.5),
    "2.5-10": (2.5, 10.0),
    "0.1-10": (0.1, 10.0),
}
STRIDE_M = 20
CROP_15 = REPO / "data" / "case7a" / "opensees_crops" / "h5" / "crop_west.h5"
PILOT_DIR = REPO / "data" / "case7a" / "opensees_crops" / "record_pilot" / "h5"
PILOTS = (
    "crop_west_qvs_d30.h5",
    "crop_west_qvs_d60.h5",
    "crop_west_svlab_d15.h5",
    "crop_west_svlab_d30.h5",
    "crop_west_svlab_d60.h5",
)
OUT = CASE_DIR / "results" / "flat_west_record_length.json"


def _ttf():
    from run_case7a import load_ttf_batch_fast

    return load_ttf_batch_fast()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def score_h5(path: Path, ttf) -> dict:
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(path, "r") as handle:
        vs = np.asarray(handle["Vs_realization_2D"][:], dtype=np.float64)
        zeta = applied_zeta(handle)
        time = np.asarray(handle["recorders/accel/time"][:], dtype=np.float64)
        acc = np.asarray(handle["recorders/accel/surface"][:], dtype=np.float64)
        params = handle["params"].attrs
        f1 = float(params["damping_freq_first"])
        f2 = float(params["damping_freq_second"])
        dx = float(handle["grid"].attrs["dx"])
        x_all = np.asarray(handle["recorders/accel/surface"].attrs["x_m"], dtype=np.float64)
        motion = float(handle["grid"].attrs["motion_freq"])
        ricker = params.get("ricker_is", "acceleration")
        if isinstance(ricker, bytes):
            ricker = ricker.decode()
        damping = params.get("damping", "qvs")
        if isinstance(damping, bytes):
            damping = damping.decode()
        duration = float(params["duration"]) if "duration" in params else float(time[-1] - time[0])
    dt = float(time[1] - time[0])
    width = vs.shape[1]
    sensor = width // 2
    cols = np.arange(0, width, STRIDE_M)
    if sensor not in cols:
        cols = np.sort(np.append(cols, sensor))
    freq, raw = tf_over_input(
        acc[:, cols],
        seiskit_input_acceleration(motion, 0.5, time, str(ricker)),
        dt,
        ttf,
        window_s=LABEL_WINDOW_S,
    )
    truth = raw / SCALE
    tf1d = np.empty_like(truth)
    for j, col in enumerate(cols):
        xi = np.stack([rayleigh_xi_target(freq, z, f1, f2) for z in zeta[:, col]])
        tf1d[j] = surface_over_outcrop(freq, vs[:, col], dz=1.0, xi=xi, rho=RHO)
    log_t = np.log(np.maximum(truth, 1e-6))
    log_d = np.log(np.maximum(tf1d, 1e-6))
    i_s = int(np.where(cols == sensor)[0][0])
    x_m = x_all[cols] if x_all.shape[0] == width else cols.astype(np.float64) * dx
    bands = {}
    for name, (lo, hi) in BANDS.items():
        sl = (freq >= lo) & (freq <= hi)
        gap = np.abs(log_t[:, sl] - log_d[:, sl])
        station_p = np.array([_pearson(log_t[j, sl], log_d[j, sl]) for j in range(cols.size)])
        bands[name] = {
            "mean_abs_log": float(gap.mean()),
            "sensor_abs_log": float(gap[i_s].mean()),
            "sensor_pearson": _pearson(log_t[i_s, sl], log_d[i_s, sl]),
            "pooled_pearson": _pearson(log_t[:, sl].ravel(), log_d[:, sl].ravel()),
            "station_abs_log": gap.mean(axis=1).astype(float).tolist(),
            "station_pearson": station_p.astype(float).tolist(),
        }
    return {
        "file": path.name,
        "damping": str(damping),
        "duration_s": duration,
        "record_s": float(time[-1] - time[0]),
        "f1_hz": f1,
        "f2_hz": f2,
        "n_stations": int(cols.size),
        "stride_m": STRIDE_M,
        "x_m": x_m.astype(float).tolist(),
        "center_x_m": float(x_m[i_s]),
        "bands": bands,
    }


def pilot_paths() -> list[Path]:
    found = [CROP_15] if CROP_15.is_file() else []
    found.extend(p for name in PILOTS if (p := PILOT_DIR / name).is_file())
    return found


def main() -> None:
    paths = pilot_paths()
    if not paths:
        raise SystemExit(f"no flat-west records under {CROP_15} or {PILOT_DIR}")
    ttf = _ttf()
    rows = [score_h5(path, ttf) for path in paths]
    missing = [name for name in PILOTS if not (PILOT_DIR / name).is_file()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"window_s": LABEL_WINDOW_S, "scale": SCALE, "rows": rows, "missing": missing}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{'file':28} {'damp':6} {'dur':6} {'band':8} {'center':8} {'pooled':8} {'min':8} {'max':8}")
    for row in rows:
        for name in ("0.1-2.5", "0.1-10"):
            band = row["bands"][name]
            station = np.asarray(band["station_pearson"])
            print(
                f"{row['file']:28} {row['damping']:6} {row['duration_s']:6.0f} {name:8} "
                f"{band['sensor_pearson']:8.3f} {band['pooled_pearson']:8.3f} "
                f"{station.min():8.3f} {station.max():8.3f}"
            )
    if missing:
        print("missing pilots:", ", ".join(missing))
        raise SystemExit(2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
