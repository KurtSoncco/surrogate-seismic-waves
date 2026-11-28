#!/usr/bin/env python3
"""Does a wider Case 7 window change the surface motion, and does damping?

Compares log |TF| of the 500/1000/1500 m Q–Vs crops to the full 4000 m Q–Vs
section, the full Q–Vs section to the full SVLAB-damping section, and that
SVLAB-damping section to the SVLAB record. Stations are every 5 m. A constant
scale is removed only for the SVLAB comparison, where the input wavelet differs.

    python context_convergence.py
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

from case7_crops import CROPS  # noqa: E402
from labels import band_mask, calibrate_input_scale, seiskit_input_acceleration, tf_over_input  # noqa: E402

CROP500 = REPO / "data" / "case7a" / "opensees_crops" / "h5"
CONTEXT = REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5"
OUT = REPO / "checkpoints" / "case7_generalization" / "context_gate"
BANDS = {"low": (0.1, 0.4), "trusted": (0.1, 2.5)}
STRIDE_M = 5
# A crop "matches" the full section when the low band is this close.
MATCH_ABS_LOG = 0.05
MATCH_PEARSON = 0.97
WEST_X = (-2000.0, -650.0)


def _ttf():
    from run_case7a import load_ttf_batch_fast

    return load_ttf_batch_fast()


def _attrs(handle) -> dict:
    out = {}
    for key, value in handle["params"].attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        elif hasattr(value, "item"):
            value = value.item()
        out[key] = value
    return out


def load_tf(path: Path, ttf) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        params = _attrs(handle)
        surface = handle["recorders/accel/surface"][:]
        x = np.asarray(handle["recorders/accel/surface"].attrs["x_m"], dtype=np.float64)
        time = handle["recorders/accel/time"][:].astype(np.float64)
        freq_motion = float(handle["grid"].attrs["motion_freq"])
    step = STRIDE_M
    keep = np.arange(0, x.size, step)
    dt = float(time[1] - time[0])
    acc = seiskit_input_acceleration(freq_motion, 0.5, time, "acceleration")
    freq, tf = tf_over_input(surface[:, keep], acc, dt, ttf)
    return {"id": path.stem, "params": params, "x": x[keep], "freq": freq, "tf": tf}


def _aligned(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xb = {round(float(x), 3): i for i, x in enumerate(b["x"])}
    ia, ib = [], []
    for i, x in enumerate(a["x"]):
        j = xb.get(round(float(x), 3))
        if j is not None:
            ia.append(i)
            ib.append(j)
    if not ia:
        raise RuntimeError(f"no shared stations between {a['id']} and {b['id']}")
    return np.array(ia), np.array(ib), a["x"][ia]


def _metrics(tf_a: np.ndarray, tf_b: np.ndarray, freq: np.ndarray, x: np.ndarray) -> dict:
    out = {}
    log_a = np.log(np.maximum(tf_a, 1e-12))
    log_b = np.log(np.maximum(tf_b, 1e-12))
    for name, band in BANDS.items():
        m = band_mask(freq, band)
        gap = np.mean(np.abs(log_a[:, m] - log_b[:, m]), axis=1)
        aa, bb = log_a[:, m], log_b[:, m]
        aa0, bb0 = aa - aa.mean(axis=1, keepdims=True), bb - bb.mean(axis=1, keepdims=True)
        pear = np.sum(aa0 * bb0, axis=1) / np.maximum(
            np.sqrt(np.sum(aa0**2, axis=1) * np.sum(bb0**2, axis=1)), 1e-12
        )
        west = (x >= WEST_X[0]) & (x < WEST_X[1])
        out[name] = {
            "mean_abs_log": float(np.mean(gap)),
            "mean_pearson": float(np.mean(pear)),
            "west_mean_abs_log": float(np.mean(gap[west])) if np.any(west) else None,
            "west_mean_pearson": float(np.mean(pear[west])) if np.any(west) else None,
        }
    return out


def _matches(row: dict, band: str = "trusted") -> bool:
    part = row[band]
    return part["mean_abs_log"] < MATCH_ABS_LOG and part["mean_pearson"] > MATCH_PEARSON


def _records_identical(a: Path, b: Path) -> bool:
    import h5py

    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        sa, sb = fa["recorders/accel/surface"], fb["recorders/accel/surface"]
        if sa.shape != sb.shape:
            return False
        return bool(np.array_equal(sa[:, ::50], sb[:, ::50]))


def _svlab(freq_ref: np.ndarray) -> dict:
    """SVLAB outcrop |TF| on the 5 m surface nodes."""
    from case7a import ensure_cache, window_node_index
    from run_case7a import outcrop_reference_transfer_function

    pack = ensure_cache()
    xs, nodes = [], []
    for v in np.arange(-2000.0, 2000.0, STRIDE_M):
        nodes.append(window_node_index(float(v)))
        xs.append(float(v))
    freq, tf, scale = outcrop_reference_transfer_function(pack, np.array(nodes, dtype=np.int64))
    if freq.shape != freq_ref.shape or not np.allclose(freq, freq_ref):
        tf = np.vstack([np.interp(freq_ref, freq, row) for row in tf])
        freq = freq_ref
    return {"id": "svlab", "x": np.array(xs), "freq": freq, "tf": tf, "scale": scale}


def decide(report: dict) -> dict:
    """Width gate on 0.1–2.5 Hz, scored on each crop's 500 m stations. The low band is a flag."""
    by_width: dict[int, list[bool]] = {}
    low_width: dict[int, list[bool]] = {}
    for row in report["crop_vs_full_qvs"]:
        by_width.setdefault(int(row["inner_m"]), []).append(_matches(row, "trusted"))
        low_width.setdefault(int(row["inner_m"]), []).append(_matches(row, "low"))
    w500 = all(by_width.get(500, []))
    w1000 = all(by_width.get(1000, []))
    w1500 = all(by_width.get(1500, []))
    if w500:
        choice = "head"
        reason = "On 0.1-2.5 Hz, 500 m crops already match the full section on the same stations."
    elif w1000 or w1500:
        choice = "vardomain"
        reason = "On 0.1-2.5 Hz, a 1000 or 1500 m window is needed to match the full section."
    else:
        choice = "whole_section"
        reason = "On 0.1-2.5 Hz, even 1500 m does not match the full section on the same stations."
    damp = report.get("damping_qvs_vs_svlab", {})
    if damp.get("identical"):
        damp_note = "full_svlab is the same record as full_qvs, so the damping comparison was not run."
    else:
        low = damp["low"]
        sv = report.get("svlab_vs_full_svlab", {}).get("low", {})
        damp_note = (
            f"Q-Vs vs SVLAB damping on the full section: low-band |dlog|={low['mean_abs_log']:.3f}, "
            f"west |dlog|={low['west_mean_abs_log']}."
        )
        if sv:
            west = sv.get("west_mean_pearson")
            if west is not None and west > 0.8:
                damp_note += " The SVLAB-damping run tracks SVLAB in the west, so the west hole is damping."
            else:
                damp_note += " The SVLAB-damping run still misses SVLAB in the west, so damping alone does not explain it."
    return {
        "choice": choice,
        "reason": reason,
        "damping": damp_note,
        "width_match": {"500": w500, "1000": w1000, "1500": w1500},
        "low_band_match": {str(k): all(v) for k, v in sorted(low_width.items())},
        "scored_on": "stations of the 500 m crop",
        "decision_band_hz": [0.1, 2.5],
    }


def main() -> None:
    import h5py

    ttf = _ttf()
    runs = {}
    for path in sorted(CROP500.glob("crop_*.h5")):
        print(f"tf {path.name}", flush=True)
        runs[path.stem] = load_tf(path, ttf)
        with h5py.File(path, "r") as handle:
            runs[path.stem]["inner_m"] = 500
            runs[path.stem]["name"] = str(handle["params"].attrs.get("name", path.stem))
    missing = []
    for path in sorted(CONTEXT.glob("*.h5")):
        print(f"tf {path.name}", flush=True)
        runs[path.stem] = load_tf(path, ttf)
        runs[path.stem]["inner_m"] = int(round(float(runs[path.stem]["params"].get("inner_m", 0))))
        runs[path.stem]["name"] = str(runs[path.stem]["params"].get("name", ""))
    for crop in CROPS:
        for width in (1000, 1500):
            sid = f"{crop['sample_id']}_w{width}"
            if sid not in runs:
                missing.append(sid)
    for sid in ("full_qvs", "full_svlab"):
        if sid not in runs:
            missing.append(sid)
    if missing:
        raise SystemExit(f"missing H5s: {missing}")

    full = runs["full_qvs"]
    rows = []
    for crop in CROPS:
        base = runs[crop["sample_id"]]
        lo, hi = float(base["x"].min()), float(base["x"].max())
        for width, sid in ((500, crop["sample_id"]), (1000, f"{crop['sample_id']}_w1000"), (1500, f"{crop['sample_id']}_w1500")):
            ia, ib, x = _aligned(runs[sid], full)
            keep = (x >= lo) & (x <= hi)
            metrics = _metrics(runs[sid]["tf"][ia][keep], full["tf"][ib][keep], full["freq"], x[keep])
            rows.append({"sample_id": sid, "name": crop["name"], "inner_m": width, "n_stations": int(keep.sum()), **metrics})
            print(
                f"{sid}: trusted |dlog|={metrics['trusted']['mean_abs_log']:.3f} "
                f"pearson={metrics['trusted']['mean_pearson']:.3f} "
                f"low |dlog|={metrics['low']['mean_abs_log']:.3f}",
                flush=True,
            )
    identical = _records_identical(CONTEXT / "full_qvs.h5", CONTEXT / "full_svlab.h5")
    if identical:
        print("full_svlab matches full_qvs; damping comparison skipped", flush=True)
        report = {"crop_vs_full_qvs": rows, "damping_qvs_vs_svlab": {"identical": True}}
    else:
        ia, ib, x = _aligned(full, runs["full_svlab"])
        damping = _metrics(full["tf"][ia], runs["full_svlab"]["tf"][ib], full["freq"], x)
        svlab = _svlab(full["freq"])
        from labels import CALIBRATION_BAND_HZ

        ia, ib, x = _aligned(runs["full_svlab"], svlab)
        east = x > 1300.0
        scale = calibrate_input_scale(
            runs["full_svlab"]["tf"][ia][east], svlab["tf"][ib][east], full["freq"], CALIBRATION_BAND_HZ
        )
        svlab_metrics = _metrics(runs["full_svlab"]["tf"][ia] / scale, svlab["tf"][ib], full["freq"], x)
        report = {
            "crop_vs_full_qvs": rows,
            "damping_qvs_vs_svlab": damping,
            "svlab_scale": scale,
            "svlab_vs_full_svlab": svlab_metrics,
        }
    report["decision"] = decide(report)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "convergence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
