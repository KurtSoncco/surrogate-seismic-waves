#!/usr/bin/env python3
"""Vertical-SH check of the Case 7 reference along the whole section.

For every 15 m window center, compares the OpenSees |TF| with the 1D column
transfer function under two denominators:

* ``downhole``: surface(x) / downhole(x = 0, 120 m), against 1D surface/within-120 m.
* ``input``: surface(x) / (c * input motion), against 1D surface/outcrop.

``c`` is calibrated in the flat east zone. Where 1D matches in the flat zones
the label is physically sound and any model error there is the model's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from case7a import ZONES, ensure_cache, window_center_x, window_node_index, zone_of
from labels import (
    CORE_BAND_HZ,
    TRUSTED_BAND_HZ,
    band_mask,
    calibrate_input_scale,
    case7_input_acceleration,
    tf_over_input,
)
from run_case7a import OUT_DIR, load_ttf_batch_fast
from section_1d import section_tf_1d

BANDS = {"full": (0.1, 10.0), "trusted": TRUSTED_BAND_HZ, "core": CORE_BAND_HZ}
CALIBRATION_ZONE = "flat_east"


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel() - np.mean(a)
    b = np.asarray(b, dtype=np.float64).ravel() - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 0 else float("nan")


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))


def zone_scores(x: np.ndarray, pred: np.ndarray, true: np.ndarray, freq: np.ndarray) -> dict:
    """Median per-window Pearson and relative L2 by zone and band."""
    zones = zone_of(x)
    out: dict = {}
    for zone in ZONES:
        rows = np.flatnonzero(zones == zone)
        out[zone] = {"n_windows": int(rows.size)}
        for band, rng in BANDS.items():
            m = band_mask(freq, rng)
            p = [_pearson(pred[i, m], true[i, m]) for i in rows]
            r = [_rel_l2(pred[i, m], true[i, m]) for i in rows]
            out[zone][band] = {"pearson_median": float(np.median(p)), "rel_l2_median": float(np.median(r))}
    return out


def compute(pack: dict) -> dict[str, np.ndarray | float]:
    ttf = load_ttf_batch_fast()
    x = window_center_x()
    nodes = np.array([window_node_index(float(v)) for v in x])
    surface = pack["surface_h"][:, nodes]
    dt = float(pack["dt"])
    freq, tf_dh = tf_over_input(surface, pack["downhole"], dt, ttf)
    acc_in = case7_input_acceleration(pack["ricker"], pack["time"])
    _, tf_in_raw = tf_over_input(surface, acc_in, dt, ttf)
    tf1d_within = section_tf_1d(freq, pack["vs"], x, "within120")
    tf1d_outcrop = section_tf_1d(freq, pack["vs"], x, "outcrop")
    calib_rows = zone_of(x) == CALIBRATION_ZONE
    scale = calibrate_input_scale(tf_in_raw[calib_rows], tf1d_outcrop[calib_rows], freq)
    return {
        "x": x,
        "freq": freq,
        "tf_downhole": tf_dh,
        "tf_input": tf_in_raw / scale,
        "tf1d_within120": tf1d_within,
        "tf1d_outcrop": tf1d_outcrop,
        "input_scale": scale,
    }


def plot(res: dict, out_path: Path) -> None:
    x, freq = res["x"], res["freq"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, constrained_layout=True)
    for ax, band in zip(axes, ("trusted", "core")):
        m = band_mask(freq, BANDS[band])
        for key, ref, color, label in (
            ("tf1d_within120", "tf_downhole", "C0", "surface / downhole vs 1D within-120 m"),
            ("tf1d_outcrop", "tf_input", "C3", "surface / input vs 1D outcrop"),
        ):
            ax.plot(x, [_pearson(res[key][i, m], res[ref][i, m]) for i in range(len(x))], color=color, lw=1.2, label=label)
        for name, (lo, hi) in ZONES.items():
            ax.axvspan(max(lo, x[0]), min(hi, x[-1]), alpha=0.06 if name != "geologic" else 0.15, color="0.3")
        ax.set_ylim(-0.2, 1.02)
        ax.set_ylabel("Pearson r")
        ax.set_title(f"1D vs OpenSees, {BANDS[band][0]}-{BANDS[band][1]} Hz")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("x (m)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "section_1d")
    args = parser.parse_args()
    pack = ensure_cache()
    res = compute(pack)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("x", "freq", "tf_downhole", "tf_input", "tf1d_within120", "tf1d_outcrop"):
        np.save(args.out_dir / f"{key}.npy", np.asarray(res[key], dtype=np.float32))
    metrics = {
        "input_scale": res["input_scale"],
        "calibration": f"median(ratio/1D) over {CALIBRATION_ZONE}, 0.4-2.0 Hz",
        "bands_hz": BANDS,
        "downhole_vs_1d_within120": zone_scores(res["x"], res["tf1d_within120"], res["tf_downhole"], res["freq"]),
        "input_vs_1d_outcrop": zone_scores(res["x"], res["tf1d_outcrop"], res["tf_input"], res["freq"]),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot(res, args.out_dir / "section_1d_pearson.png")
    print(json.dumps({k: metrics[k] for k in ("input_scale",)}, indent=2))
    for name in ("downhole_vs_1d_within120", "input_vs_1d_outcrop"):
        print(name)
        for zone, s in metrics[name].items():
            print(f"  {zone:10s} trusted P {s['trusted']['pearson_median']:.3f} L2 {s['trusted']['rel_l2_median']:.3f} | core P {s['core']['pearson_median']:.3f} L2 {s['core']['rel_l2_median']:.3f}")


if __name__ == "__main__":
    main()
