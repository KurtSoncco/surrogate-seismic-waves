#!/usr/bin/env python3
"""Estimate TREND_FREQ_SCALE = median(f_truth / f_H1D_uncalibrated).

Uses GIFNO TF cache (center recorder) when GIFNO_DATA_ROOT is set; else RV
OpenSees-2D pancake. Writes JSON + prints export line for training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP))
import config

config.setup_import_paths()

from haskell_baseline import haskell_trend_af_within  # noqa: E402

_EPS = 1e-12


def _first_peak(freq: np.ndarray, af: np.ndarray) -> float:
    af = np.asarray(af, float).ravel()
    freq = np.asarray(freq, float).ravel()
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(
            af, prominence=max(0.05 * (af.max() - af.min()), 1e-6)
        )
        if peaks.size:
            return float(freq[int(peaks[0])])
    except Exception:
        pass
    return float(freq[int(np.argmax(af))])


def calibrate_gifno(limit: int | None) -> dict:
    import csv

    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass

    freq = np.load(config.TF_FREQ_PATH)
    tf = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
    # tf: (N, R, F) — center recorder
    r_c = tf.shape[1] // 2
    rows = []
    with config.MANIFEST_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if limit is not None:
        n = min(n, limit)
    scales = []
    for i in range(n):
        row = rows[i]
        vs1 = float(row.get("Vs1") or row.get("vs1") or 0)
        H = float(row.get("H") or row.get("h") or 0)
        vs2 = float(row.get("Vs2") or row.get("vs2") or 0)
        if vs1 <= 0 or H <= 0 or vs2 <= 0:
            # attrs may live only on H5
            import h5py
            from pathlib import Path as P

            p = P(row["h5_path"])
            if not p.is_file():
                p = config.H5_DIR / p.name
            with h5py.File(p, "r") as hf:
                vs1 = float(hf["params"].attrs["Vs1"])
                H = float(hf["params"].attrs["H"])
                vs2 = float(hf["params"].attrs["Vs2"])
        truth = np.asarray(tf[i, r_c], dtype=float)
        raw = haskell_trend_af_within(
            freq, vs1=vs1, H=H, vs2=vs2, xi=config.DEFAULT_XI_TREND
        )
        ft, fr = _first_peak(freq, truth), _first_peak(freq, raw)
        if ft > _EPS and fr > _EPS:
            scales.append(ft / fr)
    scales = np.asarray(scales, float)
    return {
        "source": "gifno",
        "n": int(scales.size),
        "TREND_FREQ_SCALE": float(np.median(scales)),
        "mean": float(np.mean(scales)),
        "p25": float(np.percentile(scales, 25)),
        "p75": float(np.percentile(scales, 75)),
    }


def calibrate_rv(max_seeds: int = 5, max_sobol: int = 32) -> dict:
    import h5py

    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    sys.path.insert(0, str(config.RV_ROOT))
    os.environ.pop("RV_SMOKE", None)
    from manifest import (  # noqa: WPS433
        _hallal_block_size,
        active_rf_seeds,
        active_sobol_count,
        index_to_params,
        total_combinations,
    )

    hallal0 = _hallal_block_size(active_sobol_count())
    lookup = {}
    for i in range(hallal0, total_combinations()):
        p = index_to_params(i)
        if p.method == "opensees_2d":
            lookup[(p.sobol_id, p.seed)] = i
    seeds = active_rf_seeds()[:max_seeds]
    n_sobol = min(max_sobol, active_sobol_count())
    scales = []
    for sid in range(n_sobol):
        for seed in seeds:
            idx = lookup.get((sid, seed))
            if idx is None:
                continue
            path = config.RV_H5_DIR / f"run_{idx}.h5"
            if not path.is_file():
                continue
            with h5py.File(path, "r") as f:
                vs1 = float(f["params"].attrs["Vs1"])
                H = float(f["params"].attrs["H"])
                vs2 = float(f["params"].attrs["Vs2"])
                af = f["transfer_function"]["AF"][:].astype(float)
                freq = f["transfer_function"]["freq"][:].astype(float)
            raw = haskell_trend_af_within(
                freq, vs1=vs1, H=H, vs2=vs2, xi=config.DEFAULT_XI_TREND
            )
            ft, fr = _first_peak(freq, af), _first_peak(freq, raw)
            if ft > _EPS and fr > _EPS:
                scales.append(ft / fr)
    scales = np.asarray(scales, float)
    return {
        "source": "rv_pancake",
        "n": int(scales.size),
        "TREND_FREQ_SCALE": float(np.median(scales)),
        "mean": float(np.mean(scales)),
        "p25": float(np.percentile(scales, 25)),
        "p75": float(np.percentile(scales, 75)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("auto", "gifno", "rv"), default="auto")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--out", type=Path, default=config.DIAGNOSTICS_DIR / "trend_freq_scale.json")
    args = p.parse_args()

    src = args.source
    if src == "auto":
        src = "gifno" if config.TF_PER_SAMPLE_PATH.is_file() else "rv"

    if src == "gifno":
        if not config.TF_PER_SAMPLE_PATH.is_file():
            raise SystemExit(f"Missing TF cache: {config.TF_PER_SAMPLE_PATH}")
        result = calibrate_gifno(args.limit)
    else:
        result = calibrate_rv()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    scale = result["TREND_FREQ_SCALE"]
    print(json.dumps(result, indent=2))
    print(f"\nexport THFNO_TREND_FREQ_SCALE={scale:.8f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
