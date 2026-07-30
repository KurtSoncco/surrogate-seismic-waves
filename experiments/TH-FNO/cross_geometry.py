#!/usr/bin/env python3
"""Cross-geometry sample-efficiency protocol (AGENTS §5).

Reports:
  - Capability H5 availability (three_layer / dipping)
  - Zero-shot trend H_1D baselines on RV (center AF) as a geometry-A sanity row
  - Few-shot budgets {0,10,100,1000}: pending TF labels on capability cases
    (accel recorders present; TF cache / FFT not yet wired)

Per-recorder edge scoring is used wherever multi-recorder TFs exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
import config

config.setup_import_paths()

from haskell_baseline import H_1D_trend  # noqa: E402

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


def _rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-12))


def score_rv_trend_zeroshot(max_seeds: int = 5, max_sobol: int = 16) -> dict:
    """Center-only trend vs OpenSees AF on an RV subset (per-recorder edges N/A)."""
    h5_dir = config.RV_H5_DIR
    if not h5_dir.is_dir():
        return {"status": "missing_rv_h5"}
    # Prefer campaign index when available (opensees_2d only)
    paths: list[Path] = []
    try:
        sys.path.insert(0, str(config.RV_ROOT))
        from analysis.index_map import (  # type: ignore
            active_rf_seeds,
            active_sobol_count,
            index_to_params,
            total_combinations,
        )

        lookup = {}
        for i in range(total_combinations()):
            p = index_to_params(i)
            if p.method == "opensees_2d":
                lookup[(p.sobol_id, p.seed)] = i
        n_s = min(max_sobol, active_sobol_count())
        seeds = list(active_rf_seeds())[:max_seeds]
        for sid in range(n_s):
            for seed in seeds:
                idx = lookup.get((sid, seed))
                if idx is None:
                    continue
                path = h5_dir / f"run_{idx}.h5"
                if path.is_file():
                    paths.append(path)
    except Exception:
        paths = sorted(h5_dir.glob("run_*.h5"))[: max_seeds * max_sobol]
    if not paths:
        return {"status": "empty"}
    rels = []
    for path in paths:
        with h5py.File(path, "r") as f:
            if "transfer_function" not in f:
                continue
            freq = np.asarray(f["transfer_function"]["freq"][:], dtype=float)
            af = np.asarray(f["transfer_function"]["AF"][:], dtype=float)
            p = f["params"].attrs
            if "Vs1" not in p:
                continue
            vs1 = float(p["Vs1"])
            H = float(p["H"])
            vs2 = float(p["Vs2"])
            xi = float(p["xi"]) if "xi" in p else 0.05
        trend = H_1D_trend(freq, vs1=vs1, H=H, vs2=vs2, xi=xi)
        rels.append(_rel_l2(trend, af))
    if not rels:
        return {"status": "no_valid_pairs"}
    return {
        "status": "ok",
        "n": len(rels),
        "rel_l2_center_mean": float(np.mean(rels)),
        "rel_l2_center_std": float(np.std(rels)),
        "note": "RV AF is center-only; edge metrics require 21-recorder GIFNO/TF cache",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=config.MODEL_SAVE_PATH)
    p.add_argument("--budgets", type=str, default="0,10,100,1000")
    p.add_argument("--max-rv-pairs", type=int, default=80)
    args = p.parse_args()
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]

    out = config.RESULTS_SAVE_DIR / "cross_geometry"
    out.mkdir(parents=True, exist_ok=True)

    caps = {}
    for name in ("three_layer", "dipping"):
        d = config.CAPABILITY_ROOT / name / "h5"
        caps[name] = sorted(d.glob("case_*.h5")) if d.is_dir() else []

    rv_trend = score_rv_trend_zeroshot(
        max_seeds=max(1, args.max_rv_pairs // 16),
        max_sobol=min(16, max(1, args.max_rv_pairs)),
    )

    curves = {}
    for geo, files in caps.items():
        has_accel = False
        if files:
            with h5py.File(files[0], "r") as f:
                has_accel = "recorders" in f and "accel" in f["recorders"]
        curves[geo] = {
            str(b): {
                "n_available": len(files),
                "has_accel_recorders": has_accel,
                "status": (
                    "pending_tf_labels"
                    if files
                    else "missing_h5"
                ),
                "rel_l2_center": None,
                "rel_l2_edge": None,
            }
            for b in budgets
        }

    report = {
        "predict_mode_default": config.PREDICT_MODE,
        "d2_pass": False,
        "checkpoint_exists": args.checkpoint.is_file(),
        "capability_counts": {k: len(v) for k, v in caps.items()},
        "budgets": budgets,
        "rv_trend_zeroshot": rv_trend,
        "note": (
            "D2 FAIL → train direct |TF| on GIFNO corpus; few-shot curves need "
            "capability TF labels derived from accel or OpenSees TF cache."
        ),
        "curves": curves,
    }

    path = out / "sample_efficiency.json"
    path.write_text(json.dumps(report, indent=2))

    # Flat CSV for dashboards
    csv_path = out / "sample_efficiency_status.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["geometry", "budget", "n_available", "status"])
        for geo, by_b in curves.items():
            for b, row in by_b.items():
                w.writerow([geo, b, row["n_available"], row["status"]])

    print(f"Wrote {path}")
    print("RV trend zero-shot:", json.dumps(rv_trend, indent=2))
    print("capability:", json.dumps(report["capability_counts"], indent=2))


if __name__ == "__main__":
    main()
