#!/usr/bin/env python3
"""Why the logged iid validation error is not the test error.

Training scores the first 32 valid iid validation runs (``iid_limit`` in
``train.py``). ``split.npy`` holds 720 validation runs and 720 test runs.
The epoch is chosen by the manifold validation log-L1, stored as ``val_score``.

    python run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "scripts" / "_entry.py").is_file():
        sys.path.insert(0, str(_parent / "scripts"))
        break
from _entry import enter

enter(_HERE)

import numpy as np
import pandas as pd

from paths import CACHE, STAGE2

OUT = _HERE.parent
IID_LIMIT = 32
STATIONS = 21


def _column_mid(log_tf: np.ndarray, log_1d: np.ndarray, rows: np.ndarray, mid: np.ndarray) -> float:
    a = np.asarray(log_1d[rows], dtype=np.float64)
    b = np.asarray(log_tf[rows], dtype=np.float64)
    return float(np.mean(np.abs(a[:, :, mid] - b[:, :, mid])))


def main() -> None:
    split = np.load(CACHE / "iid" / "split.npy").astype(str)
    group = np.load(CACHE / "iid" / "group.npy")
    valid_path = CACHE / "iid" / "valid.npy"
    valid = np.load(valid_path).astype(bool) if valid_path.is_file() else np.ones(split.shape, dtype=bool)
    freq = np.load(CACHE / "freq.npy")
    mid = (freq >= 0.4) & (freq <= 2.5)
    f1 = np.load(CACHE / "iid" / "rayleigh_f1.npy")
    log_tf = np.load(CACHE / "iid" / "log_tf.npy", mmap_mode="r")
    log_1d = np.load(CACHE / "iid" / "log_tf1d.npy", mmap_mode="r")
    val = np.flatnonzero((split == "val") & valid)
    test = np.flatnonzero((split == "test") & valid)
    capped = val[:IID_LIMIT]
    history = pd.read_csv(STAGE2 / "history.csv")
    chosen = history.loc[history["epoch"] == 53].iloc[0]
    test_eval = json.loads((STAGE2 / "eval_test.json").read_text())
    payload = {
        "iid_limit_runs": IID_LIMIT,
        "stations_per_run": int(log_tf.shape[1]),
        "capped_val": {
            "runs": int(capped.size),
            "sites": int(np.unique(group[capped]).size),
            "stations_one_width": int(capped.size * STATIONS),
            "f1_median_hz": float(np.median(f1[capped])),
            "column_log_l1_mid": _column_mid(log_tf, log_1d, capped, mid),
        },
        "full_val": {
            "runs": int(val.size),
            "sites": int(np.unique(group[val]).size),
            "stations_one_width": int(val.size * STATIONS),
            "f1_median_hz": float(np.median(f1[val])),
            "column_log_l1_mid": _column_mid(log_tf, log_1d, val, mid),
        },
        "test": {
            "runs": int(test.size),
            "sites": int(np.unique(group[test]).size),
            "stations_one_width": int(test_eval["bins"]["iid_w500"]["n_stations"]),
            "f1_median_hz": float(np.median(f1[test])),
            "column_log_l1_mid": _column_mid(log_tf, log_1d, test, mid),
            "model_log_l1_mid": float(test_eval["bins"]["iid_w500"]["log_l1_mid"]),
            "model_log_l1_fullband_mean": float(test_eval["iid_mean"]["log_l1"]),
        },
        "epoch_53": {
            "val_score_manifold_log_l1": float(chosen["val_score"]),
            "iid_log_l1_on_the_32_run_cap": float(chosen["iid_log_l1"]),
            "selection": "val_score is the mean manifold validation log-L1. iid_log_l1 is not the quantity that picked the epoch.",
        },
        "why": (
            "The 32-run cap is the first 32 valid validation runs, and those runs are two sites. "
            "Their median f1 is higher and their column mid-band error is smaller than the full "
            "validation split or the test split. The test figure scores all 720 test runs."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
