#!/usr/bin/env python3
"""Whole Case 7 section versus the column at each station.

The 500 m flat-west numbers are a separate solve. This scores the 4000 m
interior (x from -2000 m to 1999 m) in full_qvs.h5. full_svlab.h5 stores
SVLAB zeta, but its surface record matches full_qvs, so it is not a second
damping. Both files are 15 s. There is no 30 s or 60 s full-section record.

    python score_full_section.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from case7a import ZONES  # noqa: E402
from score_record_length import SCALE, _ttf, score_h5  # noqa: E402
from labels import LABEL_WINDOW_S  # noqa: E402

REPO = CASE_DIR.parents[1]
FULL = REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5" / "full_qvs.h5"
OUT = CASE_DIR / "results" / "full_section_vs_column.json"
REPORT_BANDS = ("0.1-2.5", "0.1-10")
# The 500 m flat-west crop, so the same ground can be read off the full section.
WEST_CROP = (-1750.0, -1250.0)


def _zone_rows(x: np.ndarray, pear: np.ndarray, gap: np.ndarray) -> dict:
    out = {}
    spans = {**ZONES, "west_crop": WEST_CROP, "section": (-np.inf, np.inf)}
    for name, (lo, hi) in spans.items():
        m = (x >= lo) & (x < hi)
        if not np.any(m):
            continue
        out[name] = {
            "n": int(m.sum()),
            "pearson_min": float(pear[m].min()),
            "pearson_median": float(np.median(pear[m])),
            "pearson_max": float(pear[m].max()),
            "abs_log_median": float(np.median(gap[m])),
        }
    return out


def main() -> None:
    if not FULL.is_file():
        raise SystemExit(f"missing {FULL}")
    row = score_h5(FULL, _ttf())
    x = np.asarray(row["x_m"])
    zones = {}
    for name in REPORT_BANDS:
        band = row["bands"][name]
        zones[name] = _zone_rows(
            x,
            np.asarray(band["station_pearson"]),
            np.asarray(band["station_abs_log"]),
        )
    payload = {
        "window_s": LABEL_WINDOW_S,
        "scale": SCALE,
        "note": (
            "15 s Q-Vs full section. full_svlab.h5 has the same surface record, "
            "so SVLAB damping was not what OpenSees applied. No longer full-section record is on disk."
        ),
        "row": row,
        "zones": zones,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{'band':8} {'zone':12} {'n':4} {'P min':8} {'P med':8} {'P max':8} {'|log|':8}")
    for name in REPORT_BANDS:
        for zone, stats in zones[name].items():
            print(
                f"{name:8} {zone:12} {stats['n']:4d} "
                f"{stats['pearson_min']:8.3f} {stats['pearson_median']:8.3f} "
                f"{stats['pearson_max']:8.3f} {stats['abs_log_median']:8.3f}"
            )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
