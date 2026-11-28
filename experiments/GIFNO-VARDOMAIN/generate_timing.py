#!/usr/bin/env python3
"""Core-hour pilot: 20 sections at each interior width, 250 m deep.

Widths are 500, 1000, 1500, and 2000 m. Half the rows in each bucket are
``damping_family=qvs`` (zeta computed from Vs at solve time) and half are
``explicit`` (the CSV zeta). One row in four sets the Rayleigh f1 independently
of the column f0. ``duration_s`` stays 15 until the record-length pilot reports
a length. This file is the manifest; it is not a production solve.

    python generate_timing.py --out hpc/timing_widths.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CASE7A_DIR = Path(__file__).resolve().parents[1] / "case7a"
if str(CASE7A_DIR) not in sys.path:
    sys.path.insert(0, str(CASE7A_DIR))

import section_generator as sg  # noqa: E402

from generate_deep import sample_deep  # noqa: E402

WIDTHS_M = (500, 1000, 1500, 2000)
PER_WIDTH = 20


def timing_rows(per_width: int = PER_WIDTH, seed: int = 11) -> list[dict]:
    rows = []
    for i, width in enumerate(WIDTHS_M):
        batch = sample_deep(per_width, seed=seed + i, prefix=f"t{width}")
        for j, row in enumerate(batch):
            row["lx_m"] = float(width)
            row["sample_id"] = f"t{width}_{j:02d}"
            row["damping_family"] = "qvs" if j < per_width // 2 else "explicit"
            row["duration_s"] = 15.0
            if j % 4 == 0:
                f1 = float(np.geomspace(0.2, 2.0, per_width)[j])
                row["rayleigh_f1_hz"] = f1
                row["rayleigh_f2_hz"] = 3.0 * f1
        rows.extend(batch)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-width", type=int, default=PER_WIDTH)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rows = timing_rows(args.per_width, args.seed)
    sg.write_rows(rows, args.out)
    print(f"wrote {len(rows)} rows ({args.per_width} x {list(WIDTHS_M)}) -> {args.out}")


if __name__ == "__main__":
    main()
