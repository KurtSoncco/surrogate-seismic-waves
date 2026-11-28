#!/usr/bin/env python3
"""One OpenSees case with a surface recorder every metre.

Keeps the H5. The mesh is the interior plus the usual 500 m edge pads.
``hx_m`` defaults to 1, so ``recorders/accel/surface`` is one trace per metre.

    python savio_dense_recorders.py --manifest timing_widths.csv --index 0 --out-dir "$SCRATCH/vardomain/dense_recorders"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CASE_DIR = Path(os.environ.get("CASE_DIR", Path(__file__).resolve().parents[2] / "case7a"))
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from run_sections import simulate  # noqa: E402
from section_generator import read_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hx-m", type=float, default=1.0)
    args = ap.parse_args()
    row = read_rows(args.manifest)[args.index]
    dest = simulate(
        row, args.out_dir,
        duration_s=float(row.get("duration_s") or 15.0),
        hx_m=float(args.hx_m),
        keep=False,
    )
    import h5py

    with h5py.File(dest, "r") as handle:
        surface = handle["recorders/accel/surface"]
        x_m = surface.attrs.get("x_m")
        record = {
            "sample_id": str(row["sample_id"]),
            "h5": str(dest),
            "hx_m": float(args.hx_m),
            "n_surface": int(surface.shape[1]),
            "n_time": int(surface.shape[0]),
            "x0_m": None if x_m is None else float(x_m[0]),
            "x1_m": None if x_m is None else float(x_m[-1]),
            "status": "solved",
            "host": os.uname().nodename,
        }
    out = args.out_dir / "solved"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{row['sample_id']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
