#!/usr/bin/env python3
"""One 15 s OpenSees timing on a row of timing_widths.csv.

Writes the solver wall time and deletes the H5. The mesh is the interior
plus 500 m of repeated edge column on each side. This is the number in the
runtime table; the 0.5 s pilot is not.

    python savio_timing15.py --manifest timing_widths.csv --index 0 --out-dir "$SCRATCH/vardomain/timing15"
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
    args = ap.parse_args()
    row = read_rows(args.manifest)[args.index]
    width = int(round(float(row["lx_m"])))
    dest = simulate(row, args.out_dir, duration_s=float(row.get("duration_s") or 15.0))
    import h5py

    with h5py.File(dest, "r") as handle:
        vs = handle["Vs_realization_2D"]
        record = {
            "width_m": width,
            "sample_id": str(row["sample_id"]),
            "wall_s": float(handle.attrs["wall_s"]),
            "opensees_nx": int(vs.shape[1]),
            "opensees_nz": int(vs.shape[0]),
            "duration_s": float(row.get("duration_s") or 15.0),
            "status": "solved",
            "host": os.uname().nodename,
        }
    dest.unlink(missing_ok=True)
    out = args.out_dir / "walls"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"w{width}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
