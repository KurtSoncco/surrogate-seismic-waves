#!/usr/bin/env python3
"""32 deeper sections (2000 m interior x 250 m) for Stampede.

``run_sections.py`` then repeats each edge column 500 m, so OpenSees sees
3000 x 250 m. Case 7a ``DEPTH_M`` stays 150 m. Contacts are milder than
round 0 (40 m steps, 25 m blend, rock/soil Vs ratio at most 3) so gravity
can factor the larger mesh.

    python generate_deep.py --n 32 --out sections/deep250.csv
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

DEPTH_M = 250.0
SOIL_THICKNESS = (60.0, 220.0)
# Milder contacts than Case 7a so UmfPack gravity can factor 3000 x 250 m meshes.
STEP_WIDTH_M = 40.0
CONTACT_BLEND_M = 25.0
VS_SMOOTH_M = 20.0
MAX_VS_RATIO = 3.0
DH_CLIP_M = 30.0
TROUGH_MAX_M = 35.0


def sample_deep(n: int, seed: int, prefix: str = "deep") -> list[dict]:
    old_depth, old_ranges = sg.DEPTH_M, sg.RANGES
    try:
        sg.DEPTH_M = DEPTH_M
        sg.RANGES = dict(sg.RANGES)
        sg.RANGES["soil_thickness_m"] = SOIL_THICKNESS
        rows = sg.sample_rows(n, seed=seed, prefix=prefix)
    finally:
        sg.DEPTH_M = old_depth
        sg.RANGES = old_ranges
    for row in rows:
        row["depth_m"] = DEPTH_M
        row["step_width_m"] = STEP_WIDTH_M
        row["contact_blend_m"] = CONTACT_BLEND_M
        row["vs_smooth_m"] = VS_SMOOTH_M
        row["max_vs_ratio"] = MAX_VS_RATIO
        row["feature_dh_m"] = float(np.clip(row["feature_dh_m"], -DH_CLIP_M, DH_CLIP_M))
        if row["feature_name"] == "trough":
            row["feature"] = sg.FEATURES.index("step")
            row["feature_name"] = "step"
        row["trough_depth_m"] = 0.0
        cap = float(row["vs_contact_mps"]) * MAX_VS_RATIO
        if float(row["rock_vs_min_mps"]) > cap:
            row["rock_vs_min_mps"] = cap
        if float(row["rock_vs_max_mps"]) < float(row["rock_vs_min_mps"]):
            row["rock_vs_max_mps"] = float(row["rock_vs_min_mps"]) * 1.15
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = sample_deep(args.n, args.seed)
    sg.write_rows(rows, args.out)
    print(f"wrote {len(rows)} rows depth={DEPTH_M} -> {args.out}")


if __name__ == "__main__":
    main()
