#!/usr/bin/env python3
"""pylauncher lines for Case 7 crops that still lack an H5.

    python make_crop_tasks.py --meshes-dir ../data/case7a/opensees_crops/meshes --out-dir $SCRATCH/case7_opensees_crops
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

from case7_crops import INNER_NX  # noqa: E402
from run_case7_crops import h5_ok, job_list  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meshes-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    runner = CASE_DIR / "run_case7_crops.py"
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, crop in enumerate(job_list(args.meshes_dir)):
        n_inner = int(round(float(crop.get("inner_m", INNER_NX))))
        if h5_ok(args.out_dir / "h5" / f"{crop['sample_id']}.h5", n_inner):
            continue
        cmd = [
            args.python,
            "-u",
            str(runner),
            "--meshes-dir",
            str(args.meshes_dir.resolve()),
            "--index",
            str(i),
            "--out-dir",
            str(args.out_dir),
        ]
        print(" ".join(shlex.quote(c) for c in cmd) + f" > {shlex.quote(str(log_dir / crop['sample_id']))}.log 2>&1")
        n += 1
    print(f"{n} tasks", file=sys.stderr)


if __name__ == "__main__":
    main()
