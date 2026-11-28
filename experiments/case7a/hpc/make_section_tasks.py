#!/usr/bin/env python3
"""pylauncher command lines for the section rows that still lack an H5.

    python make_section_tasks.py --manifest ../sections/round0.csv --out-dir $SCRATCH/sections > tasks.txt

Each line runs one section single-threaded; ``stampede3_sections.slurm`` packs
them onto nodes with ``CORES_PER_TASK`` cores each (memory headroom, not
threads).
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

from run_sections import h5_ok  # noqa: E402
from section_generator import read_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()
    runner = CASE_DIR / "run_sections.py"
    log_dir = args.out_dir / "logs"
    n = 0
    for i, row in enumerate(read_rows(args.manifest)):
        if h5_ok(args.out_dir / "h5" / f"{row['sample_id']}.h5"):
            continue
        cmd = [args.python, "-u", str(runner), "--manifest", str(args.manifest.resolve()), "--index", str(i), "--out-dir", str(args.out_dir)]
        if args.duration:
            cmd += ["--duration", str(args.duration)]
        print(" ".join(shlex.quote(c) for c in cmd) + f" > {shlex.quote(str(log_dir / row['sample_id']))}.log 2>&1")
        n += 1
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{n} tasks", file=sys.stderr)


if __name__ == "__main__":
    main()
