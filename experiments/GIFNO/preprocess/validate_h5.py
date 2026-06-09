#!/usr/bin/env python3
"""
Validate Neural Operator run_*.h5 files before TF preprocessing.

Completion criteria mirror opensees/neural-operator/data/run_experiment.py:
  - run_{index}.h5 must exist and be readable
  - Required groups/datasets: Vs grids, grid/params attrs, recorders/accel
  - Accel data must have even channel count (base + surface lateral rows)

For cleaning interrupted *simulation* outputs (orphan raw dirs without H5), use:
  opensees/neural-operator/data/cleanup_orphan_sobol_outputs.py

This script handles the consumer side: detect corrupt/truncated H5 files that
should be deleted on the storage volume so simulations can be rerun.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py

try:
    import hdf5plugin  # noqa: F401
except ImportError as e:
    raise ImportError("hdf5plugin is required to read compressed H5 files.") from e

RUN_RE = re.compile(r"run_(\d+)\.h5$")

REQUIRED_TOP_KEYS = (
    "Vs_realization_2D",
    "Damping_zeta",
    "Vs_profile_1D",
    "grid",
    "params",
    "recorders",
)
REQUIRED_GRID_ATTRS = ("dt", "Lx", "Lz")
REQUIRED_PARAM_ATTRS = ("H_discretized", "CoV", "rf_seed", "f0_effective")
MIN_FILE_BYTES = 1024
MIN_TIME_SAMPLES = 10


@dataclass
class H5ValidationResult:
    path: Path
    run_index: int
    ok: bool
    error: str = ""
    file_bytes: int = 0


def validate_h5_file(path: Path) -> H5ValidationResult:
    """Return validation result for a single run_*.h5 file."""
    m = RUN_RE.search(path.name)
    if not m:
        return H5ValidationResult(path, -1, False, "unexpected filename")
    run_index = int(m.group(1))
    file_bytes = path.stat().st_size
    if file_bytes < MIN_FILE_BYTES:
        return H5ValidationResult(
            path, run_index, False, f"file too small ({file_bytes} B)", file_bytes
        )

    try:
        with h5py.File(path, "r") as f:
            for key in REQUIRED_TOP_KEYS:
                if key not in f:
                    raise KeyError(f"missing '{key}'")

            for attr in REQUIRED_GRID_ATTRS:
                if attr not in f["grid"].attrs:
                    raise KeyError(f"missing grid attr '{attr}'")

            for attr in REQUIRED_PARAM_ATTRS:
                if attr not in f["params"].attrs:
                    raise KeyError(f"missing params attr '{attr}'")

            acc = f["recorders/accel/data"]
            time = f["recorders/accel/time"]
            n_time, n_ch = acc.shape
            if n_time < MIN_TIME_SAMPLES:
                raise ValueError(f"accel has too few time samples: {acc.shape}")
            if n_ch < 2 or n_ch % 2 != 0:
                raise ValueError(f"accel must have even channel count: {acc.shape}")
            if time.shape[0] != n_time:
                raise ValueError(
                    f"time/accel length mismatch: {time.shape} vs {acc.shape}"
                )

            # Force-read compressed payloads (catches truncated writes).
            _ = float(f["grid"].attrs["dt"])
            _ = f["Vs_realization_2D"][0, 0]
            _ = f["Damping_zeta"][0, 0]
            _ = f["recorders/accel/data"][0, 0]

        return H5ValidationResult(path, run_index, True, file_bytes=file_bytes)
    except Exception as exc:
        return H5ValidationResult(
            path, run_index, False, str(exc), file_bytes=file_bytes
        )


def _sorted_run_files(h5_dir: Path, pattern: str = "run_*.h5") -> list[Path]:
    files = [p for p in h5_dir.glob(pattern) if p.is_file()]
    return sorted(files, key=lambda p: int(RUN_RE.search(p.name).group(1)))  # type: ignore[union-attr]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5-dir",
        type=Path,
        default=Path("/mnt/box_lab/Projects/Neural Operator/data/h5"),
        help="Directory containing run_*.h5 files",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help="Move invalid H5 files here instead of deleting",
    )
    parser.add_argument(
        "--delete-invalid",
        action="store_true",
        help="Delete invalid H5 files (use after review; allows simulation rerun)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move/delete invalid files (default is report only)",
    )
    parser.add_argument("--pattern", type=str, default="run_*.h5")
    args = parser.parse_args()

    if args.delete_invalid and args.quarantine_dir is not None:
        raise SystemExit("Use only one of --delete-invalid or --quarantine-dir")

    h5_dir = args.h5_dir
    if not h5_dir.is_dir():
        raise SystemExit(f"H5 dir not found: {h5_dir}")

    files = _sorted_run_files(h5_dir, args.pattern)
    results = [validate_h5_file(p) for p in files]
    valid = [r for r in results if r.ok]
    invalid = [r for r in results if not r.ok]

    indices = {r.run_index for r in valid}
    if indices:
        missing_in_range = [
            i for i in range(min(indices), max(indices) + 1) if i not in indices
        ]
    else:
        missing_in_range = []

    print(f"[validate_h5] scanned: {len(files)}")
    print(f"[validate_h5] valid:   {len(valid)}")
    print(f"[validate_h5] invalid: {len(invalid)}")
    print(f"[validate_h5] missing indices in [{min(indices) if indices else 'n/a'}, "
          f"{max(indices) if indices else 'n/a'}]: {len(missing_in_range)}")

    for r in invalid[:30]:
        print(f"  INVALID {r.path.name}: {r.error}")
    if len(invalid) > 30:
        print(f"  ... +{len(invalid) - 30} more")

    if missing_in_range:
        print(f"  first missing indices: {missing_in_range[:20]}")

    if not invalid:
        print("[validate_h5] All files passed — safe to run full preprocessing.")
        return 0

    if not args.apply:
        print("\nReport only. Re-run with --apply --quarantine-dir DIR or --delete-invalid.")
        return 1

    if args.quarantine_dir:
        args.quarantine_dir.mkdir(parents=True, exist_ok=True)
        for r in invalid:
            dest = args.quarantine_dir / r.path.name
            shutil.move(str(r.path), str(dest))
            print(f"  quarantined {r.path.name} -> {dest}")
    elif args.delete_invalid:
        for r in invalid:
            r.path.unlink(missing_ok=True)
            print(f"  deleted {r.path.name}")
    else:
        print("No action specified. Use --quarantine-dir or --delete-invalid with --apply.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
