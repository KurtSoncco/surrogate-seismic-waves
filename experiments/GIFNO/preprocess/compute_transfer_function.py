#!/usr/bin/env python3
"""
Compute transfer functions for GIFNO from Neural Operator H5 files.

Port of opensees/data_generation/emulator_first/code/data_processing/
compute_transfer_function_h15.py, adapted for 21-lateral recorder layout.

Reads:
  - /mnt/box_lab/Projects/Neural Operator/data/h5/run_*.h5

Outputs:
  - experiments/GIFNO/data/transfer_function_results/tf_per_sample.npy
    shape: (n_samples, N_LATERAL, N_FREQ)
  - experiments/GIFNO/data/transfer_function_results/freq.npy
  - experiments/GIFNO/data/transfer_function_results/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

try:
    import hdf5plugin  # noqa: F401
except ImportError as e:
    raise ImportError(
        "hdf5plugin is required to read compressed run_*.h5 files."
    ) from e

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import config  # noqa: E402

OPENSEES_ROOT = config.OPENSEES_ROOT
if str(OPENSEES_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENSEES_ROOT))

from seiskit.ttf.TTF import TTF_batch_fast  # noqa: E402

from validate_h5 import validate_h5_file  # noqa: E402

RUN_RE = re.compile(r"run_(\d+)\.h5$")
DZ = config.DZ
SMOOTH_COEFF = config.SMOOTH_COEFF
N_FREQ = config.N_FREQ
FREQ_START_HZ = config.FREQ_START_HZ
FREQ_END_HZ = config.FREQ_END_HZ


def _sorted_run_files(results_dir: Path, *, pattern: str) -> list[Path]:
    files = [p for p in results_dir.glob(pattern) if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No H5 files found: {results_dir}/{pattern}")

    def idx_of(p: Path) -> int:
        m = RUN_RE.search(p.name)
        if not m:
            raise ValueError(f"Unexpected filename: {p.name}")
        return int(m.group(1))

    return sorted(files, key=idx_of)


def _split_base_surf(
    data: np.ndarray, row_y_m, n_lateral: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (base_2d, surf_2d) each shape (n_lateral, n_time)."""
    if data.shape[1] != 2 * n_lateral:
        raise ValueError(
            f"Expected {2 * n_lateral} accel channels, got {data.shape[1]}"
        )
    if row_y_m is None:
        surf_2d = data[:, :n_lateral].T
        base_2d = data[:, n_lateral : 2 * n_lateral].T
    else:
        row_y = np.asarray(row_y_m, dtype=np.float64).ravel()
        if row_y.size != 2 or row_y[0] >= row_y[1]:
            raise ValueError(
                f"Unexpected row_y_m {row_y!r}: expected two ascending depths"
            )
        base_2d = data[:, :n_lateral].T
        surf_2d = data[:, n_lateral : 2 * n_lateral].T
    return base_2d, surf_2d


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute GIFNO TFs from Neural Operator run_*.h5 files"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=config.H5_DIR,
        help="Directory containing raw run_*.h5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.TF_RESULTS_DIR,
        help="Output directory for tf_per_sample.npy and manifest",
    )
    parser.add_argument("--pattern", type=str, default="run_*.h5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip H5 completeness check (not recommended before full runs)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = _sorted_run_files(args.results_dir, pattern=args.pattern)
    candidates = all_files[: args.limit] if args.limit is not None else all_files
    skipped: list[dict] = []
    run_files: list[Path] = []

    if args.skip_validation:
        run_files = candidates
    else:
        for p in candidates:
            result = validate_h5_file(p)
            if result.ok:
                run_files.append(p)
            else:
                skipped.append(
                    {
                        "run_index": result.run_index,
                        "h5_path": str(p),
                        "error": result.error,
                        "file_bytes": result.file_bytes,
                    }
                )

    if skipped:
        skipped_path = output_dir / "invalid_h5.csv"
        with open(skipped_path, "w", newline="") as sf:
            writer = csv.DictWriter(
                sf, fieldnames=["run_index", "h5_path", "error", "file_bytes"]
            )
            writer.writeheader()
            writer.writerows(skipped)
        print(
            f"[GIFNO TF] Skipped {len(skipped)} invalid H5 file(s); "
            f"see {skipped_path}",
            flush=True,
        )

    if not run_files:
        raise RuntimeError(
            "No valid H5 files to process. Run validate_h5.py to inspect failures."
        )

    first_file = run_files[0]
    with h5py.File(first_file, "r") as f:
        dt_val = float(f["grid"].attrs["dt"])
        n_ch = f["recorders/accel/data"].shape[1]
    n_lateral = n_ch // 2

    freq_ref = np.logspace(
        np.log10(FREQ_START_HZ), np.log10(FREQ_END_HZ), N_FREQ
    )
    n_samples = len(run_files)
    print(
        f"[GIFNO TF] dt={dt_val:.5f}s, n_samples={n_samples}, "
        f"n_lateral={n_lateral}, n_freq={N_FREQ}",
        flush=True,
    )

    tf_all_path = output_dir / "tf_per_sample.npy"
    tf_all = np.lib.format.open_memmap(
        tf_all_path,
        mode="w+",
        dtype="float32",
        shape=(n_samples, n_lateral, N_FREQ),
    )

    manifest_path = output_dir / "manifest.csv"
    recorder_x = config.recorder_x_indices()
    manifest_rows: list[dict] = []

    for i, p in enumerate(tqdm(run_files, desc="TF", unit="sample")):
        with h5py.File(p, "r") as f:
            dt_i = float(f["grid"].attrs["dt"])
            data = f["recorders/accel/data"][:]
            row_y_m = f["recorders/accel/data"].attrs.get("row_y_m")
            attrs = dict(f["params"].attrs)
            nz = int(f["Vs_realization_2D"].shape[0])
            run_idx = int(RUN_RE.search(p.name).group(1))  # type: ignore[union-attr]
            rf_seed = int(attrs.get("rf_seed", attrs.get("seed", -1)))
            h_val = float(attrs.get("H_discretized", attrs.get("H", 0.0)))
            cov_val = float(attrs.get("CoV", 0.0))
            f0_val = float(attrs.get("f0_effective", 0.0))

        base_2d, surf_2d = _split_base_surf(data, row_y_m, n_lateral)
        freq_i, mags_i = TTF_batch_fast(
            base_2d,
            surf_2d,
            dt=dt_i,
            dz=DZ,
            smooth_coeff=SMOOTH_COEFF,
            Vsmin=None,
            n_points=N_FREQ,
        )
        if len(freq_i) != N_FREQ:
            raise RuntimeError(f"Unexpected n_freq={len(freq_i)} for {p.name}")

        tf_all[i, :, :] = mags_i.astype("float32")
        manifest_rows.append(
            {
                "sample_idx": i,
                "run_index": run_idx,
                "h5_path": str(p),
                "rf_seed": rf_seed,
                "H_discretized": h_val,
                "CoV": cov_val,
                "f0_effective": f0_val,
                "nz_actual": nz,
                "n_lateral": n_lateral,
            }
        )
        if i % 250 == 0:
            tf_all.flush()

    tf_all.flush()
    np.save(output_dir / "freq.npy", freq_ref.astype(np.float32))

    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    np.save(output_dir / "recorder_x_idx.npy", recorder_x)
    print(f"[GIFNO TF] Saved per-sample TFs: {tf_all_path}", flush=True)
    print(f"[GIFNO TF] Saved manifest: {manifest_path}", flush=True)
    print(f"[GIFNO TF] Recorder x-indices: {recorder_x.tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
