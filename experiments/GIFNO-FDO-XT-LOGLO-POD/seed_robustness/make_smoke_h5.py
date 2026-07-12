#!/usr/bin/env python3
"""
Generate minimal multi-seed H5 files for local smoke tests (no OpenSees).

Writes run_{index}.h5 for sample_id=0, replicate_id=0..n_seeds-1 with slightly
different Vs fields (simulating rf_seed variability).

Example:
  uv run python seed_robustness/make_smoke_h5.py --n-seeds 30 --out-dir checkpoints/seed_robustness/h5_smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SEISKIT_DATA = Path.home() / "seiskit" / "neural-operator" / "data"
DEFAULT_OUT = (
    Path.home()
    / "surrogate-seismic-waves"
    / "checkpoints"
    / "seed_robustness"
    / "h5_smoke"
)


def _load_sobol(seiskit_data_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "sobol", seiskit_data_dir / "sobol.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load sobol from {seiskit_data_dir}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sobol"] = mod
    spec.loader.exec_module(mod)
    return mod


def write_h5(
    path: Path,
    *,
    entry,
    seed_offset: int,
    nz: int = 40,
    nx: int = 1500,
    n_time: int = 200,
) -> None:
    rng = np.random.default_rng(entry.rf_seed)
    base_vs = np.linspace(150.0, 800.0, nz * nx, dtype=np.float32).reshape(nz, nx)
    noise = rng.normal(0, 20.0, size=(nz, nx)).astype(np.float32)
    vs = np.clip(base_vs + noise + seed_offset * 5.0, 100.0, 2000.0)
    zeta = np.full((nz, nx), 0.025, dtype=np.float32)

    # Synthetic accel: frequency content shifts slightly per seed
    t = np.linspace(0, 2.0, n_time, dtype=np.float32)
    f0 = 3.0 + 0.05 * entry.replicate_id
    base_sig = np.sin(2 * np.pi * f0 * t)
    surf_sig = (1.2 + 0.02 * entry.replicate_id) * base_sig
    n_lateral = 21
    n_ch = 2 * n_lateral
    data = np.zeros((n_time, n_ch), dtype=np.float32)
    for i in range(n_lateral):
        phase = rng.uniform(-0.1, 0.1)
        data[:, i] = base_sig * (1.0 + phase)
        data[:, n_lateral + i] = surf_sig * (1.0 + phase)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("Vs_realization_2D", data=vs)
        f.create_dataset("Damping_zeta", data=zeta)
        grid = f.create_group("grid")
        grid.attrs["Lx"] = float(nx)
        grid.attrs["Lz"] = float(nz)
        grid.attrs["dx"] = 1.0
        grid.attrs["dz"] = 1.0
        grid.attrs["dt"] = 0.01
        f.attrs["sample_id"] = entry.sample_id
        f.attrs["replicate_id"] = entry.replicate_id
        f.attrs["rf_seed"] = entry.rf_seed
        params = f.create_group("params")
        params.attrs["Vs1"] = entry.Vs1
        params.attrs["H"] = entry.H_discretized
        params.attrs["CoV"] = entry.CoV
        params.attrs["rf_seed"] = entry.rf_seed
        params.attrs["f0_effective"] = 3.0
        accel = f.create_group("recorders").create_group("accel")
        accel.create_dataset("time", data=t)
        accel.create_dataset("data", data=data)


def main() -> int:
    p = argparse.ArgumentParser(description="Create smoke-test Sobol H5 files")
    p.add_argument("--sample-id", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=30)
    p.add_argument("--replicate-start", type=int, default=0)
    p.add_argument("--seeds-per-sample", type=int, default=30)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seiskit-data-dir", type=Path, default=DEFAULT_SEISKIT_DATA)
    args = p.parse_args()

    sobol = _load_sobol(args.seiskit_data_dir)
    manifest = sobol.build_manifest(
        sample_count=256,
        seeds_per_sample=args.seeds_per_sample,
    )
    entries = sorted(
        [e for e in manifest if e.sample_id == args.sample_id],
        key=lambda e: e.replicate_id,
    )
    entries = [
        e
        for e in entries
        if args.replicate_start <= e.replicate_id < args.replicate_start + args.n_seeds
    ]

    for entry in entries:
        out = args.out_dir / f"run_{entry.index}.h5"
        write_h5(out, entry=entry, seed_offset=entry.replicate_id)
        print(f"  wrote {out.name}  rep={entry.replicate_id}  rf_seed={entry.rf_seed}")

    print(f"[smoke] {len(entries)} H5 files -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
