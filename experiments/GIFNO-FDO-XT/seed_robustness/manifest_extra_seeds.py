#!/usr/bin/env python3
"""
Build supplemental Sobol manifests for seed-robustness OpenSees runs.

Uses the canonical 6D design in seiskit ``neural-operator/data/sobol.py``.
Extending ``seeds_per_sample`` from 30→50 preserves replicate 0..29 (same
``rf_seed`` values); only replicate_id >= 30 are new.

Example:
  uv run python seed_robustness/manifest_extra_seeds.py --sample-id 0 --seeds-per-sample 50
  uv run python seed_robustness/manifest_extra_seeds.py --sample-id 0 --extra-only --print-indices
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

DEFAULT_SEISKIT_DATA = Path.home() / "seiskit" / "neural-operator" / "data"
DEFAULT_OUT_DIR = (
    Path.home() / "surrogate-seismic-waves" / "checkpoints" / "seed_robustness"
)


def _load_sobol_module(seiskit_data_dir: Path):
    sobol_path = seiskit_data_dir / "sobol.py"
    if not sobol_path.is_file():
        raise FileNotFoundError(f"sobol.py not found: {sobol_path}")
    spec = importlib.util.spec_from_file_location("sobol", sobol_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {sobol_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sobol"] = mod
    spec.loader.exec_module(mod)
    return mod


def entries_for_sample(
    manifest: list,
    sample_id: int,
    *,
    extra_only: bool = False,
    min_replicate_id: int = 0,
) -> list:
    rows = [e for e in manifest if e.sample_id == sample_id]
    if extra_only:
        rows = [e for e in rows if e.replicate_id >= min_replicate_id]
    return sorted(rows, key=lambda e: e.replicate_id)


def write_filtered_manifest(sobol_mod, entries: list, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return sobol_mod.write_manifest_csv(out_path, entries)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build seed-robustness Sobol manifest slices"
    )
    p.add_argument("--seiskit-data-dir", type=Path, default=DEFAULT_SEISKIT_DATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--sample-id", type=int, default=0)
    p.add_argument("--sample-count", type=int, default=256)
    p.add_argument("--seeds-per-sample", type=int, default=50)
    p.add_argument(
        "--extra-only",
        action="store_true",
        help="Only replicate_id >= 30 (supplemental OpenSees runs).",
    )
    p.add_argument(
        "--min-replicate-id",
        type=int,
        default=30,
        help="With --extra-only, minimum replicate_id to include.",
    )
    p.add_argument(
        "--print-indices",
        action="store_true",
        help="Print global manifest indices for run_experiment.py.",
    )
    p.add_argument(
        "--out-name",
        type=str,
        default=None,
        help="Output CSV basename (default: sobol_manifest_s{S}_n{N}.csv).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    sobol = _load_sobol_module(args.seiskit_data_dir)

    manifest = sobol.build_manifest(
        sample_count=args.sample_count,
        seeds_per_sample=args.seeds_per_sample,
    )
    rows = entries_for_sample(
        manifest,
        args.sample_id,
        extra_only=args.extra_only,
        min_replicate_id=args.min_replicate_id,
    )
    if not rows:
        print(
            f"ERROR: no manifest rows for sample_id={args.sample_id}", file=sys.stderr
        )
        return 1

    suffix = "extra" if args.extra_only else "full"
    out_name = args.out_name or (
        f"sobol_manifest_s{args.sample_id:04d}_n{args.seeds_per_sample}_{suffix}.csv"
    )
    out_path = args.out_dir / out_name
    write_filtered_manifest(sobol, rows, out_path)

    phys = rows[0]
    print(f"[manifest] Wrote {len(rows)} rows -> {out_path}")
    print(
        f"  sample_id={phys.sample_id}  Vs1={phys.Vs1:.2f}  H={phys.H_discretized:.1f}  "
        f"CoV={phys.CoV:.3f}  rH={phys.rH:.1f}  aHV={phys.aHV:.1f}  Vs2={phys.Vs2:.1f}"
    )
    print(f"  replicate_id range: {rows[0].replicate_id}..{rows[-1].replicate_id}")

    if args.print_indices:
        indices = [e.index for e in rows]
        print(f"  indices: {indices[0]}..{indices[-1]} ({len(indices)} tasks)")
        for e in rows:
            print(
                f"    index={e.index:5d}  rep={e.replicate_id:2d}  rf_seed={e.rf_seed}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
