#!/usr/bin/env python3
"""Interior outcrop labels for the finished 150 m manifold solves.

Drops the 500 m free-field pads. The 5 m set is every complete H5. A second
pass labels every metre on 24 sections stratified by stratum and width, which
is the interpolation check and is not the training loss.

    python label_manifold.py \
        --h5-dir /path/manifold150/h5 \
        --manifest /path/manifold150/manifest.csv \
        --out ../../data/vardomain/manifold_labels
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dense_labels import label_section

SCALE = 1.9776999950408936  # acceleration Ricker, fixed-window round1
WINDOW_S = 1310.72
N_DENSE = 24


def complete_rows(manifest: Path) -> list[dict]:
    rows = [r for r in csv.DictReader(manifest.open()) if r.get("status") == "complete"]
    if not rows:
        raise SystemExit(f"no complete rows in {manifest}")
    return rows


def dense_ids(rows: list[dict], n: int, seed: int) -> list[str]:
    """One row from each (stratum, width) cell, then fill, both seeds of a pair not required."""
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        if r.get("split") == "blind":
            continue
        cells[(r["stratum"], r["lx_m"])].append(r["sample_id"])
    rng = np.random.default_rng(seed)
    keys = sorted(cells)
    for k in keys:
        rng.shuffle(cells[k])
    picked: list[str] = []
    while len(picked) < n and any(cells[k] for k in keys):
        for k in keys:
            if cells[k] and len(picked) < n:
                picked.append(cells[k].pop())
    return picked


def _run(spec: tuple) -> str:
    path, out, stride = spec
    try:
        return label_section(Path(path), Path(out), SCALE, stride, WINDOW_S, drop_pads=True)
    except Exception as exc:  # noqa: BLE001 — one bad H5 must not kill the pool
        return f"FAIL {Path(path).name}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="label only the first N complete rows (smoke)")
    ap.add_argument("--dense", type=int, default=N_DENSE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = complete_rows(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    stride5 = args.out / "stride5"
    stride1 = args.out / "stride1"
    stride5.mkdir(parents=True, exist_ok=True)
    stride1.mkdir(parents=True, exist_ok=True)
    specs = [(args.h5_dir / f"{r['sample_id']}.h5", stride5, 5.0) for r in rows]
    missing = [p for p, _, _ in specs if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} H5 missing, first {missing[0]}")
    print(f"labeling {len(specs)} interiors at 5 m with {args.workers} workers", flush=True)
    fails = 0
    with ProcessPoolExecutor(max_workers=min(args.workers, len(specs))) as pool:
        for i, msg in enumerate(pool.map(_run, specs), start=1):
            if msg.startswith("FAIL"):
                fails += 1
            if i % 10 == 0 or msg.startswith("FAIL") or i == len(specs):
                print(f"{i}/{len(specs)} {msg}", flush=True)
    ids = dense_ids(rows, args.dense, args.seed)
    (args.out / "dense_ids.json").write_text(json.dumps(ids, indent=2))
    dense_specs = [(args.h5_dir / f"{sid}.h5", stride1, 1.0) for sid in ids]
    print(f"labeling {len(dense_specs)} interiors at 1 m", flush=True)
    with ProcessPoolExecutor(max_workers=min(args.workers, max(len(dense_specs), 1))) as pool:
        for msg in pool.map(_run, dense_specs):
            print(msg, flush=True)
    meta = {
        "scale": SCALE, "window_s": WINDOW_S, "n_complete": len(rows), "n_fail": fails,
        "stride5_m": 5.0, "stride1_m": 1.0, "n_dense": len(ids), "h5_dir": str(args.h5_dir),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    if fails:
        raise SystemExit(f"{fails} sections failed")


if __name__ == "__main__":
    main()
