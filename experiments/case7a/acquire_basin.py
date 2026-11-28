#!/usr/bin/env python3
"""AL4PDE / ACCEL acquisition over the 256-run basin ensemble (plan step 7).

Same rules as acquire_sections.py: mutations of the highest-error columns plus
30 % random, then LCMD or SBAL on the residual deep ensemble. Case 7a is held
out. The written CSV is a run_basin_ensemble.py manifest (append or replace).

    python acquire_basin.py --model-dir CK0 --model-dir CK1 --done basin_ensemble.csv \
        --batch 16 --round 1 --out basin_round1.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from acquire_sections import RANDOM_SHARE, Ensemble, lcmd, sbal
from basin_ensemble import (
    COLUMNS,
    basin_rows,
    gradational_field,
    mutate_basin,
    near_case7a,
    train_rows,
)


def _read(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path)):
        out = dict(r)
        for k in (
            "soil_thickness_m", "vs_surface_mps", "vs_contact_mps", "vs_rock_mps",
            "contact_depth_west_m", "contact_depth_east_m", "zeta_soil", "zeta_rock",
            "rayleigh_f1_hz", "rayleigh_f2_hz", "holdout_case7a",
        ):
            if k in out:
                out[k] = float(out[k]) if k != "holdout_case7a" else int(float(out[k]))
        rows.append(out)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, action="append", required=True)
    ap.add_argument("--done", type=Path, default=None, help="already simulated basin CSV (default: basin_rows())")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pool-factor", type=int, default=8)
    ap.add_argument("--method", choices=("lcmd", "sbal"), default="lcmd")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if len(args.model_dir) < 2:
        raise SystemExit("deep-ensemble acquisition needs at least two members")
    rng = np.random.default_rng([args.seed, args.round])
    ens = Ensemble(args.model_dir)
    done = [r for r in (_read(args.done) if args.done else basin_rows()) if int(r.get("holdout_case7a", 0)) == 0]
    phi_done = {}
    for r in done:
        vs, rock = gradational_field(r)
        zeta = np.where(rock, float(r["zeta_rock"]), float(r["zeta_soil"]))
        phi_done[r["sample_id"]] = ens.features(vs, zeta)
    score = {sid: float(np.linalg.norm(phi)) for sid, phi in phi_done.items()}
    parents = sorted(done, key=lambda r: -score[r["sample_id"]])[: max(1, args.batch // 2)]
    print("parents:", [(p["sample_id"], round(score[p["sample_id"]], 4)) for p in parents])

    pool_n = args.pool_factor * args.batch
    n_rand = int(round(RANDOM_SHARE * pool_n))
    tag = f"br{args.round:02d}"
    pool = [mutate_basin(parents[i % len(parents)], rng, f"{tag}m_{i:04d}") for i in range(pool_n - n_rand)]
    extra = train_rows(n_rand, seed=int(rng.integers(1 << 31)))
    for i, r in enumerate(extra):
        r["sample_id"] = f"{tag}u_{i:04d}"
        if not near_case7a(r):
            pool.append(r)
    phi_pool = []
    for r in pool:
        vs, rock = gradational_field(r)
        zeta = np.where(rock, float(r["zeta_rock"]), float(r["zeta_soil"]))
        phi_pool.append(ens.features(vs, zeta))
    phi_pool = np.stack(phi_pool)
    phi_train = np.stack(list(phi_done.values()))
    picks = lcmd(phi_pool, phi_train, args.batch) if args.method == "lcmd" else sbal(phi_pool, args.batch, rng)
    out = []
    for j, i in enumerate(picks):
        r = dict(pool[i])
        r["acq_round"] = args.round
        r["acq_method"] = args.method
        r["acq_order"] = j
        r["origin"] = "mutation" if "parent_id" in r else "random"
        out.append(r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(COLUMNS) + [k for r in out for k in r if k not in COLUMNS]
    keys = list(dict.fromkeys(keys))
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"round {args.round}: {len(out)} columns -> {args.out}")


if __name__ == "__main__":
    main()
