#!/usr/bin/env python3
"""Phase 4 pilot manifest: nested crops of 8 masters.

One master geology is drawn at 2000 m. The 500, 1000, 1500 and 2000 m rows are
centered crops of that field (``master_lx_m``, ``crop_x0_m``), so a width does
not redraw the random Vs field. ``section_generator.DEPTH_M`` stays 150 m.
Every row sets ``depth_m`` to 250, with the milder contacts used by
``generate_deep.py`` (40 m steps, 25 m blend, rock/soil Vs ratio at most 3).

Design, and nothing is solved here:

    8 masters × 4 widths × 3 seeds × {gradient, layered}  +  a CoV = 0 slice

Damping is explicit and is drawn independently of the Vs parameters: zeta and
the Rayleigh pair are not Q(Vs) and are not the column quarter-wavelength
frequency. The same damping is written on every crop of a master. Record
length is omitted so a later ``--duration`` / ``--timing`` can set it.

    python pilot_manifest.py --out ../GIFNO-VARDOMAIN/hpc/pilot_g4.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

import section_generator as sg  # noqa: E402

PILOT_SEED = 20261125
N_MASTERS = 8
N_SEEDS = 3
FILLS = ("gradient", "layered")
PILOT_DEPTH_M = 250.0
SOIL_THICKNESS_M = (60.0, 220.0)
# Same mild-contact columns as generate_deep.py, so a 3000 x 250 m padded mesh can factor.
STEP_WIDTH_M = 40.0
CONTACT_BLEND_M = 25.0
VS_SMOOTH_M = 20.0
MAX_VS_RATIO = 3.0
DH_CLIP_M = 30.0
COV_RANGE = (0.10, 0.30)
ZETA_RANGE = (0.005, 0.05)
F1_RANGE_HZ = (0.2, 2.0)
HPC_DIR = CASE_DIR.parent / "GIFNO-VARDOMAIN" / "hpc"
DEFAULT_MANIFEST = HPC_DIR / "pilot_g4.csv"
DEFAULT_TIMING = HPC_DIR / "pilot_record_timing.csv"
TIMING_NOTE = (
    "Time this one 2000 m master before choosing a record length. "
    "Do not start the 8-master run from this file."
)


def _mild_contact(row: dict) -> None:
    """Stamp the generate_deep contact limits onto one sampled master."""
    row["depth_m"] = float(PILOT_DEPTH_M)
    row["lx_m"] = float(sg.MASTER_LX_M)
    row["step_width_m"] = float(STEP_WIDTH_M)
    row["contact_blend_m"] = float(CONTACT_BLEND_M)
    row["vs_smooth_m"] = float(VS_SMOOTH_M)
    row["max_vs_ratio"] = float(MAX_VS_RATIO)
    row["feature_dh_m"] = float(np.clip(float(row["feature_dh_m"]), -DH_CLIP_M, DH_CLIP_M))
    if row["feature_name"] == "trough":
        row["feature"] = int(sg.FEATURES.index("step"))
        row["feature_name"] = "step"
    row["trough_depth_m"] = 0.0
    cap = float(row["vs_contact_mps"]) * MAX_VS_RATIO
    if float(row["rock_vs_min_mps"]) > cap:
        row["rock_vs_min_mps"] = float(cap)
    if float(row["rock_vs_max_mps"]) < float(row["rock_vs_min_mps"]):
        row["rock_vs_max_mps"] = float(row["rock_vs_min_mps"]) * 1.15


def _damping_independent_of_vs(row: dict, rng: np.random.Generator) -> None:
    """Explicit zeta and Rayleigh anchors that do not follow Vs or f0."""
    zeta = float(np.exp(rng.uniform(np.log(ZETA_RANGE[0]), np.log(ZETA_RANGE[1]))))
    frac = float(rng.uniform(0.05, 1.0))
    f0 = max(0.2, float(row["f0_flat_hz"]))
    f1 = float(rng.uniform(*F1_RANGE_HZ))
    for _ in range(8):
        if abs(f1 - f0) >= 0.05:
            break
        f1 = float(rng.uniform(*F1_RANGE_HZ))
    if abs(f1 - f0) < 0.05:
        f1 = float(np.clip(f0 + 0.5, *F1_RANGE_HZ))
    if abs(f1 - f0) < 0.05:
        f1 = float(np.clip(f0 - 0.5, *F1_RANGE_HZ))
    row["zeta_soil"] = zeta
    row["zeta_rock"] = float(max(sg.ZETA_ROCK_MIN, zeta * frac))
    row["rayleigh_f1_hz"] = f1
    row["rayleigh_f2_hz"] = float(3.0 * f1)
    row["damping_family"] = "explicit"


def sample_masters(n: int = N_MASTERS, seed: int = PILOT_SEED) -> list[dict]:
    """n master templates at 2000 m x 250 m. Does not expand widths or seeds."""
    old_depth, old_ranges = sg.DEPTH_M, sg.RANGES
    try:
        sg.DEPTH_M = PILOT_DEPTH_M
        sg.RANGES = dict(sg.RANGES)
        sg.RANGES["soil_thickness_m"] = SOIL_THICKNESS_M
        rows = sg.sample_rows(n, seed=seed, prefix="p4tmp")
    finally:
        sg.DEPTH_M = old_depth
        sg.RANGES = old_ranges
    damp = np.random.default_rng(seed + 17)
    realize = np.random.default_rng(seed + 99)
    masters = []
    for i, row in enumerate(rows):
        _mild_contact(row)
        _damping_independent_of_vs(row, damp)
        master_id = f"p4m{i:02d}"
        row["master_id"] = master_id
        row["group_id"] = master_id
        row["sample_id"] = master_id
        row["stratum"] = "pilot"
        row["split"] = "pilot"
        row["vs_corr_h_m"] = 50.0
        row["vs_corr_aniso"] = 10.0
        row["n_soil_layers"] = int(realize.integers(4, 13))
        row["soilgen_seed"] = int(realize.integers(1, 2**31 - 1))
        row["field_seeds"] = [int(s) for s in realize.integers(1, 2**31 - 1, size=N_SEEDS)]
        row["vs_cov_master"] = float(realize.uniform(*COV_RANGE))
        masters.append(row)
    return masters


def _crop(master: dict, *, width: float, fill: str, vs_cov: float, field_seed: int, tag: str) -> dict:
    row = {k: v for k, v in master.items() if k not in ("field_seeds", "vs_cov_master")}
    i0, _i1 = sg.nested_crop_bounds(sg.MASTER_LX_M, width)
    master_id = str(master["master_id"])
    row.update(
        sample_id=f"{master_id}_w{int(width):04d}_{fill}_{tag}",
        master_id=master_id,
        group_id=master_id,
        lx_m=float(width),
        master_lx_m=float(sg.MASTER_LX_M),
        crop_x0_m=float(i0 * sg.DZ_M),
        depth_m=float(PILOT_DEPTH_M),
        soil_fill=fill,
        vs_cov=float(vs_cov),
        field_seed=int(field_seed),
        cov_slice=int(vs_cov == 0.0),
        damping_family="explicit",
    )
    return row


def build_pilot(seed: int = PILOT_SEED) -> list[dict]:
    """Full pilot table. Record length is not a column."""
    rows: list[dict] = []
    for master in sample_masters(N_MASTERS, seed):
        for fill in FILLS:
            for width in sg.NESTED_WIDTHS_M:
                for seed_i, field_seed in enumerate(master["field_seeds"]):
                    rows.append(_crop(
                        master, width=float(width), fill=fill,
                        vs_cov=float(master["vs_cov_master"]), field_seed=int(field_seed), tag=f"s{seed_i}",
                    ))
                rows.append(_crop(
                    master, width=float(width), fill=fill,
                    vs_cov=0.0, field_seed=0, tag="cov0",
                ))
    for row in rows:
        if "duration_s" in row:
            del row["duration_s"]
    return rows


def timing_rows(rows: list[dict] | None = None) -> list[dict]:
    """One 2000 m master, timed at 200 s and at 400 s. Not a production record."""
    rows = build_pilot() if rows is None else rows
    sample_id = "p4m00_w2000_gradient_s0"
    matches = [(i, r) for i, r in enumerate(rows) if r["sample_id"] == sample_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {sample_id} row, found {len(matches)}")
    index, row = matches[0]
    out = []
    for seconds in (200, 400):
        out.append({
            "sample_id": row["sample_id"],
            "manifest_index": int(index),
            "master_id": row["master_id"],
            "lx_m": float(row["lx_m"]),
            "timing_s": int(seconds),
            "note": TIMING_NOTE,
        })
    return out


def write_pilot(manifest: Path = DEFAULT_MANIFEST, timing: Path = DEFAULT_TIMING, seed: int = PILOT_SEED) -> tuple[Path, Path]:
    rows = build_pilot(seed)
    sg.write_rows(rows, manifest)
    sg.write_rows(timing_rows(rows), timing)
    return manifest, timing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--timing-out", type=Path, default=DEFAULT_TIMING)
    ap.add_argument("--seed", type=int, default=PILOT_SEED)
    args = ap.parse_args()
    manifest, timing = write_pilot(args.out, args.timing_out, args.seed)
    n = sum(1 for _ in open(manifest)) - 1
    print(f"wrote {n} pilot rows -> {manifest}")
    print(f"wrote 200 s and 400 s timing rows -> {timing}")
    print("record length is not in the pilot CSV; production N is not chosen")


if __name__ == "__main__":
    main()
