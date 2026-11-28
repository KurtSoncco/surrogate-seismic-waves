#!/usr/bin/env python3
"""Parameter manifest for a later gradational-basin fine-tune.

Writes sample ranges only. It does not call OpenSees or build H5 files.
The holdout row is the Case 7a center column and is not part of the train split.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

N_TRAIN = 256
SEED = 0
DEPTH_M = 150.0
WINDOW_M = 500.0
ZETA_RATIO = 10.0
DOWNHOLE_FRACTION = 0.8
SMOOTH_COEFF = 500
N_FREQ = 1000
FREQ_LO_HZ = 0.1
FREQ_HI_HZ = 10.0
N_RECORDERS = 21
RECORDER_SPACING_M = 15.0

# Measured center column from the Case 7a Vs grid and the instruction sheet.
CASE7A_SOIL_M = 85.0
CASE7A_VS_SURFACE = 140.0
CASE7A_VS_CONTACT = 353.0
CASE7A_VS_ROCK = 2371.0
CASE7A_XI = 0.0075
CASE7A_F1_HZ = 0.75
CASE7A_F2_HZ = 2.25

COLUMNS = (
    "sample_id",
    "split",
    "holdout_case7a",
    "soil_thickness_m",
    "vs_surface_mps",
    "vs_contact_mps",
    "vs_rock_mps",
    "contact_depth_west_m",
    "contact_depth_east_m",
    "rayleigh_xi",
    "rayleigh_f1_hz",
    "rayleigh_f2_hz",
    "zeta_soil",
    "zeta_rock",
    "zeta_ratio",
    "downhole_depth_fraction",
    "downhole_depth_m",
    "depth_m",
    "smooth_coeff",
    "n_freq",
    "freq_lo_hz",
    "freq_hi_hz",
    "n_recorders",
    "recorder_spacing_m",
    "label",
)


def _travel_time_s(thickness_m: float, vs_surface: float, vs_contact: float) -> float:
    """Travel time through a linear Vs gradient. Uniform Vs uses thickness/Vs."""
    if abs(vs_contact - vs_surface) < 1e-6:
        return thickness_m / vs_surface
    return thickness_m * np.log(vs_contact / vs_surface) / (vs_contact - vs_surface)


def _label_fields(thickness_m: float) -> dict[str, float | int | str]:
    return {
        "downhole_depth_fraction": DOWNHOLE_FRACTION,
        "downhole_depth_m": DOWNHOLE_FRACTION * DEPTH_M,
        "depth_m": DEPTH_M,
        "smooth_coeff": SMOOTH_COEFF,
        "n_freq": N_FREQ,
        "freq_lo_hz": FREQ_LO_HZ,
        "freq_hi_hz": FREQ_HI_HZ,
        "n_recorders": N_RECORDERS,
        "recorder_spacing_m": RECORDER_SPACING_M,
        "label": (
            "surface / downhole in rock; "
            f"Konno-Ohmachi {SMOOTH_COEFF}; "
            f"{N_FREQ} bins {FREQ_LO_HZ}-{FREQ_HI_HZ} Hz; "
            f"soil thickness {thickness_m:.1f} m is not the downhole depth"
        ),
    }


def _row(
    sample_id: str,
    split: str,
    holdout: int,
    soil_m: float,
    vs_surface: float,
    vs_contact: float,
    vs_rock: float,
    west_m: float,
    east_m: float,
    xi: float,
    f1_hz: float,
    f2_hz: float,
) -> dict:
    row = {
        "sample_id": sample_id,
        "split": split,
        "holdout_case7a": holdout,
        "soil_thickness_m": soil_m,
        "vs_surface_mps": vs_surface,
        "vs_contact_mps": vs_contact,
        "vs_rock_mps": vs_rock,
        "contact_depth_west_m": west_m,
        "contact_depth_east_m": east_m,
        "rayleigh_xi": xi,
        "rayleigh_f1_hz": f1_hz,
        "rayleigh_f2_hz": f2_hz,
        "zeta_soil": xi,
        "zeta_rock": xi / ZETA_RATIO,
        "zeta_ratio": ZETA_RATIO,
    }
    row.update(_label_fields(soil_m))
    return row


def train_rows(n_train: int = N_TRAIN, seed: int = SEED) -> list[dict]:
    """Gradational columns. Rock fills the 150 m mesh. Zeta ratio stays 10."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_train):
        soil_m = float(rng.uniform(40.0, 120.0))
        vs_surface = float(rng.uniform(100.0, 200.0))
        vs_contact = float(rng.uniform(250.0, 600.0))
        vs_rock = float(rng.uniform(800.0, 2600.0))
        dip_m = float(rng.uniform(-40.0, 40.0))
        west = float(np.clip(soil_m - 0.5 * dip_m, 20.0, 140.0))
        east = float(np.clip(soil_m + 0.5 * dip_m, 20.0, 140.0))
        xi = float(rng.uniform(0.005, 0.015))
        f0 = 0.25 / _travel_time_s(soil_m, vs_surface, vs_contact)
        rows.append(
            _row(
                f"basin_{i:04d}",
                "train",
                0,
                soil_m,
                vs_surface,
                vs_contact,
                vs_rock,
                west,
                east,
                xi,
                f0,
                3.0 * f0,
            )
        )
    return rows


DZ_M = 1.0
NX_VARIABILITY = 500


def gradational_field(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Linear Vs gradient and bedrock mask for the 500 m window.

    Row 0 is the surface element. Shape is (150, 500), one metre per cell.
    Contact depth varies linearly from the west edge to the east edge.
    """
    nz = int(DEPTH_M / DZ_M)
    nx = NX_VARIABILITY
    z_m = (np.arange(nz) + 0.5) * DZ_M
    x_frac = np.linspace(0.0, 1.0, nx)
    west_m = float(row["contact_depth_west_m"])
    east_m = float(row["contact_depth_east_m"])
    contact_m = west_m + (east_m - west_m) * x_frac
    vs_surface = float(row["vs_surface_mps"])
    vs_contact = float(row["vs_contact_mps"])
    vs_rock = float(row["vs_rock_mps"])
    frac = np.clip(z_m[:, None] / np.maximum(contact_m[None, :], 1e-6), 0.0, 1.0)
    soil = z_m[:, None] < contact_m[None, :]
    vs_soil = vs_surface + frac * (vs_contact - vs_surface)
    vs = np.where(soil, vs_soil, vs_rock)
    return vs.astype(np.float64), np.logical_not(soil)


def case7a_holdout_row() -> dict:
    """Center column of Treasure Island. Excluded from the train split."""
    return _row(
        "holdout_case7a",
        "holdout",
        1,
        CASE7A_SOIL_M,
        CASE7A_VS_SURFACE,
        CASE7A_VS_CONTACT,
        CASE7A_VS_ROCK,
        CASE7A_SOIL_M,
        CASE7A_SOIL_M,
        CASE7A_XI,
        CASE7A_F1_HZ,
        CASE7A_F2_HZ,
    )


BASIN_RANGES = {
    "soil_thickness_m": (40.0, 120.0),
    "vs_surface_mps": (100.0, 200.0),
    "vs_contact_mps": (250.0, 600.0),
    "vs_rock_mps": (800.0, 2600.0),
    "zeta_soil": (0.005, 0.015),
    "dip_m": (-40.0, 40.0),
}
CASE7A_EXCLUSION_RADIUS = 0.08


def near_case7a(row: dict) -> bool:
    d = [
        (float(row[k]) - v) / (BASIN_RANGES[k][1] - BASIN_RANGES[k][0])
        for k, v in (
            ("soil_thickness_m", CASE7A_SOIL_M),
            ("vs_surface_mps", CASE7A_VS_SURFACE),
            ("vs_contact_mps", CASE7A_VS_CONTACT),
            ("zeta_soil", CASE7A_XI),
        )
    ]
    return bool(np.linalg.norm(d) < CASE7A_EXCLUSION_RADIUS)


def _finalize_basin(row: dict, sample_id: str, split: str = "train") -> dict:
    soil = float(row["soil_thickness_m"])
    west = float(row.get("contact_depth_west_m", soil))
    east = float(row.get("contact_depth_east_m", soil))
    f0 = 0.25 / _travel_time_s(soil, float(row["vs_surface_mps"]), float(row["vs_contact_mps"]))
    return _row(
        sample_id,
        split,
        0,
        soil,
        float(row["vs_surface_mps"]),
        float(row["vs_contact_mps"]),
        float(row["vs_rock_mps"]),
        west,
        east,
        float(row["zeta_soil"]),
        f0,
        3.0 * f0,
    )


def mutate_basin(row: dict, rng: np.random.Generator, sample_id: str, step: float = 0.15) -> dict:
    """ACCEL edit of one basin parameter (plan step 7 applied to the 256-run ensemble)."""
    for _ in range(100):
        key = str(rng.choice(list(BASIN_RANGES)))
        lo, hi = BASIN_RANGES[key]
        new = dict(row)
        if key == "dip_m":
            dip = 0.5 * (float(row["contact_depth_east_m"]) - float(row["contact_depth_west_m"]))
            dip = float(np.clip(dip + rng.uniform(-step, step) * (hi - lo), lo, hi))
            soil = float(row["soil_thickness_m"])
            new["contact_depth_west_m"] = float(np.clip(soil - dip, 20.0, 140.0))
            new["contact_depth_east_m"] = float(np.clip(soil + dip, 20.0, 140.0))
        else:
            new[key] = float(np.clip(float(row[key]) + rng.uniform(-step, step) * (hi - lo), lo, hi))
        if key != "dip_m" and new[key] == row[key]:
            continue
        out = _finalize_basin(new, sample_id)
        out["parent_id"] = row["sample_id"]
        out["mutated_key"] = key
        if not near_case7a(out):
            return out
    raise RuntimeError(f"could not mutate {row['sample_id']} away from Case 7a")


def basin_rows(n_train: int = N_TRAIN, seed: int = SEED) -> list[dict]:
    return train_rows(n_train, seed) + [case7a_holdout_row()]


def write_manifest(path: Path, n_train: int = N_TRAIN, seed: int = SEED) -> Path:
    rows = basin_rows(n_train, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "basin_ensemble.csv",
    )
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    path = write_manifest(args.out, args.n_train, args.seed)
    print(f"wrote {path} ({args.n_train} train + 1 holdout)")


if __name__ == "__main__":
    main()
