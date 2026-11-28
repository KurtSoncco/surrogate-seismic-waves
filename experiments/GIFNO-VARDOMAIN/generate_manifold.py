#!/usr/bin/env python3
"""150 m manifold sections for the first OpenSees wave.

Depth is the Case 7 mesh, 150 m including the bedrock. Interiors are 500, 1000,
1500 and 2000 m. Records are 50 s, which is where the 1D truncation bias at
zeta 0.005 falls below 1e-5. Half the random-field rows are a second seed of
the same geology. Layered soil is not in this wave: every row is the gradient
fill ``section_field`` already solves.

    python generate_manifold.py --out sections/manifold150.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CASE7A_DIR = Path(__file__).resolve().parents[1] / "case7a"
if str(CASE7A_DIR) not in sys.path:
    sys.path.insert(0, str(CASE7A_DIR))

import section_generator as sg  # noqa: E402

DURATION_S = 50.0
DEPTH_M = 150.0
FEATURE_WIDTH_MIN_M = 75.0
LOW_ZETA = 0.01
LOW_F1_HZ = 0.4

# (stratum, width_m, n_rows). Pairs are counted as rows.
STRATA = (
    ("one_feature", 500, 8),
    ("one_feature", 1000, 32),
    ("one_feature", 1500, 32),
    ("one_feature", 2000, 32),
    ("two_feature", 1000, 16),
    ("two_feature", 1500, 16),
    ("two_feature", 2000, 16),
    ("dip", 1500, 16),
    ("dip", 2000, 16),
    ("low_cov", 1000, 16),
    ("low_cov", 1500, 16),
    ("low_cov", 2000, 16),
    ("train_cov", 1000, 16),
    ("train_cov", 1500, 16),
    ("train_cov", 2000, 16),
    ("matched", 1000, 8),
    ("matched", 2000, 8),
    ("blind", 0, 16),
)


def _fits(row: dict) -> bool:
    x = (np.arange(int(row["lx_m"])) + 0.5)
    h = np.full_like(x, float(row["soil_thickness_m"]))
    feats = [{k: row[k] for k in ("feature", "feature_x_frac", "feature_width_m", "feature_dh_m", "trough_depth_m")}]
    feats += json.loads(row.get("extra_features") or "[]")
    for f in feats:
        h = h + sg._feature_offset(
            x, float(row["lx_m"]), sg.FEATURES[int(f["feature"])], float(f["feature_x_frac"]),
            float(f["feature_width_m"]), float(f["feature_dh_m"]), float(f["trough_depth_m"]),
        )
    return bool(h.min() >= 15.0 and h.max() <= DEPTH_M - 5.0)


def _base(rng: np.random.Generator, lx: float, sample_id: str) -> dict:
    for _ in range(40):
        raw = {k: sg._scale(k, float(rng.random())) for k in sg.RANGES}
        row = sg.finalize(raw, sample_id, lx)
        if not sg.near_case7(row) and _fits(row) and row["feature_name"] != "trough":
            return row
    raise RuntimeError(f"no geometry fit {lx} m for {sample_id}")


def _place_feature(row: dict, rng: np.random.Generator, kind: str, x_frac: float) -> None:
    lx = float(row["lx_m"])
    row["feature"] = sg.FEATURES.index(kind)
    row["feature_name"] = kind
    row["feature_x_frac"] = x_frac
    row["feature_width_m"] = float(rng.uniform(FEATURE_WIDTH_MIN_M, max(FEATURE_WIDTH_MIN_M, 0.5 * lx)))
    row["feature_dh_m"] = float(rng.uniform(-40.0, 40.0))
    row["trough_depth_m"] = float(rng.uniform(20.0, 50.0))


def _as_dip(row: dict, rng: np.random.Generator) -> dict:
    """A ramp whose length is |dh| / tan(theta), redrawn until it fits."""
    lx = float(row["lx_m"])
    for _ in range(80):
        dh = float(rng.uniform(30.0, 80.0) * rng.choice([-1.0, 1.0]))
        theta = float(rng.uniform(np.deg2rad(2.0), np.deg2rad(15.0)))
        width = abs(dh) / np.tan(theta)
        if width < FEATURE_WIDTH_MIN_M or width > 0.7 * lx:
            continue
        row["feature"] = sg.FEATURES.index("ramp")
        row["feature_name"] = "ramp"
        row["feature_dh_m"] = dh
        row["feature_width_m"] = width
        row["feature_x_frac"] = float(rng.uniform(0.2, 0.8))
        if _fits(row):
            row["dip_deg"] = float(np.rad2deg(theta))
            return row
    raise RuntimeError(f"no dip fit {lx} m")


def _two_features(row: dict, rng: np.random.Generator) -> dict:
    lx = float(row["lx_m"])
    for _ in range(40):
        kinds = rng.choice(["ramp", "step"], size=2)
        w0 = float(rng.uniform(FEATURE_WIDTH_MIN_M, 0.3 * lx))
        w1 = float(rng.uniform(FEATURE_WIDTH_MIN_M, 0.3 * lx))
        gap = 0.5 * (w0 + w1) / lx + 0.05
        x0 = float(rng.uniform(0.15, 0.85 - gap))
        x1 = float(rng.uniform(x0 + gap, 0.85))
        _place_feature(row, rng, str(kinds[0]), x0)
        row["feature_width_m"] = w0
        extra = {
            "feature": sg.FEATURES.index(str(kinds[1])),
            "feature_x_frac": x1,
            "feature_width_m": w1,
            "feature_dh_m": float(rng.uniform(-40.0, 40.0)),
            "trough_depth_m": float(rng.uniform(20.0, 50.0)),
        }
        row["extra_features"] = json.dumps([extra])
        if _fits(row):
            return row
    raise RuntimeError(f"no two-feature fit {lx} m")


def _stamp(row: dict, *, stratum: str, group: str, family: str, cov: float, seed: int) -> dict:
    row = dict(row)
    row.update(
        stratum=stratum,
        group_id=group,
        damping_family=family,
        duration_s=DURATION_S,
        depth_m=DEPTH_M,
        vs_cov=float(cov),
        field_seed=int(seed),
        vs_corr_h_m=50.0,
        vs_corr_aniso=10.0,
        soil_fill="gradient",
        extra_features=row.get("extra_features") or "[]",
    )
    return row


def _explicit(row: dict, rng: np.random.Generator, low: bool) -> None:
    if low:
        zeta = float(np.exp(rng.uniform(np.log(0.005), np.log(LOW_ZETA))))
        f1 = float(rng.uniform(0.2, LOW_F1_HZ))
    else:
        zeta = float(np.exp(rng.uniform(np.log(0.005), np.log(0.05))))
        f1 = float(row["rayleigh_f1_hz"])
    row["zeta_soil"] = zeta
    row["zeta_rock"] = max(0.00075, zeta * float(rng.uniform(0.05, 1.0)))
    row["rayleigh_f1_hz"] = f1
    row["rayleigh_f2_hz"] = 3.0 * f1


def build(seed: int = 11) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    n = 0

    def take(stratum: str, lx: float) -> dict:
        nonlocal n
        row = _base(rng, lx, f"man_{n:04d}")
        n += 1
        return row

    for stratum, lx, count in STRATA:
        if stratum == "blind":
            for width in (500, 1000, 1500, 2000):
                for _ in range(4):
                    row = take(stratum, width)
                    rows.append(_stamp(row, stratum=stratum, group=row["sample_id"], family="qvs", cov=0.0, seed=0))
            continue
        made = 0
        while made < count:
            row = take(stratum, lx)
            group = row["sample_id"]
            if stratum == "dip":
                row = _as_dip(row, rng)
            elif stratum == "two_feature":
                row = _two_features(row, rng)
            elif stratum == "one_feature":
                _place_feature(row, rng, str(rng.choice(("flat", "ramp", "step"))), float(rng.uniform(0.15, 0.85)))
                if not _fits(row):
                    continue
            cov = 0.0
            if stratum == "low_cov":
                cov = 0.05
            elif stratum == "train_cov":
                cov = float(rng.uniform(0.10, 0.30))
            if stratum == "matched":
                left = _stamp(row, stratum=stratum, group=group, family="qvs", cov=0.0, seed=0)
                right = _stamp(dict(row, sample_id=f"man_{n:04d}"), stratum=stratum, group=group, family="explicit", cov=0.0, seed=0)
                n += 1
                _explicit(right, rng, low=False)
                rows.extend((left, right))
                made += 2
                continue
            family = "explicit" if rng.random() < 0.30 else "qvs"
            stamped = _stamp(row, stratum=stratum, group=group, family=family, cov=cov, seed=int(rng.integers(1, 2**31)))
            if family == "explicit":
                _explicit(stamped, rng, low=False)
            rows.append(stamped)
            made += 1
            if stratum in ("low_cov", "train_cov") and made <= count / 2:
                twin = _stamp(dict(row, sample_id=f"man_{n:04d}"), stratum=stratum, group=group, family=family, cov=cov, seed=int(rng.integers(1, 2**31)))
                n += 1
                if family == "explicit":
                    twin["zeta_soil"] = stamped["zeta_soil"]
                    twin["zeta_rock"] = stamped["zeta_rock"]
                    twin["rayleigh_f1_hz"] = stamped["rayleigh_f1_hz"]
                    twin["rayleigh_f2_hz"] = stamped["rayleigh_f2_hz"]
                rows.append(twin)
                made += 1

    explicit_wide = [r for r in rows if r["damping_family"] == "explicit" and r["lx_m"] in (1500.0, 2000.0) and r["stratum"] != "matched"]
    need = int(np.ceil(0.25 * len(explicit_wide)))
    have = [r for r in explicit_wide if r["zeta_soil"] <= LOW_ZETA and r["rayleigh_f1_hz"] <= LOW_F1_HZ]
    for row in explicit_wide:
        if len(have) >= need:
            break
        if row in have:
            continue
        _explicit(row, rng, low=True)
        have.append(row)
    for i, row in enumerate(rows):
        row["split"] = "blind" if row["stratum"] == "blind" else "train"
        row["sample_id"] = f"man_{i:04d}"
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = build(args.seed)
    sg.write_rows(rows, args.out)
    print(f"wrote {len(rows)} rows depth={DEPTH_M} duration={DURATION_S} -> {args.out}")


if __name__ == "__main__":
    main()
