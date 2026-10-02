#!/usr/bin/env python3
"""Importance-sampled GIFNO 6D corner design for Parts 1–2.

32 new 6D locations × 5 RF seeds (8 locations held out forever) plus 10 extra
seeds at sample 50's exact 6D. Writes seiskit-compatible manifests.

    uv run python experiments/DeepONet-Residual/response_variability/corner_is_design.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EXP = Path(__file__).resolve().parents[1]
_REPO = _EXP.parent.parent
_SEISKIT_SOBOL = Path("/home/kurt-/seiskit/neural-operator/data")
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))
if str(_SEISKIT_SOBOL) not in sys.path:
    sys.path.insert(0, str(_SEISKIT_SOBOL))

import config  # noqa: E402

from sobol import (  # noqa: E402
    DEFAULT_SAMPLER_SEED,
    MANIFEST_COLUMNS,
    ManifestEntry,
    RF_SEED_MAX,
    RF_SEED_MIN,
    derive_execution_parameters,
    generate_physical_samples,
    generate_rf_seed_matrix,
    physical_to_unit,
    unit_to_physical,
    write_manifest_csv,
)

OUT_DIR = config.RESULTS_DIR / "response_variability" / "eval_bias"
PER_SAMPLE_CSV = (
    config.RESULTS_DIR / "response_variability" / "gino_bias" / "per_sample.csv"
)

COV_LO, COV_HI = 0.255, 0.3
RH_LO, RH_HI = 80.0, 100.0
N_LOCATIONS = 32
N_HOLDOUT = 8
N_SEEDS = 5
N_POOL = 4000
N_SAMPLE50_SEEDS = 10
DESIGN_SEED = 20260909
NEAR_TOL = 0.03  # unit-cube L2 to existing Sobol corner IDs

# Nested-test sample 50 / 148 (same 6D). Seeds must not collide.
SAMPLE50 = {
    "Vs1": 264.14721504428144,
    "H": 37.0,
    "CoV": 0.2812460053712129,
    "rH": 83.556063847,
    "aHV": 10.97534288,
    "Vs2": 989.6374036043122,
}
FORBIDDEN_RF_SEEDS = frozenset({2486904, 2823621})
SAMPLE50_SAMPLE_ID = 9000


def load_per_sample_anderson(
    path: Path = PER_SAMPLE_CSV,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(CoV, r_H, Anderson) from nested-test IID+dipping rows."""
    cov: list[float] = []
    rh: list[float] = []
    gof: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                c = float(row["CoV"])
                r = float(row["rH"])
                a = float(row["gof_af"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (np.isfinite(c) and np.isfinite(r) and np.isfinite(a)):
                continue
            cov.append(c)
            rh.append(r)
            gof.append(a)
    if not cov:
        raise FileNotFoundError(f"no Anderson rows in {path}")
    return np.asarray(cov), np.asarray(rh), np.asarray(gof)


def scott_bandwidth(x: np.ndarray) -> float:
    n = max(len(x), 2)
    return float(1.06 * np.nanstd(x) * n ** (-0.2))


def kernel_anderson(
    query_cov: np.ndarray,
    query_rh: np.ndarray,
    cov: np.ndarray,
    rh: np.ndarray,
    anderson: np.ndarray,
    *,
    bw_cov: float | None = None,
    bw_rh: float | None = None,
) -> np.ndarray:
    """Nadaraya–Watson smoother of Anderson on (CoV, r_H)."""
    qc = np.asarray(query_cov, dtype=float).reshape(-1)
    qr = np.asarray(query_rh, dtype=float).reshape(-1)
    bw_c = float(bw_cov) if bw_cov is not None else max(scott_bandwidth(cov), 1e-3)
    bw_r = float(bw_rh) if bw_rh is not None else max(scott_bandwidth(rh), 1e-3)
    zc = (qc[:, None] - cov[None, :]) / bw_c
    zr = (qr[:, None] - rh[None, :]) / bw_r
    w = np.exp(-0.5 * (zc**2 + zr**2))
    num = (w * anderson[None, :]).sum(axis=1)
    den = w.sum(axis=1).clip(min=1e-12)
    return num / den


def existing_sobol_corner(
    *,
    n_sobol: int = 256,
    sampler_seed: int = DEFAULT_SAMPLER_SEED,
) -> np.ndarray:
    """Unique GIFNO Sobol 6Ds in the train-q80 CoV × r_H cell (the 13 n3000 IDs)."""
    phys = generate_physical_samples(
        target_count=n_sobol, method="sobol", sampler_seed=sampler_seed
    )
    mask = (phys[:, 2] >= COV_LO) & (phys[:, 3] >= RH_LO)
    return np.unique(np.round(phys[mask], 8), axis=0)


def truncated_lhs(n: int, rng_seed: int) -> np.ndarray:
    """LHS in the truncated 6D corner; Vs1/H/aHV/Vs2 use usual GIFNO maps."""
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=6, seed=int(rng_seed))
    unit = np.asarray(sampler.random(n=n), dtype=float)
    phys = unit_to_physical(unit)
    phys[:, 2] = COV_LO + unit[:, 2] * (COV_HI - COV_LO)
    phys[:, 3] = RH_LO + unit[:, 3] * (RH_HI - RH_LO)
    return phys


def unit_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ua = physical_to_unit(np.atleast_2d(a))
    ub = physical_to_unit(np.atleast_2d(b))
    return np.sqrt(((ua[:, None, :] - ub[None, :, :]) ** 2).sum(axis=-1))


def far_from_existing(
    cand: np.ndarray, existing: np.ndarray, tol: float = NEAR_TOL
) -> np.ndarray:
    if existing.size == 0:
        return np.ones(len(cand), dtype=bool)
    dmin = unit_l2(cand, existing).min(axis=1)
    return dmin >= float(tol)


def importance_pick(
    pool: np.ndarray,
    weights: np.ndarray,
    *,
    n: int,
    existing: np.ndarray,
    rng: np.random.Generator,
    tol: float = NEAR_TOL,
) -> np.ndarray:
    ok = far_from_existing(pool, existing, tol=tol)
    pool = pool[ok]
    weights = np.asarray(weights, dtype=float)[ok]
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0:
        weights = np.ones(len(pool))
    p = weights / weights.sum()
    order = rng.choice(len(pool), size=len(pool), replace=False, p=p)
    picked: list[np.ndarray] = []
    picked_arr = existing.copy()
    for i in order:
        row = pool[int(i)]
        if (
            picked_arr.size
            and not far_from_existing(row[None, :], picked_arr, tol=tol)[0]
        ):
            continue
        picked.append(row)
        picked_arr = np.vstack([picked_arr, row]) if picked_arr.size else row[None, :]
        if len(picked) >= n:
            break
    if len(picked) < n:
        raise RuntimeError(f"only picked {len(picked)}/{n} corner locations")
    return np.asarray(picked, dtype=float)


def maximin_holdout(
    phys: np.ndarray, n_hold: int, rng: np.random.Generator
) -> np.ndarray:
    """Greedy maximin subset (indices) for locations held out of training forever."""
    unit = physical_to_unit(phys)
    n = len(unit)
    start = int(rng.integers(0, n))
    chosen = [start]
    remaining = set(range(n)) - {start}
    while len(chosen) < n_hold and remaining:
        c = unit[np.asarray(chosen)]
        best_i, best_d = None, -1.0
        for i in remaining:
            d = float(np.sqrt(((c - unit[i]) ** 2).sum(axis=1)).min())
            if d > best_d:
                best_d, best_i = d, i
        assert best_i is not None
        chosen.append(best_i)
        remaining.remove(best_i)
    return np.asarray(sorted(chosen), dtype=int)


def extra_rf_seeds(
    n: int,
    *,
    forbidden: frozenset[int],
    rng: np.random.Generator,
) -> np.ndarray:
    seeds: list[int] = []
    seen = set(forbidden)
    while len(seeds) < n:
        s = int(rng.integers(RF_SEED_MIN, RF_SEED_MAX + 1))
        if s in seen:
            continue
        seen.add(s)
        seeds.append(s)
    return np.asarray(seeds, dtype=int)


def entry_from_phys(
    *,
    index: int,
    sample_id: int,
    replicate_id: int,
    rf_seed: int,
    phys: np.ndarray,
) -> ManifestEntry:
    vs1, h, cov, rh, ahv, vs2 = (float(x) for x in phys)
    execution = derive_execution_parameters(Vs1=vs1, H=h)
    return ManifestEntry(
        index=index,
        sample_id=sample_id,
        replicate_id=replicate_id,
        rf_seed=int(rf_seed),
        Vs1=vs1,
        H_requested=execution.H_requested,
        H_discretized=execution.H_discretized,
        soil_layer_count=execution.soil_layer_count,
        CoV=cov,
        rH=rh,
        aHV=ahv,
        Vs2=vs2,
        dz_1D=execution.dz_1D,
        bedrock_thickness=execution.bedrock_thickness,
        bedrock_thickness_discretized=execution.bedrock_thickness_discretized,
        bedrock_layer_count=execution.bedrock_layer_count,
        Lz_discretized=execution.Lz_discretized,
        motion_freq=execution.motion_freq,
        f0_effective=execution.f0_effective,
        duration=execution.duration,
        damping_freq_first=execution.damping_freq_first,
    )


def sample50_phys() -> np.ndarray:
    return np.array(
        [
            SAMPLE50["Vs1"],
            SAMPLE50["H"],
            SAMPLE50["CoV"],
            SAMPLE50["rH"],
            SAMPLE50["aHV"],
            SAMPLE50["Vs2"],
        ],
        dtype=float,
    )


def build_design(
    *,
    n_pool: int = N_POOL,
    n_loc: int = N_LOCATIONS,
    n_hold: int = N_HOLDOUT,
    n_seeds: int = N_SEEDS,
    seed: int = DESIGN_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cov, rh, gof = load_per_sample_anderson()
    existing = existing_sobol_corner()
    pool = truncated_lhs(n_pool, rng_seed=seed)
    a_hat = kernel_anderson(pool[:, 2], pool[:, 3], cov, rh, gof)
    picked = importance_pick(pool, a_hat, n=n_loc, existing=existing, rng=rng)
    hold_idx = maximin_holdout(picked, n_hold, rng)
    hold_mask = np.zeros(n_loc, dtype=bool)
    hold_mask[hold_idx] = True
    seeds = generate_rf_seed_matrix(n_loc, n_seeds, seed=seed + 1)
    corner: list[ManifestEntry] = []
    gidx = 0
    for sid, row in enumerate(picked):
        for rid, rf in enumerate(seeds[sid]):
            corner.append(
                entry_from_phys(
                    index=gidx,
                    sample_id=sid,
                    replicate_id=int(rid),
                    rf_seed=int(rf),
                    phys=row,
                )
            )
            gidx += 1
    extra_seeds = extra_rf_seeds(
        N_SAMPLE50_SEEDS,
        forbidden=FORBIDDEN_RF_SEEDS,
        rng=np.random.default_rng(seed + 2),
    )
    theta50 = sample50_phys()
    extra: list[ManifestEntry] = []
    for rid, rf in enumerate(extra_seeds):
        extra.append(
            entry_from_phys(
                index=rid,
                sample_id=SAMPLE50_SAMPLE_ID,
                replicate_id=int(rid),
                rf_seed=int(rf),
                phys=theta50,
            )
        )
    combined: list[ManifestEntry] = []
    for i, e in enumerate(corner + extra):
        combined.append(
            ManifestEntry(**{**e.to_row(), "index": i})  # type: ignore[arg-type]
        )
    return {
        "locations": picked,
        "held_out": hold_mask,
        "existing_corner": existing,
        "corner": corner,
        "extra": extra,
        "combined": combined,
        "a_hat_picked": kernel_anderson(picked[:, 2], picked[:, 3], cov, rh, gof),
    }


def _write_locations_csv(path: Path, phys: np.ndarray, held_out: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "held_out", "Vs1", "H", "CoV", "rH", "aHV", "Vs2"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(phys):
            w.writerow(
                {
                    "sample_id": i,
                    "held_out": int(held_out[i]),
                    "Vs1": float(row[0]),
                    "H": float(row[1]),
                    "CoV": float(row[2]),
                    "rH": float(row[3]),
                    "aHV": float(row[4]),
                    "Vs2": float(row[5]),
                }
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--seed", type=int, default=DESIGN_SEED)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob = build_design(seed=int(args.seed))
    write_manifest_csv(out / "corner_is_manifest.csv", blob["corner"])
    write_manifest_csv(out / "sample50_extra_seeds.csv", blob["extra"])
    write_manifest_csv(out / "corner_is_all.csv", blob["combined"])
    _write_locations_csv(
        out / "corner_is_locations.csv", blob["locations"], blob["held_out"]
    )
    n_exist = len(blob["existing_corner"])
    n_hold = int(blob["held_out"].sum())
    print(
        f"Wrote {len(blob['corner'])} corner rows "
        f"({N_LOCATIONS} loc × {N_SEEDS} seeds, {n_hold} held-out loc), "
        f"{len(blob['extra'])} sample-50 extra seeds, "
        f"excluded {n_exist} existing Sobol corner IDs → {out}",
        flush=True,
    )
    assert len(blob["corner"]) == N_LOCATIONS * N_SEEDS
    assert len(blob["extra"]) == N_SAMPLE50_SEEDS
    assert n_hold == N_HOLDOUT
    extra_seeds = {int(e.rf_seed) for e in blob["extra"]}
    assert extra_seeds.isdisjoint(FORBIDDEN_RF_SEEDS)
    _ = MANIFEST_COLUMNS


if __name__ == "__main__":
    main()
