#!/usr/bin/env python3
"""State-of-practice curves on the OpenSees damping and base condition.

OpenSees and the stored 1D column use Q(Vs) damping as Rayleigh damping matched
at the record (f1, f2), and an elastic half-space outcrop. The cached Pretell,
Toro, Passeri, and Dmult curves used a rigid base and frequency-independent
damping. This script recomputes those four arms on the matched physics.

Pretell is the geomean of the record strip. Toro and Passeri randomize Vs,
then assign one soil zeta and one rock zeta from the Taborda-Bielak Q(Vs)
harmonic mean, and propagate with the record's Rayleigh pair. Dmult multiplies
that Q(Vs) zeta by Hallal's 3-6 factors on the base profile. Cosine-bowl
sections are not in this test split.

    python matched_physics.py --check
    python matched_physics.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
VARDOMAIN = PAPER.parent
REPO = VARDOMAIN.parents[1]
CASE7A = REPO / "experiments" / "case7a"
DEEP = REPO / "experiments" / "DeepONet-Residual"
for path in (DEEP, REPO, CASE7A, VARDOMAIN):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from data import Cache  # noqa: E402

OUT = HERE / "03_state_of_practice"
CACHE = REPO / "data" / "vardomain" / "cache"
H5 = {
    "iid": REPO / "data" / "gifno_screen" / "h5",
    "ood_dipping": REPO / "data" / "gifno_screen" / "ood_dipping" / "h5",
}
RHO = 2000.0
N_PRETELL = 200
N_SEEDS = 40
DMULT = tuple(float(x) for x in np.linspace(3.0, 6.0, 10))
FAMILIES = ("iid", "ood_dipping")


def q_of(vs: np.ndarray) -> np.ndarray:
    """Taborda and Bielak (2013) Q(Vs), Vs in m/s."""
    x = np.asarray(vs, dtype=np.float64) / 1000.0
    return 10.5 - 16.0 * x + 153.0 * x**2 - 103.0 * x**3 + 34.7 * x**4 - 5.29 * x**5 + 0.31 * x**6


def qvs_zeta(vs: np.ndarray, soil_nz: int) -> np.ndarray:
    """One soil zeta and one rock zeta, the seiskit global_avg rule."""
    vs = np.asarray(vs, dtype=np.float64).ravel()
    soil_nz = int(np.clip(soil_nz, 1, vs.size))
    soil = vs[:soil_nz]
    rock = vs[soil_nz:]
    q_soil = np.maximum(q_of(soil), 1.0)
    xi_soil = 1.0 / (2.0 * (q_soil.size / np.sum(1.0 / q_soil)))
    if rock.size:
        q_rock = max(float(q_of(np.array([np.median(rock)]))[0]), 1.0)
        xi_rock = 1.0 / (2.0 * q_rock)
    else:
        xi_rock = xi_soil
    zeta = np.full(vs.shape, xi_rock, dtype=np.float64)
    zeta[:soil_nz] = xi_soil
    return zeta


def rayleigh_xi(zeta: np.ndarray, freq: np.ndarray, f1: float, f2: float) -> np.ndarray:
    """Rayleigh damping ratio. ``zeta`` is per layer; the last axis is frequency."""
    zeta = np.asarray(zeta, dtype=np.float64)
    omega = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)
    w1, w2 = 2.0 * np.pi * float(f1), 2.0 * np.pi * float(f2)
    a0 = 2.0 * zeta * w1 * w2 / (w1 + w2)
    a1 = 2.0 * zeta / (w1 + w2)
    return 0.5 * (a0.reshape(zeta.shape + (1,)) / omega + a1.reshape(zeta.shape + (1,)) * omega)


def outcrop_many(freq: np.ndarray, vs: np.ndarray, zeta: np.ndarray, f1: float, f2: float, dz: float) -> np.ndarray:
    """Elastic-halfspace |TF|. ``vs`` and ``zeta`` are (n_col, n_layer) or (n_layer,)."""
    vs = np.asarray(vs, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    single = vs.ndim == 1
    if single:
        vs = vs[None, :]
        zeta = zeta[None, :]
    if vs.shape != zeta.shape:
        raise ValueError(f"vs {vs.shape} and zeta {zeta.shape} differ")
    n_col, n_layer = vs.shape
    freq = np.asarray(freq, dtype=np.float64)
    omega = 2.0 * np.pi * freq
    xi = rayleigh_xi(zeta, freq, f1, f2)
    u = np.ones((n_col, freq.size), dtype=np.complex128)
    tau = np.zeros_like(u)
    for i in range(n_layer):
        vs_c = vs[:, i:i + 1] * np.sqrt(1.0 + 2.0j * xi[:, i, :])
        kh = omega * float(dz) / vs_c
        c, s = np.cos(kh), np.sin(kh)
        gk = 1j * omega * RHO * vs_c
        u, tau = u * c + tau * (s / gk), -u * gk * s + tau * c
    gk_hs = 1j * omega * RHO * vs[:, -1:] * np.sqrt(1.0 + 2.0j * xi[:, -1, :])
    af = 1.0 / np.maximum(np.abs(u - 1j * tau / gk_hs), 1e-12)
    return af[0] if single else af


def _median_soil_zeta(vs: np.ndarray, zeta: np.ndarray) -> float:
    mid = vs.shape[1] // 2
    column_vs = vs[:, mid]
    column_zeta = zeta[:, mid]
    soil = column_zeta[column_vs < 760.0]
    use = soil if soil.size else column_zeta
    return float(np.median(use))


def _geomean(stack: np.ndarray) -> np.ndarray:
    return np.exp(np.mean(np.log(np.clip(stack, 1e-12, None)), axis=0))


def _params(family: str, run: int) -> dict:
    import h5py

    path = H5[family] / f"run_{int(run)}.h5"
    with h5py.File(path, "r") as handle:
        attrs = handle["params"].attrs
        if "H_discretized" in attrs:
            thickness = float(attrs["H_discretized"])
        else:
            thickness = float(attrs["H"])
        soil = int(attrs["soil_layer_count"]) if "soil_layer_count" in attrs else int(round(thickness))
        return {
            "vs1": float(attrs["Vs1"]),
            "vs2": float(attrs["Vs2"]),
            "H": thickness,
            "cov": float(attrs["CoV"]),
            "soil_nz": soil,
        }


def _task(family: str, run: int, n_active: int, f1: float, f2: float) -> dict:
    params = _params(family, run)
    return {"family": family, "run": int(run), "n_active": int(n_active), "f1": float(f1), "f2": float(f2), **params}


def _check_against_column() -> None:
    from section_1d import rayleigh_xi_target, surface_over_outcrop

    freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
    vs = np.linspace(180.0, 400.0, 40)
    zeta = np.full(40, 0.03)
    zeta[-5:] = 0.004
    f1, f2 = 0.8, 2.4
    got = outcrop_many(freq, vs, zeta, f1, f2, dz=1.0)
    xi = np.stack([rayleigh_xi_target(freq, float(z), f1, f2) for z in zeta])
    ref = surface_over_outcrop(freq, vs, dz=1.0, xi=xi, rho=RHO)
    err = float(np.max(np.abs(got - ref)))
    if err > 1e-6:
        raise RuntimeError(f"outcrop mismatch {err}")
    print(f"outcrop check max abs error {err:.2e}", flush=True)


def _one(task: dict) -> dict:
    for path in (DEEP, CASE7A, Path.home() / "seiskit"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    from seiskit.profile_randomization import (
        build_base_case_profile,
        generate_tts_randomized_profile,
        generate_vs_randomized_profile,
    )
    from response_variability.seiskit_arms import hallal_config

    freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
    f1, f2 = float(task["f1"]), float(task["f2"])
    n_active = int(task["n_active"])
    vs = np.asarray(np.load(CACHE / task["family"] / "vs.npy", mmap_mode="r")[int(task["run"]), :n_active], dtype=np.float64)
    zeta = np.asarray(np.load(CACHE / task["family"] / "zeta.npy", mmap_mode="r")[int(task["run"]), :n_active], dtype=np.float64)
    cols = np.linspace(0, vs.shape[1] - 1, N_PRETELL).round().astype(int)
    pretell = _geomean(outcrop_many(freq, vs[:, cols].T, zeta[:, cols].T, f1, f2, dz=1.0))
    cfg = hallal_config(vs1=task["vs1"], H=task["H"], cov=task["cov"], vs2=task["vs2"], dz=0.5)
    soil_nz = max(1, int(round(float(task["H"]) / 0.5)))

    def _stack(kind: str) -> np.ndarray:
        gen = generate_vs_randomized_profile if kind == "toro" else generate_tts_randomized_profile
        rows = []
        for seed in range(1, N_SEEDS + 1):
            profile = np.asarray(gen(cfg, np.random.default_rng(seed)), dtype=np.float64).ravel()
            n_soil = min(soil_nz, profile.size)
            profile_zeta = qvs_zeta(profile, n_soil)
            rows.append(outcrop_many(freq, profile, profile_zeta, f1, f2, dz=0.5))
        return _geomean(np.vstack(rows))

    base = np.asarray(build_base_case_profile(cfg), dtype=np.float64).ravel()
    n_soil = min(soil_nz, base.size)
    base_zeta = qvs_zeta(base, n_soil)
    dmult = _geomean(np.vstack([
        outcrop_many(freq, base, base_zeta * factor, f1, f2, dz=0.5) for factor in DMULT
    ]))
    return {
        "run": int(task["run"]),
        "pretell": pretell.astype(np.float32),
        "toro": _stack("toro").astype(np.float32),
        "passeri": _stack("passeri").astype(np.float32),
        "dmult": dmult.astype(np.float32),
        "xi_soil": _median_soil_zeta(vs, zeta),
    }


def _family_jobs(cache: Cache, family: str, limit: int | None) -> list[dict]:
    camp = next(c for c in cache.campaigns if c.name == family)
    runs = np.flatnonzero(np.asarray(camp.split).astype(str) == "test").astype(np.int64)
    if limit is not None:
        runs = runs[:limit]
    jobs = []
    for run in runs:
        run = int(run)
        jobs.append(_task(family, run, int(camp.n_active[run]), float(camp.f1[run]), float(camp.f2[run])))
    return jobs


def _run_family(family: str, jobs: list[dict], workers: int) -> None:
    freq_n = int(np.load(CACHE / "freq.npy").size)
    out = {name: np.empty((len(jobs), freq_n), dtype=np.float32) for name in ("pretell", "toro", "passeri", "dmult")}
    runs = np.empty(len(jobs), dtype=np.int64)
    xi = np.empty(len(jobs), dtype=np.float64)
    print(f"{family} n={len(jobs)} workers={workers}", flush=True)
    ctx = mp.get_context("spawn")
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for row in pool.map(_one, jobs, chunksize=1):
            i = next(k for k, job in enumerate(jobs) if job["run"] == row["run"])
            runs[i] = row["run"]
            xi[i] = row["xi_soil"]
            for name in out:
                out[name][i] = row[name]
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"  {family} {done}/{len(jobs)}", flush=True)
    dest = OUT / f"matched_{family}.npz"
    np.savez_compressed(dest, run=runs, xi_soil=xi, **out)
    print(f"wrote {dest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--family", choices=FAMILIES, action="append")
    args = ap.parse_args()
    _check_against_column()
    if args.check:
        return
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    cache = Cache(CACHE)
    families = args.family or list(FAMILIES)
    for family in families:
        _run_family(family, _family_jobs(cache, family, args.limit), args.workers)
    (OUT / "matched_physics.json").write_text(json.dumps({
        "base": "elastic half-space outcrop",
        "damping": "Q(Vs) global_avg, Rayleigh at the record (f1, f2)",
        "pretell_columns": N_PRETELL,
        "seeds": N_SEEDS,
        "dmult_multipliers": list(DMULT),
        "excluded": "cosine-bowl troughs are not in the iid or dipping test split",
        "families": families,
        "limit": args.limit,
    }, indent=2))


if __name__ == "__main__":
    main()
