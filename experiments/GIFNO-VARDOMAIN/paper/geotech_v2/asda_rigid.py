#!/usr/bin/env python3
"""Column and state of practice with a rigid base where the dashpot sits.

OpenSees puts the bedrock in the mesh and the ASDAbsorbingBoundary2D under it.
The rigid counterpart keeps that finite bedrock and prescribes the total motion
of the base node. |TF| is surface / base. The OpenSees target stays surface /
injected wave, the same labels as asda_elastic.py.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "experiments" / "case7a"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE.parents[1]))

import data as _vardomain_data  # noqa: E402, F401  keeps Cache bound to the vardomain module

from asda_elastic import SCALE  # noqa: E402
from bootstrap_ci import bootstrap_cluster_statistic  # noqa: E402
from center_curves import FAMILIES, test_frame  # noqa: E402
from matched_physics import (  # noqa: E402
    CACHE,
    DMULT,
    N_PRETELL,
    N_SEEDS,
    RHO,
    _family_jobs,
    _geomean,
    qvs_zeta,
    rayleigh_xi,
)
from metrics import anderson, pearson  # noqa: E402

OUT = HERE / "03_state_of_practice"
METHODS = ("Column", "Pretell", "Passeri", "Dmult", "Toro")


def rigid_many(freq: np.ndarray, vs: np.ndarray, zeta: np.ndarray, f1: float, f2: float, dz: float) -> np.ndarray:
    """|u(surface) / u(base)|. The base is the bottom of ``vs``, bedrock included."""
    vs = np.asarray(vs, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    single = vs.ndim == 1
    if single:
        vs = vs[None, :]
        zeta = zeta[None, :]
    freq = np.asarray(freq, dtype=np.float64)
    omega = 2.0 * np.pi * freq
    xi = rayleigh_xi(zeta, freq, f1, f2)
    u = np.ones((vs.shape[0], freq.size), dtype=np.complex128)
    tau = np.zeros_like(u)
    for i in range(vs.shape[1]):
        vs_c = vs[:, i:i + 1] * np.sqrt(1.0 + 2.0j * xi[:, i, :])
        kh = omega * float(dz) / vs_c
        c, s = np.cos(kh), np.sin(kh)
        gk = 1j * omega * RHO * vs_c
        u, tau = u * c + tau * (s / gk), -u * gk * s + tau * c
    af = 1.0 / np.maximum(np.abs(u), 1e-12)
    return af[0] if single else af


def _one(task: dict) -> dict:
    for path in (REPO / "experiments" / "DeepONet-Residual", REPO / "experiments" / "case7a", Path.home() / "seiskit"):
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
    pretell = _geomean(rigid_many(freq, vs[:, cols].T, zeta[:, cols].T, f1, f2, dz=1.0))
    cfg = hallal_config(vs1=task["vs1"], H=task["H"], cov=task["cov"], vs2=task["vs2"], dz=0.5)
    soil_nz = max(1, int(round(float(task["H"]) / 0.5)))

    def _stack(kind: str) -> np.ndarray:
        gen = generate_vs_randomized_profile if kind == "toro" else generate_tts_randomized_profile
        rows = []
        for seed in range(1, N_SEEDS + 1):
            profile = np.asarray(gen(cfg, np.random.default_rng(seed)), dtype=np.float64).ravel()
            profile_zeta = qvs_zeta(profile, min(soil_nz, profile.size))
            rows.append(rigid_many(freq, profile, profile_zeta, f1, f2, dz=0.5))
        return _geomean(np.vstack(rows))

    base = np.asarray(build_base_case_profile(cfg), dtype=np.float64).ravel()
    base_zeta = qvs_zeta(base, min(soil_nz, base.size))
    dmult = _geomean(np.vstack([
        rigid_many(freq, base, base_zeta * factor, f1, f2, dz=0.5) for factor in DMULT
    ]))
    return {
        "run": int(task["run"]),
        "pretell": pretell.astype(np.float32),
        "toro": _stack("toro").astype(np.float32),
        "passeri": _stack("passeri").astype(np.float32),
        "dmult": dmult.astype(np.float32),
    }


def _run_family(family: str, jobs: list[dict], workers: int) -> None:
    freq_n = int(np.load(CACHE / "freq.npy").size)
    out = {name: np.empty((len(jobs), freq_n), dtype=np.float32) for name in ("pretell", "toro", "passeri", "dmult")}
    runs = np.empty(len(jobs), dtype=np.int64)
    print(f"{family} n={len(jobs)} workers={workers}", flush=True)
    ctx = mp.get_context("spawn")
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for row in pool.map(_one, jobs, chunksize=1):
            i = next(k for k, job in enumerate(jobs) if job["run"] == row["run"])
            runs[i] = row["run"]
            for name in out:
                out[name][i] = row[name]
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  {family} {done}/{len(jobs)}", flush=True)
    dest = OUT / f"asda_rigid_{family}.npz"
    np.savez_compressed(dest, run=runs, **out)
    print(f"wrote {dest}", flush=True)


def _column(family: str) -> np.ndarray:
    """Center-station rigid column, bedrock included, Rayleigh at the record (f1, f2)."""
    root = CACHE / family
    split = np.load(root / "split.npy").astype(str)
    rows = np.flatnonzero(split == "test")
    rec_x = np.load(root / "rec_x.npy")
    col = int(rec_x[10])
    vs = np.asarray(np.load(root / "vs.npy", mmap_mode="r")[rows, :, col], dtype=np.float64)
    zeta = np.asarray(np.load(root / "zeta.npy", mmap_mode="r")[rows, :, col], dtype=np.float64)
    n_active = np.asarray(np.load(root / "n_active.npy")[rows], dtype=np.int32)
    f1 = np.asarray(np.load(root / "rayleigh_f1.npy")[rows], dtype=np.float64)
    f2 = np.asarray(np.load(root / "rayleigh_f2.npy")[rows], dtype=np.float64)
    freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
    out = np.empty((rows.size, freq.size), dtype=np.float64)
    for i in range(rows.size):
        n = int(n_active[i])
        out[i] = rigid_many(freq, vs[i, :n], zeta[i, :n], float(f1[i]), float(f2[i]), dz=1.0)
    return out


def _score(freq, truth, curve, f0) -> dict[str, np.ndarray]:
    band = (freq >= 0.4) & (freq <= 2.5)
    whole = (freq >= 0.1) & (freq <= 10.0)
    log_c = np.log(np.clip(curve, 1e-12, None))
    log_t = np.log(np.clip(truth, 1e-12, None))
    return {
        "pearson": np.array([pearson(curve[i], truth[i]) for i in range(truth.shape[0])]),
        "anderson": np.array([anderson(freq, truth[i], curve[i], f0[i]) for i in range(truth.shape[0])]),
        "log_l1_mid": np.mean(np.abs(log_c[:, band] - log_t[:, band]), axis=1),
        "log_l1_full": np.mean(np.abs(log_c[:, whole] - log_t[:, whole]), axis=1),
    }


def _table() -> None:
    rows = []
    for name, family in FAMILIES.items():
        frame = test_frame(family)
        blob = np.load(OUT / f"asda_rigid_{family}.npz")
        order = {int(run): i for i, run in enumerate(np.asarray(blob["run"]))}
        c = SCALE[family]
        truth = frame["opensees"] * c
        curves = {"Column": _column(family)}
        for method in METHODS[1:]:
            curves[method] = np.stack([
                np.asarray(blob[method.lower()][order[int(run)]], dtype=np.float64) for run in frame["run"]
            ])
        print(f"{name}  c={c:.4f}", flush=True)
        for method in METHODS:
            scored = _score(frame["freq"], truth, curves[method], frame["f0"])
            stat_p = bootstrap_cluster_statistic(scored["pearson"], frame["group"], np.median, rng=np.random.default_rng(0))
            stat_a = bootstrap_cluster_statistic(scored["anderson"], frame["group"], np.median, rng=np.random.default_rng(0))
            site = pd.DataFrame({"value": scored["pearson"], "group": frame["group"]}).groupby("group")["value"].median()
            rows.append({
                "domain": name,
                "method": method,
                "n_records": int(frame["run"].size),
                "n_sites": stat_p["n_clusters"],
                "pearson_record": stat_p["point"],
                "pearson_low": stat_p["low"],
                "pearson_high": stat_p["high"],
                "pearson_site": float(site.median()),
                "anderson_record": stat_a["point"],
                "anderson_low": stat_a["low"],
                "anderson_high": stat_a["high"],
                "log_l1_0p4_2p5": float(np.median(scored["log_l1_mid"])),
                "log_l1_0p1_10": float(np.median(scored["log_l1_full"])),
                "input_scale_c": c,
            })
    table = pd.DataFrame(rows)
    dest = OUT / "table_asda_rigid.csv"
    table.to_csv(dest, index=False)
    print(table.to_string(index=False))
    print(f"wrote {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--family", choices=("iid", "ood_dipping"), action="append")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.score_only:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from data import Cache

        cache = Cache(CACHE)
        for family in args.family or ("iid", "ood_dipping"):
            _run_family(family, _family_jobs(cache, family, None), args.workers)
    _table()


if __name__ == "__main__":
    main()
