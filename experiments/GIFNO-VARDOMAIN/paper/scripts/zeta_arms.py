"""Toro and Passeri again, with each record's median soil damping.

The cached arms use a constant xi of 0.05. This cache keeps those curves and
adds the same two randomizations at the median zeta on the soil rows of the
record. Pretell already uses the record strip. Dmult keeps its own 2% base.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from center_curves import H5
from metrics import ROCK_VS
from paths import CACHE, FIG

N_SEEDS = 40


def _dest(family: str):
    return FIG / f"test_zeta_arms_{family}.npz"


def _median_zeta(family: str, run: int, soil: int) -> float:
    vs = np.asarray(np.load(CACHE / family / "vs.npy", mmap_mode="r")[int(run), :soil], dtype=np.float64)
    zeta = np.asarray(np.load(CACHE / family / "zeta.npy", mmap_mode="r")[int(run), :soil], dtype=np.float64)
    if vs.ndim == 1:
        soil_zeta = zeta[vs < ROCK_VS]
    else:
        row = np.median(vs, axis=1)
        mask = row < ROCK_VS
        soil_zeta = zeta[mask] if np.any(mask) else zeta
    if soil_zeta.size == 0:
        soil_zeta = zeta
    return float(np.median(soil_zeta))


def _task(family: str, run: int) -> dict:
    import h5py

    path = H5[family] / f"run_{int(run)}.h5"
    with h5py.File(path, "r") as handle:
        params = handle["params"].attrs
        thickness = float(params["H_discretized"] if "H_discretized" in params else params["H"])
        soil = int(params["soil_layer_count"]) if "soil_layer_count" in params else int(round(thickness))
        return {
            "family": family,
            "run": int(run),
            "vs1": float(params["Vs1"]),
            "H": thickness,
            "cov": float(params["CoV"]),
            "vs2": float(params["Vs2"]),
            "xi": _median_zeta(family, int(run), soil),
        }


def _one(task: dict) -> dict:
    import sys

    from paths import REPO

    deep = str(REPO / "experiments" / "DeepONet-Residual")
    if deep not in sys.path:
        sys.path.insert(0, deep)
    from response_variability.seiskit_arms import hallal_geomean_tf

    freq = np.load(CACHE / "freq.npy")
    curves = {}
    for kind in ("toro", "passeri"):
        curve, _sig = hallal_geomean_tf(
            freq=freq,
            vs1=float(task["vs1"]),
            H=float(task["H"]),
            cov=float(task["cov"]),
            vs2=float(task["vs2"]),
            xi=float(task["xi"]),
            n_seeds=N_SEEDS,
            kind=kind,
        )
        curves[kind] = np.asarray(curve, dtype=np.float32)
    return {"run": int(task["run"]), "xi": float(task["xi"]), **curves}


def ensure_zeta_arms(family: str, runs: np.ndarray, *, workers: int = 6) -> dict[str, np.ndarray]:
    dest = _dest(family)
    runs = np.asarray(runs, dtype=np.int64)
    if dest.is_file():
        blob = np.load(dest)
        if np.array_equal(np.asarray(blob["run"]), runs):
            print(f"loaded zeta arms {family} n={runs.size}", flush=True)
            return {
                "toro": np.asarray(blob["toro"], dtype=np.float64),
                "passeri": np.asarray(blob["passeri"], dtype=np.float64),
                "zeta": np.asarray(blob["zeta"], dtype=np.float64),
            }
    tasks = [_task(family, int(run)) for run in runs]
    n_freq = int(np.load(CACHE / "freq.npy").size)
    toro = np.empty((runs.size, n_freq), dtype=np.float32)
    passeri = np.empty((runs.size, n_freq), dtype=np.float32)
    zeta = np.empty(runs.size, dtype=np.float64)
    print(f"zeta arms {family} n={runs.size} median xi {np.median([t['xi'] for t in tasks]):.4f}", flush=True)
    ctx = mp.get_context("spawn")
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for row in pool.map(_one, tasks, chunksize=1):
            i = int(np.where(runs == row["run"])[0][0])
            toro[i] = row["toro"]
            passeri[i] = row["passeri"]
            zeta[i] = row["xi"]
            done += 1
            if done % 25 == 0 or done == runs.size:
                print(f"  zeta arms {family} {done}/{runs.size}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, run=runs, toro=toro, passeri=passeri, zeta=zeta)
    return {
        "toro": np.asarray(toro, dtype=np.float64),
        "passeri": np.asarray(passeri, dtype=np.float64),
        "zeta": zeta,
    }
