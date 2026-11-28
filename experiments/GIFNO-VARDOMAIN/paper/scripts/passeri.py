"""Passeri geomean on the same tasks as the saved Toro curves.

``kind='passeri'`` is the TTS random profile in ``hallal_geomean_tf``. Results
are cached beside the Pretell / Toro / Dmult npz files.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from center_curves import H5, passeri_path
from paths import CACHE

N_SEEDS = 40


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
            "soil_nz": soil,
        }


def _one(task: dict) -> dict:
    import sys

    from paths import REPO

    deep = str(REPO / "experiments" / "DeepONet-Residual")
    if deep not in sys.path:
        sys.path.insert(0, deep)
    from response_variability.seiskit_arms import hallal_geomean_tf

    freq = np.load(CACHE / "freq.npy")
    curve, _sig = hallal_geomean_tf(
        freq=freq,
        vs1=float(task["vs1"]),
        H=float(task["H"]),
        cov=float(task["cov"]),
        vs2=float(task["vs2"]),
        xi=0.05,
        n_seeds=N_SEEDS,
        kind="passeri",
    )
    return {"run": int(task["run"]), "passeri": np.asarray(curve, dtype=np.float32)}


def ensure_passeri(family: str, runs: np.ndarray, *, workers: int = 6) -> np.ndarray:
    dest = passeri_path(family)
    runs = np.asarray(runs, dtype=np.int64)
    if dest.is_file():
        blob = np.load(dest)
        if np.array_equal(np.asarray(blob["run"]), runs):
            print(f"loaded passeri {family} n={runs.size}", flush=True)
            return np.asarray(blob["passeri"], dtype=np.float64)
    tasks = [_task(family, int(run)) for run in runs]
    out = np.empty((runs.size, int(np.load(CACHE / "freq.npy").size)), dtype=np.float32)
    print(f"passeri {family} n={runs.size}", flush=True)
    ctx = mp.get_context("spawn")
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for row in pool.map(_one, tasks, chunksize=1):
            i = int(np.where(runs == row["run"])[0][0])
            out[i] = row["passeri"]
            done += 1
            if done % 25 == 0 or done == runs.size:
                print(f"  passeri {family} {done}/{runs.size}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, run=runs, passeri=out)
    return np.asarray(out, dtype=np.float64)
