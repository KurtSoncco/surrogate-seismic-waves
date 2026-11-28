"""One Pretell, Toro, and damping-multiplier curve. Used by a process pool."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEEP = REPO / "experiments" / "DeepONet-Residual"
CACHE = REPO / "data" / "vardomain" / "cache"
if str(DEEP) not in sys.path:
    sys.path.insert(0, str(DEEP))

_FREQ: np.ndarray | None = None


def init_worker() -> None:
    global _FREQ
    _FREQ = np.load(CACHE / "freq.npy")
    from response_variability.seiskit_arms import hallal_dmin_geomean_tf  # noqa: F401


def compute_arms(task: dict) -> dict:
    from response_variability.seiskit_arms import (
        hallal_dmin_geomean_tf,
        hallal_geomean_tf,
        pretell_haskell_tf,
    )

    freq = _FREQ if _FREQ is not None else np.load(CACHE / "freq.npy")
    run = int(task["run"])
    family = str(task["family"])
    soil = int(task["soil_nz"])
    vs = np.asarray(np.load(CACHE / family / "vs.npy", mmap_mode="r")[run, :soil], dtype=np.float64)
    zeta = np.asarray(np.load(CACHE / family / "zeta.npy", mmap_mode="r")[run, :soil], dtype=np.float64)
    vs2 = float(task["vs2"])
    pretell, _sig = pretell_haskell_tf(
        freq=freq, vs_strip=vs, zeta_strip=zeta, vs2=vs2, soil_nz=soil, dz=1.0, n_samples=200,
    )
    toro, _sig = hallal_geomean_tf(
        freq=freq,
        vs1=float(task["vs1"]),
        H=float(task["H"]),
        cov=float(task["cov"]),
        vs2=vs2,
        xi=0.05,
        n_seeds=40,
        kind="toro",
    )
    dmult, _sig = hallal_dmin_geomean_tf(
        freq=freq, vs1=float(task["vs1"]), H=float(task["H"]), cov=float(task["cov"]), vs2=vs2,
    )
    return {
        "run": run,
        "pretell": np.asarray(pretell, dtype=np.float32),
        "toro": np.asarray(toro, dtype=np.float32),
        "dmult": np.asarray(dmult, dtype=np.float32),
    }
