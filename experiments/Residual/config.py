"""Configuration for OrbitAll-style residual feature screening on GIFNO data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

# --- Paths ---
EXPERIMENT_DIR = Path(__file__).resolve().parent
_BOX_DATA_ROOT = Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data")
_LEGACY_BOX_ROOT = Path("/mnt/box_lab/Projects/Neural Operator/data")


def _resolve_data_root() -> Path:
    if env_root := os.environ.get("GIFNO_DATA_ROOT"):
        return Path(env_root)
    if _BOX_DATA_ROOT.exists():
        return _BOX_DATA_ROOT
    if _LEGACY_BOX_ROOT.exists():
        return _LEGACY_BOX_ROOT
    return EXPERIMENT_DIR / "dummy_data"


DATA_ROOT = _resolve_data_root()
H5_DIR = Path(os.environ.get("GIFNO_H5_DIR", DATA_ROOT / "h5"))
TF_RESULTS_DIR = Path(os.environ.get("GIFNO_TF_DIR", DATA_ROOT / "transfer_function"))

TF_PER_SAMPLE_PATH = TF_RESULTS_DIR / "tf_per_sample.npy"
TF_FREQ_PATH = TF_RESULTS_DIR / "freq.npy"
MANIFEST_PATH = TF_RESULTS_DIR / "manifest.csv"
RECORDER_X_IDX_PATH = TF_RESULTS_DIR / "recorder_x_idx.npy"

CACHE_DIR = EXPERIMENT_DIR / "cache"
RESULTS_DIR = EXPERIMENT_DIR / "results"

# --- Grid (match GIFNO / neural-operator generation) ---
NZ_MAX: int = 128
NX_FULL: int = 1500
LX_VARIABILITY: int = 500
BC_WIDTH: int = 500
NX: int = LX_VARIABILITY
X_SLICE_START: int = BC_WIDTH
X_SLICE_END: int = BC_WIDTH + LX_VARIABILITY
DX: float = 1.0
DZ: float = 1.0
RHO: float = 2000.0

# --- Recorders / frequency ---
N_LATERAL: int = 21
N_FREQ: int = 1000
FREQ_START_HZ: float = 0.1
FREQ_END_HZ: float = 10.0
FREQ_BAND_LOW: Tuple[float, float] = (0.1, 0.5)
FREQ_BAND_MID: Tuple[float, float] = (0.5, 2.0)
FREQ_BAND_HIGH: Tuple[float, float] = (2.0, 10.0)

# --- Screening defaults (subsample-first hard gate) ---
N_SAMPLES: int = 100
N_FREQ_SCREEN: int = 50  # log-spaced subset of N_FREQ for MI/RF table
K_XI: int = 8  # top-K spectral modes → 2K real/imag features
SEED: int = 42
DEFAULT_XI_TREND: float = 0.05  # damping for nominal single-layer Haskell
IMP_GRAD_EDGE_PERCENTILE: float = 90.0  # |imp_grad| threshold for dist_edge
N_RF_ESTIMATORS: int = 200
N_PERMUTATION_REPEATS: int = 10
CENTRAL_RECORDER_IDX: int = N_LATERAL // 2  # index into 21 recorders
N_PLOT_COMBOS: int = 9  # 3x3 diagnostic panel

for d in (CACHE_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
