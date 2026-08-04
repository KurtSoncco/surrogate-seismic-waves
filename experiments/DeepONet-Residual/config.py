"""Config for signed-residual DeepONet (field + stochastic branch)."""

from __future__ import annotations

import os
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESIDUAL_DIR = EXPERIMENT_DIR.parent / "Residual"

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

# Reuse Residual magnitude caches / sample indices when present
RESIDUAL_CACHE_DIR = RESIDUAL_DIR / "cache"
CACHE_DIR = EXPERIMENT_DIR / "cache"
RESULTS_DIR = EXPERIMENT_DIR / "results"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"

# Grid (match GIFNO / Residual)
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

N_LATERAL: int = 21
N_FREQ: int = 1000
FREQ_START_HZ: float = 0.1
FREQ_END_HZ: float = 10.0
N_FREQ_TRAIN: int = 50  # log-spaced subset for trunk queries during training
K_XI: int = 8
SEED: int = 42
DEFAULT_XI_TREND: float = 0.05  # xi_damp scalar + nominal Haskell damping

# Model
LATENT_DIM: int = 128
FIELD_CHANNELS: int = 3  # Vs_norm, zeta_norm, Z_norm
FIELD_HIDDEN: int = 48
BRANCH_HIDDEN: int = 256
TRUNK_HIDDEN: int = 256
TRUNK_LAYERS: int = 5

# Train defaults
BATCH_SIZE: int = 8
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-5
ADAMW_BETAS: tuple[float, float] = (0.9, 0.999)
SMOOTH_L1_BETA: float = 1.0  # Huber transition (PyTorch SmoothL1Loss beta)
EPOCHS: int = 300
PATIENCE: int = 50
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
NUM_WORKERS: int = 0

for d in (CACHE_DIR, RESULTS_DIR, CHECKPOINT_DIR):
    d.mkdir(parents=True, exist_ok=True)
