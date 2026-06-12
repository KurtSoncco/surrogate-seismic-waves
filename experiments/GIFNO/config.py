# config.py
"""Centralized configuration for the GIFNO grid-direct FNO experiment."""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# --- Paths ---
EXPERIMENT_DIR = Path(__file__).resolve().parent

# Data root: Box locally, Savio scratch on HPC (set GIFNO_DATA_ROOT in job script).
DATA_ROOT = Path(
    os.environ.get(
        "GIFNO_DATA_ROOT",
        "/mnt/box_lab/Projects/Neural Operator/data",
    )
)
H5_DIR = Path(os.environ.get("GIFNO_H5_DIR", DATA_ROOT / "h5"))
TF_RESULTS_DIR = Path(os.environ.get("GIFNO_TF_DIR", DATA_ROOT / "transfer_function"))
MODEL_SAVE_DIR = Path(os.environ.get("GIFNO_MODEL_DIR", TF_RESULTS_DIR / "models"))
RESULTS_SAVE_DIR = Path(os.environ.get("GIFNO_RESULTS_DIR", TF_RESULTS_DIR / "results"))

OPENSEES_ROOT = Path(os.environ.get("OPENSEES_ROOT", "/home/kurt-asus/opensees"))

TF_PER_SAMPLE_PATH = TF_RESULTS_DIR / "tf_per_sample.npy"
TF_FREQ_PATH = TF_RESULTS_DIR / "freq.npy"
MANIFEST_PATH = TF_RESULTS_DIR / "manifest.csv"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "best_model.pt"

for d in (TF_RESULTS_DIR, MODEL_SAVE_DIR, RESULTS_SAVE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- H5 / grid ---
NZ_MAX: int = 128
NX_FULL: int = 1500  # full OpenSees domain (500 m BC + 500 m variability + 500 m BC)
LX_VARIABILITY: int = 500  # center soil-variability strip width [m]
BC_WIDTH: int = 500  # each side boundary padding in full domain
NX: int = LX_VARIABILITY  # Phase 1: crop to variability region only
X_SLICE_START: int = BC_WIDTH  # column index on full 1500-wide grid
X_SLICE_END: int = BC_WIDTH + LX_VARIABILITY  # exclusive
DX: float = 1.0
DZ: float = 1.0

# --- Recorders (match neural-operator run_experiment.py) ---
N_LATERAL: int = 21
NODES_EACH_SIDE: int = 10
LATERAL_SPACING_M: float = 15.0

# --- Transfer function preprocessing (match compute_transfer_function_h15.py) ---
N_FREQ: int = 1000
FREQ_START_HZ: float = 0.1
FREQ_END_HZ: float = 10.0
SMOOTH_COEFF: float = 500

# --- Model ---
IN_CHANNELS: int = 4
LATENT_CHANNELS: int = 64
FNO_MODES: Tuple[int, int] = (32, 32)
NUM_FNO_LAYERS: int = 5

# --- Training ---
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 500
BATCH_SIZE: int = 4
TRAIN_SPLIT: float = 0.7
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15
SEED: int = 42
EARLY_STOP_PATIENCE: int = 80
GRAD_CLIP_NORM: float = 1.0
NUM_WORKERS: int = 2

# --- W&B ---
WANDB_PROJECT: str = "gifno_fno"
WANDB_RUN_NAME: str = "gifno_grid_fno_run"

# --- Evaluation plots ---
EVAL_N_HEATMAPS: int = 3
EVAL_N_CENTRAL_CURVES: int = 4
EVAL_N_WORST_SAMPLES: int = 3

# --- Loss (composite) ---
LOSS_REL_WEIGHT: float = 1.0
LOSS_H1_WEIGHT: float = 0.0
LOSS_FREQ_WEIGHT: float = 0.00
LOSS_P: int = 2
HARD_MINING: bool = False
HARD_MINING_POWER: float = 1.0
FREQ_LOSS_LOG_WEIGHT: bool = True


def recorder_x_indices(
    nx: int = NX,
    nodes_each_side: int = NODES_EACH_SIDE,
    spacing_m: float = LATERAL_SPACING_M,
    dx: float = DX,
) -> np.ndarray:
    """Integer x-indices on the cropped NX grid (center ± nodes_each_side @ spacing)."""
    center = nx // 2
    step = max(1, int(round(spacing_m / dx)))
    lo = center - nodes_each_side * step
    hi = center + nodes_each_side * step
    return np.arange(lo, hi + 1, step, dtype=np.int64)
