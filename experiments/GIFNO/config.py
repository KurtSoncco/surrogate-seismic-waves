# config.py
"""Centralized configuration for the GIFNO grid-direct FNO experiment."""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# --- Paths ---
EXPERIMENT_DIR = Path(__file__).resolve().parent
_BOX_DATA_ROOT = Path("/mnt/box_lab/Projects/Neural Operator/data")
_DUMMY_DATA_ROOT = EXPERIMENT_DIR / "dummy_data"


def _resolve_data_root() -> Path:
    """Prefer explicit env, then writable Box mount, else local dummy data."""
    if env_root := os.environ.get("GIFNO_DATA_ROOT"):
        return Path(env_root)
    if _BOX_DATA_ROOT.exists() and os.access(_BOX_DATA_ROOT, os.W_OK | os.X_OK):
        return _BOX_DATA_ROOT
    return _DUMMY_DATA_ROOT


# Data root: Box locally, Savio scratch on HPC (set GIFNO_DATA_ROOT in job script).
DATA_ROOT = _resolve_data_root()
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
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        # CI and other environments may not have access to remote data mounts.
        pass

# --- Input normalization (per sample; TF targets stay in physical units) ---
NORMALIZE_VS_SURFACE: bool = True  # Vs / Vs(surface, x) per column
NORMALIZE_ZETA: bool = True  # zeta / max(zeta) over active depth rows
VS_NORM_EPS: float = 1e-6
ZETA_NORM_EPS: float = 1e-12

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
LATENT_CHANNELS: int = 96
FNO_MODES: Tuple[int, int] = (32, 32)
NUM_FNO_LAYERS: int = 5

# --- Training ---
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 1500
BATCH_SIZE: int = 16  # A100-friendly; override with GIFNO_BATCH_SIZE if OOM
TRAIN_SPLIT: float = 0.7
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15
SEED: int = 42
EARLY_STOP_PATIENCE: int = 80
GRAD_CLIP_NORM: float = 1.0
NUM_WORKERS: int = (
    4  # match Slurm --cpus-per-task=4; more workers triggers dataloader warnings
)
USE_AMP: bool = False  # FNO spectral FFT fails in fp16/bf16 on (128, 500) grid
TORCH_COMPILE: bool = (
    False  # off by default (FNO spectral layers may not compile cleanly)
)

# --- W&B ---
WANDB_PROJECT: str = "gifno_fno"
WANDB_RUN_NAME: str = "gifno_grid_fno_run"

# --- Evaluation plots ---
EVAL_N_HEATMAPS: int = 3
EVAL_N_CENTRAL_CURVES: int = 4
EVAL_N_WORST_SAMPLES: int = 3

# --- Loss (composite) ---
LOSS_REL_WEIGHT: float = 1.0
LOSS_H1_WEIGHT: float = 0.25
LOSS_FREQ_WEIGHT: float = 0.00
LOSS_P: int = 2
HARD_MINING: bool = False
HARD_MINING_POWER: float = 2.0
FREQ_LOSS_LOG_WEIGHT: bool = True
LOG_TF_LOSS: bool = False  # train relative L2 on log(|TF|)
LOSS_LINF_WEIGHT: float = 0.0  # relative max per-frequency error
VALLEY_LOSS_WEIGHT: float = 0.0  # extra weight on TF valley bins in rel loss
VALLEY_PERCENTILE: float = 20.0  # bottom percentile of log(|TF|) per recorder curve


def _parse_gifno_env_value(key: str, raw: str):
    """Parse GIFNO_<KEY> env string into the correct Python type."""
    raw = raw.strip()
    if key == "FNO_MODES":
        parts = [int(x.strip()) for x in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(f"FNO_MODES expects two integers, got {raw!r}")
        return tuple(parts)
    if key in (
        "HARD_MINING",
        "NORMALIZE_VS_SURFACE",
        "NORMALIZE_ZETA",
        "FREQ_LOSS_LOG_WEIGHT",
        "LOG_TF_LOSS",
        "USE_AMP",
        "TORCH_COMPILE",
    ):
        return raw.lower() in ("1", "true", "yes", "on")
    if key in (
        "NUM_FNO_LAYERS",
        "LATENT_CHANNELS",
        "BATCH_SIZE",
        "EARLY_STOP_PATIENCE",
        "NUM_EPOCHS",
        "SEED",
        "NUM_WORKERS",
        "LOSS_P",
    ):
        return int(raw)
    if key in (
        "LEARNING_RATE",
        "WEIGHT_DECAY",
        "LOSS_REL_WEIGHT",
        "LOSS_H1_WEIGHT",
        "LOSS_FREQ_WEIGHT",
        "LOSS_LINF_WEIGHT",
        "VALLEY_LOSS_WEIGHT",
        "VALLEY_PERCENTILE",
        "HARD_MINING_POWER",
        "GRAD_CLIP_NORM",
    ):
        return float(raw)
    if key == "WANDB_RUN_NAME":
        return raw
    raise KeyError(f"Unknown config override key: {key}")


_OVERRIDABLE_KEYS = (
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "NUM_EPOCHS",
    "BATCH_SIZE",
    "SEED",
    "EARLY_STOP_PATIENCE",
    "GRAD_CLIP_NORM",
    "NUM_WORKERS",
    "LATENT_CHANNELS",
    "FNO_MODES",
    "NUM_FNO_LAYERS",
    "LOSS_REL_WEIGHT",
    "LOSS_H1_WEIGHT",
    "LOSS_FREQ_WEIGHT",
    "LOSS_LINF_WEIGHT",
    "LOSS_P",
    "HARD_MINING",
    "HARD_MINING_POWER",
    "LOG_TF_LOSS",
    "VALLEY_LOSS_WEIGHT",
    "VALLEY_PERCENTILE",
    "USE_AMP",
    "TORCH_COMPILE",
    "WANDB_RUN_NAME",
)


def apply_env_overrides() -> list[str]:
    """Apply GIFNO_<KEY> and WANDB_RUN_NAME env overrides to module-level config."""
    applied: list[str] = []
    g = globals()
    for key in _OVERRIDABLE_KEYS:
        env_key = "WANDB_RUN_NAME" if key == "WANDB_RUN_NAME" else f"GIFNO_{key}"
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        g[key] = _parse_gifno_env_value(key, raw)
        applied.append(f"{key}={g[key]!r}")
    if applied:
        print("[GIFNO config] env overrides: " + ", ".join(applied), flush=True)
    return applied


apply_env_overrides()


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
