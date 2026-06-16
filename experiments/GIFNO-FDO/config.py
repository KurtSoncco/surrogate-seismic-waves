# config.py
"""Configuration for GIFNO-FDO: FNO encoder + recorder DeepONet head."""

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
_GIFNO_DIR = EXPERIMENT_DIR.parent / "GIFNO"
_BOX_DATA_ROOT = Path("/mnt/box_lab/Projects/Neural Operator/data")
_DUMMY_DATA_ROOT = _GIFNO_DIR / "dummy_data"


def _resolve_data_root() -> Path:
    if env_root := os.environ.get("GIFNO_DATA_ROOT"):
        return Path(env_root)
    if _BOX_DATA_ROOT.exists() and os.access(_BOX_DATA_ROOT, os.W_OK | os.X_OK):
        return _BOX_DATA_ROOT
    return _DUMMY_DATA_ROOT


DATA_ROOT = _resolve_data_root()
H5_DIR = Path(os.environ.get("GIFNO_H5_DIR", DATA_ROOT / "h5"))
TF_RESULTS_DIR = Path(os.environ.get("GIFNO_TF_DIR", DATA_ROOT / "transfer_function"))
MODEL_SAVE_DIR = Path(
    os.environ.get("GIFNO_MODEL_DIR", TF_RESULTS_DIR / "models" / "fdo")
)
RESULTS_SAVE_DIR = Path(
    os.environ.get("GIFNO_RESULTS_DIR", TF_RESULTS_DIR / "results" / "fdo")
)

TF_PER_SAMPLE_PATH = TF_RESULTS_DIR / "tf_per_sample.npy"
TF_FREQ_PATH = TF_RESULTS_DIR / "freq.npy"
MANIFEST_PATH = TF_RESULTS_DIR / "manifest.csv"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "best_model.pt"
PREPROCESS_SCRIPT = _GIFNO_DIR / "preprocess" / "compute_transfer_function.py"

for d in (TF_RESULTS_DIR, MODEL_SAVE_DIR, RESULTS_SAVE_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# --- Grid / recorders (same as GIFNO) ---
NORMALIZE_VS_SURFACE: bool = True
NORMALIZE_ZETA: bool = True
VS_NORM_EPS: float = 1e-6
ZETA_NORM_EPS: float = 1e-12
NZ_MAX: int = 128
NX: int = 500
DX: float = 1.0
DZ: float = 1.0
N_LATERAL: int = 21
NODES_EACH_SIDE: int = 10
LATERAL_SPACING_M: float = 15.0
N_FREQ: int = 1000

# --- FNO encoder ---
IN_CHANNELS: int = 4
LATENT_CHANNELS: int = 96
FNO_MODES: Tuple[int, int] = (32, 32)
NUM_FNO_LAYERS: int = 5

# --- DeepONet readout ---
BRANCH_MODE: str = "surface"  # surface | depth
DEEPONET_LATENT_DIM: int = 64
TRUNK_HIDDEN: int = 128
TRUNK_LAYERS: int = 4

# --- Training ---
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 1500
BATCH_SIZE: int = 16
TRAIN_SPLIT: float = 0.7
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15
SEED: int = 42
EARLY_STOP_PATIENCE: int = 80
GRAD_CLIP_NORM: float = 1.0
NUM_WORKERS: int = 4
USE_AMP: bool = False
TORCH_COMPILE: bool = False

# --- W&B (separate project from GIFNO grid-FNO) ---
WANDB_PROJECT: str = "gifno_fno_deeponet"
WANDB_RUN_NAME: str = "fdo_run"

# --- Eval plots ---
EVAL_N_HEATMAPS: int = 3
EVAL_N_CENTRAL_CURVES: int = 4
EVAL_N_WORST_SAMPLES: int = 3

# --- Loss (round-3 winner: lw_nm_h1) ---
LOSS_REL_WEIGHT: float = 1.0
LOSS_H1_WEIGHT: float = 0.25
LOSS_FREQ_WEIGHT: float = 0.0
LOSS_P: int = 2
HARD_MINING: bool = False
HARD_MINING_POWER: float = 2.0
FREQ_LOSS_LOG_WEIGHT: bool = True
LOG_TF_LOSS: bool = False
LOSS_LINF_WEIGHT: float = 0.0
VALLEY_LOSS_WEIGHT: float = 0.0
VALLEY_PERCENTILE: float = 20.0


def _parse_env_value(key: str, raw: str):
    raw = raw.strip()
    if key == "FNO_MODES":
        parts = [int(x.strip()) for x in raw.split(",")]
        return tuple(parts)
    if key == "BRANCH_MODE":
        mode = raw.lower()
        if mode not in ("surface", "depth"):
            raise ValueError(f"BRANCH_MODE must be surface or depth, got {raw!r}")
        return mode
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
        "DEEPONET_LATENT_DIM",
        "TRUNK_HIDDEN",
        "TRUNK_LAYERS",
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
    "BRANCH_MODE",
    "DEEPONET_LATENT_DIM",
    "TRUNK_HIDDEN",
    "TRUNK_LAYERS",
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
    applied: list[str] = []
    g = globals()
    for key in _OVERRIDABLE_KEYS:
        env_key = "WANDB_RUN_NAME" if key == "WANDB_RUN_NAME" else f"GIFNO_{key}"
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        g[key] = _parse_env_value(key, raw)
        applied.append(f"{key}={g[key]!r}")
    if applied:
        print("[GIFNO-FDO config] env overrides: " + ", ".join(applied), flush=True)
    return applied


apply_env_overrides()


def recorder_x_indices(
    nx: int = NX,
    nodes_each_side: int = NODES_EACH_SIDE,
    spacing_m: float = LATERAL_SPACING_M,
    dx: float = DX,
) -> np.ndarray:
    center = nx // 2
    step = max(1, int(round(spacing_m / dx)))
    lo = center - nodes_each_side * step
    hi = center + nodes_each_side * step
    return np.arange(lo, hi + 1, step, dtype=np.int64)


def setup_import_paths() -> None:
    """Ensure FDO config is used when importing shared GIFNO modules."""
    fdo = str(EXPERIMENT_DIR)
    gifno = str(_GIFNO_DIR)
    if fdo not in sys.path:
        sys.path.insert(0, fdo)
    if gifno not in sys.path:
        sys.path.insert(1, gifno)
