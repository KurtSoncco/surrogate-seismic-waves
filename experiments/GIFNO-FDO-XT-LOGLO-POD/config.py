# config.py
"""Configuration for GIFNO-FDO-XT-LOGLO-POD: dual-path LOGLO encoder + POD-DeepONet."""

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
    os.environ.get("GIFNO_MODEL_DIR", TF_RESULTS_DIR / "models" / "fdo_xt_loglo_pod")
)
RESULTS_SAVE_DIR = Path(
    os.environ.get("GIFNO_RESULTS_DIR", TF_RESULTS_DIR / "results" / "fdo_xt_loglo_pod")
)

TF_PER_SAMPLE_PATH = TF_RESULTS_DIR / "tf_per_sample.npy"
TF_FREQ_PATH = TF_RESULTS_DIR / "freq.npy"
MANIFEST_PATH = TF_RESULTS_DIR / "manifest.csv"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "best_model.pt"
PREPROCESS_SCRIPT = _GIFNO_DIR / "preprocess" / "compute_transfer_function.py"
POD_PREPROCESS_SCRIPT = _GIFNO_DIR / "preprocess" / "compute_pod_basis.py"
# POD basis lives under the (per-variant) model dir so sweep variants with
# different POD_NUM_MODES / limits never share or overwrite a single cache.
POD_MODES_PATH = MODEL_SAVE_DIR / "pod_modes.npy"
POD_MEAN_PATH = MODEL_SAVE_DIR / "pod_mean.npy"

for d in (TF_RESULTS_DIR, MODEL_SAVE_DIR, RESULTS_SAVE_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# --- Grid / recorders ---
NORMALIZE_VS_SURFACE: bool = True
NORMALIZE_ZETA: bool = True
VS_NORM_EPS: float = 1e-6
ZETA_NORM_EPS: float = 1e-12
NZ_MAX: int = 128
NX_FULL: int = 1500
LX_VARIABILITY: int = 500
BC_WIDTH: int = 500
NX: int = LX_VARIABILITY
X_SLICE_START: int = BC_WIDTH
X_SLICE_END: int = BC_WIDTH + LX_VARIABILITY
DX: float = 1.0
DZ: float = 1.0
N_LATERAL: int = 21
NODES_EACH_SIDE: int = 10
LATERAL_SPACING_M: float = 15.0
N_FREQ: int = 1000

# --- LOGLO encoder ---
IN_CHANNELS: int = 4
LATENT_CHANNELS: int = 128
FNO_MODES: Tuple[int, int] = (32, 32)
NUM_FNO_LAYERS: int = 5
LOGLO_PATCH_SIZE: Tuple[int, int] = (16, 20)
LOGLO_HFP_KERNEL: int = 4
LOGLO_HFP_STRIDE: int = 4
LOGLO_HF_NOISE_ALPHA: float = 0.025

# --- POD-DeepONet readout ---
BRANCH_MODE: str = "surface"
POD_NUM_MODES: int = 32
POD_BRANCH_HIDDEN: int = 256
DEEPONET_LATENT_DIM: int = 128
TRUNK_HIDDEN: int = 128
TRUNK_LAYERS: int = 4
X_COORD_MODE: str = "normalized"

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
# bf16 autocast: FFT/spectral/POD stay fp32 (verified rel-diff ~1e-5 vs fp32).
USE_AMP: bool = True
TORCH_COMPILE: bool = False
# Preload all samples into RAM once (avoids per-epoch H5 reads from slow disk).
CACHE_DATASET: bool = True
# Compute expensive per-recorder tail metrics only every N epochs (logging only;
# model selection uses val_loss every epoch).
VAL_TAIL_EVERY: int = 10

# --- Optimizer ---
OPTIMIZER: str = "adam"
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.999
ADAM_EPS: float = 1e-8
AMSGRAD: bool = True

# --- W&B ---
WANDB_PROJECT: str = "gifno_fdo_xt_loglo_pod"
WANDB_RUN_NAME: str = "loglo_pod_run"

# --- Eval plots ---
EVAL_N_HEATMAPS: int = 3
EVAL_N_CENTRAL_CURVES: int = 4
EVAL_N_WORST_SAMPLES: int = 3
EVAL_STRAT_BINS: int = 4  # quantile bins for CoV/H/rH stratified breakdowns

# --- Loss ---
LOSS_REL_WEIGHT: float = 1.0
LOSS_H1_WEIGHT: float = 0.25
LOSS_FREQ_WEIGHT: float = 0.0
LOSS_RADIAL_WEIGHT: float = 0.25
RADIAL_I_LOW: int = 4
RADIAL_I_HIGH: int = 12
LOSS_P: int = 1
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
    if key == "LOGLO_PATCH_SIZE":
        parts = [int(x.strip()) for x in raw.split(",")]
        return tuple(parts)
    if key == "BRANCH_MODE":
        mode = raw.lower()
        if mode not in ("surface", "depth"):
            raise ValueError(f"BRANCH_MODE must be surface or depth, got {raw!r}")
        return mode
    if key == "X_COORD_MODE":
        mode = raw.lower()
        if mode not in ("normalized", "meters"):
            raise ValueError(f"X_COORD_MODE must be normalized or meters, got {raw!r}")
        return mode
    if key == "OPTIMIZER":
        opt = raw.lower()
        if opt not in ("adam", "adamw"):
            raise ValueError(f"OPTIMIZER must be adam or adamw, got {raw!r}")
        return opt
    if key in (
        "HARD_MINING",
        "NORMALIZE_VS_SURFACE",
        "NORMALIZE_ZETA",
        "FREQ_LOSS_LOG_WEIGHT",
        "LOG_TF_LOSS",
        "USE_AMP",
        "TORCH_COMPILE",
        "AMSGRAD",
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
        "POD_NUM_MODES",
        "POD_BRANCH_HIDDEN",
        "LOGLO_HFP_KERNEL",
        "LOGLO_HFP_STRIDE",
        "RADIAL_I_LOW",
        "RADIAL_I_HIGH",
    ):
        return int(raw)
    if key in (
        "LEARNING_RATE",
        "WEIGHT_DECAY",
        "LOSS_REL_WEIGHT",
        "LOSS_H1_WEIGHT",
        "LOSS_FREQ_WEIGHT",
        "LOSS_RADIAL_WEIGHT",
        "LOSS_LINF_WEIGHT",
        "VALLEY_LOSS_WEIGHT",
        "VALLEY_PERCENTILE",
        "HARD_MINING_POWER",
        "GRAD_CLIP_NORM",
        "ADAM_BETA1",
        "ADAM_BETA2",
        "ADAM_EPS",
        "LOGLO_HF_NOISE_ALPHA",
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
    "LOGLO_PATCH_SIZE",
    "LOGLO_HFP_KERNEL",
    "LOGLO_HFP_STRIDE",
    "LOGLO_HF_NOISE_ALPHA",
    "POD_NUM_MODES",
    "POD_BRANCH_HIDDEN",
    "BRANCH_MODE",
    "DEEPONET_LATENT_DIM",
    "TRUNK_HIDDEN",
    "TRUNK_LAYERS",
    "X_COORD_MODE",
    "LOSS_REL_WEIGHT",
    "LOSS_H1_WEIGHT",
    "LOSS_FREQ_WEIGHT",
    "LOSS_RADIAL_WEIGHT",
    "RADIAL_I_LOW",
    "RADIAL_I_HIGH",
    "LOSS_LINF_WEIGHT",
    "LOSS_P",
    "HARD_MINING",
    "HARD_MINING_POWER",
    "LOG_TF_LOSS",
    "VALLEY_LOSS_WEIGHT",
    "VALLEY_PERCENTILE",
    "USE_AMP",
    "TORCH_COMPILE",
    "OPTIMIZER",
    "ADAM_BETA1",
    "ADAM_BETA2",
    "ADAM_EPS",
    "AMSGRAD",
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
        print(
            "[GIFNO-FDO-XT-LOGLO-POD config] env overrides: " + ", ".join(applied),
            flush=True,
        )
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


def domain_half_width_m(nx: int = NX, dx: float = DX) -> float:
    return float(nx // 2) * dx


def recorder_x_trunk_coords(
    recorder_x: np.ndarray | None = None,
    *,
    nx: int = NX,
    dx: float = DX,
    mode: str = X_COORD_MODE,
) -> np.ndarray:
    if recorder_x is None:
        recorder_x = recorder_x_indices(nx=nx, dx=dx)
    center = nx // 2
    x_m = (recorder_x.astype(np.float64) - center) * dx
    if mode == "meters":
        return x_m.astype(np.float32)
    half_w = domain_half_width_m(nx=nx, dx=dx)
    if half_w < 1e-8:
        return np.zeros_like(x_m, dtype=np.float32)
    return (x_m / half_w).astype(np.float32)


def setup_import_paths() -> None:
    xt = str(EXPERIMENT_DIR)
    gifno = str(_GIFNO_DIR)
    if xt not in sys.path:
        sys.path.insert(0, xt)
    if gifno not in sys.path:
        sys.path.insert(1, gifno)
