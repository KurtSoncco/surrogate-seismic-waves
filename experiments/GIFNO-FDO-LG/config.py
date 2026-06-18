# config.py
"""GIFNO-FDO-LG: FNO global + U-Net local branch fused before XT DeepONet head."""

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
_GIFNO_DIR = EXPERIMENT_DIR.parent / "GIFNO"
_XT_DIR = EXPERIMENT_DIR.parent / "GIFNO-FDO-XT"
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
    os.environ.get("GIFNO_MODEL_DIR", TF_RESULTS_DIR / "models" / "fdo_lg")
)
RESULTS_SAVE_DIR = Path(
    os.environ.get("GIFNO_RESULTS_DIR", TF_RESULTS_DIR / "results" / "fdo_lg")
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

# --- FNO encoder ---
IN_CHANNELS: int = 4
LATENT_CHANNELS: int = 128
FNO_MODES: Tuple[int, int] = (32, 32)
NUM_FNO_LAYERS: int = 5

# --- U-Net local branch ---
UNET_BASE_CHANNELS: int = 64
FUSION_MODE: str = "concat"  # concat | gated

# --- DeepONet readout (2D trunk: log f + x) ---
BRANCH_MODE: str = "surface"
DEEPONET_LATENT_DIM: int = 128
TRUNK_HIDDEN: int = 128
TRUNK_LAYERS: int = 4
X_COORD_MODE: str = "normalized"

# --- Transfer learning from XT ---
PRETRAIN_CHECKPOINT: str = ""
XT_ANCHOR_CHECKPOINT: str = ""  # optional XT ckpt for L2-SP when init from LG phase ckpt
TRAIN_PHASE: int = 1  # 1: unet+fusion | 2: +head | 3: all
L2SP_WEIGHT: float = 1e-4
PHASE1_LR: float = 1e-3
PHASE2_LR: float = 3e-4
PHASE3_FNO_LR: float = 1e-4
PHASE3_OTHER_LR: float = 3e-4

# --- Training (XT winner defaults) ---
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE: float = PHASE1_LR
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 200
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

OPTIMIZER: str = "adam"
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.999
ADAM_EPS: float = 1e-8
AMSGRAD: bool = True

WANDB_PROJECT: str = "gifno_fdo_lg"
WANDB_RUN_NAME: str = "lg_transfer"

EVAL_N_HEATMAPS: int = 3
EVAL_N_CENTRAL_CURVES: int = 4
EVAL_N_WORST_SAMPLES: int = 3

LOSS_REL_WEIGHT: float = 1.0
LOSS_H1_WEIGHT: float = 0.25
LOSS_FREQ_WEIGHT: float = 0.0
LOSS_P: int = 1
HARD_MINING: bool = False
HARD_MINING_POWER: float = 2.0
FREQ_LOSS_LOG_WEIGHT: bool = True
LOG_TF_LOSS: bool = False
LOSS_LINF_WEIGHT: float = 0.0
VALLEY_LOSS_WEIGHT: float = 0.0
VALLEY_PERCENTILE: float = 20.0

# --- TF engineering metric bands (Hz) — shared with GIFNO metrics ---
FREQ_BAND_LOW: Tuple[float, float] = (0.1, 0.5)
FREQ_BAND_MID: Tuple[float, float] = (0.5, 2.0)
FREQ_BAND_HIGH: Tuple[float, float] = (2.0, 10.0)


def _parse_env_value(key: str, raw: str):
    raw = raw.strip()
    if key == "FNO_MODES":
        return tuple(int(x.strip()) for x in raw.split(","))
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
    if key == "FUSION_MODE":
        mode = raw.lower()
        if mode not in ("concat", "gated"):
            raise ValueError(f"FUSION_MODE must be concat or gated, got {raw!r}")
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
        "UNET_BASE_CHANNELS",
        "DEEPONET_LATENT_DIM",
        "TRUNK_HIDDEN",
        "TRUNK_LAYERS",
        "BATCH_SIZE",
        "EARLY_STOP_PATIENCE",
        "NUM_EPOCHS",
        "SEED",
        "NUM_WORKERS",
        "LOSS_P",
        "TRAIN_PHASE",
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
        "ADAM_BETA1",
        "ADAM_BETA2",
        "ADAM_EPS",
        "L2SP_WEIGHT",
        "PHASE1_LR",
        "PHASE2_LR",
        "PHASE3_FNO_LR",
        "PHASE3_OTHER_LR",
    ):
        return float(raw)
    if key in ("WANDB_RUN_NAME", "PRETRAIN_CHECKPOINT", "XT_ANCHOR_CHECKPOINT"):
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
    "UNET_BASE_CHANNELS",
    "FNO_MODES",
    "NUM_FNO_LAYERS",
    "BRANCH_MODE",
    "DEEPONET_LATENT_DIM",
    "TRUNK_HIDDEN",
    "TRUNK_LAYERS",
    "X_COORD_MODE",
    "FUSION_MODE",
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
    "OPTIMIZER",
    "ADAM_BETA1",
    "ADAM_BETA2",
    "ADAM_EPS",
    "AMSGRAD",
    "WANDB_RUN_NAME",
    "PRETRAIN_CHECKPOINT",
    "XT_ANCHOR_CHECKPOINT",
    "TRAIN_PHASE",
    "L2SP_WEIGHT",
    "PHASE1_LR",
    "PHASE2_LR",
    "PHASE3_FNO_LR",
    "PHASE3_OTHER_LR",
)


def apply_env_overrides() -> list[str]:
    applied: list[str] = []
    g = globals()
    for key in _OVERRIDABLE_KEYS:
        if key in ("WANDB_RUN_NAME", "PRETRAIN_CHECKPOINT", "XT_ANCHOR_CHECKPOINT"):
            env_key = key
        else:
            env_key = f"GIFNO_{key}"
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        g[key] = _parse_env_value(key, raw)
        applied.append(f"{key}={g[key]!r}")
    if applied:
        print("[GIFNO-FDO-LG config] env overrides: " + ", ".join(applied), flush=True)
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
    lg = str(EXPERIMENT_DIR)
    gifno = str(_GIFNO_DIR)
    if lg not in sys.path:
        sys.path.insert(0, lg)
    if gifno not in sys.path:
        sys.path.insert(1, gifno)


def phase_learning_rate(phase: int) -> float:
    if phase <= 1:
        return PHASE1_LR
    if phase == 2:
        return PHASE2_LR
    return PHASE3_OTHER_LR
