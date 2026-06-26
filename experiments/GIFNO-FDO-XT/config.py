# config.py
"""Configuration for GIFNO-FDO-XT: FNO encoder + position-aware 2D DeepONet trunk."""

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
    os.environ.get("GIFNO_MODEL_DIR", TF_RESULTS_DIR / "models" / "fdo_xt")
)
RESULTS_SAVE_DIR = Path(
    os.environ.get("GIFNO_RESULTS_DIR", TF_RESULTS_DIR / "results" / "fdo_xt")
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
LATENT_CHANNELS: int = 96
FNO_MODES: Tuple[int, int] = (32, 32)
NUM_FNO_LAYERS: int = 5

# --- DeepONet readout (2D trunk: log f + x) ---
BRANCH_MODE: str = "surface"  # surface | depth
DEEPONET_LATENT_DIM: int = 128
TRUNK_HIDDEN: int = 128
TRUNK_LAYERS: int = 4
X_COORD_MODE: str = "normalized"  # normalized: x / domain half-width | meters

# --- Training (wide_h1_p1_amsgrad winner defaults) ---
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

# --- Optimizer ---
OPTIMIZER: str = "adam"
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.999
ADAM_EPS: float = 1e-8
AMSGRAD: bool = True

# --- W&B ---
WANDB_PROJECT: str = "gifno_fdo_xt"
WANDB_RUN_NAME: str = "xt_run"

# --- Eval plots ---
EVAL_N_HEATMAPS: int = 3
EVAL_N_CENTRAL_CURVES: int = 4
EVAL_N_WORST_SAMPLES: int = 3

# --- Loss ---
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

# --- Frequency bands (Hz) ---
# Shared by band metrics (metrics.per_sample_bandwise_rel_l2_numpy) and the
# frequency-band loss curriculum below.
FREQ_BAND_LOW: Tuple[float, float] = (0.1, 0.5)
FREQ_BAND_MID: Tuple[float, float] = (0.5, 2.0)
FREQ_BAND_HIGH: Tuple[float, float] = (2.0, 10.0)

# --- Frequency-band loss curriculum ---
# Speed-safe schedule that reweights the relative loss across frequency bands as
# training progresses: emphasize low log-f early, ramp mid, then high. Weights
# are normalized to mean 1 over the freq grid so the loss scale stays stable.
# Default off -> identical to the baseline. Schedule is retuned to start the mid
# and high ramps earlier so they reach target weight before early-stop fires.
BAND_CURRICULUM: bool = False
BAND_CURRICULUM_FLOOR: float = 0.25
# Curriculum mode:
#   "time"        -> Tier 1: per-frequency weight schedule on the main rel loss
#                    (BAND_CURRICULUM_MID_START/HIGH_START/RAMP, epoch fractions).
#   "convergence" -> Tier 2: closed-loop controller that activates bands as the
#                    band-balanced val metric plateaus, driving the band-balanced
#                    loss term and warm-restarting the optimizer/LR per phase.
BAND_CURRICULUM_MODE: str = "time"
BAND_CURRICULUM_MID_START: float = 0.20
BAND_CURRICULUM_HIGH_START: float = 0.50
BAND_CURRICULUM_RAMP: float = 0.20
# Convergence-mode controls.
BAND_CURRICULUM_PHASE_PATIENCE: int = 30
BAND_CURRICULUM_MIN_DELTA: float = 1e-4
BAND_CURRICULUM_RAMP_EPOCHS: int = 10
BAND_CURRICULUM_LR_RESTART: bool = True
BAND_CURRICULUM_LR_RESTART_SCALE: float = 1.0
# Rebuild the optimizer on phase advance so Adam/AMSGRAD moment state resets;
# raising LR alone is defeated by AMSGRAD's max-v denominator.
BAND_CURRICULUM_RESET_OPT_STATE: bool = True

# --- Band-balanced loss term (Tier 2) ---
# Adds band_balanced_weight * mean_b(w_b * relL2_band_b / ||t_band_b||) to the
# loss; each band is normalized by its OWN energy so the low-energy high band is
# not suppressed. 0 -> off. The convergence curriculum auto-enables it.
LOSS_BAND_BALANCED_WEIGHT: float = 0.0

# --- Model selection / LR schedule ---
# SELECTION_METRIC drives best-checkpoint and early-stop:
#   "val_loss"      -> neutral relative-L2 (baseline behavior)
#   "band_balanced" -> mean of per-band rel-L2 (low/mid/high) so high-band gains
#                      are rewarded even though the high band carries low energy.
# The LR scheduler always tracks neutral val_loss; its patience/factor are tunable
# so curriculum runs can train long enough for mid/high bands to see gradient.
SELECTION_METRIC: str = "val_loss"
LR_SCHED_PATIENCE: int = 20
LR_SCHED_FACTOR: float = 0.5


def _parse_env_value(key: str, raw: str):
    raw = raw.strip()
    if key == "FNO_MODES":
        parts = [int(x.strip()) for x in raw.split(",")]
        return tuple(parts)
    if key in ("FREQ_BAND_LOW", "FREQ_BAND_MID", "FREQ_BAND_HIGH"):
        parts = [float(x.strip()) for x in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(f"{key} must be 'lo,hi', got {raw!r}")
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
    if key == "SELECTION_METRIC":
        metric = raw.lower()
        if metric not in ("val_loss", "band_balanced"):
            raise ValueError(
                f"SELECTION_METRIC must be val_loss or band_balanced, got {raw!r}"
            )
        return metric
    if key == "BAND_CURRICULUM_MODE":
        mode = raw.lower()
        if mode not in ("time", "convergence"):
            raise ValueError(
                f"BAND_CURRICULUM_MODE must be time or convergence, got {raw!r}"
            )
        return mode
    if key in (
        "HARD_MINING",
        "NORMALIZE_VS_SURFACE",
        "NORMALIZE_ZETA",
        "FREQ_LOSS_LOG_WEIGHT",
        "LOG_TF_LOSS",
        "USE_AMP",
        "TORCH_COMPILE",
        "AMSGRAD",
        "BAND_CURRICULUM",
        "BAND_CURRICULUM_LR_RESTART",
        "BAND_CURRICULUM_RESET_OPT_STATE",
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
        "LR_SCHED_PATIENCE",
        "BAND_CURRICULUM_PHASE_PATIENCE",
        "BAND_CURRICULUM_RAMP_EPOCHS",
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
        "BAND_CURRICULUM_FLOOR",
        "BAND_CURRICULUM_MID_START",
        "BAND_CURRICULUM_HIGH_START",
        "BAND_CURRICULUM_RAMP",
        "BAND_CURRICULUM_MIN_DELTA",
        "BAND_CURRICULUM_LR_RESTART_SCALE",
        "LOSS_BAND_BALANCED_WEIGHT",
        "LR_SCHED_FACTOR",
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
    "X_COORD_MODE",
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
    "FREQ_BAND_LOW",
    "FREQ_BAND_MID",
    "FREQ_BAND_HIGH",
    "BAND_CURRICULUM",
    "BAND_CURRICULUM_FLOOR",
    "BAND_CURRICULUM_MODE",
    "BAND_CURRICULUM_MID_START",
    "BAND_CURRICULUM_HIGH_START",
    "BAND_CURRICULUM_RAMP",
    "BAND_CURRICULUM_PHASE_PATIENCE",
    "BAND_CURRICULUM_MIN_DELTA",
    "BAND_CURRICULUM_RAMP_EPOCHS",
    "BAND_CURRICULUM_LR_RESTART",
    "BAND_CURRICULUM_LR_RESTART_SCALE",
    "BAND_CURRICULUM_RESET_OPT_STATE",
    "LOSS_BAND_BALANCED_WEIGHT",
    "SELECTION_METRIC",
    "LR_SCHED_PATIENCE",
    "LR_SCHED_FACTOR",
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
        print("[GIFNO-FDO-XT config] env overrides: " + ", ".join(applied), flush=True)
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
    """Half-width of the lateral model strip [m]; domain edges normalize to ±1."""
    return float(nx // 2) * dx


def recorder_x_trunk_coords(
    recorder_x: np.ndarray | None = None,
    *,
    nx: int = NX,
    dx: float = DX,
    mode: str = X_COORD_MODE,
) -> np.ndarray:
    """
    Lateral coordinate for trunk input.

    meters: offset from domain center in metres.
    normalized: x_m / domain_half_width so model edges are ±1 (recorders lie inside).
    """
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
    """Ensure XT config is used when importing shared GIFNO modules."""
    xt = str(EXPERIMENT_DIR)
    gifno = str(_GIFNO_DIR)
    if xt not in sys.path:
        sys.path.insert(0, xt)
    if gifno not in sys.path:
        sys.path.insert(1, gifno)
