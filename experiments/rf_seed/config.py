# config.py
"""Centralized configuration for the rf_seed 2D Vs to transfer function FNO model."""

from pathlib import Path
from typing import List, Tuple

import torch

# --- Paths ---
EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
MODEL_SAVE_DIR = EXPERIMENT_DIR / "models"
RESULTS_SAVE_DIR = EXPERIMENT_DIR / "results"

DATA_PATH = DATA_DIR / "transfer_functions_dict.pkl"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "best_model.pt"

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Dict keys for loading data (try in order) ---
VS_KEYS: Tuple[str, ...] = ("vs_array", "vs_2d", "vs", "Vs_profile", "Vs")
TF_KEYS: Tuple[str, ...] = ("tf_magnitude", "tf", "transfer_function", "TTF", "ttf")

# --- Model: 2D Vs input shape and output (matches data: 67 x 1500) ---
INPUT_SHAPE: Tuple[int, int] = (67, 1500)  # (H, W) for 2D Vs profile
OUTPUT_SIZE: int = 1000  # n_freq for transfer function
LATENT_DIM: int = 500

# --- 2D CNN Encoder ---
ENCODER_CHANNELS: List[int] = [1, 32, 64, 128, 256]
ENCODER_KERNEL_SIZE: int = 3
ENCODER_POOL_SIZE: int = 2

# --- FNO OperatorDecoder ---
FNO_MODES: int = 16
FNO_WIDTH: int = 150
NUM_FNO_LAYERS: int = 5

# --- Training ---
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE: float = 5e-3
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 5000
BATCH_SIZE: int = 64
TRAIN_SPLIT: float = 0.7
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15
SEED: int = 42

# --- Early Stopping ---
EARLY_STOP_PATIENCE: int = 500

# --- Gradient Clipping ---
GRAD_CLIP_NORM: float = 1.0

# --- W&B Logging ---
WANDB_PROJECT: str = "rf_seed_fno"
WANDB_RUN_NAME: str = "rf_seed_2d_fno_run"
