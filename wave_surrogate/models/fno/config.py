# config.py
"""Centralized configuration for the FNO model."""

from pathlib import Path
from typing import List

import torch

# --- Data Configuration ---
# Paths to the original pickle and CSV files
VS_PICKLE_PATH = "data/1D Profiles/TF_HLC/Vs_values_1000.pt"
TTF_PICKLE_PATH = "data/1D Profiles/TF_HLC/TTF_data_1000.pt"
FREQ_PATH = "data/1D Profiles/TF_HLC/TTF_freq_1000.csv"
MODEL_SAVE_PATH = "outputs/models/Soil_Bedrock/best_fno_ttf_model.pt"
RESULTS_SAVE_PATH = "outputs/figures/model"
Path(RESULTS_SAVE_PATH).mkdir(parents=True, exist_ok=True)

# --- Data Preprocessing ---
F0_FILTER_THRESHOLD: float = 2.0  # Filter out profiles with f0 >= this value

# --- Model Hyperparameters ---
INPUT_SIZE: int = 29  # Corresponds to the maximum number of layers in Vs profiles
LATENT_DIM: int = 1000  # Latent dimension for the encoder
OUTPUT_SIZE: int = 1000  # Size of the output transfer function
FNO_MODES: int = 16
FNO_WIDTH: int = 50
NUM_FNO_LAYERS: int = 3
DROPOUT_RATE: float = 0.1
ENCODER_CHANNELS: List[int] = [1, 32, 64, 128, 256, 512,]
## END OF LIST ##
ENCODER_KERNEL_SIZE: int = 3
ENCODER_POOL_SIZE: int = 2

# --- Training Configuration ---
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE: float = 5e-3
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 5000
BATCH_SIZE: int = 100
TRAIN_SPLIT: float = 0.5
VAL_SPLIT: float = 0.25
# TEST_SPLIT is inferred as 1.0 - TRAIN_SPLIT - VAL_SPLIT
SEED: int = 42  # For reproducibility

# --- Early Stopping ---
EARLY_STOP_PATIENCE: int = 1500

# --- Gradient Clipping ---
GRAD_CLIP_NORM: float = 1.0  # Max norm for gradient clipping

# --- W&B Logging ---
WANDB_PROJECT: str = "ttf-prediction"
WANDB_RUN_NAME: str = "B1-Encoder-Deeper"
