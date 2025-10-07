# config.py
"""Centralized configuration for the FNO model."""

from pathlib import Path

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
F0_FILTER_THRESHOLD = 2.0  # Filter out profiles with f0 >= this value

# --- Model Hyperparameters ---
INPUT_SIZE = 29  # Corresponds to the maximum number of layers in Vs profiles
LATENT_DIM = 1000  # Latent dimension for the encoder
OUTPUT_SIZE = 1000  # Size of the output transfer function
FNO_MODES = 16
FNO_WIDTH = 50
DROPOUT_RATE = 0.1

ENCODER_CHANNELS = [1, 32, 64, 128, 256]  # Channels for each conv layer in the encoder
ENCODER_KERNEL_SIZE = 3
ENCODER_POOL_SIZE = 2

# --- Training Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 5000
BATCH_SIZE = 100
TRAIN_SPLIT = 0.5
VAL_SPLIT = 0.25
# TEST_SPLIT is inferred as 1.0 - TRAIN_SPLIT - VAL_SPLIT
SEED = 42  # For reproducibility

# --- Early Stopping ---
EARLY_STOP_PATIENCE = 1500

# --- Gradient Clipping ---
GRAD_CLIP_NORM = 1.0  # Max norm for gradient clipping

# --- W&B Logging ---
WANDB_PROJECT = "ttf-prediction"
WANDB_RUN_NAME = "fno-refactored-run"
