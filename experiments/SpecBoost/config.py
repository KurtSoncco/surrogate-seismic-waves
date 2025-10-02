# config.py
"""Centralized configuration for the SpecBoost FNO model."""

from pathlib import Path

import torch

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# --- Data Configuration ---
VS_PICKLE_PATH = str(
    PROJECT_ROOT / "data" / "1D Profiles" / "TF_HLC" / "Vs_values_1000.pt"
)
TTF_PICKLE_PATH = str(
    PROJECT_ROOT / "data" / "1D Profiles" / "TF_HLC" / "TTF_data_1000.pt"
)
FREQ_PATH = str(PROJECT_ROOT / "data" / "1D Profiles" / "TF_HLC" / "TTF_freq_1000.csv")

# --- Model Save Paths ---
# Get absolute path relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_SAVE_PATH = PROJECT_ROOT / "experiments" / "SpecBoost" / "models"
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
MODEL_A_SAVE_PATH = str(MODEL_SAVE_PATH / "specboost_model_A.pt")
MODEL_B_SAVE_PATH = str(MODEL_SAVE_PATH / "specboost_model_B.pt")
# --- Data Preprocessing ---
F0_FILTER_THRESHOLD = 2.0  # Filter out profiles with f0 >= this value

# --- Model A Hyperparameters (Standard Encoder-Decoder) ---
INPUT_SIZE = 29  # Vs profile length
LATENT_DIM = 1000  # Latent dimension for Model A encoder
OUTPUT_SIZE = 1000  # Transfer function output size
FNO_MODES = 16
FNO_WIDTH = 50
DROPOUT_RATE = 0.1

# Channels for Model A's encoder (single input channel)
ENCODER_CHANNELS_A = [1, 32, 64, 128, 256]
ENCODER_KERNEL_SIZE = 3
ENCODER_POOL_SIZE = 2

# --- Model B Hyperparameters (Fusion Towers) ---
# Model B uses a dual-tower fusion architecture to handle shape mismatch:
# - Tower 1: Encodes Vs profile (29,)
# - Tower 2: Encodes Model A prediction (1000,)
# - Fusion Head: Combines latents to predict residual

# Tower 1: Vs Profile Encoder
VS_ENCODER_CHANNELS = ENCODER_CHANNELS_A  # Channels for short sequence (29,)

# Tower 2: Model A Prediction Encoder
PRED_ENCODER_CHANNELS = ENCODER_CHANNELS_A  # Channels for long sequence (1000,)

# Fusion configuration
FUSION_LATENT_DIM = LATENT_DIM  # Each tower outputs this dimension
FUSION_HIDDEN_DIMS = [512, 1024, 1024, 512]  # Hidden layers in fusion MLP
FUSION_DROPOUT = 0.2  # Dropout for regularization

# Note: For comparison with paper's approach using concatenated inputs,
# you can also configure Model B with 2-channel input encoder:
ENCODER_CHANNELS_B = [2, 32, 64, 128, 256]  # For [x, ŷ] concatenation approach

# --- Training Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-3
BOOSTING_LEARNING_RATE = 1e-3  # Lower LR for boosting stages
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
BATCH_SIZE = 100
TRAIN_SPLIT = 0.5
VAL_SPLIT = 0.25
# TEST_SPLIT is inferred as 1.0 - TRAIN_SPLIT - VAL_SPLIT = 0.25

# --- Boosting Configuration ---
NUM_BOOSTING_STAGES = 3  # Total stages (Model A + Model B + ...)
BOOSTING_ETA = 0.1  # Shrinkage factor for boosting


# --- Early Stopping ---
EARLY_STOP_PATIENCE = 50

# --- Gradient Clipping ---
GRAD_CLIP_NORM = 1.0  # Max norm for gradient clipping

# --- Weights & Biases Logging ---
WANDB_PROJECT = "ttf-prediction-specboost"
WANDB_RUN_NAME = "specboost-fusion-towers"

# --- Architecture Selection ---
# Set to 'fusion' to use dual-tower fusion (handles shape mismatch)
# Set to 'concat' to use concatenated input approach (as in paper)
ARCHITECTURE_MODE = "fusion"  # 'fusion' or 'concat'
