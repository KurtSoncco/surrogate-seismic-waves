# config.py
"""Centralized configuration for the DeepONet model."""

import os
from pathlib import Path

import torch

# --- Data Configuration ---
# Paths to the original pickle and CSV files
VS_PICKLE_PATH = "data/1D Profiles/TF_HLC/Vs_values_1000.pt"
TTF_PICKLE_PATH = "data/1D Profiles/TF_HLC/TTF_data_1000.pt"
FREQ_PATH = "data/1D Profiles/TF_HLC/TTF_freq_1000.csv"
# Updated model save path
MODEL_PARAM_SAVE_PATH = "experiments/Multi-Input Operator/models"
MODEL_SAVE_PATH = os.path.join(MODEL_PARAM_SAVE_PATH, "best_deeponet_model.pth")
RESULTS_SAVE_PATH = "experiments/Multi-Input Operator/results/"

Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(RESULTS_SAVE_PATH).mkdir(parents=True, exist_ok=True)

# --- Data Preprocessing ---
F0_FILTER_THRESHOLD = 5.0  # Filter out profiles with f0 >= this value

# --- Model Hyperparameters ---
INPUT_SIZE = (
    29  # Corresponds to the maximum number of layers in Vs profiles (Input to Branch)
)
OUTPUT_SIZE = (
    1000  # Size of the output transfer function (Number of query points/frequencies)
)

# DeepONet Latent Dimension (Size of basis functions/coefficients)
LATENT_DIM = 256  # Reduced latent dim from 1000 for better generalization/extrapolation

# Branch Network (The Encoder)
ENCODER_CHANNELS = [1, 32, 64, 128]  # Channels for each conv layer in the branch
ENCODER_KERNEL_SIZE = 3
ENCODER_POOL_SIZE = 2

# Trunk Network (MLP for Frequency Coordinates)
TRUNK_INPUT_DIM = 1  # Frequency is 1D
TRUNK_LAYERS = 4
TRUNK_NEURONS = 256

# --- Training Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 5000  # Increased epochs for better convergence
BATCH_SIZE = 100
TRAIN_SPLIT = 0.5
VAL_SPLIT = 0.25
# TEST_SPLIT is inferred as 1.0 - TRAIN_SPLIT - VAL_SPLIT
SEED = 42


# --- Extrapolation/PI Loss Configuration ---
# Note: Since the exact governing PDE is not provided, we set the PI weight to 0.0.
# For physics-informed extrapolation, set this > 0.0 and implement L_PDE in train.py.
PI_LOSS_WEIGHT = 0.0

# --- Early Stopping ---
EARLY_STOP_PATIENCE = 500  # Reduced patience to reflect lower LR/smaller changes

# --- Gradient Clipping ---
GRAD_CLIP_NORM = 1.0  # Max norm for gradient clipping

# --- W&B Logging ---
WANDB_PROJECT = "ttf-prediction-deeponet"
WANDB_RUN_NAME = "deeponet-extrapolation-run"
