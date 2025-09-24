# pino_config_v2.py
"""Centralized configuration for the updated PINO model."""

import torch

# --- Data Configuration ---
VS_PICKLE_PATH = "data/1D Profiles/TF_HLC/Vs_values_1000.pt"
RHO_PICKLE_PATH = "data/1D Profiles/TF_HLC/Rho_values_1000.pt"
TTF_PICKLE_PATH = "data/1D Profiles/TF_HLC/TTF_data_1000.pt"
FREQ_PATH = "data/1D Profiles/TF_HLC/TTF_freq_1000.csv"
MODEL_SAVE_PATH = "outputs/models/Soil_Bedrock/best_pino_v2_model.pt"

F0_FILTER_THRESHOLD = 2.0
INPUT_SIZE = 29

# --- Model Hyperparameters ---
# The first channel is now 2 (Vs and Rho)
ENCODER_CHANNELS = [2, 16, 32, 64]
LATENT_DIM = 128
ENCODER_KERNEL_SIZE = 3
ENCODER_POOL_SIZE = 2

# --- PINO-Specific Hyperparameters ---
SPATIAL_POINTS = 64  # Number of spatial discretization points
TIMESTEPS = 500  # Increased for better frequency resolution
DT = 1e-3  # Time step
T_MAX = (TIMESTEPS - 1) * DT  # (TIMESTEPS-1)*DT for cleaner FFT

FNO_MODES = 16
FNO_WIDTH = 32
NUM_FNO_LAYERS = 4

# --- Training Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 3000
BATCH_SIZE = 4
TRAIN_SPLIT = 0.5
VAL_SPLIT = 0.25
EARLY_STOP_PATIENCE = 500
GRAD_CLIP_NORM = 1.0
PHYSICS_LOSS_WEIGHT = 1e-2

# --- W&B Logging ---
WANDB_PROJECT = "ttf-prediction-pino"
WANDB_RUN_NAME = "pino-v2-variable-density"

# --- Physical Constants ---
LAYER_THICKNESS = 5.0  # meters
