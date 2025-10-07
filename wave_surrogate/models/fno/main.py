# main.py
"""Main script to run the FNO training and evaluation pipeline."""

import pickle
import random

import config
import numpy as np
import torch
from data_loader import get_data_loaders
from evaluate import evaluate_model
from train import train_model
from utils import f0_calc

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()


def set_seed(seed):
    """Sets the random seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline():
    """Executes the full pipeline using original pickle files."""
    # --- Set Seed for Reproducibility ---
    logger.info(f"Setting random seed to {config.SEED} for reproducibility.")
    set_seed(config.SEED)

    # --- Load Data from Pickle and CSV ---
    logger.info("Loading data from original pickle files...")
    with open(config.TTF_PICKLE_PATH, "rb") as f:
        ttf_data = np.array(pickle.load(f))

    with open(config.VS_PICKLE_PATH, "rb") as f:
        vs_profiles = np.array(pickle.load(f))

    freq_data = np.genfromtxt(config.FREQ_PATH, delimiter=",")

    logger.info(f"Loaded {len(vs_profiles)} profiles before filtering.")

    # --- Preprocessing: Filter based on f0 ---
    logger.info(f"Filtering profiles with f0 >= {config.F0_FILTER_THRESHOLD} Hz...")
    f0_values = np.array([f0_calc(profile) for profile in vs_profiles])

    # Get indices of profiles to keep
    keep_indices = np.where(f0_values < config.F0_FILTER_THRESHOLD)[0]

    # Filter the datasets
    vs_profiles_filtered = vs_profiles[keep_indices]
    ttf_data_filtered = ttf_data[keep_indices]

    logger.info(f"Kept {len(vs_profiles_filtered)} profiles after filtering.")

    # Convert numpy arrays to lists for the data loader
    vs_list = [arr for arr in vs_profiles_filtered]
    ttf_list = [arr for arr in ttf_data_filtered]

    train_loader, val_loader, test_loader = get_data_loaders(
        vs_list, ttf_list, config.BATCH_SIZE
    )

    # --- Train Model ---
    logger.info("Starting model training...")
    run = train_model(train_loader, val_loader)

    # --- Evaluate Model ---
    logger.info("Starting model evaluation...")
    evaluate_model(test_loader, freq_data, run=run)


if __name__ == "__main__":
    run_pipeline()
