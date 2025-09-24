# pino_main.py
"""Main script to run the PINO training and evaluation pipeline."""

import pickle

import numpy as np
import pino_config as config
from data_loader import get_data_loaders
from pino_train import train_pino_model

from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.fno.utils import f0_calc

logger = setup_logging()


def run_pino_pipeline():
    """Executes the full PINO pipeline."""
    logger.info("Loading data for PINO training...")
    with open(config.VS_PICKLE_PATH, "rb") as f:
        vs_profiles = np.array(pickle.load(f))
    with open(config.RHO_PICKLE_PATH, "rb") as f:
        rho_profiles = np.array(pickle.load(f))
    with open(config.TTF_PICKLE_PATH, "rb") as f:
        ttf_data = np.array(pickle.load(f))
    freq_data = np.genfromtxt(config.FREQ_PATH, delimiter=",")

    logger.info(f"Filtering profiles with f0 >= {config.F0_FILTER_THRESHOLD} Hz...")
    f0_values = np.array([f0_calc(profile) for profile in vs_profiles])
    keep_indices = np.where(f0_values < config.F0_FILTER_THRESHOLD)[0]

    vs_profiles_filtered = vs_profiles[keep_indices]
    ttf_data_filtered = ttf_data[keep_indices]
    rho_profiles_filtered = rho_profiles[keep_indices]
    logger.info(f"Kept {len(vs_profiles_filtered)} profiles after filtering.")

    train_loader, val_loader, _ = get_data_loaders(
        vs_profiles_filtered,
        ttf_data_filtered,
        rho_profiles_filtered,
        batch_size=config.BATCH_SIZE,
    )

    logger.info("Starting PINO model training...")
    train_pino_model(train_loader, val_loader, freq_data)


if __name__ == "__main__":
    run_pino_pipeline()
