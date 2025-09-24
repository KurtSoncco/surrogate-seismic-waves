# data_loader_v2.py
"""Handles data loading for Vs and Rho profiles."""

import os
import pickle

import numpy as np
import pino_config as config
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()


def pad_array(arr, max_len):
    """Pad or truncate array to max_len."""
    if len(arr) > max_len:
        return arr[:max_len]
    elif len(arr) < max_len:
        padded_arr = np.zeros(max_len)
        padded_arr[: len(arr)] = arr
        return padded_arr
    else:
        return arr


class VsRhoTTFDataset(Dataset):
    """PyTorch Dataset for Vs/Rho profiles and Transfer Functions."""

    def __init__(
        self, vs_profiles, rho_profiles, ttf_values, max_len=config.INPUT_SIZE
    ):
        self.vs_profiles = vs_profiles
        self.rho_profiles = rho_profiles
        self.ttf_values = [torch.tensor(t, dtype=torch.float32) for t in ttf_values]

    def __len__(self):
        return len(self.ttf_values)

    def __getitem__(self, idx):
        vs_arr = np.nan_to_num(self.vs_profiles[idx])
        rho_arr = np.nan_to_num(self.rho_profiles[idx])

        # Pad or truncate to INPUT_SIZE
        vs_padded = pad_array(vs_arr, config.INPUT_SIZE)
        rho_padded = pad_array(rho_arr, config.INPUT_SIZE)

        # Stack Vs and Rho to create a 2-channel input
        input_tensor = torch.tensor(
            np.stack([vs_padded, rho_padded]), dtype=torch.float32
        )

        return input_tensor, self.ttf_values[idx]


def get_data_loaders(vs_profiles, rho_profiles, ttf_data, batch_size):
    """Creates DataLoaders for the v2 PINO."""
    dataset = VsRhoTTFDataset(vs_profiles, rho_profiles, ttf_data)
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = int(config.VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    vs_profiles = pickle.load(open(config.VS_PICKLE_PATH, "rb"))
    if os.path.exists(config.RHO_PICKLE_PATH):
        rho_profiles = pickle.load(open(config.RHO_PICKLE_PATH, "rb"))
    else:
        logger.info("Rho profiles file not found. Creating dummy Rho profiles.")
        rho_assumed = 2000  # kg/m^3, typical value for soil/rock
        rho_profiles = [np.full_like(vs, rho_assumed) for vs in vs_profiles]
        # Save the dummy Rho profiles for future use
        with open(config.RHO_PICKLE_PATH, "wb") as f:
            pickle.dump(rho_profiles, f)
            logger.info(f"Dummy Rho profiles saved to {config.RHO_PICKLE_PATH}")

    ttf_data = pickle.load(open(config.TTF_PICKLE_PATH, "rb"))

    train_loader, val_loader, test_loader = get_data_loaders(
        vs_profiles, rho_profiles, ttf_data, config.BATCH_SIZE
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
