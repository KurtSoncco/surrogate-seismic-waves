# data_loader.py
"""Handles data loading and preparation for the FNO model."""

import numpy as np
import torch
from config import INPUT_SIZE, TRAIN_SPLIT, VAL_SPLIT
from torch.utils.data import DataLoader, Dataset, random_split

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()


def pad_array(arr, max_len):
    """Pads a numpy array with zeros to a specified max length."""
    padded_arr = np.zeros(max_len)
    padded_arr[: len(arr)] = arr
    return padded_arr


class TTFDataset(Dataset):
    """PyTorch Dataset for Transfer Function prediction."""

    def __init__(self, vs_profiles, ttf_values, max_len=INPUT_SIZE):
        self.vs_profiles = [
            torch.tensor(
                pad_array(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), max_len),
                dtype=torch.float32,
            )
            for p in vs_profiles
        ]
        self.ttf_values = [torch.tensor(t, dtype=torch.float32) for t in ttf_values]

    def __len__(self):
        return len(self.ttf_values)

    def __getitem__(self, idx):
        return self.vs_profiles[idx], self.ttf_values[idx]


def get_data_loaders(vs_profiles, ttf_data, batch_size):
    """Creates and returns train, validation, and test DataLoaders."""
    dataset = TTFDataset(vs_profiles, ttf_data)

    logger.info(f"Sample of padded Vs profile: {dataset[0][0]}")
    logger.info(f"Sample of TTF profile: {dataset[0][1]}")

    # Define split sizes
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = int(VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    logger.info(f"Dataset Size: {len(dataset)}")
    logger.info(f"Train/Val/Test Split: {train_size}/{val_size}/{test_size}")

    return train_loader, val_loader, test_loader
