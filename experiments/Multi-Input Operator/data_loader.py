# data_loader.py
"""
Handles data loading and preparation for the DeepONet model.

This module:
 - Pads or truncates Vs profiles to INPUT_SIZE.
 - Constructs train/val/test splits.
 - Fits a MinMaxScaler on training TTF values (per-frequency).
 - Builds PyTorch DataLoaders for the splits.
 - Saves the fitted scaler to disk using joblib.
"""

import os
from typing import List

import config
import numpy as np
import torch
from joblib import dump as joblib_dump
from torch.utils.data import DataLoader, Dataset, random_split

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()


def pad_array(arr: np.ndarray, max_len: int) -> np.ndarray:
    """Pad or truncate 1D numpy array to length max_len using zeros for padding."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) >= max_len:
        return np.array(arr[:max_len], dtype=float)
    out = np.zeros(max_len, dtype=float)
    out[: len(arr)] = arr
    return out


class ArcsinhScaler:
    """
    Custom scaler that applies arcsinh transformation followed by MinMax scaling.
    This is useful for data with a wide dynamic range including negative values.
    """

    def __init__(self):
        self.fitted = False

    def fit(self, data: np.ndarray):
        """Fit the scaler on the provided data."""
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply arcsinh transformation followed by MinMax scaling."""
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted")
        return np.arcsinh(X)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform the data back to original scale."""
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted")
        return np.sinh(data)


class DeepONetDataset(Dataset):
    """
    Dataset wrapper that accepts raw numpy lists, performs padding/truncation,
    converts to tensors, and holds the data.
    The TTF values are scaled externally before use.
    """

    def __init__(
        self,
        vs_profiles: List[np.ndarray],
        ttf_values: List[np.ndarray],
        freq_data: np.ndarray,
        max_len: int = config.INPUT_SIZE,
    ):
        self.max_len = max_len

        # Pre-process all VS profiles and stack into a single Tensor
        self.vs_tensors = torch.stack(
            [
                torch.tensor(
                    pad_array(np.asarray(p, dtype=float), max_len),
                    dtype=torch.float32,
                ).unsqueeze(0)
                for p in vs_profiles
            ]
        )  # Shape: (N, 1, L_in)

        # Targets: TTF arrays per sample (convert to tensor (L_out))
        self.ttf_tensors = torch.stack(
            [
                torch.tensor(
                    np.nan_to_num(np.asarray(t, dtype=float)), dtype=torch.float32
                )
                for t in ttf_values
            ]
        )  # Shape: (N, L_out). This is the *unscaled* data initially.

        # Frequency coordinates (same for all samples)
        freq = np.asarray(freq_data)
        if freq.ndim == 2 and freq.shape[1] == 1:
            freq = freq.squeeze(-1)
        if freq.ndim != 1 or freq.shape[0] != config.OUTPUT_SIZE:
            raise ValueError(
                f"Frequency data length ({freq.shape[0]}) must match OUTPUT_SIZE ({config.OUTPUT_SIZE})"
            )
        self.freq_data = torch.tensor(freq, dtype=torch.float32)  # Shape: (L_out,)

        if self.vs_tensors.shape[0] != self.ttf_tensors.shape[0]:
            raise ValueError("Number of Vs profiles and TTF targets must match")

        # Will hold the scaled TTF data, initialized after scaling in get_data_loaders
        self.ttf_tensors_scaled = None

    def set_scaled_ttf(self, scaled_ttf_tensors: torch.Tensor):
        """Sets the internally stored TTF data to the scaled version."""
        assert isinstance(self.ttf_tensors, torch.Tensor)
        if scaled_ttf_tensors.shape != self.ttf_tensors.shape:
            raise ValueError("Scaled TTF shape mismatch")
        # Replace the unscaled data with the scaled version
        self.ttf_tensors_scaled = scaled_ttf_tensors
        self.ttf_tensors = None  # Free memory from unscaled data

    def __len__(self):
        return self.vs_tensors.shape[0]

    def __getitem__(self, idx):
        # Returns tuple: (vs_profile [1, INPUT_SIZE], ttf_target [OUTPUT_SIZE], freq_coords [OUTPUT_SIZE])
        if self.ttf_tensors_scaled is None:
            raise RuntimeError(
                "Scaled TTF data has not been set. Call set_scaled_ttf first."
            )
        return self.vs_tensors[idx], self.ttf_tensors_scaled[idx], self.freq_data


def get_data_loaders(
    vs_profiles: List[np.ndarray],
    ttf_data: List[np.ndarray],
    freq_data: np.ndarray,
    batch_size: int,
    seed: int = config.SEED,
):
    """Creates and returns train, validation, and test DataLoaders."""

    # 1. Initialize dataset with pre-processed Tensors
    dataset = DeepONetDataset(
        vs_profiles, ttf_data, freq_data, max_len=config.INPUT_SIZE
    )

    # 2. Split the full dataset (Correct use of random_split)
    n_total = len(dataset)
    train_size = int(config.TRAIN_SPLIT * n_total)
    val_size = int(config.VAL_SPLIT * n_total)
    test_size = n_total - train_size - val_size

    if train_size <= 0 or test_size <= 0:
        raise ValueError(
            "TRAIN_SPLIT and VAL_SPLIT lead to invalid split sizes. Check config."
        )

    generator = torch.Generator().manual_seed(seed)

    # random_split correctly returns Subset objects
    train_subset, val_subset, test_subset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # 3. Scaling (done once on the full stacked numpy array for efficiency)
    full_ttf_arr = (
        dataset.ttf_tensors.numpy() if dataset.ttf_tensors is not None else np.array([])
    )

    # Use indices from the Subset objects to get the training data slice
    train_indices = train_subset.indices
    train_ttf_arr = full_ttf_arr[train_indices]

    scaler = ArcsinhScaler()
    scaler.fit(train_ttf_arr)  # scales each column (frequency) across samples

    # Transform all data
    full_ttf_scaled_arr = scaler.transform(full_ttf_arr)
    full_ttf_scaled_tensors = torch.tensor(
        full_ttf_scaled_arr, dtype=torch.float32
    )  # (N, L_out)

    # 4. Update the dataset with the scaled data
    dataset.set_scaled_ttf(full_ttf_scaled_tensors)

    # 5. Save Scaler
    save_dir = os.path.abspath(config.MODEL_PARAM_SAVE_PATH)
    os.makedirs(save_dir, exist_ok=True)
    scaler_path = os.path.join(save_dir, "ttf_scaler.joblib")
    try:
        joblib_dump(scaler, scaler_path)
        logger.info(f"Saved TTF scaler to: {scaler_path}")
    except Exception as e:
        logger.warning(f"Failed to save scaler to {scaler_path}: {e}")

    # 6. Create DataLoaders using the Subset objects
    pin_memory = torch.cuda.is_available() and config.DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )

    logger.info(f"Dataset Size: {n_total}")
    logger.info(
        f"Train/Val/Test Split: {len(train_subset)}/{len(val_subset)}/{len(test_subset)}"
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick smoke test to verify data loader behaves correctly.
    import pickle

    logger.info("Loading data for smoke test...")
    # NOTE: Added dtype=object for loading arrays of varying lengths
    try:
        with open(config.TTF_PICKLE_PATH, "rb") as f:
            ttf_data = np.array(pickle.load(f), dtype=object)
        with open(config.VS_PICKLE_PATH, "rb") as f:
            vs_profiles = np.array(pickle.load(f), dtype=object)
    except FileNotFoundError:
        logger.warning("Could not load real data for smoke test. Skipping.")
        exit()

    freq_data = np.genfromtxt(config.FREQ_PATH, delimiter=",")
    if freq_data.ndim == 2 and freq_data.shape[1] == 1:
        freq_data = freq_data.squeeze(-1)

    vs_list = [arr for arr in vs_profiles]
    ttf_list = [arr for arr in ttf_data]

    train_loader, val_loader, test_loader = get_data_loaders(
        vs_list, ttf_list, freq_data, config.BATCH_SIZE
    )
    # inspect a batch
    for vs_b, ttf_b, freq_b in train_loader:
        logger.info(
            f"vs batch: {vs_b.shape}, ttf batch: {ttf_b.shape}, freq shape: {freq_b.shape}"
        )

        # Check normalization
        logger.info(
            f"ttf batch stats: min {ttf_b.min().item():.4f}, max {ttf_b.max().item():.4f}, mean {ttf_b.mean().item():.4f}"
        )
        break
