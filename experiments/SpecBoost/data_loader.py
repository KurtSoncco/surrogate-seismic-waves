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
    """PyTorch Dataset for the initial (Stage 1) training."""

    def __init__(self, vs_profiles, ttf_values, max_len=INPUT_SIZE):
        # Correctly process the input profiles
        processed_profiles = []
        for p in vs_profiles:
            # 1. First, ensure the profile is a numeric numpy array.
            #    This crucial step converts any `None` values to `np.nan`.
            numeric_p = np.array(p, dtype=float)

            # 2. Now, safely replace np.nan and inf values with 0.0.
            cleaned_p = np.nan_to_num(numeric_p, nan=0.0, posinf=0.0, neginf=0.0)

            # 3. Pad the cleaned array and convert to a tensor.
            padded_p = pad_array(cleaned_p, max_len)
            tensor_p = torch.tensor(padded_p, dtype=torch.float32)
            processed_profiles.append(tensor_p)

        self.vs_profiles = processed_profiles

        # The rest of your __init__ method remains the same
        self.ttf_values = [
            torch.tensor(
                np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32
            )
            for t in ttf_values
        ]

    def __len__(self):
        return len(self.ttf_values)

    def __getitem__(self, idx):
        return self.vs_profiles[idx], self.ttf_values[idx]


class ResidualDataset(Dataset):
    """
    PyTorch Dataset for SpecBoost Stage 2 (Intermediate Fusion).
    It prepares the dual inputs for the fusion model and the residual target.
    """

    def __init__(self, original_dataset, model_a_preds):
        self.original_dataset = original_dataset
        self.model_a_preds = torch.tensor(model_a_preds, dtype=torch.float32)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Get the original data pair from the initial dataset split
        vs_profile, original_target = self.original_dataset[idx]

        # Get the corresponding prediction from Model A
        model_a_pred = self.model_a_preds[idx]

        # The target for Model B is the residual error
        residual_target = original_target - model_a_pred

        # Return the two inputs as a tuple for the fusion towers
        dual_inputs = (vs_profile, model_a_pred)

        return dual_inputs, residual_target


class NormalizedResidualDataset(torch.utils.data.Dataset):
    """
    Dataset for the iterative boosting pipeline using the ResidualFNOModel.
    It provides the previous prediction as input and the normalized residual as the target.
    """

    def __init__(self, original_dataset, cumulative_preds, residual_mean, residual_std):
        self.original_dataset = original_dataset
        self.cumulative_preds = torch.tensor(cumulative_preds, dtype=torch.float32)
        self.residual_mean = residual_mean
        self.residual_std = residual_std

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # We only need the original target to calculate the true residual
        _, original_target = self.original_dataset[idx]
        cumulative_pred = self.cumulative_preds[idx]

        # Calculate and normalize the target residual
        residual = original_target - cumulative_pred
        normalized_residual = (residual - self.residual_mean) / (
            self.residual_std + 1e-8
        )

        # The input is ONLY the cumulative prediction from the previous stage
        return cumulative_pred, normalized_residual


def get_primary_loaders(vs_profiles, ttf_data, batch_size):
    """
    Performs the initial data split and returns the DataLoaders for Stage 1,
    plus the raw Dataset objects needed for Stage 2.
    """
    dataset = TTFDataset(vs_profiles, ttf_data)

    # Define split sizes
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = int(VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset ONCE to ensure consistency
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    logger.info("Created primary data splits.")
    logger.info(
        f"Train/Val/Test Split: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}"
    )

    # Create DataLoaders for Stage 1
    train_loader_A = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader_A = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return train_loader_A, val_loader_A, test_loader, train_dataset, val_dataset


def get_residual_loaders(
    train_dataset, val_dataset, train_preds_a, val_preds_a, batch_size
):
    """
    Creates and returns the DataLoaders for Stage 2 using the original
    dataset splits and the predictions from Model A.
    """
    logger.info("Creating residual datasets for Stage 2.")

    # Create the residual datasets using the original splits
    residual_train_dataset = ResidualDataset(train_dataset, train_preds_a)
    residual_val_dataset = ResidualDataset(val_dataset, val_preds_a)

    # Create DataLoaders for Stage 2
    train_loader_B = DataLoader(
        residual_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader_B = DataLoader(
        residual_val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    logger.info(
        f"Created residual loaders. Train size: {len(residual_train_dataset)}, Val size: {len(residual_val_dataset)}"
    )

    return train_loader_B, val_loader_B


if __name__ == "__main__":
    ## Let's do a quick test with the data loader and the paths

    import pickle

    from config import FREQ_PATH, TTF_PICKLE_PATH, VS_PICKLE_PATH

    from wave_surrogate.models.fno.utils import f0_calc

    # Load data
    with open(TTF_PICKLE_PATH, "rb") as f:
        ttf_data = np.array(pickle.load(f))
    with open(VS_PICKLE_PATH, "rb") as f:
        vs_profiles = np.array(
            pickle.load(f), dtype=object
        )  # Use dtype=object for ragged arrays
    freq_data = np.genfromtxt(FREQ_PATH, delimiter=",")
    f0_values = np.array([f0_calc(profile) for profile in vs_profiles])
    keep_indices = np.where(f0_values < 2.0)[0]
    vs_profiles_filtered = vs_profiles[keep_indices]
    ttf_data_filtered = ttf_data[keep_indices]
    vs_list = [arr for arr in vs_profiles_filtered]
    ttf_list = [arr for arr in ttf_data_filtered]

    train_loader, val_loader, test_loader, train_dataset, val_dataset = (
        get_primary_loaders(vs_list, ttf_list, batch_size=32)
    )
    # Quick check of the dataloader
    for inputs, targets in train_loader:
        print("Input batch shape:", inputs.shape)
        print("One profile shape (should be (1, 29)):", inputs[0])
        print("Target batch shape:", targets.shape)
        break
    logger.info("DataLoader test completed successfully.")

    # Dataloader for ResidualDataset test
    # Get the shape of a single target tensor from the dataset
    target_len = train_dataset[0][1].shape[0]  # type: ignore
    dummy_model_a_preds_train = np.random.rand(len(train_dataset), target_len).astype(
        np.float32
    )
    dummy_model_a_preds_val = np.random.rand(len(val_dataset), target_len).astype(
        np.float32
    )
    train_loader_B, val_loader_B = get_residual_loaders(
        train_dataset,
        val_dataset,
        dummy_model_a_preds_train,
        dummy_model_a_preds_val,
        batch_size=32,
    )
    for (inputs, model_a_preds), residual_targets in train_loader_B:
        print("Dual input batch shapes:", inputs.shape, model_a_preds.shape)
        print("Residual target batch shape:", residual_targets.shape)
        break
    logger.info("Residual DataLoader test completed successfully.")

    ## CHeck if there any nans in the data
    for inputs, targets in train_loader:
        if torch.isnan(inputs).any():
            logger.error("Input contains NaNs!")
        if torch.isnan(targets).any():
            logger.error("Targets contain NaNs!")
        break
    logger.info("NaN check completed successfully.")

    ## Check if there any nans in the data for residual loader
    for (inputs, model_a_preds), residual_targets in train_loader_B:
        if torch.isnan(inputs).any():
            logger.error("Input contains NaNs!")
        if torch.isnan(model_a_preds).any():
            logger.error("Model A predictions contain NaNs!")
        if torch.isnan(residual_targets).any():
            logger.error("Residual targets contain NaNs!")
        break
    logger.info("NaN check for residual loader completed successfully.")
