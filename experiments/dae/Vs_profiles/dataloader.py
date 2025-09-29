import os
import pickle
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils import TrainingConfig

# --- VsDataset Class (with added NaN handling) ---

Profile = Union[List[float], np.ndarray]


class VsDataset(Dataset):
    """
    PyTorch Dataset for Vs profiles, for DAE training.
    This implementation preprocesses all profiles in a vectorized manner upon
    initialization for maximum efficiency during training.
    """

    def __init__(
        self,
        vs_profiles: List[Profile],
        original_dz: float = 5.0,
        new_dz: float = 1.0,
    ):
        # The final tensor representing all processed profiles
        self.vs_profiles: torch.Tensor
        # The calculated final length of the sequences
        self.final_len: int

        if not vs_profiles:
            self.final_len = 0
            self.vs_profiles = torch.empty(0, 0, dtype=torch.float32)
            return

        # Step 1: Calculate the final length internally
        non_empty_profiles = [p for p in vs_profiles if p is not None and len(p) > 0]
        if not non_empty_profiles:
            # If all profiles are empty, we can't determine a length. Default to 0.
            self.final_len = 0
            self.vs_profiles = torch.zeros(len(vs_profiles), 0, dtype=torch.float32)
            return

        max_raw_len = max(len(p) for p in non_empty_profiles)
        interp_factor = int(original_dz / new_dz)
        if original_dz % new_dz != 0:
            raise ValueError("original_dz must be a multiple of new_dz.")
        self.final_len = max_raw_len * interp_factor

        # Step 2: Pad raw profiles to the max raw length
        padded_raw_profiles = np.zeros(
            (len(vs_profiles), max_raw_len), dtype=np.float32
        )
        for i, p in enumerate(vs_profiles):
            if p is None or len(p) == 0:
                continue

            p_arr = np.nan_to_num(np.asarray(p, dtype=np.float32))
            padding_val = p_arr[-1]
            padded_raw_profiles[i, : len(p_arr)] = p_arr
            padded_raw_profiles[i, len(p_arr) :] = padding_val

        # Step 3: Interpolate the entire batch at once.
        # The result is already correctly padded to the final length.
        final_profiles_np = np.repeat(padded_raw_profiles, interp_factor, axis=1)

        # ✅ The old truncation/padding logic is no longer needed!

        # Step 4: Convert to a PyTorch tensor
        self.vs_profiles = torch.from_numpy(final_profiles_np)

    def __len__(self) -> int:
        return len(self.vs_profiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        profile = self.vs_profiles[idx]
        return profile, profile


# --- More Efficient Dataloader Function ---


def get_vs_dataloaders(
    vs_profiles: List[Profile],
    batch_size: int,
    save_path: str,
    config: TrainingConfig,
    train_split: float = 0.70,
    val_split: float = 0.15,
):
    """
    Creation of train, validation, and test DataLoaders.
    This version splits the raw data first and scales the resulting tensors
    directly, avoiding costly back-and-forth conversions.

    Args:
        vs_profiles: List of Vs profiles (each profile is a list or array of floats).
        batch_size: Batch size for the DataLoaders.
        save_path: Directory to save the fitted scaler.
        train_split: Proportion of data to use for training (default 0.70).
        val_split: Proportion of data to use for validation (default 0.15).
    Returns:
        A tuple of (train_loader, val_loader, test_loader, scaler).
    """
    # 1. Split indices first
    dataset_size = len(vs_profiles)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)  # Shuffle before splitting

    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # 2. Create datasets for each split using the raw data
    train_raw = [vs_profiles[i] for i in train_indices]
    val_raw = [vs_profiles[i] for i in val_indices]
    test_raw = [vs_profiles[i] for i in test_indices]

    train_dataset = VsDataset(
        train_raw, original_dz=config.original_dz, new_dz=config.new_dz
    )
    val_dataset = VsDataset(
        val_raw, original_dz=config.original_dz, new_dz=config.new_dz
    )
    test_dataset = VsDataset(
        test_raw, original_dz=config.original_dz, new_dz=config.new_dz
    )

    # 3. Fit scaler ONLY on the training data tensor
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Reshape tensor for scaler: (num_samples * seq_len, 1)
    train_data_flat = train_dataset.vs_profiles.flatten().reshape(-1, 1)
    scaler.fit(train_data_flat.numpy())

    # Save the scaler
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "vs_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # 4. Define a helper to transform data tensors and create TensorDatasets
    def scale_tensor_and_create_dataset(
        dataset: VsDataset, scaler: MinMaxScaler
    ) -> TensorDataset:
        data_tensor = dataset.vs_profiles
        if data_tensor.numel() == 0:  # Handle empty dataset
            return TensorDataset(data_tensor, data_tensor)

        original_shape = data_tensor.shape
        scaled_data = scaler.transform(data_tensor.flatten().reshape(-1, 1).numpy())
        scaled_tensor = torch.FloatTensor(scaled_data).reshape(original_shape)
        return TensorDataset(scaled_tensor, scaled_tensor)

    # 5. Create scaled datasets and dataloaders
    train_loader = DataLoader(
        scale_tensor_and_create_dataset(train_dataset, scaler),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        scale_tensor_and_create_dataset(val_dataset, scaler),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        scale_tensor_and_create_dataset(test_dataset, scaler),
        batch_size=batch_size,
        shuffle=False,
    )  # No need to shuffle test set

    print(f"Vs Dataset Size: {dataset_size}")
    print(
        f"Vs Train/Val/Test Split: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}"
    )

    return train_loader, val_loader, test_loader, scaler


# --- Main Execution Block (with the fix) ---
if __name__ == "__main__":
    from utils import TrainingConfig

    from wave_surrogate.logging_setup import setup_logging

    config = TrainingConfig()
    logger = setup_logging()

    dataset_file = config.vs_data_path

    try:
        data = pd.read_parquet(dataset_file)
        vs_profiles_list = data["model_data"].tolist()
        logger.info(
            f"Loaded and converted {len(vs_profiles_list)} profiles from {dataset_file}."
        )
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_file}")
        raise

    train_loader, val_loader, test_loader, scaler = get_vs_dataloaders(
        vs_profiles=vs_profiles_list,
        batch_size=config.batch_size,
        save_path=config.model_dir,
        config=config,
    )

    # Quick check of the dataloader
    print("\nFirst batch from train_loader:")
    for batch in train_loader:
        inputs, targets = batch
        print(f"  Batch shape: {inputs.shape}")
        print(f"  Batch min value: {inputs.min():.4f}, max value: {inputs.max():.4f}")
        break
