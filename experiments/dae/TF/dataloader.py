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


class TFDataset(Dataset):
    """
    PyTorch Dataset for TF profiles, for DAE training.
    This implementation preprocesses all profiles in a vectorized manner upon
    initialization for maximum efficiency during training.
    """

    def __init__(
        self,
        tf_profiles: List[Profile],
    ):
        # Step 3: Convert to a PyTorch tensor
        self.tf_profiles = torch.from_numpy(np.array(tf_profiles, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.tf_profiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        profile = self.tf_profiles[idx]
        return profile, profile


# --- More Efficient Dataloader Function ---


def get_tf_dataloaders(
    tf_profiles: List[Profile],
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
        tf_profiles: List of TF profiles (each profile is a list or array of floats).
        batch_size: Batch size for the DataLoaders.
        save_path: Directory to save the fitted scaler.
        train_split: Proportion of data to use for training (default 0.70).
        val_split: Proportion of data to use for validation (default 0.15).
    Returns:
        A tuple of (train_loader, val_loader, test_loader, scaler).
    """
    # 1. Split indices first
    dataset_size = len(tf_profiles)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)  # Shuffle before splitting

    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # 2. Create datasets for each split using the raw data
    train_raw = [tf_profiles[i] for i in train_indices]
    val_raw = [tf_profiles[i] for i in val_indices]
    test_raw = [tf_profiles[i] for i in test_indices]

    train_dataset = TFDataset(train_raw)
    val_dataset = TFDataset(val_raw)
    test_dataset = TFDataset(test_raw)

    # 3. Fit scaler ONLY on the training data tensor
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Reshape tensor for scaler: (num_samples * seq_len, 1)
    train_data_flat = train_dataset.tf_profiles.flatten().reshape(-1, 1)
    scaler.fit(train_data_flat.numpy())

    # Save the scaler
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "tf_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # 4. Define a helper to transform data tensors and create TensorDatasets
    def scale_tensor_and_create_dataset(
        dataset: TFDataset, scaler: MinMaxScaler
    ) -> TensorDataset:
        data_tensor = dataset.tf_profiles
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

    print(f"TF Dataset Size: {dataset_size}")
    print(
        f"TF Train/Val/Test Split: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}"
    )

    return train_loader, val_loader, test_loader, scaler


# --- Main Execution Block (with the fix) ---
if __name__ == "__main__":
    from utils import TrainingConfig

    from wave_surrogate.logging_setup import setup_logging

    config = TrainingConfig()
    logger = setup_logging()

    dataset_file = config.tf_data_path

    try:
        data = pd.read_parquet(dataset_file)
        vs_profiles_list = data["model_data"].tolist()
        logger.info(
            f"Loaded and converted {len(vs_profiles_list)} profiles from {dataset_file}."
        )
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_file}")
        raise

    train_loader, val_loader, test_loader, scaler = get_tf_dataloaders(
        tf_profiles=vs_profiles_list,
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
