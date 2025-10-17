import os
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils import TrainingConfig

from wave_surrogate.logging_setup import setup_logging

# --- Profile Type Hint ---
Profile = Optional[Union[List[float], np.ndarray]]


# --- MaterialsDataset Class (No changes needed, it's already well-designed) ---
class MaterialsDataset(Dataset):
    """
    PyTorch Dataset for material profiles (Vs, Vp, Rho).
    Produces tensors of shape (N, C=3, L).
    Handles NaNs and padding. NOTE: does NOT perform interpolation/resampling.
    Input profiles are padded to the maximum profile length found in the batch.
    """

    def __init__(
        self,
        vs_profiles: List[Profile],
        vp_profiles: List[Profile],
        rho_profiles: List[Profile],
    ):
        # final tensor: shape (N, C, L)
        self.materials: torch.Tensor
        self.final_len: int

        num_samples = len(vs_profiles)
        if num_samples == 0:
            self.final_len = 0
            self.materials = torch.empty(0, 3, 0, dtype=torch.float32)
            return

        if not (len(vp_profiles) == num_samples and len(rho_profiles) == num_samples):
            raise ValueError(
                "vs_profiles, vp_profiles and rho_profiles must have equal length"
            )

        # Determine max length from all provided profiles
        lengths = [
            len(p)
            for p_list in [vs_profiles, vp_profiles, rho_profiles]
            for p in p_list
            if p is not None and len(p) > 0
        ]

        if not lengths:
            self.final_len = 0
            self.materials = torch.zeros(num_samples, 3, 0, dtype=torch.float32)
            return

        max_raw_len = max(lengths)
        self.final_len = max_raw_len

        def build_padded_array(profiles):
            # Convert list of profiles to list of 1D torch tensors, nan-safe
            tensors = []
            lengths_list = []
            last_vals = []
            for p in profiles:
                if p is None or len(p) == 0:
                    tensors.append(torch.empty(0, dtype=torch.float32))
                    lengths_list.append(0)
                    last_vals.append(torch.tensor(0.0, dtype=torch.float32))
                else:
                    arr = torch.from_numpy(np.nan_to_num(np.asarray(p, dtype=np.float32)))
                    tensors.append(arr)
                    lengths_list.append(min(arr.numel(), max_raw_len))
                    last_vals.append(arr[-1])

            # Pad to (num_samples, max_raw_len) with zeros first
            padded = torch.nn.utils.rnn.pad_sequence(
                [t[:max_raw_len] for t in tensors], batch_first=True, padding_value=0.0
            )

            # Ensure correct width even if all inputs are empty
            if padded.size(1) < max_raw_len:
                pad_cols = max_raw_len - padded.size(1)
                padded = torch.nn.functional.pad(padded, (0, pad_cols))

            lengths_tensor = torch.tensor(lengths_list, dtype=torch.long)
            last_vals_tensor = torch.stack(last_vals).to(dtype=torch.float32)

            # Build mask for padded positions per row
            arange_cols = torch.arange(max_raw_len).unsqueeze(0).expand(num_samples, -1)
            pad_mask = arange_cols >= lengths_tensor.unsqueeze(1)

            # For rows where length == 0, fill entire row with 0 (already 0); else fill padded with last value
            fill_vals = last_vals_tensor.view(-1, 1).expand(-1, max_raw_len)
            padded = torch.where(pad_mask, fill_vals, padded)

            return padded.numpy().astype(np.float32)

        vs_padded = build_padded_array(vs_profiles)
        vp_padded = build_padded_array(vp_profiles)
        rho_padded = build_padded_array(rho_profiles)

        stacked = np.stack([vs_padded, vp_padded, rho_padded], axis=1)
        self.materials = torch.from_numpy(stacked).float()

    def __len__(self) -> int:
        return len(self.materials)

    def __getitem__(self, idx: int):
        mat = self.materials[idx]  # shape (3, L)
        return mat, mat


# --- VsDataset Class (Unchanged, but no longer used in main) ---
class VsDataset(Dataset):
    # ... (Your VsDataset class code here, no changes needed)
    pass


# --- MODIFIED Dataloader Function for Multi-Channel Data ---
def get_material_dataloaders(
    dataset_path: str,
    batch_size: int,
    save_path: str,
    train_split: float = 0.70,
    val_split: float = 0.15,
    spacing_z: float = 5.0,
    rng_seed: int = 42,
):
    """
    Creates train, validation, and test DataLoaders for multi-channel material data.
    - Reads a parquet file with 'vs_profile', 'vp_profile', 'rho_profile' columns.
    - Creates a depth coordinate grid based on fixed spacing (e.g., 5m).
    - Fits a separate scaler for each channel (Vs, Vp, Rho) on the training set.
    - Saves all scalers to a single pickle file.

    Args:
        dataset_path: Path to the input parquet file.
        batch_size: Batch size for the DataLoaders.
        save_path: Directory to save the fitted scalers.
        train_split: Proportion of data for training.
        val_split: Proportion of data for validation.
    Returns:
        A tuple of (train_loader, val_loader, test_loader, scalers_dict).
    """
    # 1. Load data from the parquet file
    try:
        df = pd.read_parquet(dataset_path)
        vs_profiles = df["vs_profile"].tolist()
        vp_profiles = df["vp_profile"].tolist()
        rho_profiles = df["rho_profile"].tolist()
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading data: {e}")
        print(
            "Ensure the parquet file exists and contains 'vs_profile', 'vp_profile', and 'rho_profile' columns."
        )
        raise

    # 2. Split indices
    dataset_size = len(vs_profiles)
    indices = list(range(dataset_size))
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(indices)

    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # 3. Create raw data splits for each channel
    train_vs, val_vs, test_vs = (
        [vs_profiles[i] for i in train_indices],
        [vs_profiles[i] for i in val_indices],
        [vs_profiles[i] for i in test_indices],
    )
    train_vp, val_vp, test_vp = (
        [vp_profiles[i] for i in train_indices],
        [vp_profiles[i] for i in val_indices],
        [vp_profiles[i] for i in test_indices],
    )
    train_rho, val_rho, test_rho = (
        [rho_profiles[i] for i in train_indices],
        [rho_profiles[i] for i in val_indices],
        [rho_profiles[i] for i in test_indices],
    )

    # 4. Create PyTorch Datasets for each split
    train_dataset = MaterialsDataset(train_vs, train_vp, train_rho)
    val_dataset = MaterialsDataset(val_vs, val_vp, val_rho)
    test_dataset = MaterialsDataset(test_vs, test_vp, test_rho)

    # 5. Compute per-channel min/max on TRAINING data (vectorized, torch-only)
    # Shapes: mins/maxs -> (3,)
    if train_dataset.materials.numel() > 0:
        channel_mins = torch.amin(train_dataset.materials, dim=(0, 2))
        channel_maxs = torch.amax(train_dataset.materials, dim=(0, 2))
    else:
        channel_mins = torch.zeros(3, dtype=torch.float32)
        channel_maxs = torch.ones(3, dtype=torch.float32)

    # Small epsilon to avoid division by zero
    eps = torch.finfo(torch.float32).eps

    # Save the scalers dictionary (torch tensors are pickleable)
    scalers: Dict[str, torch.Tensor] = {"mins": channel_mins, "maxs": channel_maxs}

    # Save the scalers dictionary
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "material_scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    # 6. Helper to scale a multi-channel tensor and create a TensorDataset
    def scale_tensor_and_create_dataset(
        dataset: MaterialsDataset, scalers_dict: Dict[str, torch.Tensor]
    ) -> TensorDataset:
        data_tensor = dataset.materials
        if data_tensor.numel() == 0:
            return TensorDataset(data_tensor, data_tensor)

        # Vectorized per-channel min-max scaling: (B, C, L)
        mins = scalers_dict["mins"].view(1, -1, 1)
        maxs = scalers_dict["maxs"].view(1, -1, 1)
        denom = (maxs - mins).clamp_min(eps)
        scaled_tensor = (data_tensor - mins) / denom

        # Create the depth coordinate grid based on fixed spacing (vectorized)
        num_points = data_tensor.shape[2]
        depth_grid = torch.linspace(0, spacing_z * (num_points - 1), num_points, dtype=data_tensor.dtype, device=data_tensor.device)
        grid_tensor = depth_grid.unsqueeze(0).repeat(data_tensor.shape[0], 1)

        # Concatenate grid as an extra channel: output shape (B, 4, L)
        grid_channel = grid_tensor.unsqueeze(1)  # (B, 1, L)
        features_plus_grid = torch.cat([grid_channel, scaled_tensor], dim=1)

        # Targets: mirror the same (B, 4, L) unless you prefer only features
        return TensorDataset(features_plus_grid, features_plus_grid)

    # 7. Create scaled datasets and DataLoaders
    train_loader = DataLoader(
        scale_tensor_and_create_dataset(train_dataset, scalers),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        scale_tensor_and_create_dataset(val_dataset, scalers),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        scale_tensor_and_create_dataset(test_dataset, scalers),
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Materials Dataset Size: {dataset_size}")
    print(
        f"Train/Val/Test Split: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}"
    )

    return train_loader, val_loader, test_loader, scalers


# --- UPDATED Main Execution Block ---
if __name__ == "__main__":
    config = TrainingConfig()
    logger = setup_logging()

    # Assumes config.vs_data_path now points to the new multi-channel parquet file
    # e.g., 'data/processed/materials_dataset.parquet'
    dataset_file = config.materials_data_path

    logger.info(f"Loading multi-channel dataset from: {dataset_file}")

    # The new function handles everything: loading, splitting, scaling, and dataloader creation
    train_loader, val_loader, test_loader, scalers = get_material_dataloaders(
        dataset_path=dataset_file,
        batch_size=config.batch_size,
        save_path=str(config.model_dir),
    )

    logger.info(
        f"Dataloaders created successfully. Scalers saved to '{str(config.model_dir)}'."
    )
    logger.info(f"Scaler keys: {list(scalers.keys())}")

    # Verification step: check the shape of a batch
    try:
        sample_batch, _ = next(iter(train_loader))
        logger.info(f"Sample batch shape: {sample_batch.shape}")
        # Expected output: torch.Size([batch_size, 3, sequence_length])
    except StopIteration:
        logger.warning("Train loader is empty, cannot verify batch shape.")
