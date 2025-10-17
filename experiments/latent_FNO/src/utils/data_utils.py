# data_utils.py
"""
Data loading and preprocessing utilities for the latent FNO experiments.
"""

import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class SeismicDataLoader:
    """
    Data loader for seismic data with preprocessing and normalization.
    """

    def __init__(
        self,
        vs_data_path: str,
        ttf_data_path: str,
        freq_data_path: str,
        input_dim: int = 29,
        device: str = "cpu",
    ):
        """
        Initialize data loader.

        Args:
            vs_data_path: Path to Vs profile data
            ttf_data_path: Path to transfer function data
            freq_data_path: Path to frequency data
            input_dim: Expected input dimension for Vs profiles
            device: Device to use for tensors
        """
        self.vs_data_path = vs_data_path
        self.ttf_data_path = ttf_data_path
        self.freq_data_path = freq_data_path
        self.input_dim = input_dim
        self.device = device

        # Data will be loaded and preprocessed
        self.vs_data = None
        self.ttf_data = None
        self.freq_data = None
        self.normalization_params = {}

    def load_raw_data(self) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Load raw data from files.

        Returns:
            Tuple of (vs_data, ttf_data, freq_data)
        """
        import pickle
        
        # Load Vs profiles using pickle (as used in SpecBoost)
        with open(self.vs_data_path, "rb") as f:
            vs_profiles = np.array(pickle.load(f), dtype=object)
        
        # Process Vs profiles to match expected input dimension
        processed_profiles = []
        for profile in vs_profiles:
            # Convert to numeric array and handle NaN/inf values
            numeric_p = np.array(profile, dtype=float)
            cleaned_p = np.nan_to_num(numeric_p, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Pad or truncate to desired input dimension
            if len(cleaned_p) < self.input_dim:
                padded = np.pad(cleaned_p, (0, self.input_dim - len(cleaned_p)), mode='edge')
            else:
                padded = cleaned_p[:self.input_dim]
            
            processed_profiles.append(padded)
        
        self.vs_data = torch.tensor(np.array(processed_profiles), dtype=torch.float32)

        # Load transfer functions using pickle
        with open(self.ttf_data_path, "rb") as f:
            ttf_data_raw = np.array(pickle.load(f))
        
        # Process TTF data
        processed_ttf = []
        for ttf in ttf_data_raw:
            cleaned_ttf = np.nan_to_num(ttf, nan=0.0, posinf=0.0, neginf=0.0)
            processed_ttf.append(cleaned_ttf)
        
        self.ttf_data = torch.tensor(np.array(processed_ttf), dtype=torch.float32)

        # Load frequency data
        self.freq_data = np.loadtxt(self.freq_data_path, delimiter=",")

        return self.vs_data, self.ttf_data, self.freq_data

    def preprocess_data(
        self,
        filter_f0_threshold: float = 2.0,
        normalize: bool = True,
        remove_outliers: bool = True,
        outlier_threshold: float = 3.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the loaded data.

        Args:
            filter_f0_threshold: Threshold for filtering based on fundamental frequency
            normalize: Whether to normalize the data
            remove_outliers: Whether to remove outliers
            outlier_threshold: Threshold for outlier detection (in standard deviations)

        Returns:
            Tuple of (processed_vs_data, processed_ttf_data)
        """
        if self.vs_data is None or self.ttf_data is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")

        vs_data = self.vs_data.clone()
        ttf_data = self.ttf_data.clone()

        # Filter based on fundamental frequency if available
        if hasattr(self, "freq_data") and self.freq_data is not None:
            vs_data, ttf_data = self._filter_by_fundamental_frequency(
                vs_data, ttf_data, filter_f0_threshold
            )

        # Remove outliers
        if remove_outliers:
            vs_data, ttf_data = self._remove_outliers(
                vs_data, ttf_data, outlier_threshold
            )

        # Normalize data
        if normalize:
            vs_data, ttf_data, norm_params = self._normalize_data(vs_data, ttf_data)
            self.normalization_params = norm_params

        return vs_data, ttf_data

    def _filter_by_fundamental_frequency(
        self, vs_data: torch.Tensor, ttf_data: torch.Tensor, threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter data based on fundamental frequency."""
        # This is a placeholder - implement based on your specific needs
        # For now, we'll just return the data as-is
        return vs_data, ttf_data

    def _remove_outliers(
        self, vs_data: torch.Tensor, ttf_data: torch.Tensor, threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove outliers from the data."""
        # Calculate statistics for outlier detection
        vs_std = torch.std(vs_data, dim=1)
        ttf_std = torch.std(ttf_data, dim=1)

        # Keep samples within threshold standard deviations
        vs_mask = vs_std < (torch.mean(vs_std) + threshold * torch.std(vs_std))
        ttf_mask = ttf_std < (torch.mean(ttf_std) + threshold * torch.std(ttf_std))

        # Combined mask
        mask = vs_mask & ttf_mask

        return vs_data[mask], ttf_data[mask]

    def _normalize_data(
        self, vs_data: torch.Tensor, ttf_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Normalize data using z-score normalization."""
        # Calculate statistics
        vs_mean = torch.mean(vs_data, dim=0)
        vs_std = torch.std(vs_data, dim=0)
        ttf_mean = torch.mean(ttf_data, dim=0)
        ttf_std = torch.std(ttf_data, dim=0)

        # Avoid division by zero
        vs_std = torch.where(vs_std == 0, torch.ones_like(vs_std), vs_std)
        ttf_std = torch.where(ttf_std == 0, torch.ones_like(ttf_std), ttf_std)

        # Normalize
        vs_normalized = (vs_data - vs_mean) / vs_std
        ttf_normalized = (ttf_data - ttf_mean) / ttf_std

        # Store normalization parameters
        norm_params = {
            "vs_mean": vs_mean,
            "vs_std": vs_std,
            "ttf_mean": ttf_mean,
            "ttf_std": ttf_std,
        }

        return vs_normalized, ttf_normalized, norm_params

    def denormalize_predictions(
        self, predictions: torch.Tensor, data_type: str = "ttf"
    ) -> torch.Tensor:
        """
        Denormalize predictions back to original scale.

        Args:
            predictions: Normalized predictions
            data_type: Type of data ("vs" or "ttf")

        Returns:
            Denormalized predictions
        """
        if not self.normalization_params:
            return predictions

        if data_type == "ttf":
            mean = self.normalization_params["ttf_mean"]
            std = self.normalization_params["ttf_std"]
        elif data_type == "vs":
            mean = self.normalization_params["vs_mean"]
            std = self.normalization_params["vs_std"]
        else:
            raise ValueError("data_type must be 'vs' or 'ttf'")

        return predictions * std + mean

    def create_data_loaders(
        self,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.

        Args:
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
            batch_size: Batch size for data loaders
            shuffle: Whether to shuffle training data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.vs_data is None or self.ttf_data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        # Create dataset
        dataset = TensorDataset(self.vs_data, self.ttf_data)

        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size

        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader

    def save_normalization_params(self, path: str):
        """Save normalization parameters to file."""
        with open(path, "wb") as f:
            pickle.dump(self.normalization_params, f)

    def load_normalization_params(self, path: str):
        """Load normalization parameters from file."""
        with open(path, "rb") as f:
            self.normalization_params = pickle.load(f)


class DataAugmentation:
    """
    Data augmentation techniques for seismic data.
    """

    @staticmethod
    def add_noise(
        data: torch.Tensor, noise_level: float = 0.01, noise_type: str = "gaussian"
    ) -> torch.Tensor:
        """
        Add noise to the data.

        Args:
            data: Input data
            noise_level: Level of noise to add
            noise_type: Type of noise ("gaussian", "uniform")

        Returns:
            Data with added noise
        """
        if noise_type == "gaussian":
            noise = torch.randn_like(data) * noise_level
        elif noise_type == "uniform":
            noise = (torch.rand_like(data) - 0.5) * 2 * noise_level
        else:
            raise ValueError("noise_type must be 'gaussian' or 'uniform'")

        return data + noise

    @staticmethod
    def time_warping(data: torch.Tensor, warp_factor: float = 0.1) -> torch.Tensor:
        """
        Apply time warping to the data.

        Args:
            data: Input data (batch_size, sequence_length)
            warp_factor: Factor for time warping

        Returns:
            Time-warped data
        """
        batch_size, seq_len = data.shape

        # Generate warping parameters
        warped_data = []
        for i in range(batch_size):
            # Create warping function
            x = torch.linspace(0, seq_len - 1, seq_len)

            # Simple linear warping (can be made more complex)
            warp = 1 + warp_factor * torch.sin(2 * np.pi * x / seq_len)
            warp = warp / torch.mean(warp)  # Normalize to preserve mean

            # Apply warping (simplified - in practice you'd use interpolation)
            warped = data[i] * warp
            warped_data.append(warped)

        return torch.stack(warped_data)

    @staticmethod
    def magnitude_scaling(
        data: torch.Tensor, scale_range: Tuple[float, float] = (0.8, 1.2)
    ) -> torch.Tensor:
        """
        Apply magnitude scaling to the data.

        Args:
            data: Input data
            scale_range: Range for scaling factors

        Returns:
            Magnitude-scaled data
        """
        batch_size = data.shape[0]
        scale_factors = (
            torch.rand(batch_size) * (scale_range[1] - scale_range[0]) + scale_range[0]
        )

        # Apply scaling to each sample
        scaled_data = data * scale_factors.unsqueeze(-1)

        return scaled_data


def create_enhanced_dataset(
    vs_data: torch.Tensor,
    ttf_data: torch.Tensor,
    augmentation_config: Optional[Dict[str, Any]] = None,
) -> TensorDataset:
    """
    Create an enhanced dataset with data augmentation.

    Args:
        vs_data: Vs profile data
        ttf_data: Transfer function data
        augmentation_config: Configuration for data augmentation

    Returns:
        Enhanced dataset
    """
    if augmentation_config is None:
        return TensorDataset(vs_data, ttf_data)

    # Apply augmentations
    augmented_vs = vs_data.clone()
    augmented_ttf = ttf_data.clone()

    if augmentation_config.get("add_noise", False):
        noise_level = augmentation_config.get("noise_level", 0.01)
        augmented_vs = DataAugmentation.add_noise(augmented_vs, noise_level)
        augmented_ttf = DataAugmentation.add_noise(augmented_ttf, noise_level)

    if augmentation_config.get("magnitude_scaling", False):
        scale_range = augmentation_config.get("scale_range", (0.8, 1.2))
        augmented_vs = DataAugmentation.magnitude_scaling(augmented_vs, scale_range)
        augmented_ttf = DataAugmentation.magnitude_scaling(augmented_ttf, scale_range)

    return TensorDataset(augmented_vs, augmented_ttf)


def load_experiment_data(
    data_config: Dict[str, str], preprocess_config: Optional[Dict[str, Any]] = None
) -> Tuple[SeismicDataLoader, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Convenience function to load and preprocess experiment data.

    Args:
        data_config: Configuration with data paths
        preprocess_config: Configuration for preprocessing

    Returns:
        Tuple of (data_loader, vs_data, ttf_data, freq_data)
    """
    # Create data loader
    data_loader = SeismicDataLoader(
        vs_data_path=data_config["vs_data_path"],
        ttf_data_path=data_config["ttf_data_path"],
        freq_data_path=data_config["freq_data_path"],
    )

    # Load raw data
    vs_data, ttf_data, freq_data = data_loader.load_raw_data()

    # Preprocess data
    if preprocess_config:
        vs_data, ttf_data = data_loader.preprocess_data(**preprocess_config)
    else:
        vs_data, ttf_data = data_loader.preprocess_data()

    return data_loader, vs_data, ttf_data, freq_data
