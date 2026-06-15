# dataset.py
import os
import pickle
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import FNOLatentConfig, load_frozen_model

from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.dae.architectures import Encoder

logger = setup_logging()

# Profile type alias for clarity
Profile = Union[List[float], np.ndarray, pd.Series]


class VsDataset(Dataset):
    """
    PyTorch Dataset for Vs profiles, for DAE training.

    This implementation preprocesses all profiles in a vectorized manner upon
    initialization for maximum efficiency during training. It handles NaN
    filling, padding to a uniform length, and upsampling.
    """

    def __init__(
        self,
        vs_profiles: List[Profile],
        original_dz: float = 5.0,
        new_dz: float = 1.0,
    ):
        if not vs_profiles:
            self.vs_profiles = torch.empty((0, 0), dtype=torch.float32)
            self.profile_length = 0
            return

        interp_factor = self._resolve_interp_factor(original_dz, new_dz)
        final_np = self._preprocess_profiles(vs_profiles, interp_factor)
        self.profile_length = final_np.shape[1]
        self.vs_profiles = torch.from_numpy(final_np)

    @staticmethod
    def _resolve_interp_factor(
        original_dz: float, new_dz: float, tol: float = 1e-8
    ) -> int:
        """
        Return integer interpolation factor original_dz/new_dz with tolerance.
        Raises ValueError if not close to an integer or if inputs are invalid.
        """
        if not (new_dz > 0 and original_dz > 0):
            raise ValueError("original_dz and new_dz must be positive numbers.")

        ratio = original_dz / new_dz
        if not np.isclose(ratio, round(ratio), atol=tol):
            raise ValueError(
                f"original_dz/new_dz must be a near-integer ratio (tol={tol}). "
                f"Got ratio={ratio} for original_dz={original_dz}, new_dz={new_dz}"
            )
        factor = int(round(ratio))
        if factor < 1:
            raise ValueError(f"Interpolation factor must be >= 1, but got {factor}.")
        return factor

    @staticmethod
    def _preprocess_profiles(
        vs_profiles: List[Profile], interp_factor: int
    ) -> np.ndarray:
        """
        Handles NaN filling, padding, and upsampling for all profiles.

        - Fills NaNs using forward- and back-fill for robustness.
        - Pads all profiles to the maximum length using the last valid value.
        - Upsamples by repeating each element `interp_factor` times.

        Returns a float32 numpy array of shape (N, final_len).
        """
        non_empty_profiles = [p for p in vs_profiles if p is not None and len(p) > 0]
        if not non_empty_profiles:
            # Return an array with the correct number of samples but zero length
            return np.zeros((len(vs_profiles), 0), dtype=np.float32)

        max_raw_len = max(len(p) for p in non_empty_profiles)
        padded_profiles = np.zeros((len(vs_profiles), max_raw_len), dtype=np.float32)

        for i, p in enumerate(vs_profiles):
            if p is None or len(p) == 0:
                # Leaves the row as zeros, which is a reasonable default for empty data
                continue

            series = pd.Series(p, dtype=np.float32)
            # ffill handles NaNs after a value, bfill handles leading NaNs.
            # fillna(0) is a fallback if the entire profile is NaN.
            filled_series = series.ffill().bfill().fillna(0)
            p_arr = filled_series.to_numpy()

            # Pad profiles shorter than the max length using the edge value
            if len(p_arr) < max_raw_len:
                p_arr = np.pad(p_arr, (0, max_raw_len - len(p_arr)), mode="edge")

            padded_profiles[i, :] = p_arr[:max_raw_len]

        # Upsample via repetition: a fast, vectorized nearest-neighbor upscale
        return np.repeat(padded_profiles, interp_factor, axis=1)

    def __len__(self) -> int:
        return len(self.vs_profiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        profile = self.vs_profiles[idx]
        return profile, profile


class TFDataset(Dataset):
    """
    PyTorch Dataset for TF profiles, for DAE training.
    Assumes profiles are pre-processed and of uniform length.
    """

    def __init__(self, tf_profiles: List[Profile]):
        if not tf_profiles:
            self.tf_profiles = torch.empty((0, 0), dtype=torch.float32)
        else:
            # This conversion is clean and efficient.
            numpy_stack = np.array(tf_profiles, dtype=np.float32)
            self.tf_profiles = torch.from_numpy(numpy_stack)

    def __len__(self) -> int:
        return len(self.tf_profiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        profile = self.tf_profiles[idx]
        return profile, profile


def _batch_encode(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 256,
    device: str = "cpu",
    scaler: Optional[Any] = None,
) -> torch.Tensor:
    """
    Batch-encode a Dataset using the provided encoder model.

    Args:
        model: The encoder model to use.
        dataset: The dataset to encode.
        batch_size: The batch size for encoding.
        device: The device to run the model on ('cpu' or 'cuda').
        scaler: An optional scaler (e.g., from scikit-learn) with a `transform`
                method that operates on NumPy arrays.

    Returns:
        A CPU tensor with stacked encoded outputs.
    """
    if len(dataset) == 0:  # type: ignore
        logger.error("Dataset is empty. Cannot encode.")
        return torch.empty((0, 0), dtype=torch.float32)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_z = []
    device_torch = torch.device(device)
    model.to(device_torch)
    model.eval()

    with torch.no_grad():
        for batch_data, _ in loader:
            if scaler is not None:
                try:
                    # --- MODIFICATION FOR GLOBAL SCALER ---
                    original_shape = batch_data.shape
                    batch_numpy = batch_data.cpu().numpy()

                    # Reshape for scaler: (batch, features) -> (batch * features, 1)
                    reshaped_numpy = batch_numpy.reshape(-1, 1)
                    scaled_reshaped = scaler.transform(reshaped_numpy)

                    # Reshape back to original batch shape
                    scaled_numpy = scaled_reshaped.reshape(original_shape)
                    batch_data = torch.from_numpy(scaled_numpy)
                    # --- END MODIFICATION ---
                except (ValueError, AttributeError) as e:
                    logger.debug(f"Scaler transform failed: {e}. Passing raw tensor.")

            batch_tensor = batch_data.to(device_torch, dtype=torch.float32)
            z = model(batch_tensor)
            all_z.append(z.cpu())

    if not all_z:
        logger.warning("Data encoding resulted in an empty list of tensors.")
        return torch.empty((0, 0), dtype=torch.float32)

    return torch.cat(all_z, dim=0)


class _BaseProfileDataset(Dataset):
    """
    A private base class to encapsulate common data/scaler loading logic.
    """

    @staticmethod
    def _load_dataframe(path: str, name: str) -> pd.DataFrame:
        """Loads a parquet dataframe and validates its content."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} data path not found: {path}")
        df = pd.read_parquet(path)
        if "model_data" not in df.columns:
            raise KeyError(f"Expected 'model_data' column in {name} file: {path}")
        return df

    @staticmethod
    def _load_scaler(path: Optional[str], name: str) -> Optional[Any]:
        """Loads a pickled scaler object from path if it exists."""
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(
                    f"Failed to load {name} scaler from {path}: {e}. "
                    "Continuing without scaler."
                )
        else:
            if path:
                logger.debug(f"{name} scaler not found at path: {path}")
            else:
                logger.debug(f"{name} scaler path not provided. Skipping.")
        return None


class LatentDataset(_BaseProfileDataset):
    """
    Dataset that encodes raw Vs and TF data into latent vectors (z_vs, z_tf)
    using frozen encoders. Encoding is performed once at initialization.
    """

    def __init__(self, config: FNOLatentConfig):
        super().__init__()
        self.config = config
        logger.info("Initializing LatentDataset...")

        # Load models, scalers, and data
        self.vs_encoder, self.tf_encoder = self._load_encoders()
        self.vs_scaler = self._load_scaler(config.Vs_scaler_path, "Vs")
        self.tf_scaler = self._load_scaler(config.TF_scaler_path, "TF")

        vs_profiles_list, tf_profiles_list = self._load_profile_lists()
        self._validate_profile_counts(vs_profiles_list, tf_profiles_list)

        # Prepare intermediate datasets for encoding
        vs_dataset = VsDataset(vs_profiles_list, config.original_dz, config.new_dz)
        tf_dataset = TFDataset(tf_profiles_list)

        # Encode data into latent representations
        self.z_vs, self.z_tf = self._encode_datasets(vs_dataset, tf_dataset)

        if len(self.z_vs) != len(self.z_tf):
            raise ValueError(
                f"Post-encoding latent sets have mismatched lengths: "
                f"{len(self.z_vs)} vs {len(self.z_tf)}"
            )

        logger.info("LatentDataset initialization complete.")

    def _load_encoders(self) -> Tuple[nn.Module, nn.Module]:
        logger.info("Loading pre-trained encoders...")
        vs_encoder = load_frozen_model(
            Encoder,
            self.config.Vs_encoder_path,
            input_dim=self.config.input_dim,
            hidden_dim=self.config.Vs_dim_encoder,
            latent_dim=self.config.Vs_latent_dim,
        )
        tf_encoder = load_frozen_model(
            Encoder,
            self.config.TF_encoder_path,
            input_dim=self.config.output_dim,
            hidden_dim=self.config.TF_dim_encoder,
            latent_dim=self.config.TF_latent_dim,
        )
        return vs_encoder, tf_encoder

    def _load_profile_lists(self) -> Tuple[List[Profile], List[Profile]]:
        logger.info("Loading and preprocessing raw data...")
        vs_df = self._load_dataframe(self.config.vs_data_path, "Vs")
        tf_df = self._load_dataframe(self.config.tf_data_path, "TF")
        return vs_df["model_data"].tolist(), tf_df["model_data"].tolist()

    @staticmethod
    def _validate_profile_counts(list1: list, list2: list):
        if len(list1) != len(list2):
            raise ValueError(
                "Vs and TF datasets must be the same length and aligned row-wise. "
                f"Got {len(list1)} Vs profiles and {len(list2)} TF profiles."
            )
        logger.info(f"Loaded {len(list1)} entries from datasets.")

    def _encode_datasets(
        self, vs_dataset: Dataset, tf_dataset: Dataset
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info("Encoding data into latent spaces (batched)...")
        batch_size = self.config.batch_size
        device = self.config.device

        z_vs = _batch_encode(
            self.vs_encoder, vs_dataset, batch_size, device, self.vs_scaler
        )
        logger.info(f"Encoded Vs -> z_vs shape: {tuple(z_vs.shape)}")

        z_tf = _batch_encode(
            self.tf_encoder, tf_dataset, batch_size, device, self.tf_scaler
        )
        logger.info(f"Encoded TF -> z_tf shape: {tuple(z_tf.shape)}")
        return z_vs, z_tf

    def __len__(self) -> int:
        return len(self.z_vs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.z_vs[idx], self.z_tf[idx]


class VsTFDataset(_BaseProfileDataset):
    """
    Dataset that provides raw (Vs, TF) profile pairs, fully preprocessed and
    scaled at initialization for maximum performance during training/evaluation.
    """

    def __init__(self, config: FNOLatentConfig):
        super().__init__()
        logger.info("Initializing VsTFDataset with raw data...")

        # 1. Load raw data from files
        vs_df = self._load_dataframe(config.vs_data_path, "Vs")
        tf_df = self._load_dataframe(config.tf_data_path, "TF")

        vs_profiles_list = vs_df["model_data"].tolist()
        tf_profiles_list = tf_df["model_data"].tolist()

        if len(vs_profiles_list) != len(tf_profiles_list):
            raise ValueError("Vs and TF datasets must have the same length.")

        # 2. Use existing Dataset classes for robust preprocessing (composition)
        vs_preprocessor = VsDataset(vs_profiles_list, config.original_dz, config.new_dz)
        tf_preprocessor = TFDataset(tf_profiles_list)

        self.vs_profiles = vs_preprocessor.vs_profiles
        self.tf_profiles = tf_preprocessor.tf_profiles

        # 3. Load and apply scalers once during initialization
        self.vs_scaler = self._load_scaler(config.Vs_scaler_path, "Vs")
        self.tf_scaler = self._load_scaler(config.TF_scaler_path, "TF")

        self.vs_profiles = self._apply_scaler(self.vs_profiles, self.vs_scaler, "Vs")
        self.tf_profiles = self._apply_scaler(self.tf_profiles, self.tf_scaler, "TF")

    def __len__(self) -> int:
        return len(self.vs_profiles)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of (Vs, TF) tensors. This is a highly efficient
        indexing operation as all preprocessing and scaling is done upfront.
        """
        return self.vs_profiles[idx], self.tf_profiles[idx]


class SoilProfileDataset(_BaseProfileDataset):
    """
    Dataset for soil profile data that provides both material properties (Vs, Vp, Rho)
    and their corresponding depth coordinates for OT Encoder training.
    """

    def __init__(self, config: FNOLatentConfig):
        super().__init__()
        logger.info("Initializing SoilProfileDataset for OT Encoder...")

        # 1. Load raw data from files
        vs_df = self._load_dataframe(config.vs_data_path, "Vs")
        tf_df = self._load_dataframe(config.tf_data_path, "TF")

        vs_profiles_list = vs_df["model_data"].tolist()
        tf_profiles_list = tf_df["model_data"].tolist()

        if len(vs_profiles_list) != len(tf_profiles_list):
            raise ValueError("Vs and TF datasets must have the same length.")

        # 2. Process soil profiles to create multi-channel data (Vs, Vp, Rho)
        self.soil_profiles, self.depth_grids, self.tf_profiles = (
            self._process_soil_profiles(vs_profiles_list, tf_profiles_list, config)
        )

        # 3. Load and apply scalers
        self.vs_scaler = self._load_scaler(config.Vs_scaler_path, "Vs")
        self.tf_scaler = self._load_scaler(config.TF_scaler_path, "TF")

        # Apply scaling to soil profiles and TF
        self.soil_profiles = self._apply_scaler(
            self.soil_profiles, self.vs_scaler, "Vs"
        )
        self.tf_profiles = self._apply_scaler(self.tf_profiles, self.tf_scaler, "TF")

        logger.info(f"Processed {len(self.soil_profiles)} soil profiles with shapes:")
        logger.info(f"  Soil profiles: {self.soil_profiles.shape}")
        logger.info(f"  Depth grids: {self.depth_grids.shape}")
        logger.info(f"  TF profiles: {self.tf_profiles.shape}")

    def _process_soil_profiles(self, vs_profiles_list, tf_profiles_list, config):
        """
        Process soil profiles to create multi-channel data and depth grids.

        For now, we'll use Vs data and create synthetic Vp and Rho based on empirical relationships:
        - Vp ≈ 1.73 * Vs (for typical soil conditions)
        - Rho ≈ 1.8 + 0.1 * (Vs / 1000) (density relationship)
        """
        # Use existing VsDataset for preprocessing
        vs_preprocessor = VsDataset(vs_profiles_list, config.original_dz, config.new_dz)
        tf_preprocessor = TFDataset(tf_profiles_list)

        vs_data = vs_preprocessor.vs_profiles  # Shape: (N, profile_length)
        tf_data = tf_preprocessor.tf_profiles  # Shape: (N, tf_length)

        # Create synthetic multi-channel soil profiles
        batch_size, profile_length = vs_data.shape

        # Initialize multi-channel tensor: (N, 3, profile_length)
        soil_profiles = torch.zeros(batch_size, 3, profile_length, dtype=torch.float32)

        # Channel 0: Vs (shear wave velocity)
        soil_profiles[:, 0, :] = vs_data

        # Channel 1: Vp (compressional wave velocity) - empirical relationship
        soil_profiles[:, 1, :] = 1.73 * vs_data

        # Channel 2: Rho (density) - empirical relationship
        soil_profiles[:, 2, :] = 1.8 + 0.1 * (vs_data / 1000.0)

        # Create depth grids for each profile
        # Assuming uniform spacing from 0 to max_depth
        max_depth = profile_length * config.new_dz  # meters
        depth_grids = torch.linspace(0, max_depth, profile_length, dtype=torch.float32)
        depth_grids = depth_grids.unsqueeze(0).repeat(
            batch_size, 1
        )  # (N, profile_length)

        return soil_profiles, depth_grids, tf_data

    def __len__(self) -> int:
        return len(self.soil_profiles)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns a tuple of ((soil_profile, depth_grid), tf_profile).

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing:
            - soil_profile: (3, profile_length) tensor with Vs, Vp, Rho
            - depth_grid: (profile_length,) tensor with depth coordinates
            - tf_profile: (tf_length,) tensor with transfer function
        """
        soil_profile = self.soil_profiles[idx]  # (3, profile_length)
        depth_grid = self.depth_grids[idx]  # (profile_length,)
        tf_profile = self.tf_profiles[idx]  # (tf_length,)

        return (soil_profile, depth_grid), tf_profile

    @staticmethod
    def _apply_scaler(
        data: torch.Tensor, scaler: Optional[Any], name: str
    ) -> torch.Tensor:
        """
        Applies a global scaler to the data tensor by reshaping the data.
        """
        if scaler and data.numel() > 0:
            try:
                # --- CORRECTED LOGIC FOR GLOBAL SCALER ---
                original_shape = data.shape
                numpy_data = data.numpy()

                # 1. Flatten the data and "create a dimension" to make it a column vector
                # Shape changes from (n_samples, n_features) to (n_samples * n_features, 1)
                reshaped_data = numpy_data.reshape(-1, 1)

                # 2. Apply the scaler transform on the single column
                scaled_reshaped_data = scaler.transform(reshaped_data)

                # 3. Reshape the scaled data back to its original shape
                scaled_data = scaled_reshaped_data.reshape(original_shape)

                return torch.from_numpy(scaled_data).float()
                # --- END CORRECTED LOGIC ---
            except (ValueError, AttributeError) as e:
                logger.warning(
                    f"{name} global scaling failed: {e}. Check if the scaler "
                    "was fit on flattened data. Using unscaled data."
                )
        return data.float()  # Ensure float type even if not scaled


if __name__ == "__main__":
    # Quick smoke test - requires a valid config setup
    logger.info("Running smoke test...")
    try:
        # Assumes a default FNOLatentConfig() can be instantiated
        # and points to valid (even if empty) data files.
        config = FNOLatentConfig()
        ds = VsTFDataset(config)
        logger.info(f"VsTFDataset size: {len(ds)}")
        if len(ds) > 0:
            vs_sample, tf_sample = ds[0]
            logger.info(
                f"Sample Vs shape: {vs_sample.shape}, TF shape: {tf_sample.shape}"
            )
            logger.info(
                f"Sample Vs dtype: {vs_sample.dtype}, TF dtype: {tf_sample.dtype}"
            )

        lat_ds = LatentDataset(config)
        logger.info(f"\nLatentDataset size: {len(lat_ds)}")
        if len(lat_ds) > 0:
            z_vs, z_tf = lat_ds[0]
            logger.info(f"Sample z_vs shape: {z_vs.shape}, z_tf shape: {z_tf.shape}")

    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.error(
            f"\nSmoke test failed. Please ensure config paths are valid. Error: {e}"
        )
    except Exception as e:
        logger.error(f"\nUnexpected error during smoke test: {e}")
