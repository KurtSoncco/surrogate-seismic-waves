import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import torch


@dataclass
class FNOLatentConfig:
    # OT Encoder parameters
    latent_dim: int = 64  # Fixed latent dimension for OT encoder
    ot_steps: int = 15  # Number of OT optimization steps
    ot_lr: float = 0.1  # Learning rate for OT optimization

    # DAE model parameters (for compatibility)
    Vs_latent_dim: int = 64  # Dimensionality of the latent space
    Vs_dim_encoder: Union[int, List[int]] = field(
        default_factory=lambda: [64, 128, 256, 512]
    )
    Vs_dim_decoder: Union[int, List[int]] = field(
        default_factory=lambda: [512, 256, 128, 64]
    )

    TF_latent_dim: int = 200  # Dimensionality of the latent space
    TF_dim_encoder: Union[int, List[int]] = field(
        default_factory=lambda: [1024, 512, 256, 128]
    )
    TF_dim_decoder: Union[int, List[int]] = field(
        default_factory=lambda: [128, 256, 512, 1024]
    )

    ## Model DAE paths
    Vs_encoder_path: str = "./experiments/dae/Vs_profiles/models/encoder.pth"
    Vs_decoder_path: str = "./experiments/dae/Vs_profiles/models/decoder.pth"
    Vs_scaler_path: str = "./experiments/dae/Vs_profiles/models/vs_scaler.pkl"

    TF_encoder_path: str = "./experiments/dae/TF/models/encoder.pth"
    TF_decoder_path: str = "./experiments/dae/TF/models/decoder.pth"
    TF_scaler_path: str = "./experiments/dae/TF/models/tf_scaler.pkl"

    # FNO model parameters
    modes: int = 16
    width: int = 64
    fno_layers: int = 4
    fno_modes: int = 8  # Must be <= latent_dim / 2
    fno_width: int = 64

    # Training parameters
    batch_size: int = 128
    epochs: int = 1000
    lr: float = 5e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-5
    betas: tuple = (0.6, 0.8)  # Adam optimizer betas
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42  # Random seed for reproducibility
    patience: int = 100  # Patience for learning rate scheduler
    early_stopping_patience: int = 150  # Patience for early stopping
    train_size: float = 0.7  # Proportion of data for training
    val_size: float = 0.15  # Proportion of data for validation
    log_interval: int = 10  # Epoch interval for logging

    ## Unfreeze autoencoders for fine-tuning
    fine_tune_autoencoders: bool = True  # Whether to fine-tune the DAE components

    # Paths
    ## Data
    dataset_path: str = "./data/1D Profiles/Soil_Bedrock"
    vs_data_path: str = os.path.join(dataset_path, "model_arrays_HLC.parquet")
    tf_data_path: str = os.path.join(dataset_path, "TTF_data_1000.parquet")
    freq_data_path: str = os.path.join(dataset_path, "TTF_freq_1000.csv")
    original_dz: float = 5.0  # Original depth interval in meters
    new_dz: float = 5.0  # New depth interval in meters
    input_dim: int = int(155 // new_dz)  # Length of input Vs profiles
    output_dim: int = 1000  # Dimensionality of output TF profiles

    ## Model saving and artifacts
    model_dir: str = "./experiments/dae/FNO_latent/models"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    fno_model_path: str = os.path.join(model_dir, "fno_model.pth")
    # Checkpoints and artifacts
    checkpoint_path: str = os.path.join(model_dir, "fno_best.pth")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    # Results and plots
    results_path: str = "./experiments/dae/FNO_latent/images"
    Path(results_path).mkdir(parents=True, exist_ok=True)

    run_name: str = "FNO_Latent_Experiment"
    use_wandb: bool = True
    wandb_project: str | None = "FNO-Latent"
    # Seed for reproducibility
    seed: int = 42


# Helper functions


def load_frozen_model(model_class, model_path, **kwargs):
    """Helper function to load a model and its weights, then freeze it."""
    model = model_class(**kwargs)
    # Ensure model weights are loaded to the correct device context initially, e.g., 'cpu'
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
