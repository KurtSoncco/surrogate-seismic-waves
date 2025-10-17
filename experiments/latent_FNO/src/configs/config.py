# config.py
"""
Configuration settings for the latent FNO experiments.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass
class LatentFNOConfig:
    """Configuration for latent FNO experiments."""

    # Project paths
    project_root: Path = Path(__file__).parent.parent.parent.parent.parent
    data_dir: Path = project_root / "data" / "1D Profiles" / "TF_HLC"
    model_save_dir: Path = project_root / "experiments" / "latent_FNO" / "models"
    results_dir: Path = project_root / "experiments" / "latent_FNO" / "results"

    # Data configuration
    vs_data_path: str = str(data_dir / "Vs_values_1000.pt")
    ttf_data_path: str = str(data_dir / "TTF_data_1000.pt")
    freq_data_path: str = str(data_dir / "TTF_freq_1000.csv")

    # Model dimensions
    input_dim: int = 29  # Vs profile length
    output_dim: int = 1000  # Transfer function output size
    latent_dim: int = 128  # Latent space dimension

    # Model architecture
    encoder_type: str = "mlp"  # Options: mlp, cnn, transformer, autoencoder
    decoder_type: str = (
        "mlp"  # Options: mlp, cnn, transformer, autoencoder, fno_operator
    )
    fno_processor_type: str = (
        "simple"  # Options: simple, sequence, multiscale, adaptive, conditional
    )

    # Encoder configuration
    encoder_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_dims": [512, 256, 128],
            "dropout_rate": 0.1,
            "activation": "relu",
        }
    )

    # Decoder configuration
    decoder_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_dims": [128, 256, 512],
            "dropout_rate": 0.1,
            "activation": "relu",
        }
    )

    # FNO processor configuration
    fno_processor_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "fno_modes": 16,
            "fno_width": 64,
            "num_fno_layers": 3,
            "dropout_rate": 0.1,
            "use_residual": True,
        }
    )

    # Training configuration
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 1000
    patience: int = 100
    early_stopping_patience: int = 200

    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Device and reproducibility
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Logging and monitoring
    log_interval: int = 10
    save_interval: int = 50
    use_wandb: bool = False
    wandb_project: str = "latent_fno"
    wandb_run_name: str = "latent-fno-baseline"

    # Experiment specific
    experiment_name: str = "latent_fno_baseline"
    description: str = "Baseline latent FNO experiment"

    def __post_init__(self):
        """Create directories after initialization."""
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# Predefined configurations for different experiments
CONFIGS = {
    "baseline": LatentFNOConfig(),
    "cnn_encoder": LatentFNOConfig(
        encoder_type="cnn",
        encoder_config={
            "channels": [1, 32, 64, 128],
            "kernel_sizes": [3, 3, 3],
            "pool_sizes": [2, 2, 2],
            "dropout_rate": 0.1,
            "use_adaptive_pool": True,
        },
    ),
    "transformer_encoder": LatentFNOConfig(
        encoder_type="transformer",
        encoder_config={
            "d_model": 128,
            "nhead": 8,
            "num_layers": 4,
            "dropout_rate": 0.1,
        },
    ),
    "fno_decoder": LatentFNOConfig(
        decoder_type="fno_operator",
        decoder_config={"fno_modes": 16, "fno_width": 64, "num_fno_layers": 3},
    ),
    "sequence_fno": LatentFNOConfig(
        fno_processor_type="sequence",
        fno_processor_config={
            "sequence_length": 64,
            "fno_modes": 16,
            "fno_width": 64,
            "num_fno_layers": 3,
            "dropout_rate": 0.1,
            "use_residual": True,
        },
    ),
    "multiscale_fno": LatentFNOConfig(
        fno_processor_type="multiscale",
        fno_processor_config={
            "scales": [1, 2, 4],
            "fno_modes": 16,
            "fno_width": 64,
            "num_fno_layers": 2,
            "dropout_rate": 0.1,
            "use_residual": True,
        },
    ),
    "adaptive_fno": LatentFNOConfig(
        fno_processor_type="adaptive",
        fno_processor_config={
            "fno_modes": 16,
            "fno_width": 64,
            "num_fno_layers": 3,
            "dropout_rate": 0.1,
            "use_residual": True,
        },
    ),
    "high_latent_dim": LatentFNOConfig(
        latent_dim=256,
        fno_processor_config={
            "fno_modes": 32,
            "fno_width": 128,
            "num_fno_layers": 4,
            "dropout_rate": 0.1,
            "use_residual": True,
        },
    ),
    "low_latent_dim": LatentFNOConfig(
        latent_dim=64,
        fno_processor_config={
            "fno_modes": 8,
            "fno_width": 32,
            "num_fno_layers": 2,
            "dropout_rate": 0.1,
            "use_residual": True,
        },
    ),
}


def get_config(config_name: str = "baseline") -> LatentFNOConfig:
    """Get a predefined configuration."""
    if config_name not in CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}"
        )

    return CONFIGS[config_name]


def create_custom_config(**kwargs) -> LatentFNOConfig:
    """Create a custom configuration with specified parameters."""
    config = LatentFNOConfig()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

    return config
