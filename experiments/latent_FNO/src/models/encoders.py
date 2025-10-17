# encoders.py
"""
Modular encoder implementations for the latent FNO pipeline.
Each encoder takes input data and produces a latent representation.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for all encoders in the latent FNO pipeline.
    """

    def __init__(self, input_dim: int, latent_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Latent tensor of shape (batch_size, latent_dim)
        """
        pass


class MLPEncoder(BaseEncoder):
    """
    Multi-layer perceptron encoder with configurable architecture.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list = [512, 256, 128],
        dropout_rate: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(input_dim, latent_dim)

        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build the network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Final layer to latent dimension
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(activation, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNNEncoder(BaseEncoder):
    """
    Convolutional neural network encoder for sequence data.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        channels: list = [1, 32, 64, 128],
        kernel_sizes: list = [3, 3, 3],
        pool_sizes: list = [2, 2, 2],
        dropout_rate: float = 0.1,
        use_adaptive_pool: bool = True,
        **kwargs,
    ):
        super().__init__(input_dim, latent_dim)

        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.use_adaptive_pool = use_adaptive_pool

        # Build convolutional layers
        conv_layers = []
        for i in range(len(channels) - 1):
            conv_layers.extend(
                [
                    nn.Conv1d(
                        channels[i],
                        channels[i + 1],
                        kernel_sizes[i],
                        padding=kernel_sizes[i] // 2,
                    ),
                    nn.BatchNorm1d(channels[i + 1]),
                    nn.ReLU(),
                    nn.MaxPool1d(pool_sizes[i])
                    if not use_adaptive_pool
                    else nn.Identity(),
                    nn.Dropout(dropout_rate),
                ]
            )

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate the output size after convolutions
        if use_adaptive_pool:
            self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
            conv_output_dim = channels[-1]
        else:
            # Calculate size after pooling operations
            conv_output_dim = self._calculate_conv_output_size()

        # Final projection to latent dimension
        self.projection = nn.Linear(conv_output_dim, latent_dim)

    def _calculate_conv_output_size(self):
        """Calculate the output size after convolutions and pooling."""
        size = self.input_dim
        for pool_size in self.pool_sizes:
            size = size // pool_size
        return self.channels[-1] * size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension if needed: (batch, seq_len) -> (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Apply convolutional layers
        x = self.conv_layers(x)

        # Apply adaptive pooling if enabled
        if self.use_adaptive_pool:
            x = self.adaptive_pool(x)

        # Flatten and project to latent dimension
        x = x.view(x.size(0), -1)
        return self.projection(x)


class TransformerEncoder(BaseEncoder):
    """
    Transformer-based encoder for sequence data.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(input_dim, latent_dim)

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(input_dim, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input: (batch, seq_len) -> (batch, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # Project to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0)

        # Apply transformer
        x = self.transformer(x)

        # Global average pooling and project to latent dimension
        x = x.mean(dim=1)  # (batch, d_model)
        return self.output_projection(x)


class AutoEncoderEncoder(BaseEncoder):
    """
    Encoder that uses a pre-trained autoencoder.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_path: Optional[str] = None,
        freeze_encoder: bool = True,
        **kwargs,
    ):
        super().__init__(input_dim, latent_dim)

        # This would load a pre-trained encoder
        # For now, we'll create a placeholder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        if encoder_path:
            self.load_pretrained_encoder(encoder_path)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def load_pretrained_encoder(self, encoder_path: str):
        """Load pre-trained encoder weights."""
        try:
            state_dict = torch.load(encoder_path, map_location="cpu")
            self.encoder.load_state_dict(state_dict)
            print(f"Loaded pre-trained encoder from {encoder_path}")
        except Exception as e:
            print(f"Warning: Could not load encoder from {encoder_path}: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# Registry for easy encoder creation
ENCODER_REGISTRY = {
    "mlp": MLPEncoder,
    "cnn": CNNEncoder,
    "transformer": TransformerEncoder,
    "autoencoder": AutoEncoderEncoder,
}


def create_encoder(
    encoder_type: str, input_dim: int, latent_dim: int, **kwargs
) -> BaseEncoder:
    """
    Factory function to create encoders.

    Args:
        encoder_type: Type of encoder to create
        input_dim: Input dimension
        latent_dim: Latent dimension
        **kwargs: Additional arguments for the encoder

    Returns:
        Encoder instance
    """
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. Available: {list(ENCODER_REGISTRY.keys())}"
        )

    return ENCODER_REGISTRY[encoder_type](input_dim, latent_dim, **kwargs)
