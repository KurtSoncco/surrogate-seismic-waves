# decoders.py
"""
Modular decoder implementations for the latent FNO pipeline.
Each decoder takes a latent representation and produces output data.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for all decoders in the latent FNO pipeline.
    """

    def __init__(self, latent_dim: int, output_dim: int, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            x: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        pass


class MLPDecoder(BaseDecoder):
    """
    Multi-layer perceptron decoder with configurable architecture.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list = [128, 256, 512],
        dropout_rate: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(latent_dim, output_dim)

        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build the network
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Final layer to output dimension
        layers.append(nn.Linear(prev_dim, output_dim))

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


class CNNDecoder(BaseDecoder):
    """
    Convolutional neural network decoder for sequence generation.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        channels: list = [128, 64, 32, 1],
        kernel_sizes: list = [3, 3, 3],
        upsampling_factors: list = [2, 2, 2],
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(latent_dim, output_dim)

        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.upsampling_factors = upsampling_factors

        # Initial projection from latent to first channel dimension
        self.initial_projection = nn.Linear(latent_dim, channels[0] * (output_dim // 8))

        # Build transposed convolutional layers
        conv_layers = []
        for i in range(len(channels) - 1):
            conv_layers.extend(
                [
                    nn.ConvTranspose1d(
                        channels[i],
                        channels[i + 1],
                        kernel_sizes[i],
                        stride=upsampling_factors[i],
                        padding=kernel_sizes[i] // 2,
                        output_padding=upsampling_factors[i] - 1,
                    ),
                    nn.BatchNorm1d(channels[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )

        self.conv_layers = nn.Sequential(*conv_layers)

        # Final adjustment layer if needed
        if output_dim % 8 != 0:
            self.final_adjustment = nn.Conv1d(
                channels[-1], channels[-1], kernel_size=1, stride=1
            )
        else:
            self.final_adjustment = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project to initial shape
        x = self.initial_projection(x)

        # Reshape for convolutions: (batch, channels*length) -> (batch, channels, length)
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.channels[0], -1)

        # Apply transposed convolutions
        x = self.conv_layers(x)

        # Final adjustment
        x = self.final_adjustment(x)

        # Remove channel dimension if output should be 1D
        if x.size(1) == 1:
            x = x.squeeze(1)

        # Ensure correct output size
        if x.size(-1) != self.output_dim:
            x = F.interpolate(
                x.unsqueeze(1) if x.dim() == 2 else x,
                size=self.output_dim,
                mode="linear",
                align_corners=False,
            )
            if x.dim() == 3:
                x = x.squeeze(1)

        return x


class TransformerDecoder(BaseDecoder):
    """
    Transformer-based decoder for sequence generation.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(latent_dim, output_dim)

        self.d_model = d_model

        # Input projection from latent to model dimension
        self.input_projection = nn.Linear(latent_dim, d_model)

        # Positional encoding for output sequence
        self.pos_encoding = nn.Parameter(torch.randn(output_dim, d_model))

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Project latent to model dimension
        memory = self.input_projection(x).unsqueeze(1)  # (batch, 1, d_model)

        # Create target sequence with positional encoding
        tgt = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply transformer decoder
        output = self.transformer(tgt, memory)

        # Project to final output
        output = self.output_projection(output)

        return output.squeeze(-1)  # (batch, output_dim)


class AutoEncoderDecoder(BaseDecoder):
    """
    Decoder that uses a pre-trained autoencoder.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        decoder_path: Optional[str] = None,
        freeze_decoder: bool = True,
        **kwargs,
    ):
        super().__init__(latent_dim, output_dim)

        # This would load a pre-trained decoder
        # For now, we'll create a placeholder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        if decoder_path:
            self.load_pretrained_decoder(decoder_path)

        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def load_pretrained_decoder(self, decoder_path: str):
        """Load pre-trained decoder weights."""
        try:
            state_dict = torch.load(decoder_path, map_location="cpu", weights_only=False)
            self.decoder.load_state_dict(state_dict)
            print(f"Loaded pre-trained decoder from {decoder_path}")
        except Exception as e:
            print(f"Warning: Could not load decoder from {decoder_path}: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class FNOOperatorDecoder(BaseDecoder):
    """
    FNO-based operator decoder that treats the latent as a frequency representation.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        fno_modes: int = 16,
        fno_width: int = 64,
        num_fno_layers: int = 3,
        fft_length: int = None,
        **kwargs,
    ):
        super().__init__(latent_dim, output_dim)

        self.fno_modes = fno_modes
        self.fno_width = fno_width
        
        # Use a FFT-friendly length if not specified
        if fft_length is None:
            # Find the smallest power of 2 >= output_dim
            fft_length = 2 ** int(np.ceil(np.log2(output_dim)))
        
        self.fft_length = fft_length

        # Project latent to FNO width
        self.latent_projection = nn.Linear(latent_dim, fno_width)

        # Create FNO layers
        from wave_surrogate.models.fno.model import FNO1D

        fno_layers = []
        for _ in range(num_fno_layers):
            fno_layers.append(FNO1D(modes=fno_modes, width=fno_width))

        self.fno_layers = nn.Sequential(*fno_layers)

        # Project from FNO width to output
        self.output_projection = nn.Linear(fno_width, 1)

        # Create frequency grid for FNO
        self.register_buffer(
            "freq_grid", torch.linspace(0, 1, fft_length).unsqueeze(0).unsqueeze(-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Project latent to FNO width
        x = self.latent_projection(x)  # (batch, fno_width)

        # Expand to FFT-friendly sequence: (batch, fno_width) -> (batch, fft_length, fno_width)
        x = x.unsqueeze(1).expand(-1, self.fft_length, -1)

        # Apply FNO layers
        x = self.fno_layers(x)

        # Project to output
        x = self.output_projection(x)

        # Resize to desired output dimension if needed
        if self.fft_length != self.output_dim:
            # x is (batch, fft_length, 1), we need (batch, 1, fft_length) for interpolation
            x = F.interpolate(
                x.transpose(1, 2),  # (batch, fft_length, 1) -> (batch, 1, fft_length)
                size=self.output_dim,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)  # (batch, 1, output_dim) -> (batch, output_dim, 1)

        return x.squeeze(-1)  # (batch, output_dim)


# Registry for easy decoder creation
DECODER_REGISTRY = {
    "mlp": MLPDecoder,
    "cnn": CNNDecoder,
    "transformer": TransformerDecoder,
    "autoencoder": AutoEncoderDecoder,
    "fno_operator": FNOOperatorDecoder,
}


def create_decoder(
    decoder_type: str, latent_dim: int, output_dim: int, **kwargs
) -> BaseDecoder:
    """
    Factory function to create decoders.

    Args:
        decoder_type: Type of decoder to create
        latent_dim: Latent dimension
        output_dim: Output dimension
        **kwargs: Additional arguments for the decoder

    Returns:
        Decoder instance
    """
    if decoder_type not in DECODER_REGISTRY:
        raise ValueError(
            f"Unknown decoder type: {decoder_type}. Available: {list(DECODER_REGISTRY.keys())}"
        )

    return DECODER_REGISTRY[decoder_type](latent_dim, output_dim, **kwargs)
