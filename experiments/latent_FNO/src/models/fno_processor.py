# fno_processor.py
"""
FNO processor implementations for latent space operations.
This is the core component that operates in the latent space using Fourier Neural Operators.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn

from wave_surrogate.models.fno.model import FNO1D


class BaseFNOProcessor(nn.Module, ABC):
    """
    Abstract base class for FNO processors in the latent space.
    """

    def __init__(self, latent_dim: int, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FNO processor.

        Args:
            x: Input latent tensor of shape (batch_size, latent_dim)

        Returns:
            Processed latent tensor of shape (batch_size, latent_dim)
        """
        pass


class SimpleFNOProcessor(BaseFNOProcessor):
    """
    Simple FNO processor that treats the latent vector as a 1D sequence.
    """

    def __init__(
        self,
        latent_dim: int,
        fno_modes: int = 16,
        fno_width: int = 64,
        num_fno_layers: int = 3,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(latent_dim)

        self.fno_modes = fno_modes
        self.fno_width = fno_width

        # Project latent to FNO width
        self.input_projection = nn.Linear(latent_dim, fno_width)

        # Create FNO layers
        fno_layers = []
        for i in range(num_fno_layers):
            fno_layers.append(FNO1D(modes=fno_modes, width=fno_width))
            if i < num_fno_layers - 1:  # Add dropout between layers except last
                fno_layers.append(nn.Dropout(dropout_rate))

        self.fno_layers = nn.Sequential(*fno_layers)

        # Project back to latent dimension
        self.output_projection = nn.Linear(fno_width, latent_dim)

        # Residual connection
        self.use_residual = kwargs.get("use_residual", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for residual connection
        residual = x

        # Project to FNO width
        x = self.input_projection(x)  # (batch, fno_width)

        # Reshape for FNO: (batch, fno_width) -> (batch, 1, fno_width)
        # We treat the latent as a 1D sequence of length 1
        x = x.unsqueeze(1)

        # Apply FNO layers
        x = self.fno_layers(x)

        # Reshape back: (batch, 1, fno_width) -> (batch, fno_width)
        x = x.squeeze(1)

        # Project back to latent dimension
        x = self.output_projection(x)

        # Add residual connection if enabled
        if self.use_residual:
            x = x + residual

        return x


class SequenceFNOProcessor(BaseFNOProcessor):
    """
    FNO processor that treats the latent vector as a sequence and processes it.
    """

    def __init__(
        self,
        latent_dim: int,
        sequence_length: Optional[int] = None,
        fno_modes: int = 16,
        fno_width: int = 64,
        num_fno_layers: int = 3,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(latent_dim)

        # If sequence_length is not provided, use latent_dim
        self.sequence_length = sequence_length or latent_dim
        self.fno_modes = fno_modes
        self.fno_width = fno_width

        # Project latent to sequence of FNO width
        self.input_projection = nn.Linear(latent_dim, self.sequence_length * fno_width)

        # Create FNO layers
        fno_layers = []
        for i in range(num_fno_layers):
            fno_layers.append(FNO1D(modes=fno_modes, width=fno_width))
            if i < num_fno_layers - 1:  # Add dropout between layers except last
                fno_layers.append(nn.Dropout(dropout_rate))

        self.fno_layers = nn.Sequential(*fno_layers)

        # Project back to latent dimension
        self.output_projection = nn.Linear(self.sequence_length * fno_width, latent_dim)

        # Residual connection
        self.use_residual = kwargs.get("use_residual", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for residual connection
        residual = x

        # Project to sequence of FNO width
        x = self.input_projection(x)  # (batch, sequence_length * fno_width)

        # Reshape for FNO: (batch, sequence_length * fno_width) -> (batch, sequence_length, fno_width)
        x = x.reshape(x.size(0), self.sequence_length, self.fno_width)

        # Apply FNO layers
        x = self.fno_layers(x)

        # Reshape back: (batch, sequence_length, fno_width) -> (batch, sequence_length * fno_width)
        x = x.reshape(x.size(0), -1)

        # Project back to latent dimension
        x = self.output_projection(x)

        # Add residual connection if enabled
        if self.use_residual:
            x = x + residual

        return x


class MultiScaleFNOProcessor(BaseFNOProcessor):
    """
    Multi-scale FNO processor that processes the latent at different scales.
    """

    def __init__(
        self,
        latent_dim: int,
        scales: List[int] = [1, 2, 4],
        fno_modes: int = 16,
        fno_width: int = 64,
        num_fno_layers: int = 2,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(latent_dim)

        self.scales = scales
        self.fno_modes = fno_modes
        self.fno_width = fno_width

        # Create processors for each scale
        self.scale_processors = nn.ModuleList()
        self.scale_projections = nn.ModuleList()

        for scale in scales:
            # Project to FNO width for this scale
            projection = nn.Linear(latent_dim, fno_width)
            self.scale_projections.append(projection)

            # Create FNO layers for this scale
            fno_layers = []
            for i in range(num_fno_layers):
                fno_layers.append(FNO1D(modes=fno_modes, width=fno_width))
                if i < num_fno_layers - 1:
                    fno_layers.append(nn.Dropout(dropout_rate))

            processor = nn.Sequential(*fno_layers)
            self.scale_processors.append(processor)

        # Fusion layer to combine multi-scale features
        self.fusion = nn.Linear(fno_width * len(scales), latent_dim)

        # Residual connection
        self.use_residual = kwargs.get("use_residual", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for residual connection
        residual = x

        scale_features = []

        for i, scale in enumerate(self.scales):
            # Project to FNO width
            scale_x = self.scale_projections[i](x)  # (batch, fno_width)

            # Reshape for FNO based on scale
            if scale == 1:
                # Treat as single point
                scale_x = scale_x.unsqueeze(1)  # (batch, 1, fno_width)
            else:
                # Repeat to create sequence
                scale_x = scale_x.unsqueeze(1).repeat(
                    1, scale, 1
                )  # (batch, scale, fno_width)

            # Apply FNO processing
            scale_x = self.scale_processors[i](scale_x)

            # Global average pooling to get fixed-size feature
            scale_x = scale_x.mean(dim=1)  # (batch, fno_width)

            scale_features.append(scale_x)

        # Concatenate multi-scale features
        x = torch.cat(scale_features, dim=1)  # (batch, fno_width * num_scales)

        # Fusion
        x = self.fusion(x)

        # Add residual connection if enabled
        if self.use_residual:
            x = x + residual

        return x


class AdaptiveFNOProcessor(BaseFNOProcessor):
    """
    Adaptive FNO processor that learns the optimal processing strategy.
    This processor uses attention mechanisms to adaptively focus on different parts of the latent space.

    Args:
        latent_dim: Dimension of the latent space
        fno_modes: Number of Fourier modes to use
        fno_width: Width of the FNO layers
        num_fno_layers: Number of FNO layers
        dropout_rate: Dropout rate between FNO layers
        use_residual: Whether to use residual connections

    **kwargs: Additional arguments
    """

    def __init__(
        self,
        latent_dim: int,
        fno_modes: int = 16,
        fno_width: int = 64,
        num_fno_layers: int = 3,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(latent_dim)

        self.fno_modes = fno_modes
        self.fno_width = fno_width

        # Attention mechanism to adapt processing
        self.attention = nn.MultiheadAttention(
            embed_dim=fno_width, num_heads=8, dropout=dropout_rate, batch_first=True
        )

        # Project latent to FNO width
        self.input_projection = nn.Linear(latent_dim, fno_width)

        # Create FNO layers
        fno_layers = []
        for i in range(num_fno_layers):
            fno_layers.append(FNO1D(modes=fno_modes, width=fno_width))
            if i < num_fno_layers - 1:
                fno_layers.append(nn.Dropout(dropout_rate))

        self.fno_layers = nn.Sequential(*fno_layers)

        # Project back to latent dimension
        self.output_projection = nn.Linear(fno_width, latent_dim)

        # Residual connection
        self.use_residual = kwargs.get("use_residual", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for residual connection
        residual = x

        # Project to FNO width
        x = self.input_projection(x)  # (batch, fno_width)

        # Create query, key, value for attention
        # We'll use the input as both query and key/value
        x_seq = x.unsqueeze(1)  # (batch, 1, fno_width)

        # Apply attention
        attended_x, _ = self.attention(x_seq, x_seq, x_seq)
        x = attended_x.squeeze(1)  # (batch, fno_width)

        # Reshape for FNO: (batch, fno_width) -> (batch, 1, fno_width)
        x = x.unsqueeze(1)

        # Apply FNO layers
        x = self.fno_layers(x)

        # Reshape back: (batch, 1, fno_width) -> (batch, fno_width)
        x = x.squeeze(1)

        # Project back to latent dimension
        x = self.output_projection(x)

        # Add residual connection if enabled
        if self.use_residual:
            x = x + residual

        return x


class ConditionalFNOProcessor(BaseFNOProcessor):
    """
    Conditional FNO processor that adapts based on input conditions.
    This processor takes an additional condition vector to modulate the FNO processing.

    Args:
        latent_dim: Dimension of the latent space
        condition_dim: Dimension of the condition vector
        fno_modes: Number of Fourier modes to use
        fno_width: Width of the FNO layers
        num_fno_layers: Number of FNO layers
        dropout_rate: Dropout rate between FNO layers
        use_residual: Whether to use residual connections

    **kwargs: Additional arguments
    """

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        fno_modes: int = 16,
        fno_width: int = 64,
        num_fno_layers: int = 3,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(latent_dim)

        self.condition_dim = condition_dim
        self.fno_modes = fno_modes
        self.fno_width = fno_width

        # Condition processing
        self.condition_processor = nn.Sequential(
            nn.Linear(condition_dim, fno_width),
            nn.ReLU(),
            nn.Linear(fno_width, fno_width),
        )

        # Project latent to FNO width
        self.input_projection = nn.Linear(latent_dim, fno_width)

        # Create FNO layers with conditional modulation
        self.fno_layers = nn.ModuleList()
        for i in range(num_fno_layers):
            fno_layer = FNO1D(modes=fno_modes, width=fno_width)
            self.fno_layers.append(fno_layer)

            # Conditional modulation layers
            modulation = nn.Sequential(nn.Linear(fno_width, fno_width), nn.Sigmoid())
            self.fno_layers.append(modulation)

        # Project back to latent dimension
        self.output_projection = nn.Linear(fno_width, latent_dim)

        # Residual connection
        self.use_residual = kwargs.get("use_residual", True)

    def forward(
        self, x: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Store input for residual connection
        residual = x

        # Process condition
        if condition is not None:
            condition_features = self.condition_processor(condition)
        else:
            condition_features = torch.zeros(x.size(0), self.fno_width, device=x.device)

        # Project to FNO width
        x = self.input_projection(x)  # (batch, fno_width)

        # Reshape for FNO: (batch, fno_width) -> (batch, 1, fno_width)
        x = x.unsqueeze(1)

        # Apply FNO layers with conditional modulation
        for i in range(0, len(self.fno_layers), 2):
            fno_layer = self.fno_layers[i]
            modulation = self.fno_layers[i + 1]

            # Apply FNO
            x = fno_layer(x)

            # Apply conditional modulation
            modulation_weights = modulation(condition_features).unsqueeze(1)
            x = x * modulation_weights

        # Reshape back: (batch, 1, fno_width) -> (batch, fno_width)
        x = x.squeeze(1)

        # Project back to latent dimension
        x = self.output_projection(x)

        # Add residual connection if enabled
        if self.use_residual:
            x = x + residual

        return x


# Registry for easy FNO processor creation
FNO_PROCESSOR_REGISTRY = {
    "simple": SimpleFNOProcessor,
    "sequence": SequenceFNOProcessor,
    "multiscale": MultiScaleFNOProcessor,
    "adaptive": AdaptiveFNOProcessor,
    "conditional": ConditionalFNOProcessor,
}


def create_fno_processor(
    processor_type: str, latent_dim: int, **kwargs
) -> BaseFNOProcessor:
    """
    Factory function to create FNO processors.

    Args:
        processor_type: Type of FNO processor to create
        latent_dim: Latent dimension
        **kwargs: Additional arguments for the processor

    Returns:
        FNO processor instance
    """
    if processor_type not in FNO_PROCESSOR_REGISTRY:
        raise ValueError(
            f"Unknown FNO processor type: {processor_type}. Available: {list(FNO_PROCESSOR_REGISTRY.keys())}"
        )

    return FNO_PROCESSOR_REGISTRY[processor_type](latent_dim, **kwargs)
