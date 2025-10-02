from typing import List, Optional

import torch
import torch.nn as nn

from wave_surrogate.models.fno.model import OperatorDecoder


class TowerEncoder(nn.Module):
    """
    A flexible CNN-based encoder tower for processing sequences of varying lengths.

    This encoder can be configured for different input sequence lengths and
    complexity requirements through the channels and pooling strategy.

    Args:
        channels (List[int]): List of channel sizes for convolutional layers.
                             First element should be the input channels.
        latent_dim (int): Dimension of the output latent representation.
        kernel_size (int): Kernel size for convolutional layers.
        pool_size (int): Pooling size for MaxPool1d layers.
        use_adaptive_pool (bool): If True, uses AdaptiveAvgPool1d at the end
                                  for fixed-size output regardless of input length.

    Input Shape:
        (batch_size, channels[0], sequence_length)

    Output Shape:
        (batch_size, latent_dim)
    """

    def __init__(
        self,
        channels: List[int],
        latent_dim: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        use_adaptive_pool: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not channels or len(channels) < 2:
            raise ValueError(
                "channels must have at least 2 elements (input and output)"
            )

        # Build convolutional tower
        layers = []
        for i in range(len(channels) - 1):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_size,
                        padding="same",
                    ),
                    # nn.BatchNorm1d(channels[i + 1]),  # Added batch norm for stability
                    nn.GroupNorm(
                        num_groups=max(1, min(32, channels[i + 1]) // 1),
                        num_channels=channels[i + 1],
                    ),
                    nn.ReLU(),
                    nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
                    nn.MaxPool1d(pool_size),
                ]
            )

        # Optional adaptive pooling for variable-length inputs
        if use_adaptive_pool:
            layers.append(nn.AdaptiveAvgPool1d(1))

        self.conv_tower = nn.Sequential(*layers)

        # Lazy linear layer to handle varying flattened sizes
        self.fc = nn.LazyLinear(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the tower encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)

        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        x = self.conv_tower(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class FusionHead(nn.Module):
    """
    Multi-layer perceptron for fusing latent representations.

    Args:
        input_dim (int): Combined dimension of input latent vectors.
        hidden_dims (List[int]): List of hidden layer dimensions.
        output_dim (int): Output dimension (e.g., residual sequence length).
        dropout (float): Dropout probability for regularization.
        use_residual (bool): If True, adds residual connection when input_dim == output_dim.

    Input Shape:
        (batch_size, input_dim)

    Output Shape:
        (batch_size, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
        use_residual: bool = False,
    ):
        super().__init__()

        self.use_residual = use_residual and (input_dim == output_dim)

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fusion head.

        Args:
            x: Combined latent tensor of shape (batch_size, input_dim)

        Returns:
            Fused output of shape (batch_size, output_dim)
        """
        out = self.mlp(x)
        if self.use_residual:
            out = out + x
        return out


class SpecBoostModelB(nn.Module):
    """
    Dual-tower fusion model for SpecBoost Stage 2 residual prediction.

    This model processes two inputs through separate encoding towers:
    - Tower 1: Processes the original Vs profile (short sequence)
    - Tower 2: Processes Model A's prediction (long sequence)

    The latent representations are concatenated and passed through a fusion
    head to predict the residual correction.

    Architecture:
        Input 1 (Vs profile) --> Tower 1 --> Latent 1 ---|
                                                          |--> Concat --> Fusion Head --> Residual
        Input 2 (Model A pred) -> Tower 2 --> Latent 2 ---|

    Args:
        vs_encoder_channels (List[int]): Channel configuration for Vs tower.
        pred_encoder_channels (List[int]): Channel configuration for prediction tower.
        latent_dim (int): Dimension of each tower's latent output.
        output_size (int): Length of the output residual sequence.
        fusion_hidden_dims (List[int]): Hidden dimensions for fusion MLP.
        dropout (float): Dropout probability for regularization.
        use_batch_norm (bool): Whether to use batch normalization in towers.

    Input:
        vs_profile: Shape (batch_size, vs_length)
        model_a_pred: Shape (batch_size, pred_length)

    Output:
        residual_pred: Shape (batch_size, output_size)
    """

    def __init__(
        self,
        vs_encoder_channels: Optional[List[int]] = None,
        pred_encoder_channels: Optional[List[int]] = None,
        latent_dim: int = 128,
        output_size: int = 1000,
        fusion_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        vs_kernel_size: int = 3,
        pred_kernel_size: int = 5,
    ):
        super().__init__()

        # Default configurations
        if vs_encoder_channels is None:
            vs_encoder_channels = [1, 32, 64, 128]
        if pred_encoder_channels is None:
            pred_encoder_channels = [1, 64, 128, 256]
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [256, 512]

        self.latent_dim = latent_dim
        self.output_size = output_size

        # --- Tower 1: Vs Profile Encoder ---
        # For short sequences (e.g., 29 elements)
        self.vs_tower = TowerEncoder(
            channels=vs_encoder_channels,
            latent_dim=latent_dim,
            kernel_size=vs_kernel_size,
            pool_size=2,
            use_adaptive_pool=False,
            dropout=dropout,
        )

        # --- Tower 2: Model A Prediction Encoder ---
        # For long sequences (e.g., 1000 elements)
        # Uses adaptive pooling to handle variable lengths gracefully
        self.pred_tower = TowerEncoder(
            channels=pred_encoder_channels,
            latent_dim=latent_dim,
            kernel_size=pred_kernel_size,
            pool_size=2,
            use_adaptive_pool=True,  # Important for long sequences
            dropout=dropout,
        )

        # --- Fusion Head ---
        # Combines both latent representations
        self.fusion_head = FusionHead(
            input_dim=latent_dim * 2,
            hidden_dims=fusion_hidden_dims,
            output_dim=output_size,
            dropout=dropout,
        )

    def forward(
        self, vs_profile: torch.Tensor, model_a_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the dual-tower fusion model.

        Args:
            vs_profile: Velocity profile, shape (batch_size, vs_length)
            model_a_pred: Model A's prediction, shape (batch_size, pred_length)

        Returns:
            Predicted residual, shape (batch_size, output_size)
        """
        # --- Process through Tower 1 (Vs Profile) ---
        # Add channel dimension: (B, L) -> (B, 1, L)
        vs_latent = self.vs_tower(vs_profile.unsqueeze(1))

        # --- Process through Tower 2 (Model A Prediction) ---
        # Add channel dimension: (B, L) -> (B, 1, L)
        pred_input = model_a_pred.unsqueeze(1)
        pred_latent = self.pred_tower(pred_input)

        # --- Fuse latent representations ---
        combined_latent = torch.cat([vs_latent, pred_latent], dim=1)

        # --- Generate residual prediction ---
        residual_pred = self.fusion_head(combined_latent)

        return residual_pred

    def get_latent_representations(
        self, vs_profile: torch.Tensor, model_a_pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract latent representations from both towers (useful for analysis).

        Args:
            vs_profile: Velocity profile, shape (batch_size, vs_length)
            model_a_pred: Model A's prediction, shape (batch_size, pred_length)

        Returns:
            Tuple of (vs_latent, pred_latent), each of shape (batch_size, latent_dim)
        """
        with torch.no_grad():
            vs_input = vs_profile.unsqueeze(1)  # Add channel dim: (B, 1, L)
            vs_latent = self.vs_tower(vs_input)

            pred_input = model_a_pred.unsqueeze(1)
            pred_latent = self.pred_tower(pred_input)

        return vs_latent, pred_latent


# Factory function for easy model creation with preset configurations
def create_specboost_model_b(
    config_name: str = "default", latent_dim: int = 128, output_size: int = 1000
) -> SpecBoostModelB:
    """
    Factory function to create SpecBoostModelB with preset configurations.

    Args:
        config_name: One of ["default", "light", "heavy"]
        latent_dim: Latent dimension for both towers
        output_size: Output sequence length

    Returns:
        Configured SpecBoostModelB instance
    """
    configs = {
        "default": {
            "vs_encoder_channels": [1, 32, 64, 128],
            "pred_encoder_channels": [1, 64, 128, 256],
            "fusion_hidden_dims": [256, 512],
            "dropout": 0.2,
        },
        "light": {
            "vs_encoder_channels": [1, 16, 32, 64],
            "pred_encoder_channels": [1, 32, 64, 128],
            "fusion_hidden_dims": [128, 256],
            "dropout": 0.15,
        },
        "heavy": {
            "vs_encoder_channels": [1, 64, 128, 256, 512],
            "pred_encoder_channels": [1, 128, 256, 512, 512],
            "fusion_hidden_dims": [512, 1024, 512],
            "dropout": 0.25,
        },
    }

    if config_name not in configs:
        raise ValueError(f"config_name must be one of {list(configs.keys())}")

    config = configs[config_name]
    return SpecBoostModelB(
        vs_encoder_channels=config["vs_encoder_channels"],
        pred_encoder_channels=config["pred_encoder_channels"],
        latent_dim=latent_dim,
        output_size=output_size,
        fusion_hidden_dims=config["fusion_hidden_dims"],
        dropout=config["dropout"],
    )


class ResidualFNOModel(nn.Module):
    """
    The final, validated model for boosting stages.
    - Encodes the previous stage's prediction into a latent vector using a CNN tower.
    - Decodes the latent vector into a residual prediction using an FNO.
    """

    def __init__(
        self,
        pred_encoder_channels,
        latent_dim,
        output_size,
        fno_modes,
        fno_width,
        dropout,
    ):
        super().__init__()
        self.encoder = TowerEncoder(
            channels=pred_encoder_channels,
            latent_dim=latent_dim,
            kernel_size=5,
            pool_size=2,
            use_adaptive_pool=True,
            dropout=dropout,
        )
        self.decoder = OperatorDecoder(
            latent_dim=latent_dim,
            output_size=output_size,
            fno_modes=fno_modes,
            fno_width=fno_width,
        )

    def forward(self, pred_input: torch.Tensor):
        latent_vec = self.encoder(pred_input.unsqueeze(1))
        residual_pred = self.decoder(latent_vec)
        return residual_pred
