# experiment_models.py
"""
Contains simplified model architectures for debugging the SpecBoost framework.
Each model uses only one of the two inputs to isolate which signal, if any,
is learnable.
"""

import torch
import torch.nn as nn
from specboost import FusionHead, TowerEncoder


class SimplifiedVsOnlyModel(nn.Module):
    """
    A simplified model that attempts to predict the residual using ONLY the
    original Vs profile. It ignores the prediction from Model A.
    """

    def __init__(
        self, vs_encoder_channels, latent_dim, fusion_hidden_dims, output_size, dropout
    ):
        super().__init__()
        # --- Only use the Vs Profile Encoder ---
        self.vs_tower = TowerEncoder(
            channels=vs_encoder_channels,
            latent_dim=latent_dim,
            kernel_size=3,
            pool_size=2,
            use_adaptive_pool=False,
            dropout=dropout,
        )

        # --- Simplified Fusion Head ---
        # Takes input only from the single tower
        self.fusion_head = FusionHead(
            input_dim=latent_dim,  # Note: not latent_dim * 2
            hidden_dims=fusion_hidden_dims,
            output_dim=output_size,
            dropout=dropout,
        )
        print("Initialized SimplifiedVsOnlyModel")

    def forward(self, vs_profile: torch.Tensor):
        """
        Forward pass. Note: The `model_a_pred` argument is ignored to test
        the signal from the Vs profile alone.
        """
        # Add channel dimension for CNN: (B, L) -> (B, 1, L)
        vs_latent = self.vs_tower(vs_profile.unsqueeze(1))
        residual_pred = self.fusion_head(vs_latent)
        return residual_pred


class SimplifiedPredOnlyModel(nn.Module):
    """
    A simplified model that attempts to predict the residual using ONLY the
    prediction from the previous stage. It ignores the original Vs profile.
    """

    def __init__(
        self,
        pred_encoder_channels,
        latent_dim,
        fusion_hidden_dims,
        output_size,
        dropout,
    ):
        super().__init__()
        # --- Only use the Prediction Encoder ---
        self.pred_tower = TowerEncoder(
            channels=pred_encoder_channels,
            latent_dim=latent_dim,
            kernel_size=5,
            pool_size=2,
            use_adaptive_pool=True,  # Important for long sequences
            dropout=dropout,
        )

        # --- Simplified Fusion Head ---
        self.fusion_head = FusionHead(
            input_dim=latent_dim,  # Note: not latent_dim * 2
            hidden_dims=fusion_hidden_dims,
            output_dim=output_size,
            dropout=dropout,
        )
        print("Initialized SimplifiedPredOnlyModel")

    def forward(
        self,
        pred_input: torch.Tensor,
    ):
        """
        Forward pass. Note: The `vs_profile` argument is ignored to test
        the signal from the prediction alone.
        """
        # Add channel dimension for CNN: (B, L) -> (B, 1, L)
        pred_latent = self.pred_tower(pred_input.unsqueeze(1))
        residual_pred = self.fusion_head(pred_latent)
        return residual_pred
