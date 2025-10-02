# experiment_fno_model.py
"""
Defines a residual prediction model that uses a CNN Encoder (`TowerEncoder`)
followed by a Fourier Neural Operator Decoder (`OperatorDecoder`).
This architecture directly tests the hypothesis from the SpecBoost paper.
"""

import torch
import torch.nn as nn
from specboost import TowerEncoder

from wave_surrogate.models.fno.model import OperatorDecoder


class ResidualFNOModel(nn.Module):
    """
    An Encoder-Operator model designed to learn residuals.
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
        # Encoder part (same as the successful MLP-based experiment)
        self.encoder = TowerEncoder(
            channels=pred_encoder_channels,
            latent_dim=latent_dim,
            kernel_size=5,
            pool_size=2,
            use_adaptive_pool=True,
            dropout=dropout,
        )

        # Decoder part (this is the new component we are testing)
        self.decoder = OperatorDecoder(
            latent_dim=latent_dim,
            output_size=output_size,
            fno_modes=fno_modes,
            fno_width=fno_width,
        )
        print("Initialized ResidualFNOModel (Encoder -> FNO Decoder)")

    def forward(self, pred_input: torch.Tensor):
        """
        Forward pass: Encodes the prediction and decodes using the FNO.
        """
        # Pass input through the CNN encoder tower to get the latent vector
        # Input shape: (B, L) -> unsqueeze to (B, 1, L) for CNN
        latent_vec = self.encoder(pred_input.unsqueeze(1))

        # Pass the latent vector through the FNO decoder to get the residual prediction
        residual_pred = self.decoder(latent_vec)

        return residual_pred
