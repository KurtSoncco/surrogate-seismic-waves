# pino_model.py
"""Defines the PINO model architecture."""

import torch
import torch.nn as nn

# Assuming FNO1D is defined in this path
from wave_surrogate.models.fno.model import FNO1D


class PINO(nn.Module):
    """
    Physics-Informed Neural Operator for Wave Propagation.

    This model learns the operator that maps physical properties (Vs/Rho) and
    an initial boundary condition (input motion) to the full spatiotemporal
    wavefield u(z, t).
    """

    def __init__(
        self,
        vs_encoder: nn.Module,
        latent_dim: int,  # CHANGED: Decoupled from vs_encoder implementation
        fno_width: int,
        fno_modes: int,
        num_fno_layers: int,
        # REMOVED: timesteps and spatial_points are inferred from grid shape
    ):
        super().__init__()
        self.vs_encoder = vs_encoder
        self.fno_width = fno_width

        # Lifting layer: maps input motion + grid + latent params to FNO width
        # Input: input_motion(t)[1] + grid(z,t)[2] + params[latent_dim]
        self.lift = nn.Linear(1 + 2 + latent_dim, fno_width)

        # FNO stack for processing the spatiotemporal field
        # The FNO processes along the spatial dimension (Z) for each timestep
        fno_stack = [
            FNO1D(modes=fno_modes, width=fno_width) for _ in range(num_fno_layers)
        ]
        self.fno_stack = nn.Sequential(*fno_stack)

        # Projection layers: map from FNO width back to a single value (displacement)
        self.project = nn.Sequential(
            nn.Linear(fno_width, fno_width * 2),
            nn.GELU(),  # ADDED: Non-linearity for better representation
            nn.Linear(fno_width * 2, 1),
        )

    def forward(
        self, profiles: torch.Tensor, input_motion: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the PINO.

        Args:
            profiles (Tensor): Batch of Vs/Rho profiles (B, 2, L_profiles).
            input_motion (Tensor): Input ground motion at bedrock (B, T).
            grid (Tensor): Spatiotemporal grid coords (B, Z, T, 2). MUST require grad.

        Returns:
            Tensor: Predicted wavefield u(z, t) of shape (B, Z, T).
        """
        # Infer grid dimensions
        batch_size, spatial_points, timesteps, _ = grid.shape

        # 1. Encode Vs and Rho profiles to get physical latent parameters
        latent_params = self.vs_encoder(profiles)  # (B, D_lat)

        # 2. Prepare the input field for the FNO by expanding all inputs
        # to match the spatiotemporal grid dimensions.
        latent_params = latent_params.view(-1, 1, 1, latent_params.shape[-1]).expand(
            -1, spatial_points, timesteps, -1
        )
        input_motion = input_motion.view(-1, 1, timesteps, 1).expand(
            -1, spatial_points, -1, -1
        )

        # 3. Concatenate all inputs and lift to the FNO's channel space
        x = torch.cat((grid, input_motion, latent_params), dim=-1)
        x = self.lift(x)  # (B, Z, T, W)

        # 4. Process through FNO layers.
        # The FNO1D operates on the spatial dimension (Z). We treat each timestep
        # as a separate sample in a large batch.
        # Reshape: (B, Z, T, W) -> (B, T, Z, W) -> (B*T, Z, W)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size * timesteps, spatial_points, self.fno_width)

        x = self.fno_stack(x)  # FNO processes each of the (B*T) spatial slices

        # Reshape back: (B*T, Z, W) -> (B, T, Z, W) -> (B, Z, T, W)
        x = x.view(batch_size, timesteps, spatial_points, self.fno_width)
        x = x.permute(0, 2, 1, 3)

        # 5. Project back to the solution space (1 channel for displacement)
        u_pred = self.project(x).squeeze(-1)  # (B, Z, T)

        return u_pred


if __name__ == "__main__":
    # This example script is now corrected and demonstrates the proper usage.
    # A pino_config.py file is assumed to exist with the necessary parameters.
    import pino_config as config
    from physics_loss import WaveEquationLoss

    from wave_surrogate.models.fno.model import Encoder

    # --- 1. Model Initialization ---
    vs_encoder = Encoder(
        channels=config.ENCODER_CHANNELS, latent_dim=config.LATENT_DIM
    ).to(config.DEVICE)

    model = PINO(
        vs_encoder=vs_encoder,
        latent_dim=config.LATENT_DIM,  # CHANGED: Pass latent_dim directly
        fno_width=32,
        fno_modes=16,
        num_fno_layers=4,
    ).to(config.DEVICE)
    print(model)

    # --- 2. Example Input Tensors ---
    BATCH_SIZE = 4
    profiles = torch.randn(BATCH_SIZE, 2, config.INPUT_SIZE).to(config.DEVICE)
    input_motion = torch.randn(BATCH_SIZE, config.TIMESTEPS).to(config.DEVICE)

    # --- 3. Grid Creation and GRAD REQUIREMENT ---
    z = torch.linspace(0, 1, config.SPATIAL_POINTS, device=config.DEVICE)
    t = torch.linspace(0, 1, config.TIMESTEPS, device=config.DEVICE)
    zz, tt = torch.meshgrid(z, t, indexing="ij")
    grid = torch.stack([zz, tt], dim=-1).unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1)

    # CRITICAL FIX: Enable gradient tracking on the grid *before* the model call.
    grid.requires_grad_(True)

    # --- 4. Model Forward Pass ---
    u_pred = model(profiles, input_motion, grid)
    print("Output shape:", u_pred.shape)

    # --- 5. Physics Loss Computation ---
    # Ensure loss parameters are consistent with data dimensions
    total_depth = config.INPUT_SIZE * config.LAYER_THICKNESS
    total_time = config.TIMESTEPS  # Assuming this is defined in your config

    physics_criterion = WaveEquationLoss(
        layer_thickness=config.LAYER_THICKNESS,
        total_depth=total_depth,
        total_time=total_time,
        device=config.DEVICE,
    )
    # This will now work without error
    loss = physics_criterion(u_pred, profiles, grid)
    print(f"Physics loss: {loss.item():.6f}")
