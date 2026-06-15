import torch
import torch.nn as nn
from geomloss import SamplesLoss


class OTDecoder(nn.Module):
    """
    A decoder module that uses Optimal Transport to map features from a
    fixed-size, uniform latent grid back to an arbitrary 1D grid.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        ot_steps: int = 15,
        ot_lr: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.ot_steps = ot_steps
        self.ot_lr = ot_lr

        # Create the latent grid (uniformly spaced in [0, 1])
        # This MUST match the encoder's latent grid
        self.latent_grid = torch.linspace(0, 1, latent_dim, device=self.device).view(
            1, latent_dim, 1
        )

        # Initialize the OT solver (Sinkhorn divergence)
        self.ot_solver = SamplesLoss(
            loss="sinkhorn", p=2, blur=0.05, backend="tensorized"
        )

    def forward(
        self, latent_features: torch.Tensor, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Projects latent features back onto the original target grid.

        Accepts either:
        - Combined target tensor of shape (B, 4, N) where target[:,0,:] is the grid
        - Separate target grid of shape (B, N)

        Args:
            latent_features (torch.Tensor): Latent features of shape (B, C, latent_dim)
            target (torch.Tensor | None): Combined (B, 4, N) or grid (B, N)

        Returns:
            torch.Tensor: Reconstructed features of shape (B, C, N)
        """
        if target is None:
            raise ValueError(
                "Decoder requires a target grid or combined tensor (B, 4, N)"
            )
        if target.dim() == 3 and target.size(1) >= 2:
            target_grid = target[:, 0, :]
        else:
            target_grid = target

        # Device/dtype alignment
        target_grid = target_grid.to(
            device=latent_features.device, dtype=latent_features.dtype
        )
        batch_size, num_channels, _ = latent_features.shape
        num_points_target = target_grid.shape[1]

        # --- Prepare Tensors for OT Solver ---

        # 1. Source (The Latent Space)
        y_features = latent_features.permute(0, 2, 1).contiguous()  # (B, latent_dim, C)
        y_locations = self.latent_grid.repeat(batch_size, 1, 1)  # (B, latent_dim, 1)
        y_augmented = torch.cat(
            [y_locations, y_features], dim=2
        )  # (B, latent_dim, 1+C)

        # 2. Target (The Original Grid)
        x_locations = target_grid.unsqueeze(-1)  # (B, N, 1)

        # We will optimize the features on the target grid
        reconstructed_features = torch.randn(
            batch_size,
            num_points_target,
            num_channels,
            device=self.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([reconstructed_features], lr=self.ot_lr)

        # Weights
        alpha_weights = (  # Source weights (latent)
            torch.ones(batch_size, self.latent_dim, device=self.device)
            / self.latent_dim
        )
        beta_weights = (  # Target weights (original)
            torch.ones(batch_size, num_points_target, device=self.device)
            / num_points_target
        )

        for _ in range(self.ot_steps):
            optimizer.zero_grad()

            # Create the augmented target tensor
            x_augmented = torch.cat([x_locations, reconstructed_features], dim=2)

            # Note the order: Source (y) -> Target (x)
            loss = self.ot_solver(
                alpha_weights,
                y_augmented,  # Source (locations, features)
                beta_weights,
                x_augmented,  # Target (locations, features_to_be_optimized)
            )

            # Aggregate batch loss and backprop
            loss.mean().backward()
            optimizer.step()

        reconstruction = reconstructed_features.detach()

        return reconstruction.permute(0, 2, 1)  # (B, C, N)
