import torch
import torch.nn as nn
from geomloss import SamplesLoss


class OTEncoder(nn.Module):
    """
    An encoder module that uses Optimal Transport to map features from an
    arbitrary 1D grid to a fixed-size, uniform latent grid.

    This is achieved by finding the barycentric projection of the input features
    onto the latent grid, which minimizes the Sinkhorn divergence (a proxy for
    the Wasserstein-2 distance).
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
        self.latent_grid = torch.linspace(0, 1, latent_dim, device=self.device).view(
            1, latent_dim, 1
        )

        # Initialize the OT solver (Sinkhorn divergence)
        self.ot_solver = SamplesLoss(
            loss="sinkhorn", p=2, blur=0.05, backend="tensorized"
        )

    def forward(
        self, x: torch.Tensor, input_grid: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Projects input features onto the latent grid using Optimal Transport.

        Accepts either:
        - Combined input `x` of shape (B, 4, N) where x[:,0,:] is the grid and x[:,1:,:] are features
        - Separate inputs where `x` is features (B, C, N) and `input_grid` is (B, N)

        Returns:
            torch.Tensor: Latent representation of shape (B, C, latent_dim)
        """
        if input_grid is None:
            # Combined path: x -> (B, 4, N)
            assert x.dim() == 3 and x.size(1) >= 2, (
                "Expected combined tensor (B, >=2, N)"
            )
            input_grid = x[:, 0, :]
            input_features = x[:, 1:, :]
        else:
            input_features = x

        # Device/dtype alignment
        input_grid = input_grid.to(
            device=input_features.device, dtype=input_features.dtype
        )

        batch_size, num_channels, num_points = input_features.shape

        # --- Prepare Tensors for OT Solver ---
        x_features = input_features.permute(0, 2, 1).contiguous()  # (B, num_points, C)
        x_locations = input_grid.unsqueeze(-1)  # (B, num_points, 1)
        y_locations = self.latent_grid.repeat(batch_size, 1, 1)  # (B, latent_dim, 1)

        x_augmented = torch.cat(
            [x_locations, x_features], dim=-1
        )  # (B, num_points, 1+C)

        # --- Inner Optimization Loop ---
        y_features = torch.randn(
            batch_size,
            self.latent_dim,
            num_channels,
            device=self.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([y_features], lr=self.ot_lr)

        alpha_weights = (
            torch.ones(batch_size, num_points, device=self.device) / num_points
        )
        beta_weights = (
            torch.ones(batch_size, self.latent_dim, device=self.device)
            / self.latent_dim
        )

        for _ in range(self.ot_steps):
            optimizer.zero_grad()

            # We now pass tuples: (locations, features)
            # The cost is computed on the locations (1D grid).
            # The features (3D vectors) are what get transported.
            y_augmented = torch.cat([y_locations, y_features], dim=2)
            loss = self.ot_solver(
                alpha_weights,
                x_augmented,  # Source (locations, features)
                beta_weights,
                y_augmented,  # Target (locations, features)
            )

            loss.mean().backward()
            optimizer.step()

        latent_representation = y_features.detach()

        return latent_representation.permute(0, 2, 1)  # (B, C, latent_dim)


if __name__ == "__main__":
    # --- Configuration ---
    BATCH_SIZE = 8
    NUM_CHANNELS = 3  # Vs, Vp, Rho
    NUM_POINTS_IN = 250  # Number of points in the original soil profile
    LATENT_DIM_OUT = 128  # Desired fixed size of the latent representation
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on device: {DEVICE}")

    # --- Create the OTEncoder ---
    ot_encoder = OTEncoder(latent_dim=LATENT_DIM_OUT, device=DEVICE)
    ot_encoder.to(DEVICE)

    # --- Create Dummy Data ---
    # Imagine 8 soil profiles, each with 3 channels and 250 data points.
    input_features = torch.randn(BATCH_SIZE, NUM_CHANNELS, NUM_POINTS_IN, device=DEVICE)

    # Create a grid assuming the spacing between each point is 5 meters
    spacing = 5.0
    input_grid = (
        torch.linspace(0, spacing * (NUM_POINTS_IN - 1), NUM_POINTS_IN, device=DEVICE)
        .unsqueeze(0)
        .repeat(BATCH_SIZE, 1)
    )

    print(f"Input features shape: {input_features.shape}")
    print(f"Input grid shape:     {input_grid.shape}")

    # --- Forward Pass ---
    latent_representation = ot_encoder(input_features, input_grid)

    print(f"\nOutput latent shape:  {latent_representation.shape}")
    print(f"Expected latent shape:  ({BATCH_SIZE}, {NUM_CHANNELS}, {LATENT_DIM_OUT})")

    # Check if the output shape is as expected
    assert latent_representation.shape == (BATCH_SIZE, NUM_CHANNELS, LATENT_DIM_OUT)
    print("\nSuccessfully encoded features to the latent space!")
