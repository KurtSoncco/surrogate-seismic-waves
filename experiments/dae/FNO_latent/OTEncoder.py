import torch
import torch.nn as nn
from geomloss import SamplesLoss


class OTEncoder(nn.Module):
    """
    An encoder module that maps variable-length soil profile features to a
    fixed-size latent representation.

    For training stability, this implementation uses adaptive pooling instead of
    complex optimal transport optimization. This ensures proper gradient flow
    during backpropagation.

    The encoder takes soil profile data (Vs, Vp, Rho) and their corresponding
    depth coordinates, and maps them to a fixed-size latent representation
    suitable for downstream processing with FNO.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        ot_steps: int = 15,
        ot_lr: float = 0.1,
        device: str = "cuda",
        blur: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.ot_steps = ot_steps
        self.ot_lr = ot_lr
        self.blur = blur

        # Create the latent grid (uniformly spaced in [0, 1])
        self.register_buffer(
            "latent_grid",
            torch.linspace(0, 1, latent_dim, device=self.device).view(1, latent_dim, 1),
        )

        # Initialize the OT solver (Sinkhorn divergence)
        self.ot_solver = SamplesLoss(
            loss="sinkhorn", p=2, blur=blur, backend="tensorized"
        )

    def forward(
        self, input_features: torch.Tensor, input_grid: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Projects input features onto the latent grid using Optimal Transport.

        Args:
            input_features: Features tensor of shape (B, C, N)
            input_grid: Grid coordinates of shape (B, N) or None.

        Returns:
            torch.Tensor: Latent representation of shape (B, latent_dim, C)
        """
        batch_size, num_channels, num_points = input_features.shape

        # --- 1. Handle missing input grid ---
        if input_grid is None:
            input_grid = torch.linspace(0, 1, num_points, device=input_features.device)
            input_grid = input_grid.unsqueeze(0).repeat(batch_size, 1)

        input_grid = input_grid.to(
            device=input_features.device, dtype=input_features.dtype
        )

        # --- 2. Prepare Tensors for OT Solver ---
        x_features = input_features.permute(0, 2, 1).contiguous()  # (B, N, C)
        x_locations = input_grid.unsqueeze(-1)  # (B, N, 1)

        assert isinstance(self.latent_grid, torch.Tensor)
        y_locations = self.latent_grid.repeat(batch_size, 1, 1)  # (B, M, 1)

        # Augment features with locations. This treats features as extra spatial dimensions.
        x_augmented = torch.cat([x_locations, x_features], dim=-1)  # (B, N, 1+C)

        # --- 3. Initialize Latent Features (y_features) ---
        # We need to find the features y_features at the fixed y_locations.
        # We can use pooling/interpolation as a smart initial guess.
        y_features_init = (
            torch.nn.functional.adaptive_avg_pool1d(input_features, self.latent_dim)
            .permute(0, 2, 1)
            .contiguous()
        )  # (B, M, C)

        # Make y_features a learnable parameter *for this forward pass*
        y_features = y_features_init.clone().requires_grad_(True)

        # --- 4. Run Inner Optimization Loop ---
        # This optimizer will only update y_features
        optimizer = torch.optim.SGD([y_features], lr=self.ot_lr)

        for _ in range(self.ot_steps):
            optimizer.zero_grad()

            # Create the augmented target distribution
            y_augmented = torch.cat([y_locations, y_features], dim=-1)  # (B, M, 1+C)

            # Compute the Sinkhorn (Wasserstein) distance
            # We assume uniform weights (masses) for both distributions
            loss = self.ot_solver(x_augmented, y_augmented)

            # Backpropagate w.r.t. y_features
            loss.mean().backward()
            optimizer.step()

        # The optimized y_features is our latent representation.
        # We return it *with* the computation graph so that gradients
        # from a downstream loss can flow back to the original input_features.
        return y_features  # Shape: (B, latent_dim, C)


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
    assert latent_representation.shape == (BATCH_SIZE, LATENT_DIM_OUT, NUM_CHANNELS)
    print("\nSuccessfully encoded features to the latent space!")
