# model.py
"""Defines the DeepONet model architecture: Branch, Trunk, and the Operator."""

from typing import List, Tuple

import config
import torch
import torch.nn as nn


class BranchNetwork(nn.Module):
    """
    The Branch Network (previously Encoder) processes the input function (Vs profile).
    It maps the Vs profile to a set of coefficients (the latent vector).

    Input Shape: (batch_size, 1, sequence_length) -> Vs profile
    Output Shape: (batch_size, latent_dim) -> Coefficients (p)
    """

    def __init__(
        self,
        channels: List[int],
        latent_dim: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ):
        super().__init__()

        if not channels or channels[0] < 1:
            raise ValueError(
                "The 'channels' list must start with at least 1 input channel."
            )

        layers = []
        # Dynamically create the convolutional blocks
        for i in range(len(channels) - 1):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_size,
                        padding="same",
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(pool_size),
                ]
            )

        self.conv_layers = nn.Sequential(*layers)

        # Use a LazyLinear layer to automatically infer the input size
        # This layer outputs the latent vector (coefficients p)
        self.fc = nn.LazyLinear(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input should be (B, C, L) where C is the number of input channels
        x = self.conv_layers(x)
        # Flatten all dimensions except the batch dimension
        x = x.flatten(1)
        # Final fully connected layer to get coefficients p
        return self.fc(x)


class TrunkNetwork(nn.Module):
    """
    The Trunk Network processes the domain coordinates (frequencies, y).
    It maps the coordinates to a set of basis functions (the latent vector).

    Input Shape: (batch_size * output_size, 1) -> Frequency coordinate
    Output Shape: (batch_size * output_size, latent_dim) -> Basis functions (b)
    """

    def __init__(self, input_dim: int, latent_dim: int, num_layers: int, neurons: int):
        super().__init__()
        layers = []
        # First layer
        layers.extend([nn.Linear(input_dim, neurons), nn.ReLU()])
        # Intermediate layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(neurons, neurons), nn.ReLU()])
        # Output layer produces the basis functions (b)
        layers.append(nn.Linear(neurons, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y should be (B * L_out, D_y=1) where L_out=OUTPUT_SIZE
        return self.net(y)


class DeepONetModel(nn.Module):
    """
    The full Deep Operator Network combining the Branch and Trunk.

    The model approximates the operator G such that G[u](y) ≈ sum( p_k(u) * b_k(y) )
    where p are coefficients from the Branch (Vs profile) and b are basis functions
    from the Trunk (Frequency coordinates).
    """

    def __init__(self, branch: nn.Module, trunk: nn.Module, output_size: int):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.output_size = output_size
        self.latent_dim = config.LATENT_DIM

        # Optional: A final trainable weight to slightly adjust the sum output
        # self.final_linear = nn.Linear(self.latent_dim, 1) # alternative combination

    def forward(self, u_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_hat: (B, C, L_in) - Vs profile (input function)
            y: (B, L_out) - Frequency points (query coordinates)
        Returns:
            out: (B, L_out) - Predicted TTF (output function)
        """
        batch_size = u_hat.shape[0]

        # 1. Branch: u_hat -> p (Coefficients)
        # Shape: (B, L_latent)
        p = self.branch(u_hat)

        # 2. Trunk: y must be reshaped for the MLP, then processed
        # Reshape y: (B, L_out) -> (B * L_out, 1)
        y_flat = y.reshape(-1, 1)

        # y_flat -> b (Basis functions)
        # Shape: (B * L_out, L_latent)
        b_flat = self.trunk(y_flat)

        # Reshape b_flat back to (B, L_out, L_latent)
        # This is the set of basis functions evaluated at each query point (frequency)
        b = b_flat.reshape(batch_size, self.output_size, self.latent_dim)

        # 3. DeepONet combination (element-wise multiplication and summation)
        # Reshape p to be (B, 1, L_latent) for broadcasting
        p = p.unsqueeze(1)

        # Output: (B, L_out, L_latent) * (B, 1, L_latent) -> (B, L_out, L_latent)
        # Sum over the latent dimension: (B, L_out, L_latent) -> (B, L_out)
        out = torch.sum(p * b, dim=-1)

        # The output now is the full predicted TTF for all frequencies
        return out


def build_deeponet_components(config) -> Tuple[nn.Module, nn.Module]:
    """Builds and returns the BranchNetwork and TrunkNetwork based on config."""

    branch = BranchNetwork(
        channels=config.ENCODER_CHANNELS,
        latent_dim=config.LATENT_DIM,
        kernel_size=config.ENCODER_KERNEL_SIZE,
        pool_size=config.ENCODER_POOL_SIZE,
    )

    trunk = TrunkNetwork(
        input_dim=config.TRUNK_INPUT_DIM,
        latent_dim=config.LATENT_DIM,
        num_layers=config.TRUNK_LAYERS,
        neurons=config.TRUNK_NEURONS,
    )

    return branch, trunk


if __name__ == "__main__":
    ## Simple test of one forward pass with dummy data
    branch, trunk = build_deeponet_components(config)

    model = DeepONetModel(
        branch=branch, trunk=trunk, output_size=config.OUTPUT_SIZE
    ).to(config.DEVICE)

    dummy_vs = torch.randn(2, 1, config.INPUT_SIZE).to(config.DEVICE)  # (B=2)
    dummy_freq = torch.linspace(0.1, 50, config.OUTPUT_SIZE).to(config.DEVICE)
    dummy_freq = dummy_freq.unsqueeze(0).repeat(2, 1)  # (B=2, L_out)

    dummy_output = model(dummy_vs, dummy_freq)
    print(f"Dummy output shape: {dummy_output.shape}")  # Should be (2, OUTPUT_SIZE)
    assert dummy_output.shape == (2, config.OUTPUT_SIZE), (
        f"Output shape error: {dummy_output.shape}"
    )
