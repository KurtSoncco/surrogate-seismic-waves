from typing import List, Sequence, Union

import torch
import torch.nn as nn


def _ensure_list(hidden: Union[int, Sequence[int], None]) -> List[int]:
    """Normalize hidden specification to a list of ints.

    Accepts an int (treated as single hidden layer) or a sequence of ints.
    Raises ValueError for empty lists.
    """
    if isinstance(hidden, int):
        return [hidden]
    if hidden is None:
        return []
    hidden_list = list(hidden)
    if len(hidden_list) == 0:
        raise ValueError("hidden must be a non-empty int or sequence of ints")
    return [int(h) for h in hidden_list]


class Encoder(nn.Module):
    """Flexible encoder that accepts a list of hidden layer sizes.

    Args:
        input_dim: Number of input features.
        hidden_dim: Either an int (single hidden layer) or a sequence of ints
            describing the sizes of successive hidden layers.
        latent_dim: Size of the latent embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[int, Sequence[int], None],
        latent_dim: int,
    ):
        super().__init__()
        hidden_dims = _ensure_list(hidden_dim)

        layers = []
        in_features = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h

        # final projection to latent space
        layers.append(nn.Linear(in_features, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Flexible decoder that mirrors Encoder: list of hidden sizes from latent to output.

    Args:
        latent_dim: Size of latent vector.
        hidden_dim: Either an int or sequence of ints for decoder hidden layers.
        output_dim: Final output dimensionality.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Union[int, Sequence[int], None],
        output_dim: int,
    ):
        super().__init__()
        hidden_dims = _ensure_list(hidden_dim)

        layers = []
        in_features = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h

        layers.append(nn.Linear(in_features, output_dim))
        layers.append(nn.Tanh())  # To match the [-1, 1] range of the normalized data
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class WeakerDecoder(nn.Module):
    """A weaker decoder with dropout for regularization in Stage 1.

    Accepts the same hidden_dim formats as Decoder.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Union[int, Sequence[int], None],
        output_dim: int,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        hidden_dims = _ensure_list(hidden_dim)

        layers = []
        in_features = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = h

        layers.append(nn.Linear(in_features, output_dim))
        layers.append(nn.Tanh())  # To match the [-1, 1] range of the normalized data

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# --- OT Encoder and Decoder ---

# Conceptual Code for an OT Encoder Module
import torch.nn as nn
from geomloss import SamplesLoss


class OTEncoder(nn.Module):
    def __init__(self, latent_dim=128, device="cuda"):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        # Create the canonical (uniform) latent grid. This is fixed.
        self.latent_grid = torch.linspace(
            0, 1, latent_dim, device=self.device
        ).unsqueeze(0)

        # The OT "cost" function. We use Sinkhorn divergence which is fast and stable.
        # This is not a traditional "loss" but a way to compute the OT plan.
        self.ot_solver = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

    def forward(self, input_features, input_grid):
        """
        Args:
            input_features (Tensor): Batch of material properties.
                                     Shape: (batch_size, num_channels, num_points)
            input_grid (Tensor): Batch of coordinates for the input features.
                                 Shape: (batch_size, num_points)
        """
        batch_size = input_features.shape[0]

        # Ensure the latent grid is ready for the batch
        latent_grid_batch = self.latent_grid.repeat(batch_size, 1)

        # 1. Compute the OT mapping between the input grid and the latent grid.
        # This is the core of the OT process. geomloss handles the complexity.
        # We get a transport plan 'T'.
        # Note: The actual implementation involves a bit more nuance with gradients,
        # but this is the high-level idea. The library handles the backward pass.

        # 2. Push-forward the features using the transport plan 'T'.
        # This step effectively interpolates/moves the `input_features` onto the `latent_grid`.
        # The result is your latent representation.
        # A library function or a custom interpolation kernel would do this.

        # Let's assume a function `apply_transport` exists for clarity:
        # latent_representation = self.apply_transport(input_features, input_grid, latent_grid_batch, T)

        # A simpler, more direct approach for 1D is to use differentiable interpolation
        # guided by the OT cost, but let's stick to the core concept.
        # For now, we'll placeholder this logic.

        # Placeholder for the actual push-forward operation
        # In practice, you would use a library or implement a differentiable interpolation
        # based on the transport plan.
        latent_representation = torch.randn(
            batch_size, input_features.shape[1], self.latent_dim, device=self.device
        )

        return latent_representation
