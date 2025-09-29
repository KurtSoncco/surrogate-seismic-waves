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
        layers.append(nn.ReLU())  # Ensure non-negative latent features

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
        # layers.append(nn.Tanh())  # To match the [-1, 1] range of the normalized data
        layers.append(nn.ReLU())  # Vs should be non-negative
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
        # layers.append(nn.Tanh())
        layers.append(nn.ReLU())  # Vs should be non-negative

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# --- NEW CONVOLUTIONAL ARCHITECTURES ---


class ConvEncoder(nn.Module):
    """1D Convolutional Encoder.

    Args:
        input_dim: Length of the input sequence.
        hidden_dim: Sequence of channel sizes for successive convolutional layers.
        latent_dim: Size of the latent embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[int, Sequence[int], None],
        latent_dim: int,
    ):
        super().__init__()
        hidden_channels = _ensure_list(hidden_dim)

        layers = []
        in_channels = 1  # Start with 1 channel for the raw signal
        for h_chan in hidden_channels:
            layers.append(
                nn.Conv1d(in_channels, h_chan, kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.ReLU())
            in_channels = h_chan
        self.conv_stack = nn.Sequential(*layers)

        # Calculate the flattened size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            conv_output = self.conv_stack(dummy_input)
            self.flattened_size = conv_output.view(1, -1).shape[1]

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        # Add a channel dimension: (batch, seq_len) -> (batch, 1, seq_len)
        x = x.unsqueeze(1)
        x = self.conv_stack(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z


class ConvDecoder(nn.Module):
    """1D Convolutional Decoder.

    Args:
        latent_dim: Size of the latent vector.
        hidden_dim: Sequence of channel sizes for deconvolutional layers (in reverse).
        output_dim: Final output sequence length.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Union[int, Sequence[int], None],
        output_dim: int,
    ):
        super().__init__()
        hidden_channels = _ensure_list(hidden_dim)
        # We need the flattened size and final channel count from a dummy encoder
        # This makes the decoder robust to changes in the encoder architecture
        dummy_encoder = ConvEncoder(output_dim, hidden_channels, latent_dim)
        self.flattened_size = dummy_encoder.flattened_size
        # The last Conv1d layer has out_channels = hidden_channels[-1]
        self.initial_channels = hidden_channels[-1] if hidden_channels else 1
        self.unflatten_shape = dummy_encoder.conv_stack(
            torch.zeros(1, 1, output_dim)
        ).shape[1:]

        self.fc = nn.Linear(latent_dim, self.flattened_size)

        layers = []
        in_channels = self.initial_channels

        # Build deconvolutional layers in reverse order of encoder
        for h_chan in hidden_channels[-2::-1]:
            layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    h_chan,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            layers.append(nn.ReLU())
            in_channels = h_chan

        # Final layer to produce 1 channel output
        layers.append(
            nn.ConvTranspose1d(
                in_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            )
        )
        layers.append(nn.Tanh())

        self.deconv_stack = nn.Sequential(*layers)
        self.final_adapter = nn.AdaptiveAvgPool1d(
            output_dim
        )  # Adapter to fix size mismatches

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, *self.unflatten_shape)  # Reshape for deconvolution
        x = self.deconv_stack(x)
        x = self.final_adapter(x)  # Ensure exact output_dim
        recon = x.squeeze(
            1
        )  # Remove channel dim: (batch, 1, seq_len) -> (batch, seq_len)
        return recon


class WeakerConvDecoder(ConvDecoder):
    """A weaker convolutional decoder with dropout."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Union[int, Sequence[int], None],
        output_dim: int,
        dropout_rate: float = 0.5,
    ):
        # Initialize the parent ConvDecoder
        super().__init__(latent_dim, hidden_dim, output_dim)

        # Rebuild the deconv_stack with dropout
        hidden_channels = _ensure_list(hidden_dim)
        layers = []
        in_channels = self.initial_channels

        for h_chan in hidden_channels[-2::-1]:
            layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    h_chan,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Add dropout
            in_channels = h_chan

        layers.append(
            nn.ConvTranspose1d(
                in_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            )
        )
        layers.append(nn.Tanh())
        self.deconv_stack = nn.Sequential(*layers)
