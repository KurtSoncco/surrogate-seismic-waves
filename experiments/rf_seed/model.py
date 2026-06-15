# model.py
"""2D CNN Encoder + OperatorDecoder (FNO1D) for 2D Vs profile to transfer function."""

from typing import List, Tuple

import torch
import torch.nn as nn

from wave_surrogate.models.fno.model import EncoderOperatorModel, OperatorDecoder


class Encoder2D(nn.Module):
    """
    2D CNN encoder for 2D Vs profiles.
    Maps (B, 1, H, W) -> (B, latent_dim).
    """

    def __init__(
        self,
        channels: List[int],
        latent_dim: int,
        input_shape: Tuple[int, int],
        kernel_size: int = 3,
        pool_size: int = 2,
    ):
        super().__init__()
        if not channels or channels[0] < 1:
            raise ValueError(
                "The 'channels' list must start with at least 1 input channel."
            )

        layers = []
        for i in range(len(channels) - 1):
            layers.extend(
                [
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(),
                    nn.MaxPool2d(pool_size),
                ]
            )
        self.conv_layers = nn.Sequential(*layers)

        h, w = input_shape
        for _ in range(len(channels) - 1):
            h = h // pool_size  # MaxPool2d uses floor division
            w = w // pool_size
        self.flat_size = channels[-1] * h * w
        self.fc = nn.Linear(self.flat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.flatten(1)
        return self.fc(x)


def create_model(
    input_shape: Tuple[int, int],
    output_size: int,
    latent_dim: int,
    encoder_channels: List[int],
    encoder_kernel_size: int = 3,
    encoder_pool_size: int = 2,
    fno_modes: int = 16,
    fno_width: int = 50,
    num_fno_layers: int = 3,
) -> EncoderOperatorModel:
    """Create the full 2D Vs -> transfer function model."""
    encoder = Encoder2D(
        channels=encoder_channels,
        latent_dim=latent_dim,
        input_shape=input_shape,
        kernel_size=encoder_kernel_size,
        pool_size=encoder_pool_size,
    )
    decoder = OperatorDecoder(
        latent_dim=latent_dim,
        output_size=output_size,
        fno_modes=fno_modes,
        fno_width=fno_width,
        num_fno_layers=num_fno_layers,
    )
    return EncoderOperatorModel(encoder=encoder, decoder=decoder)
