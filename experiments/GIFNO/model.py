# model.py
"""GIFNO grid-direct model: ChannelLift -> FNOBlocks -> surface slice -> FrequencyHead."""

from typing import Tuple

import torch
import torch.nn as nn
from neuralop.layers.fno_block import FNOBlocks


class ChannelLift(nn.Module):
    """Project input channels to latent grid representation."""

    def __init__(self, in_channels: int, latent_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FrequencyHead(nn.Module):
    """Map surface row (B, C, Nx) -> TF field (B, Nx, N_freq)."""

    def __init__(self, latent_channels: int, n_freq: int):
        super().__init__()
        self.proj = nn.Conv1d(latent_channels, n_freq, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, Nx) -> (B, N_freq, Nx) -> (B, Nx, N_freq)
        return self.proj(x).permute(0, 2, 1)


class GIFNOGridModel(nn.Module):
    """
    Grid-direct FNO pipeline (Phase 1).

    Input:  (B, C_in, Nz, Nx)
    Output: (B, Nx, N_freq)
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        n_freq: int,
        fno_modes: Tuple[int, int],
        num_fno_layers: int,
    ):
        super().__init__()
        self.lift = ChannelLift(in_channels, latent_channels)
        self.fno = FNOBlocks(
            n_modes=fno_modes,
            in_channels=latent_channels,
            out_channels=latent_channels,
            n_layers=num_fno_layers,
            non_linearity=nn.functional.gelu,
        )
        self.freq_head = FrequencyHead(latent_channels, n_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        x = self.fno(x)
        surface = x[:, :, 0, :]
        return self.freq_head(surface)


def create_model(
    in_channels: int = 4,
    latent_channels: int = 64,
    n_freq: int = 1000,
    fno_modes: Tuple[int, int] = (32, 32),
    num_fno_layers: int = 5,
) -> GIFNOGridModel:
    return GIFNOGridModel(
        in_channels=in_channels,
        latent_channels=latent_channels,
        n_freq=n_freq,
        fno_modes=fno_modes,
        num_fno_layers=num_fno_layers,
    )
