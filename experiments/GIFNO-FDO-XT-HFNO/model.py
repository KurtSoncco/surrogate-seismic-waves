# model.py
"""H-FNO hybrid encoder + XT DeepONet head."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

import config
from spectral_layers import HybridFNOStack
from xt_readout import ChannelLift, RecorderDeepONetHeadXT


def _head_kwargs(
    latent_channels: int,
    n_freq: int,
    nx: int,
    recorder_x: np.ndarray,
    deeponet_dim: int,
    branch_mode: str,
    trunk_hidden: int,
    trunk_layers: int,
    freq: np.ndarray | None,
) -> dict:
    log_f = None
    x_trunk = None
    if freq is not None:
        log_f = np.log(np.maximum(freq, 1e-8).astype(np.float32))
    x_trunk = config.recorder_x_trunk_coords(
        recorder_x, nx=nx, mode=config.X_COORD_MODE
    )
    return dict(
        latent_channels=latent_channels,
        nx=nx,
        n_freq=n_freq,
        recorder_x=recorder_x,
        deeponet_dim=deeponet_dim,
        branch_mode=branch_mode,
        trunk_hidden=trunk_hidden,
        trunk_layers=trunk_layers,
        x_trunk=x_trunk,
        log_f=log_f,
    )


class GIFNOXTHFNOModel(nn.Module):
    """ChannelLift -> HybridFNOStack -> RecorderDeepONetHeadXT."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        n_freq: int,
        nx: int,
        fno_modes: Tuple[int, int],
        num_fno_layers: int,
        deeponet_dim: int,
        branch_mode: str = "surface",
        trunk_hidden: int = 128,
        trunk_layers: int = 4,
        recorder_x: np.ndarray | None = None,
        freq: np.ndarray | None = None,
    ):
        super().__init__()
        if recorder_x is None:
            recorder_x = config.recorder_x_indices()
        self.lift = ChannelLift(in_channels, latent_channels)
        self.fno = HybridFNOStack(
            n_modes=fno_modes,
            channels=latent_channels,
            n_layers=num_fno_layers,
        )
        self.head = RecorderDeepONetHeadXT(
            **_head_kwargs(
                latent_channels,
                n_freq,
                nx,
                recorder_x,
                deeponet_dim,
                branch_mode,
                trunk_hidden,
                trunk_layers,
                freq,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        x = self.fno(x)
        return self.head(x)


def create_model(
    in_channels: int = config.IN_CHANNELS,
    latent_channels: int = config.LATENT_CHANNELS,
    n_freq: int = config.N_FREQ,
    nx: int = config.NX,
    fno_modes: Tuple[int, int] = config.FNO_MODES,
    num_fno_layers: int = config.NUM_FNO_LAYERS,
    deeponet_dim: int = config.DEEPONET_LATENT_DIM,
    branch_mode: str = config.BRANCH_MODE,
    trunk_hidden: int = config.TRUNK_HIDDEN,
    trunk_layers: int = config.TRUNK_LAYERS,
) -> GIFNOXTHFNOModel:
    freq = None
    if config.TF_FREQ_PATH.is_file():
        freq = np.load(config.TF_FREQ_PATH)
    return GIFNOXTHFNOModel(
        in_channels=in_channels,
        latent_channels=latent_channels,
        n_freq=n_freq,
        nx=nx,
        fno_modes=fno_modes,
        num_fno_layers=num_fno_layers,
        deeponet_dim=deeponet_dim,
        branch_mode=branch_mode,
        trunk_hidden=trunk_hidden,
        trunk_layers=trunk_layers,
        freq=freq,
    )
