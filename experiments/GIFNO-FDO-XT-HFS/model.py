# model.py
"""Standard FNO encoder + post-FNO HFS + XT DeepONet head."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

import config
from spectral_layers import FNOEncoderLoop, HFSModule
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
    x_trunk = config.recorder_x_trunk_coords(
        recorder_x, nx=nx, mode=config.X_COORD_MODE
    )
    if freq is not None:
        log_f = np.log(np.maximum(freq, 1e-8).astype(np.float32))
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


class GIFNOXTHFSModel(nn.Module):
    """ChannelLift -> FNO (all layers) -> HFSModule -> RecorderDeepONetHeadXT."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        n_freq: int,
        nx: int,
        fno_modes: Tuple[int, int],
        num_fno_layers: int,
        deeponet_dim: int,
        hfs_patch_size: int = 4,
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
        self.fno = FNOEncoderLoop(
            n_modes=fno_modes,
            in_channels=latent_channels,
            out_channels=latent_channels,
            n_layers=num_fno_layers,
        )
        self.hfs = HFSModule(latent_channels, patch_size=hfs_patch_size)
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
        x = self.hfs(x)
        return self.head(x)


def create_model(
    in_channels: int = config.IN_CHANNELS,
    latent_channels: int = config.LATENT_CHANNELS,
    n_freq: int = config.N_FREQ,
    nx: int = config.NX,
    fno_modes: Tuple[int, int] = config.FNO_MODES,
    num_fno_layers: int = config.NUM_FNO_LAYERS,
    deeponet_dim: int = config.DEEPONET_LATENT_DIM,
    hfs_patch_size: int = config.HFS_PATCH_SIZE,
    branch_mode: str = config.BRANCH_MODE,
    trunk_hidden: int = config.TRUNK_HIDDEN,
    trunk_layers: int = config.TRUNK_LAYERS,
) -> GIFNOXTHFSModel:
    freq = None
    if config.TF_FREQ_PATH.is_file():
        freq = np.load(config.TF_FREQ_PATH)
    return GIFNOXTHFSModel(
        in_channels=in_channels,
        latent_channels=latent_channels,
        n_freq=n_freq,
        nx=nx,
        fno_modes=fno_modes,
        num_fno_layers=num_fno_layers,
        deeponet_dim=deeponet_dim,
        hfs_patch_size=hfs_patch_size,
        branch_mode=branch_mode,
        trunk_hidden=trunk_hidden,
        trunk_layers=trunk_layers,
        freq=freq,
    )
