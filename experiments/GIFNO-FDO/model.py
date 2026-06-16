# model.py
"""FNO encoder + recorder-aligned DeepONet head for spatial transfer functions."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from neuralop.layers.fno_block import FNOBlocks

import config

_EPS = 1e-8


class ChannelLift(nn.Module):
    def __init__(self, in_channels: int, latent_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TrunkNetwork(nn.Module):
    """MLP trunk on log-frequency coordinates -> basis (F, D)."""

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        layers.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, log_f: torch.Tensor) -> torch.Tensor:
        """log_f: (F,) or (F, 1) -> (F, D)."""
        if log_f.ndim == 1:
            log_f = log_f.unsqueeze(-1)
        return self.net(log_f)


class RecorderDeepONetHead(nn.Module):
    """
    Branch from FNO latent at recorder columns; trunk on log(f).
    Output (B, Nx, N_freq) with zeros off-recorder (loss uses recorder mask).
    """

    def __init__(
        self,
        latent_channels: int,
        nx: int,
        n_freq: int,
        recorder_x: np.ndarray,
        deeponet_dim: int = 64,
        branch_mode: str = "surface",
        trunk_hidden: int = 128,
        trunk_layers: int = 4,
        freq: np.ndarray | None = None,
    ):
        super().__init__()
        self.nx = nx
        self.n_freq = n_freq
        self.branch_mode = branch_mode
        self.register_buffer(
            "_recorder_x",
            torch.as_tensor(recorder_x, dtype=torch.long),
            persistent=False,
        )
        if freq is None:
            freq = np.logspace(-1, 1, n_freq)
        log_f = np.log(np.maximum(freq, _EPS).astype(np.float32))
        self.register_buffer(
            "_log_f",
            torch.from_numpy(log_f),
            persistent=False,
        )

        hidden = max(latent_channels, deeponet_dim)
        self.branch = nn.Sequential(
            nn.Linear(latent_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, deeponet_dim),
        )
        self.trunk = TrunkNetwork(
            latent_dim=deeponet_dim,
            hidden=trunk_hidden,
            num_layers=trunk_layers,
        )

    def _branch_features(self, latent: torch.Tensor) -> torch.Tensor:
        """FNO latent (B, C, Nz, Nx) -> (B, R, C)."""
        if self.branch_mode == "depth":
            cols = latent.index_select(3, self._recorder_x)
            return cols.mean(dim=2).permute(0, 2, 1)
        return latent[:, :, 0, :].index_select(2, self._recorder_x).permute(0, 2, 1)

    def _scatter_recorders(self, tf_rec: torch.Tensor) -> torch.Tensor:
        """(B, R, F) -> (B, Nx, F)."""
        out = tf_rec.new_zeros(tf_rec.shape[0], self.nx, self.n_freq)
        out.index_copy_(1, self._recorder_x, tf_rec)
        return out

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        feat = self._branch_features(latent)
        branch = self.branch(feat)
        trunk = self.trunk(self._log_f)
        tf_rec = torch.einsum("brd,fd->brf", branch, trunk)
        return self._scatter_recorders(tf_rec)


class GIFNOFDOModel(nn.Module):
    """
    FNO encoder + DeepONet readout at lateral recorders.

    Input:  (B, C_in, Nz, Nx)
    Output: (B, Nx, N_freq)
    """

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
        self.fno = FNOBlocks(
            n_modes=fno_modes,
            in_channels=latent_channels,
            out_channels=latent_channels,
            n_layers=num_fno_layers,
            non_linearity=nn.functional.gelu,
        )
        self.head = RecorderDeepONetHead(
            latent_channels=latent_channels,
            nx=nx,
            n_freq=n_freq,
            recorder_x=recorder_x,
            deeponet_dim=deeponet_dim,
            branch_mode=branch_mode,
            trunk_hidden=trunk_hidden,
            trunk_layers=trunk_layers,
            freq=freq,
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
) -> GIFNOFDOModel:
    freq = None
    if config.TF_FREQ_PATH.is_file():
        freq = np.load(config.TF_FREQ_PATH)
    return GIFNOFDOModel(
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
