# xt_readout.py
"""Shared XT DeepONet readout (FNO latent -> lateral TF at recorders)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

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


class TrunkNetwork2D(nn.Module):
    """MLP trunk on (log f, x) -> basis (N, D)."""

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        layers.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class RecorderDeepONetHeadXT(nn.Module):
    """Branch from FNO latent at recorder columns; 2D trunk on (log f, x)."""

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
        x_trunk: np.ndarray | None = None,
        log_f: np.ndarray | None = None,
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
        if log_f is None:
            freq = np.logspace(-1, 1, n_freq)
            log_f = np.log(np.maximum(freq, _EPS).astype(np.float32))
        self.register_buffer(
            "_log_f",
            torch.as_tensor(log_f, dtype=torch.float32),
            persistent=False,
        )
        if x_trunk is None:
            x_trunk = np.zeros(len(recorder_x), dtype=np.float32)
        self.register_buffer(
            "_x_trunk",
            torch.as_tensor(x_trunk, dtype=torch.float32),
            persistent=False,
        )

        hidden = max(latent_channels, deeponet_dim)
        self.branch = nn.Sequential(
            nn.Linear(latent_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, deeponet_dim),
        )
        self.trunk = TrunkNetwork2D(
            latent_dim=deeponet_dim,
            hidden=trunk_hidden,
            num_layers=trunk_layers,
        )

    def _branch_features(self, latent: torch.Tensor) -> torch.Tensor:
        if self.branch_mode == "depth":
            cols = latent.index_select(3, self._recorder_x)
            return cols.mean(dim=2).permute(0, 2, 1)
        return latent[:, :, 0, :].index_select(2, self._recorder_x).permute(0, 2, 1)

    def _trunk_basis(self) -> torch.Tensor:
        r = self._x_trunk.shape[0]
        f = self._log_f.shape[0]
        log_f_grid = self._log_f.unsqueeze(0).expand(r, f).reshape(-1, 1)
        x_grid = self._x_trunk.unsqueeze(1).expand(r, f).reshape(-1, 1)
        coords = torch.cat([log_f_grid, x_grid], dim=-1)
        return self.trunk(coords).view(r, f, -1)

    def _scatter_recorders(self, tf_rec: torch.Tensor) -> torch.Tensor:
        out = tf_rec.new_zeros(tf_rec.shape[0], self.nx, self.n_freq)
        out.index_copy_(1, self._recorder_x, tf_rec)
        return out

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        branch = self.branch(self._branch_features(latent))
        trunk = self._trunk_basis()
        tf_rec = torch.einsum("brd,rfd->brf", branch, trunk)
        return self._scatter_recorders(tf_rec)

    def predict_recorders(self, latent: torch.Tensor) -> torch.Tensor:
        """Return (B, R, F) without scattering to full grid."""
        branch = self.branch(self._branch_features(latent))
        trunk = self._trunk_basis()
        return torch.einsum("brd,rfd->brf", branch, trunk)
