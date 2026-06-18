# model.py
"""FNO global encoder + U-Net local branch + XT DeepONet head."""

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


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalUNetBranch(nn.Module):
    """
    Lightweight U-Net with x-only pooling for lateral heterogeneity.

    Input/output: (B, C, Nz, Nx).
    """

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64):
        super().__init__()
        c1 = base_channels
        c2 = out_channels
        self.enc1 = ConvBlock2d(in_channels, c1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.enc2 = ConvBlock2d(c1, c2)
        self.up = nn.Upsample(scale_factor=(1, 2), mode="bilinear", align_corners=False)
        self.dec = ConvBlock2d(c1 + c2, out_channels)
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d = self.up(e2)
        if d.shape[-1] != e1.shape[-1]:
            d = nn.functional.interpolate(
                d, size=e1.shape[-2:], mode="bilinear", align_corners=False
            )
        d = self.dec(torch.cat([e1, d], dim=1))
        return self.out_proj(d)


class LatentFusion(nn.Module):
    """Fuse FNO and U-Net latent fields."""

    def __init__(self, channels: int, mode: str = "concat"):
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.proj = nn.Conv2d(2 * channels, channels, kernel_size=1)
        elif mode == "gated":
            self.gate = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unknown FUSION_MODE={mode!r}")

    def forward(self, z_fno: torch.Tensor, z_unet: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            return self.proj(torch.cat([z_fno, z_unet], dim=1))
        return z_fno + self.gate(z_unet) * z_unet


class TrunkNetwork2D(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 128, num_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        layers.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class RecorderDeepONetHeadXT(nn.Module):
    """Branch from fused latent at recorder columns; 2D trunk on (log f, x)."""

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
        x_coord_mode: str = "normalized",
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
        self.register_buffer("_log_f", torch.from_numpy(log_f), persistent=False)
        x_trunk = config.recorder_x_trunk_coords(recorder_x, nx=nx, mode=x_coord_mode)
        self.register_buffer("_x_trunk", torch.from_numpy(x_trunk), persistent=False)

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


class GIFNOLGModel(nn.Module):
    """
    FNO global + U-Net local fused latent -> XT DeepONet head.

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
        unet_base_channels: int = 64,
        fusion_mode: str = "concat",
        branch_mode: str = "surface",
        trunk_hidden: int = 128,
        trunk_layers: int = 4,
        x_coord_mode: str = "normalized",
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
        self.lift_unet = ChannelLift(in_channels, unet_base_channels)
        self.unet = LocalUNetBranch(
            in_channels=unet_base_channels,
            out_channels=latent_channels,
            base_channels=unet_base_channels,
        )
        self.fusion = LatentFusion(latent_channels, mode=fusion_mode)
        self.head = RecorderDeepONetHeadXT(
            latent_channels=latent_channels,
            nx=nx,
            n_freq=n_freq,
            recorder_x=recorder_x,
            deeponet_dim=deeponet_dim,
            branch_mode=branch_mode,
            trunk_hidden=trunk_hidden,
            trunk_layers=trunk_layers,
            x_coord_mode=x_coord_mode,
            freq=freq,
        )

    def _fno_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.fno.parameters())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self._fno_frozen():
            with torch.no_grad():
                z_fno = self.fno(self.lift(x))
        else:
            z_fno = self.fno(self.lift(x))
        z_unet = self.unet(self.lift_unet(x))
        return self.fusion(z_fno, z_unet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def create_model(
    in_channels: int = config.IN_CHANNELS,
    latent_channels: int = config.LATENT_CHANNELS,
    n_freq: int = config.N_FREQ,
    nx: int = config.NX,
    fno_modes: Tuple[int, int] = config.FNO_MODES,
    num_fno_layers: int = config.NUM_FNO_LAYERS,
    deeponet_dim: int = config.DEEPONET_LATENT_DIM,
    unet_base_channels: int = config.UNET_BASE_CHANNELS,
    fusion_mode: str = config.FUSION_MODE,
    branch_mode: str = config.BRANCH_MODE,
    trunk_hidden: int = config.TRUNK_HIDDEN,
    trunk_layers: int = config.TRUNK_LAYERS,
    x_coord_mode: str = config.X_COORD_MODE,
) -> GIFNOLGModel:
    freq_arr: np.ndarray | None = None
    if config.TF_FREQ_PATH.is_file():
        loaded = np.load(config.TF_FREQ_PATH)
        if len(loaded) == n_freq:
            freq_arr = loaded
    if freq_arr is None:
        freq_arr = np.logspace(-1, 1, n_freq)
    return GIFNOLGModel(
        in_channels=in_channels,
        latent_channels=latent_channels,
        n_freq=n_freq,
        nx=nx,
        fno_modes=fno_modes,
        num_fno_layers=num_fno_layers,
        deeponet_dim=deeponet_dim,
        unet_base_channels=unet_base_channels,
        fusion_mode=fusion_mode,
        branch_mode=branch_mode,
        trunk_hidden=trunk_hidden,
        trunk_layers=trunk_layers,
        x_coord_mode=x_coord_mode,
        freq=freq_arr,
    )
