# model.py
"""Frozen XT base + band-targeted TF residual corrector."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

import config
from spectral_layers import FNOEncoderLoop
from xt_readout import ChannelLift, RecorderDeepONetHeadXT, TrunkNetwork2D

_EPS = 1e-8


class ResidualTFHeadXT(nn.Module):
    """
    Fusion-tower residual head: branch(latent, base TF) + 2D trunk -> delta at recorders.
    """

    def __init__(
        self,
        latent_channels: int,
        n_freq: int,
        nx: int,
        recorder_x: np.ndarray,
        deeponet_dim: int = 64,
        pred_hidden: int = 128,
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

        self.pred_tower = nn.Sequential(
            nn.Linear(n_freq, pred_hidden),
            nn.GELU(),
            nn.Linear(pred_hidden, latent_channels),
        )
        fused_in = latent_channels * 2
        hidden = max(fused_in, deeponet_dim)
        self.branch = nn.Sequential(
            nn.Linear(fused_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, deeponet_dim),
        )
        self.trunk = TrunkNetwork2D(
            latent_dim=deeponet_dim,
            hidden=trunk_hidden,
            num_layers=trunk_layers,
        )

    def _latent_branch(self, latent: torch.Tensor) -> torch.Tensor:
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

    def forward(self, latent: torch.Tensor, base_tf_rec: torch.Tensor) -> torch.Tensor:
        """Return scattered delta TF on grid (B, Nx, F)."""
        z_lat = self._latent_branch(latent)
        z_pred = self.pred_tower(base_tf_rec)
        fused = torch.cat([z_lat, z_pred], dim=-1)
        branch = self.branch(fused)
        trunk = self._trunk_basis()
        delta_rec = torch.einsum("brd,rfd->brf", branch, trunk)
        return self._scatter_recorders(delta_rec)


class XTBaseEncoder(nn.Module):
    """Lift + full-depth FNO + XT head (matches pretrained XT keys)."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        n_freq: int,
        nx: int,
        fno_modes: Tuple[int, int],
        num_fno_layers: int,
        deeponet_dim: int,
        branch_mode: str,
        trunk_hidden: int,
        trunk_layers: int,
        recorder_x: np.ndarray,
        freq: np.ndarray | None,
    ):
        super().__init__()
        log_f = None
        if freq is not None:
            log_f = np.log(np.maximum(freq, _EPS).astype(np.float32))
        x_trunk = config.recorder_x_trunk_coords(
            recorder_x, nx=nx, mode=config.X_COORD_MODE
        )
        self.lift = ChannelLift(in_channels, latent_channels)
        self.fno = FNOEncoderLoop(
            n_modes=fno_modes,
            in_channels=latent_channels,
            out_channels=latent_channels,
            n_layers=num_fno_layers,
        )
        self.head = RecorderDeepONetHeadXT(
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fno(self.lift(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


class GIFNOXTBoostModel(nn.Module):
    """TF = TF_base + eta * delta; base frozen during boost training."""

    def __init__(
        self,
        base: XTBaseEncoder,
        residual_head: ResidualTFHeadXT,
        boost_eta: float = 0.1,
    ):
        super().__init__()
        self.base = base
        self.residual_head = residual_head
        self.boost_eta = boost_eta

    def forward_base(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.base.encode(x)
        tf_base = self.base.head(latent)
        tf_base_rec = self.base.head.predict_recorders(latent)
        return tf_base, tf_base_rec, latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not any(p.requires_grad for p in self.base.parameters()):
            with torch.no_grad():
                tf_base, tf_base_rec, latent = self.forward_base(x)
        else:
            tf_base, tf_base_rec, latent = self.forward_base(x)
        delta = self.residual_head(latent, tf_base_rec)
        return tf_base + self.boost_eta * delta


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
    boost_eta: float = config.BOOST_ETA,
    pred_hidden: int = config.BOOST_PRED_HIDDEN,
) -> GIFNOXTBoostModel:
    recorder_x = config.recorder_x_indices()
    freq = np.load(config.TF_FREQ_PATH) if config.TF_FREQ_PATH.is_file() else None
    log_f = None
    if freq is not None:
        log_f = np.log(np.maximum(freq, _EPS).astype(np.float32))
    x_trunk = config.recorder_x_trunk_coords(
        recorder_x, nx=nx, mode=config.X_COORD_MODE
    )

    base = XTBaseEncoder(
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
        recorder_x=recorder_x,
        freq=freq,
    )
    residual = ResidualTFHeadXT(
        latent_channels=latent_channels,
        n_freq=n_freq,
        nx=nx,
        recorder_x=recorder_x,
        deeponet_dim=deeponet_dim,
        pred_hidden=pred_hidden,
        branch_mode=branch_mode,
        trunk_hidden=trunk_hidden,
        trunk_layers=trunk_layers,
        x_trunk=x_trunk,
        log_f=log_f,
    )
    return GIFNOXTBoostModel(base, residual, boost_eta=boost_eta)
