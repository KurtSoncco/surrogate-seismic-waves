# model.py
"""FNO encoder + position-aware 2D DeepONet trunk (log f, x) at recorders.

Supports optional scale-split dual-path encoding (macro Vs vs RF residual) and
output positivity activations for seed-conditional TF training.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.fno_block import FNOBlocks

import config

_EPS = 1e-8

# Channel layout when SCALE_SPLIT_VS:
#   0:Vs  1:Vs_macro  2:Vs_rf  3:zeta  4:x  5:z
_MACRO_CH = (1, 3, 4, 5)
_RF_CH = (2, 3, 4, 5)


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
        """coords: (N, 2) with columns [log_f, x] -> (N, D)."""
        return self.net(coords)


class RecorderDeepONetHeadXT(nn.Module):
    """
    Branch from FNO latent at recorder columns; 2D trunk on (log f, x).
    Output (B, Nx, N_freq) with zeros off-recorder.
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
        x_coord_mode: str = "normalized",
        freq: np.ndarray | None = None,
        output_activation: str = "none",
    ):
        super().__init__()
        self.nx = nx
        self.n_freq = n_freq
        self.branch_mode = branch_mode
        self.output_activation = output_activation
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
        x_trunk = config.recorder_x_trunk_coords(recorder_x, nx=nx, mode=x_coord_mode)
        self.register_buffer(
            "_x_trunk",
            torch.from_numpy(x_trunk),
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
        """FNO latent (B, C, Nz, Nx) -> (B, R, C)."""
        if self.branch_mode == "depth":
            cols = latent.index_select(3, self._recorder_x)
            return cols.mean(dim=2).permute(0, 2, 1)
        return latent[:, :, 0, :].index_select(2, self._recorder_x).permute(0, 2, 1)

    def _trunk_basis(self) -> torch.Tensor:
        """(R, F, D) trunk output for all recorder-frequency pairs."""
        r = self._x_trunk.shape[0]
        f = self._log_f.shape[0]
        log_f_grid = self._log_f.unsqueeze(0).expand(r, f).reshape(-1, 1)
        x_grid = self._x_trunk.unsqueeze(1).expand(r, f).reshape(-1, 1)
        coords = torch.cat([log_f_grid, x_grid], dim=-1)
        return self.trunk(coords).view(r, f, -1)

    def _scatter_recorders(self, tf_rec: torch.Tensor) -> torch.Tensor:
        """(B, R, F) -> (B, Nx, F)."""
        out = tf_rec.new_zeros(tf_rec.shape[0], self.nx, self.n_freq)
        out.index_copy_(1, self._recorder_x, tf_rec)
        return out

    def _apply_output_activation(self, tf: torch.Tensor) -> torch.Tensor:
        act = self.output_activation
        if act == "none":
            return tf
        if act == "softplus":
            return F.softplus(tf)
        if act == "exp":
            return torch.exp(tf.clamp(max=20.0))
        raise ValueError(f"Unknown OUTPUT_ACTIVATION={act!r}")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        branch = self.branch(self._branch_features(latent))
        trunk = self._trunk_basis()
        tf_rec = torch.einsum("brd,rfd->brf", branch, trunk)
        tf_rec = self._apply_output_activation(tf_rec)
        return self._scatter_recorders(tf_rec)


class DualPathFNOEncoder(nn.Module):
    """Macro FNO + RF FNO with gated fusion to a shared latent."""

    def __init__(
        self,
        latent_channels: int,
        fno_modes_macro: Tuple[int, int],
        num_fno_layers_macro: int,
        fno_modes_rf: Tuple[int, int],
        num_fno_layers_rf: int,
        macro_in_channels: int = 4,
        rf_in_channels: int = 4,
    ):
        super().__init__()
        self.macro_lift = ChannelLift(macro_in_channels, latent_channels)
        self.rf_lift = ChannelLift(rf_in_channels, latent_channels)
        self.macro_fno = FNOBlocks(
            n_modes=fno_modes_macro,
            in_channels=latent_channels,
            out_channels=latent_channels,
            n_layers=num_fno_layers_macro,
            non_linearity=nn.functional.gelu,
        )
        self.rf_fno = FNOBlocks(
            n_modes=fno_modes_rf,
            in_channels=latent_channels,
            out_channels=latent_channels,
            n_layers=num_fno_layers_rf,
            non_linearity=nn.functional.gelu,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2 * latent_channels, latent_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(2 * latent_channels, latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, Nz, Nx) scale-split input."""
        if x.shape[1] < 6:
            raise ValueError(
                f"Dual-path encoder expects 6-channel scale-split input, got C={x.shape[1]}"
            )
        macro_in = x[:, list(_MACRO_CH), :, :]
        rf_in = x[:, list(_RF_CH), :, :]
        h_m = self.macro_fno(self.macro_lift(macro_in))
        h_r = self.rf_fno(self.rf_lift(rf_in))
        cat = torch.cat([h_m, h_r], dim=1)
        g = self.gate(cat)
        fused = self.fuse(cat)
        return g * fused + (1.0 - g) * h_m


class GIFNOXTModel(nn.Module):
    """
    FNO encoder + 2D DeepONet readout at lateral recorders.

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
        x_coord_mode: str = "normalized",
        recorder_x: np.ndarray | None = None,
        freq: np.ndarray | None = None,
        output_activation: str = "none",
        dual_path: bool = False,
        fno_modes_rf: Tuple[int, int] | None = None,
        num_fno_layers_rf: int | None = None,
    ):
        super().__init__()
        if recorder_x is None:
            recorder_x = config.recorder_x_indices()
        self.dual_path = dual_path
        if dual_path:
            self.encoder = DualPathFNOEncoder(
                latent_channels=latent_channels,
                fno_modes_macro=fno_modes,
                num_fno_layers_macro=num_fno_layers,
                fno_modes_rf=fno_modes_rf or fno_modes,
                num_fno_layers_rf=num_fno_layers_rf or num_fno_layers,
            )
            self.lift = None
            self.fno = None
        else:
            self.lift = ChannelLift(in_channels, latent_channels)
            self.fno = FNOBlocks(
                n_modes=fno_modes,
                in_channels=latent_channels,
                out_channels=latent_channels,
                n_layers=num_fno_layers,
                non_linearity=nn.functional.gelu,
            )
            self.encoder = None
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
            output_activation=output_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dual_path:
            latent = self.encoder(x)
        else:
            latent = self.fno(self.lift(x))
        return self.head(latent)


def create_model(
    in_channels: int | None = None,
    latent_channels: int | None = None,
    n_freq: int | None = None,
    nx: int | None = None,
    fno_modes: Tuple[int, int] | None = None,
    num_fno_layers: int | None = None,
    deeponet_dim: int | None = None,
    branch_mode: str | None = None,
    trunk_hidden: int | None = None,
    trunk_layers: int | None = None,
    x_coord_mode: str | None = None,
    output_activation: str | None = None,
    dual_path: bool | None = None,
) -> GIFNOXTModel:
    """Factory that reads live ``config`` values (env / recipe overrides)."""
    if in_channels is None:
        in_channels = config.IN_CHANNELS
    if latent_channels is None:
        latent_channels = config.LATENT_CHANNELS
    if n_freq is None:
        n_freq = config.N_FREQ
    if nx is None:
        nx = config.NX
    if fno_modes is None:
        fno_modes = config.FNO_MODES
    if num_fno_layers is None:
        num_fno_layers = config.NUM_FNO_LAYERS
    if deeponet_dim is None:
        deeponet_dim = config.DEEPONET_LATENT_DIM
    if branch_mode is None:
        branch_mode = config.BRANCH_MODE
    if trunk_hidden is None:
        trunk_hidden = config.TRUNK_HIDDEN
    if trunk_layers is None:
        trunk_layers = config.TRUNK_LAYERS
    if x_coord_mode is None:
        x_coord_mode = config.X_COORD_MODE
    freq = None
    if config.TF_FREQ_PATH.is_file():
        freq = np.load(config.TF_FREQ_PATH)
    if output_activation is None:
        output_activation = getattr(config, "OUTPUT_ACTIVATION", "none")
    if dual_path is None:
        dual_path = bool(
            getattr(config, "DUAL_PATH_ENCODER", False)
            and getattr(config, "SCALE_SPLIT_VS", False)
        )
    return GIFNOXTModel(
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
        x_coord_mode=x_coord_mode,
        freq=freq,
        output_activation=output_activation,
        dual_path=dual_path,
        fno_modes_rf=getattr(config, "FNO_MODES_RF", fno_modes),
        num_fno_layers_rf=getattr(config, "NUM_FNO_LAYERS_RF", num_fno_layers),
    )
