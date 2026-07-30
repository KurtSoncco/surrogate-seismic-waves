"""TH-FNO model: direct |TF|(x, log f) on the central-500 strip (AGENTS §2).

Residual+gate path retained for ablations only (`predict_mode=\"residual\"`).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from neuralop.layers.fno_block import FNOBlocks

import config
from context_features import residual_gate_torch


class ChannelLift(nn.Module):
    def __init__(self, in_channels: int, latent_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=1),
            nn.GELU(),
        )
        # Strengthen local skip (AGENTS: pointwise W path)
        self.local = nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return h + self.local(h)


class FourierFeatureTrunk(nn.Module):
    """Tancik Fourier features on (log_f, x) → DeepONet trunk basis."""

    def __init__(
        self,
        latent_dim: int,
        n_fourier: int = 8,
        hidden: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.register_buffer(
            "B",
            torch.randn(2, n_fourier) * 2.0 * np.pi,
            persistent=True,
        )
        in_dim = 2 * n_fourier
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        layers.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords (N, 2)
        proj = coords @ self.B
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(ff)


class RecorderDeepONetFourier(nn.Module):
    """Branch from FNO latent; Fourier trunk on (log f, x)."""

    def __init__(
        self,
        latent_channels: int,
        nx: int,
        n_freq: int,
        recorder_x: np.ndarray,
        deeponet_dim: int = 64,
        branch_mode: str = "surface",
        trunk_hidden: int = 128,
        trunk_layers: int = 3,
        n_fourier: int = 8,
        use_fourier: bool = True,
        x_coord_mode: str = "normalized",
        freq: np.ndarray | None = None,
        physics_dim: int = 0,
    ):
        super().__init__()
        self.nx = nx
        self.n_freq = n_freq
        self.branch_mode = branch_mode
        self.physics_dim = physics_dim
        self.register_buffer(
            "_recorder_x",
            torch.as_tensor(recorder_x, dtype=torch.long),
            persistent=False,
        )
        if freq is None:
            freq = np.logspace(-1, 1, n_freq)
        log_f = np.log(np.maximum(freq.astype(np.float32), 1e-8))
        self.register_buffer("_log_f", torch.from_numpy(log_f), persistent=False)
        x_trunk = config.recorder_x_trunk_coords(
            recorder_x, nx=nx, mode=x_coord_mode
        )
        self.register_buffer(
            "_x_trunk", torch.from_numpy(x_trunk.astype(np.float32)), persistent=False
        )

        hidden = max(latent_channels, deeponet_dim)
        branch_in = latent_channels + physics_dim
        self.branch = nn.Sequential(
            nn.Linear(branch_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, deeponet_dim),
        )
        if use_fourier:
            self.trunk = FourierFeatureTrunk(
                latent_dim=deeponet_dim,
                n_fourier=n_fourier,
                hidden=trunk_hidden,
                num_layers=trunk_layers,
            )
        else:
            layers = [nn.Linear(2, trunk_hidden), nn.GELU()]
            for _ in range(trunk_layers - 1):
                layers.extend([nn.Linear(trunk_hidden, trunk_hidden), nn.GELU()])
            layers.append(nn.Linear(trunk_hidden, deeponet_dim))
            self.trunk = nn.Sequential(*layers)

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

    def forward(
        self, latent: torch.Tensor, physics: torch.Tensor | None = None
    ) -> torch.Tensor:
        feat = self._branch_features(latent)
        if self.physics_dim > 0:
            if physics is None:
                physics = feat.new_zeros(feat.shape[0], self.physics_dim)
            p = physics.unsqueeze(1).expand(-1, feat.shape[1], -1)
            feat = torch.cat([feat, p], dim=-1)
        branch = self.branch(feat)
        trunk = self._trunk_basis()
        tf_rec = torch.einsum("brd,rfd->brf", branch, trunk)
        out = tf_rec.new_zeros(tf_rec.shape[0], self.nx, self.n_freq)
        out.index_copy_(1, self._recorder_x, tf_rec)
        return out


def soft_clamp_log_delta(
    gated_delta: torch.Tensor, c: float
) -> torch.Tensor:
    """Bound g·Δ via Δ_eff = C * tanh(g·Δ / C). Residual factor ∈ [e^{-C}, e^{C}].

    With C = ln(3), correction is physically ≤ 3× (Session N+1 C1).
    """
    c_t = gated_delta.new_tensor(float(c))
    return c_t * torch.tanh(gated_delta / c_t)


class GatedDeltaModel(nn.Module):
    """
    residual: |TF|_pred = H_1D_trend * exp(soft_clamp(g·Δ))   (log_mult)
              |TF|_pred = H_1D_trend + g·Δ                     (additive)
    direct:   |TF|_pred = softplus(head(field, physics))

    Gate exact 0 when cov=0 and dip_rms=0 (residual mode only).
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
        trunk_layers: int = 3,
        x_coord_mode: str = "normalized",
        recorder_x: np.ndarray | None = None,
        freq: np.ndarray | None = None,
        gate_cov_ref: float = 0.1,
        gate_dip_ref: float = 0.05,
        use_fourier: bool = True,
        n_fourier: int = 8,
        physics_dim: int = 0,
        residual_mode: str = "additive",
        log_delta_c: float | None = None,
        log_delta_clamp: float | None = None,  # alias for log_delta_c
        predict_mode: str = "residual",
        zero_init_residual_head: bool = False,
    ):
        super().__init__()
        if recorder_x is None:
            recorder_x = config.recorder_x_indices()
        self.nx = nx
        self.n_freq = n_freq
        self.gate_cov_ref = gate_cov_ref
        self.gate_dip_ref = gate_dip_ref
        self.residual_mode = residual_mode
        if log_delta_c is None:
            log_delta_c = (
                float(log_delta_clamp)
                if log_delta_clamp is not None
                else float(np.log(3.0))
            )
        self.log_delta_c = float(log_delta_c)
        self.log_delta_clamp = self.log_delta_c  # compat
        self.predict_mode = predict_mode
        self.lift = ChannelLift(in_channels, latent_channels)
        self.fno = FNOBlocks(
            n_modes=fno_modes,
            in_channels=latent_channels,
            out_channels=latent_channels,
            n_layers=num_fno_layers,
            non_linearity=nn.functional.gelu,
        )
        self.head = RecorderDeepONetFourier(
            latent_channels=latent_channels,
            nx=nx,
            n_freq=n_freq,
            recorder_x=recorder_x,
            deeponet_dim=deeponet_dim,
            branch_mode=branch_mode,
            trunk_hidden=trunk_hidden,
            trunk_layers=trunk_layers,
            n_fourier=n_fourier,
            use_fourier=use_fourier,
            x_coord_mode=x_coord_mode,
            freq=freq,
            physics_dim=physics_dim,
        )
        if zero_init_residual_head and predict_mode == "residual":
            self.zero_init_residual_head()

    def zero_init_residual_head(self) -> None:
        """Final branch Linear → 0 so DeepONet raw Δ = 0 at init (Session N+1 C2).

        Then ``exp(g·Δ)=1`` (log_mult) or ``Δ=0`` (additive): prediction starts
        at the calibrated trend.
        """
        last = self.head.branch[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError(
                f"expected branch final Linear, got {type(last).__name__}"
            )
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(
        self,
        x: torch.Tensor,
        haskell_tf: torch.Tensor,
        cov: torch.Tensor,
        dip_rms: torch.Tensor,
        physics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = self.fno(self.lift(x))
        raw = self.head(latent, physics=physics)
        if self.predict_mode == "direct":
            # Direct |TF| — softplus keeps amplitudes ≥ 0
            return torch.nn.functional.softplus(raw)
        gate = residual_gate_torch(
            cov, dip_rms, cov_ref=self.gate_cov_ref, dip_ref=self.gate_dip_ref
        )
        while gate.ndim < raw.ndim:
            gate = gate.unsqueeze(-1)
        if self.residual_mode == "log_mult":
            # Soft-clamp g·Δ (not raw alone) so gate cannot sneak past the bound.
            delta_eff = soft_clamp_log_delta(gate * raw, self.log_delta_c)
            return haskell_tf * torch.exp(delta_eff)
        return haskell_tf + gate * raw


def create_model(
    n_freq: int | None = None,
    latent_channels: int | None = None,
    deeponet_dim: int | None = None,
    predict_mode: str | None = None,
) -> GatedDeltaModel:
    freq = None
    if config.TF_FREQ_PATH.is_file():
        freq = np.load(config.TF_FREQ_PATH)
    phys = config.PHYSICS_LATENT_DIM if config.USE_PHYSICS_HEAD else 0
    mode = predict_mode or config.PREDICT_MODE
    zero_init = bool(
        getattr(config, "ZERO_INIT_RESIDUAL_HEAD", True)
    ) and mode == "residual"
    return GatedDeltaModel(
        in_channels=config.IN_CHANNELS,
        latent_channels=latent_channels or config.LATENT_CHANNELS,
        n_freq=n_freq or config.N_FREQ,
        nx=config.NX,
        fno_modes=config.FNO_MODES,
        num_fno_layers=config.NUM_FNO_LAYERS,
        deeponet_dim=deeponet_dim or config.DEEPONET_LATENT_DIM,
        branch_mode=config.BRANCH_MODE,
        trunk_hidden=config.TRUNK_HIDDEN,
        trunk_layers=config.TRUNK_LAYERS,
        x_coord_mode=config.X_COORD_MODE,
        freq=freq,
        gate_cov_ref=config.GATE_COV_REF,
        gate_dip_ref=config.GATE_DIP_REF,
        use_fourier=config.USE_FOURIER_FEATURES,
        n_fourier=config.FOURIER_FREQS,
        physics_dim=phys,
        residual_mode=config.RESIDUAL_MODE,
        log_delta_c=float(getattr(config, "LOG_DELTA_C", config.LOG_DELTA_CLAMP)),
        predict_mode=mode,
        zero_init_residual_head=zero_init,
    )
