"""Single- vs multi-branch DeepONet for signed residual R(x, f)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConvFieldEncoder(nn.Module):
    """Shallow Conv2d encoder: stacked material fields (B, C, Nz, Nr) → vector."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 32,
        out_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(hidden * 2 * 4 * 4, out_dim),
            nn.GELU(),
        )

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        return self.net(fields)


# Backward-compatible alias
FieldEncoder = ConvFieldEncoder


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.res = _ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.res(self.proj(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False)
        self.res = _ResBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(self.proj(x))


class ResUNetFieldEncoder(nn.Module):
    """Residual U-Net field encoder → global vector for DeepONet branch.

    Input: (B, C, Nz, Nr) material stack (typically Nz=128, Nr=21).
    Output: (B, out_dim).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base: int = 32,
        out_dim: int = 128,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.GELU(),
            _ResBlock(base),
        )
        self.down1 = _Down(base, base * 2)
        self.down2 = _Down(base * 2, base * 4)
        self.bottleneck = _ResBlock(base * 4)
        self.up1 = _Up(base * 4, base * 2, base * 2)
        self.up2 = _Up(base * 2, base, base)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base, out_dim),
            nn.GELU(),
        )

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        # Pad so depth/recorder dims tolerate two 2× pools (need even sizes).
        _, _, h, w = fields.shape
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h or pad_w:
            fields = F.pad(fields, (0, pad_w, 0, pad_h), mode="replicate")
        # Second down also benefits from even size after first pool — pad again if needed
        e0 = self.stem(fields)
        e1 = self.down1(e0)
        _, _, h1, w1 = e1.shape
        pad_h1 = (2 - h1 % 2) % 2
        pad_w1 = (2 - w1 % 2) % 2
        e1_p = (
            F.pad(e1, (0, pad_w1, 0, pad_h1), mode="replicate")
            if (pad_h1 or pad_w1)
            else e1
        )
        e2 = self.down2(e1_p)
        b = self.bottleneck(e2)
        d1 = self.up1(b, e1)
        d0 = self.up2(d1, e0)
        return self.head(d0)


FieldEncoderKind = Literal[
    "conv", "resunet", "gno", "attn", "gat", "identity", "kernel"
]
ColEncKind = Literal["conv", "mlp", "attn"]
StochInjectKind = Literal["mlp", "concat"]
FuseKind = Literal["mlp", "add"]
COL_ENC_POOL_BINS = 16
FNOKind = Literal[
    "vanilla", "ufno", "ffno", "afno", "wno", "fno1d", "loglo", "tf", "band2"
]


def build_field_encoder(
    kind: FieldEncoderKind,
    *,
    in_channels: int,
    hidden: int,
    out_dim: int,
) -> nn.Module:
    if kind == "gno":
        raise ValueError("gno is a full DeepONet, not a vector field encoder")
    if kind == "resunet":
        return ResUNetFieldEncoder(
            in_channels=in_channels, base=hidden, out_dim=out_dim
        )
    return ConvFieldEncoder(in_channels=in_channels, hidden=hidden, out_dim=out_dim)


class TrunkMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        layers.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


# Full trunk: f_star, sin_f, cos_f, x_over_lambda [, log TF1D]. Scale coords only.
FULL_TRUNK_COORD_DIMS: tuple[int, ...] = (0, 3)


def mscale_factors(n_scales: int) -> tuple[float, ...]:
    """Fixed dyadic scales β or α = 1, 2, 4, … (arXiv:2504.10932)."""
    n = max(int(n_scales), 1)
    return tuple(float(2**i) for i in range(n))


def mscale_hidden(hidden: int, n_scales: int) -> int:
    """Shrink each subnet so S parallel trunks stay near the single-trunk budget."""
    n = max(int(n_scales), 1)
    if n <= 1:
        return int(hidden)
    return max(32, int(hidden) // n)


def mscale_split_latent(latent_dim: int, n_scales: int) -> list[int]:
    """Per-subnet output widths that concatenate back to ``latent_dim``."""
    n = max(int(n_scales), 1)
    base, rem = divmod(int(latent_dim), n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _scale_coord_channels(
    y: torch.Tensor, beta: float, coord_dims: tuple[int, ...]
) -> torch.Tensor:
    if abs(float(beta) - 1.0) < 1e-12 or not coord_dims:
        return y
    y_s = y.clone()
    for d in coord_dims:
        if 0 <= d < y_s.shape[-1]:
            y_s[..., d] = float(beta) * y[..., d]
    return y_s


class MscaleTrunk(nn.Module):
    """Parallel trunk MLPs on scaled coordinate channels (arXiv:2504.10932 eq. 12).

    ``S=1`` is not used: callers keep a plain ``TrunkMLP`` so shipped checkpoints
    stay byte-identical. ``S>1`` concatenates subnet outputs to ``latent_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden: int = 128,
        num_layers: int = 4,
        n_scales: int = 4,
        coord_dims: tuple[int, ...] = FULL_TRUNK_COORD_DIMS,
    ):
        super().__init__()
        self.n_scales = max(int(n_scales), 1)
        self.betas = mscale_factors(self.n_scales)
        self.coord_dims = tuple(int(i) for i in coord_dims)
        self.latent_dim = int(latent_dim)
        widths = mscale_split_latent(latent_dim, self.n_scales)
        h = mscale_hidden(hidden, self.n_scales)
        self.subnets = nn.ModuleList(
            [TrunkMLP(input_dim, w, h, num_layers) for w in widths]
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        parts = [
            net(_scale_coord_channels(y, beta, self.coord_dims))
            for net, beta in zip(self.subnets, self.betas)
        ]
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]


def _build_trunk(
    input_dim: int,
    latent_dim: int,
    hidden: int,
    num_layers: int,
    n_scales: int,
    coord_dims: tuple[int, ...],
) -> nn.Module:
    if max(int(n_scales), 1) <= 1:
        return TrunkMLP(input_dim, latent_dim, hidden, num_layers)
    return MscaleTrunk(
        input_dim,
        latent_dim,
        hidden,
        num_layers,
        n_scales=n_scales,
        coord_dims=coord_dims,
    )


def _make_fuse(in_dim: int, hidden: int, out_dim: int, *, deep: bool) -> nn.Sequential:
    if deep:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class SingleBranchDeepONet(nn.Module):
    """Park et al. style: shared branch fuses fields + stochastic early."""

    def __init__(
        self,
        *,
        field_channels: int,
        stoch_dim: int,
        trunk_dim: int,
        latent_dim: int = 64,
        field_hidden: int = 32,
        branch_hidden: int = 128,
        trunk_hidden: int = 128,
        trunk_layers: int = 4,
        use_fields: bool = True,
        use_stoch: bool = True,
        field_encoder: FieldEncoderKind = "conv",
        n_mscale_trunk: int = 1,
        n_mscale_branch: int = 1,
        mscale_coord_dims: tuple[int, ...] = FULL_TRUNK_COORD_DIMS,
    ):
        super().__init__()
        if not use_fields and not use_stoch:
            raise ValueError("Need at least fields or stochastic inputs")
        self.use_fields = use_fields
        self.use_stoch = use_stoch
        self.latent_dim = latent_dim
        self.field_encoder_kind = field_encoder
        self.n_mscale_trunk = max(int(n_mscale_trunk), 1)
        self.n_mscale_branch = max(int(n_mscale_branch), 1)
        self.branch_alphas = mscale_factors(self.n_mscale_branch)

        field_out = latent_dim if use_fields else 0
        self.field_enc = (
            build_field_encoder(
                field_encoder,
                in_channels=field_channels,
                hidden=field_hidden,
                out_dim=field_out,
            )
            if use_fields
            else None
        )
        in_dim = field_out + (stoch_dim if use_stoch else 0)
        self.fuse = _make_fuse(in_dim, branch_hidden, latent_dim, deep=True)
        if self.n_mscale_branch > 1:
            self.fuses = nn.ModuleList(
                [
                    _make_fuse(in_dim, branch_hidden, latent_dim, deep=True)
                    for _ in range(self.n_mscale_branch)
                ]
            )
            self.fuse = self.fuses[0]
        else:
            self.fuses = None
        self.trunk = _build_trunk(
            trunk_dim,
            latent_dim,
            trunk_hidden,
            trunk_layers,
            self.n_mscale_trunk,
            mscale_coord_dims,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def _fuse_one(
        self,
        fuse: nn.Module,
        fields: torch.Tensor | None,
        stoch: torch.Tensor | None,
        alpha: float,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.use_fields:
            assert fields is not None
            scaled = fields if abs(alpha - 1.0) < 1e-12 else fields * alpha
            parts.append(self.field_enc(scaled))
        if self.use_stoch:
            assert stoch is not None
            parts.append(stoch)
        return fuse(torch.cat(parts, dim=-1))

    def branch(
        self,
        fields: torch.Tensor | None,
        stoch: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.fuses is None:
            return self._fuse_one(self.fuse, fields, stoch, 1.0)
        p = self._fuse_one(self.fuses[0], fields, stoch, self.branch_alphas[0])
        for fuse, alpha in zip(self.fuses[1:], self.branch_alphas[1:]):
            p = p + self._fuse_one(fuse, fields, stoch, alpha)
        return p

    def forward(
        self,
        fields: torch.Tensor | None,
        stoch: torch.Tensor | None,
        trunk_y: torch.Tensor,
    ) -> torch.Tensor:
        p = self.branch(fields, stoch)
        bq = self.trunk(trunk_y.reshape(-1, trunk_y.shape[-1])).reshape(
            trunk_y.shape[0], trunk_y.shape[1], self.latent_dim
        )
        return (p.unsqueeze(1) * bq).sum(dim=-1) + self.bias


class MultiBranchDeepONet(nn.Module):
    """MIONet-style: separate encoders per field channel + stochastic."""

    def __init__(
        self,
        *,
        n_field_channels: int,
        stoch_dim: int,
        trunk_dim: int,
        latent_dim: int = 64,
        field_hidden: int = 32,
        branch_hidden: int = 128,
        trunk_hidden: int = 128,
        trunk_layers: int = 4,
        field_encoder: FieldEncoderKind = "conv",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_field_channels = n_field_channels
        self.field_encs = nn.ModuleList(
            [
                build_field_encoder(
                    field_encoder,
                    in_channels=1,
                    hidden=field_hidden,
                    out_dim=latent_dim,
                )
                for _ in range(n_field_channels)
            ]
        )
        self.stoch_mlp = nn.Sequential(
            nn.Linear(stoch_dim, branch_hidden),
            nn.GELU(),
            nn.Linear(branch_hidden, latent_dim),
        )
        self.trunk = TrunkMLP(trunk_dim, latent_dim, trunk_hidden, trunk_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        fields: torch.Tensor,
        stoch: torch.Tensor,
        trunk_y: torch.Tensor,
    ) -> torch.Tensor:
        p = self.stoch_mlp(stoch)
        for c, enc in enumerate(self.field_encs):
            p = p * enc(fields[:, c : c + 1])
        bq = self.trunk(trunk_y.reshape(-1, trunk_y.shape[-1])).reshape(
            trunk_y.shape[0], trunk_y.shape[1], self.latent_dim
        )
        return (p.unsqueeze(1) * bq).sum(dim=-1) + self.bias


class _ColumnEncoder(nn.Module):
    """Per-recorder 1D conv down the depth: (B, C, Nz, Nr) → (B, Nr, out_dim).

    ``depth_tokens=1`` (ship) globally pools each column. ``depth_tokens>1``
    keeps that many depth bins in the node vector so RF structure along z
    is not discarded before GNO.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_dim: int,
        depth_tokens: int = 1,
    ):
        super().__init__()
        self.depth_tokens = max(1, int(depth_tokens))
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(self.depth_tokens),
        )
        self.proj = nn.Linear(hidden * 2 * self.depth_tokens, out_dim)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        b, c, nz, nr = fields.shape
        x = fields.permute(0, 3, 1, 2).reshape(b * nr, c, nz)
        h = self.net(x).reshape(b * nr, -1)
        return self.proj(h).view(b, nr, -1)


class _ColumnMLPEncoder(nn.Module):
    """Pool depth to ``n_bins`` then Linear(C×bins → out_dim). No Conv1d."""

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        n_bins: int = COL_ENC_POOL_BINS,
    ):
        super().__init__()
        self.n_bins = int(n_bins)
        self.pool = nn.AdaptiveAvgPool1d(self.n_bins)
        self.proj = nn.Linear(in_channels * self.n_bins, out_dim)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        b, c, nz, nr = fields.shape
        x = fields.permute(0, 3, 1, 2).reshape(b * nr, c, nz)
        h = self.pool(x).reshape(b * nr, -1)
        return self.proj(h).view(b, nr, -1)


class _ColumnAttnEncoder(nn.Module):
    """Pool depth to tokens, 2-layer TransformerEncoder, mean-pool, project."""

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        n_bins: int = COL_ENC_POOL_BINS,
        d_model: int = 64,
        n_layers: int = 2,
        nhead: int = 4,
    ):
        super().__init__()
        self.n_bins = int(n_bins)
        self.pool = nn.AdaptiveAvgPool1d(self.n_bins)
        self.in_proj = nn.Linear(in_channels, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(4 * d_model, 32),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        b, c, nz, nr = fields.shape
        x = fields.permute(0, 3, 1, 2).reshape(b * nr, c, nz)
        tok = self.pool(x).permute(0, 2, 1)
        h = self.enc(self.in_proj(tok)).mean(dim=1)
        return self.out_proj(h).view(b, nr, -1)


def _build_column_encoder(
    kind: ColEncKind,
    *,
    in_channels: int,
    hidden: int,
    out_dim: int,
    depth_tokens: int,
) -> nn.Module:
    if kind == "mlp":
        return _ColumnMLPEncoder(in_channels, out_dim)
    if kind == "attn":
        return _ColumnAttnEncoder(in_channels, out_dim)
    return _ColumnEncoder(in_channels, hidden, out_dim, depth_tokens=depth_tokens)


class _ChainGNO(nn.Module):
    """kNN=2 message passing along the recorder line (no periodic wrap).

    Optional ``rh_dilate`` adds a second skip to recorders ±d, with
    d = clip(round(r_H / 25 m), 1, 8). Local hops stay kNN=2 so 3-layer
    M7680 weights load; dilated MLPs are randomly initialized.
    """

    def __init__(self, dim: int, n_layers: int = 3, rh_dilate: bool = False):
        super().__init__()
        self.rh_dilate = bool(rh_dilate)
        self._dilation: torch.Tensor | None = None
        self.layers = nn.ModuleList([_gno_update_mlp(dim) for _ in range(n_layers)])
        self.dilate_layers = (
            nn.ModuleList([_gno_update_mlp(dim) for _ in range(n_layers)])
            if self.rh_dilate
            else None
        )

    @staticmethod
    def _neighbors(x: torch.Tensor) -> torch.Tensor:
        left = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        right = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
        return 0.5 * (left + right)

    @staticmethod
    def _gather_shift(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        steps = (
            d.reshape(-1).to(device=x.device, dtype=torch.long).clamp(1, max(n - 1, 1))
        )
        if steps.numel() == 1 and b > 1:
            steps = steps.expand(b)
        idx = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)
        left_i = (idx - steps.unsqueeze(1)).clamp(0, n - 1)
        right_i = (idx + steps.unsqueeze(1)).clamp(0, n - 1)
        left = torch.gather(x, 1, left_i.unsqueeze(-1).expand(b, n, c))
        right = torch.gather(x, 1, right_i.unsqueeze(-1).expand(b, n, c))
        return 0.5 * (left + right)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            msg = self._neighbors(x)
            x = x + layer(torch.cat([x, msg], dim=-1))
            if self.rh_dilate and self.dilate_layers is not None:
                d = self._dilation
                if d is None:
                    d = torch.ones(x.shape[0], dtype=torch.long, device=x.device)
                dmsg = self._gather_shift(x, d)
                x = x + self.dilate_layers[i](torch.cat([x, dmsg], dim=-1))
        return x


LATERAL_SPACING_M = float(config.LATERAL_SPACING_M)  # 15 m array
# r_H-dilated chain GNO was trained with 25 m hops ("21 stations / 500 m").
# That is not the recorder spacing. Kernel GNO uses physical x (m) instead.
GNO_DILATION_SPACING_M = 25.0
RECORDER_SPACING_M = GNO_DILATION_SPACING_M
GNO_DILATION_LO = 1
GNO_DILATION_HI = 8
KERNEL_K = 2
KERNEL_TAU_M = LATERAL_SPACING_M
N_LATENT_X = 32


def gno_dilation_steps(
    rH: float,
    *,
    spacing: float = RECORDER_SPACING_M,
    lo: int = GNO_DILATION_LO,
    hi: int = GNO_DILATION_HI,
) -> int:
    """Recorder hops for an r_H-scaled dilation skip."""
    rh = float(rH)
    if not np.isfinite(rh) or rh <= 0:
        return int(lo)
    return int(np.clip(np.round(rh / float(spacing)), lo, hi))


def apply_gno_dilation(module: nn.Module, rH: torch.Tensor | None) -> None:
    """Push per-sample r_H dilation onto every dilated ``_ChainGNO``."""
    d: torch.Tensor | None = None
    if rH is not None:
        if not torch.is_tensor(rH):
            rH = torch.as_tensor(rH, dtype=torch.float32)
        finite = torch.nan_to_num(rH.reshape(-1).to(dtype=torch.float32), nan=0.0)
        d = (
            torch.round(finite / GNO_DILATION_SPACING_M)
            .clamp(GNO_DILATION_LO, GNO_DILATION_HI)
            .long()
        )
    for m in module.modules():
        if isinstance(m, _ChainGNO):
            if d is None or not m.rh_dilate:
                m._dilation = None
                continue
            param = next(m.parameters(), None)
            device = param.device if param is not None else d.device
            m._dilation = d.to(device=device, dtype=torch.long)


def _gno_update_mlp(dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2 * dim, dim),
        nn.GELU(),
        nn.Linear(dim, dim),
    )


def _as_batch_x(x: torch.Tensor, batch: int) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0).expand(batch, -1).contiguous()
    return x


class KernelGNO(nn.Module):
    """Distance-kNN kernel integral: p(x_q) from support nodes, not fixed adjacency.

    Neighbors are the k nearest support columns in physical metres. Each
    neighbor is ``MLP([node, Δx, |Δx|])`` then softmax-weighted by ``-|Δx|/τ``.
    Equal distances break ties by physical x so a permutation of support is a
    no-op. Residual MLPs after the gather match the chain-GNO layer count.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 3,
        k: int = KERNEL_K,
        tau_m: float = KERNEL_TAU_M,
    ):
        super().__init__()
        self.k = max(1, int(k))
        self.tau_m = float(tau_m) if float(tau_m) > 0 else float(KERNEL_TAU_M)
        self.mix = nn.Sequential(
            nn.Linear(dim + 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        extra = max(int(n_layers) - 1, 0)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
                for _ in range(extra)
            ]
        )

    def forward(
        self,
        nodes: torch.Tensor,
        support_x: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        b, n_s, _dim = nodes.shape
        sx = _as_batch_x(support_x.to(device=nodes.device, dtype=nodes.dtype), b)
        qx = _as_batch_x(query_x.to(device=nodes.device, dtype=nodes.dtype), b)
        n_q = int(qx.shape[1])
        k = min(self.k, n_s)
        dist = (qx.unsqueeze(-1) - sx.unsqueeze(1)).abs()
        # Tie-break kNN by physical x so a permutation of support is a no-op.
        x_ord = sx.argsort(dim=-1).argsort(dim=-1).to(dtype=dist.dtype)
        _, idx = (dist * float(n_s + 1) + x_ord.unsqueeze(1)).topk(
            k, dim=-1, largest=False
        )
        b_ix = torch.arange(b, device=nodes.device).view(b, 1, 1).expand(b, n_q, k)
        neigh = nodes[b_ix, idx]
        sx_k = sx[b_ix, idx]
        dx = qx.unsqueeze(-1) - sx_k
        feat = torch.cat([neigh, dx.unsqueeze(-1), dx.abs().unsqueeze(-1)], dim=-1)
        h = self.mix(feat)
        d_k = dist.gather(-1, idx)
        w = torch.softmax(-d_k / self.tau_m, dim=-1)
        p = (w.unsqueeze(-1) * h).sum(dim=2)
        for layer in self.layers:
            p = p + layer(p)
        return p


def _n_heads(dim: int) -> int:
    for h in (8, 4, 2, 1):
        if dim % h == 0:
            return h
    return 1


class _RecorderAttn(nn.Module):
    """GNOT / Transolver-lite: self-attention over the 21-recorder line.

    Full mesh Transolver is for irregular PDE grids; here the leftover lives on
    a short 1D station chain, so token attention is cheap (Nr²) and replaces
    kNN=2 message passing.
    """

    def __init__(self, dim: int, n_layers: int = 3, max_nodes: int = 64):
        super().__init__()
        nhead = _n_heads(dim)
        self.pos = nn.Parameter(torch.zeros(1, max_nodes, dim))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=nhead,
                    dim_feedforward=max(4 * dim, 32),
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nr = x.shape[1]
        x = x + self.pos[:, :nr]
        for layer in self.layers:
            x = layer(x)
        return x


class _GATLayer(nn.Module):
    """Local GAT on {left, self, right} — keeps the chain graph GNO uses."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, dim, bias=False)
        self.a = nn.Linear(2 * dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w(x)
        left = torch.cat([h[:, :1], h[:, :-1]], dim=1)
        right = torch.cat([h[:, 1:], h[:, -1:]], dim=1)
        e = torch.cat(
            [
                self.a(torch.cat([h, h], dim=-1)),
                self.a(torch.cat([h, left], dim=-1)),
                self.a(torch.cat([h, right], dim=-1)),
            ],
            dim=-1,
        )
        alpha = e.softmax(dim=-1)
        out = alpha[..., 0:1] * h + alpha[..., 1:2] * left + alpha[..., 2:3] * right
        return F.gelu(out)


class _RecorderGAT(nn.Module):
    """Veličković GAT on the kNN=2 recorder line (local, unlike dense attn)."""

    def __init__(self, dim: int, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([_GATLayer(dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class RecorderGNODeepONet(nn.Module):
    """DeepONet whose branch is per-recorder after chain GNO (lateral leftover).

    Optional OrbitAll-style physics tokens: Haskell log|TF1D| (last trunk channel
    when serial) and layer/dip flags are fused into GNO nodes. Optional learned
    1D head predicts Haskell TF1D from the same latent (must stay faithful to
    Haskell, not OpenSees).
    """

    def __init__(
        self,
        *,
        field_channels: int,
        stoch_dim: int,
        trunk_dim: int,
        latent_dim: int = 64,
        field_hidden: int = 32,
        branch_hidden: int = 128,
        trunk_hidden: int = 128,
        trunk_layers: int = 4,
        n_gno_layers: int = 3,
        node_mixer: Literal["gno", "attn", "gat", "identity", "kernel"] = "gno",
        physics_tokens: bool = False,
        geom_flag_dim: int = 0,
        learned_1d: bool = False,
        col_enc_depth_tokens: int = 1,
        n_mscale_trunk: int = 1,
        n_mscale_branch: int = 1,
        mscale_coord_dims: tuple[int, ...] = FULL_TRUNK_COORD_DIMS,
        col_enc: ColEncKind = "conv",
        stoch_inject: StochInjectKind = "mlp",
        fuse_kind: FuseKind = "mlp",
        gno_rh_dilate: bool = False,
        kernel_k: int = KERNEL_K,
        kernel_tau_m: float = KERNEL_TAU_M,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.physics_tokens = bool(physics_tokens)
        self.geom_flag_dim = int(geom_flag_dim)
        self.learned_1d = bool(learned_1d)
        self.accepts_geom_flags = self.geom_flag_dim > 0
        self.n_mscale_trunk = max(int(n_mscale_trunk), 1)
        self.n_mscale_branch = max(int(n_mscale_branch), 1)
        self.branch_alphas = mscale_factors(self.n_mscale_branch)
        self.col_enc_kind = str(col_enc)
        self.stoch_inject = str(stoch_inject)
        self.fuse_kind = str(fuse_kind)
        self.gno_rh_dilate = bool(gno_rh_dilate)
        self.kernel_k = max(1, int(kernel_k))
        self.kernel_tau_m = float(kernel_tau_m)
        if self.fuse_kind == "add" and self.stoch_inject != "mlp":
            raise ValueError(
                "fuse=add requires stoch-inject=mlp so node and stoch dims match"
            )
        self.field_encoder_kind = (
            "attn"
            if node_mixer == "attn"
            else (
                "gat"
                if node_mixer == "gat"
                else (
                    "identity"
                    if node_mixer == "identity"
                    else ("kernel" if node_mixer == "kernel" else "gno")
                )
            )
        )
        self.accepts_query_x = node_mixer == "kernel"
        self.col_enc = _build_column_encoder(
            col_enc,
            in_channels=field_channels,
            hidden=field_hidden,
            out_dim=latent_dim,
            depth_tokens=col_enc_depth_tokens,
        )
        if node_mixer == "attn":
            self.gno = _RecorderAttn(latent_dim, n_layers=n_gno_layers)
        elif node_mixer == "gat":
            self.gno = _RecorderGAT(latent_dim, n_layers=n_gno_layers)
        elif node_mixer == "identity":
            self.gno = nn.Identity()
        elif node_mixer == "kernel":
            self.gno = KernelGNO(
                latent_dim,
                n_layers=n_gno_layers,
                k=self.kernel_k,
                tau_m=self.kernel_tau_m,
            )
        else:
            self.gno = _ChainGNO(
                latent_dim, n_layers=n_gno_layers, rh_dilate=self.gno_rh_dilate
            )
        if self.stoch_inject == "concat":
            self.stoch_mlp = None
            fuse_in = latent_dim + int(stoch_dim)
        else:
            self.stoch_mlp = nn.Sequential(nn.Linear(stoch_dim, latent_dim), nn.GELU())
            fuse_in = 2 * latent_dim
        if self.fuse_kind == "add":
            self.fuse = nn.Identity()
            self.fuses = None
        else:
            self.fuse = _make_fuse(fuse_in, branch_hidden, latent_dim, deep=False)
            if self.n_mscale_branch > 1:
                self.fuses = nn.ModuleList(
                    [
                        _make_fuse(fuse_in, branch_hidden, latent_dim, deep=False)
                        for _ in range(self.n_mscale_branch)
                    ]
                )
                self.fuse = self.fuses[0]
            else:
                self.fuses = None
        self.trunk = _build_trunk(
            trunk_dim,
            latent_dim,
            trunk_hidden,
            trunk_layers,
            self.n_mscale_trunk,
            mscale_coord_dims,
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.phys_mlp: nn.Module | None = None
        if self.physics_tokens:
            self.phys_mlp = nn.Sequential(
                nn.Linear(32, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
        self.geom_mlp: nn.Module | None = None
        if self.geom_flag_dim > 0:
            self.geom_mlp = nn.Sequential(
                nn.Linear(self.geom_flag_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
        self.tf1d_head: nn.Module | None = None
        if self.learned_1d:
            self.tf1d_head = nn.Linear(latent_dim, 1)
        self.last_nodes: torch.Tensor | None = None
        self.last_tf1d_hat: torch.Tensor | None = None

    def _encode_nodes(
        self,
        fields: torch.Tensor,
        trunk_y: torch.Tensor,
        geom_flags: torch.Tensor | None,
        n_query: int,
        n_freq: int,
        query_x: torch.Tensor | None = None,
        support_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nodes = self.col_enc(fields)
        if self.geom_mlp is not None:
            flags = geom_flags
            if flags is None:
                flags = torch.zeros(
                    trunk_y.shape[0],
                    self.geom_flag_dim,
                    device=trunk_y.device,
                    dtype=trunk_y.dtype,
                )
            nodes = nodes + self.geom_mlp(flags).unsqueeze(1)
        if isinstance(self.gno, KernelGNO):
            if query_x is None:
                query_x = torch.arange(
                    nodes.shape[1], device=nodes.device, dtype=nodes.dtype
                )
            if support_x is None:
                support_x = torch.arange(
                    nodes.shape[1], device=nodes.device, dtype=nodes.dtype
                )
            return self.gno(nodes, support_x, query_x)
        if self.phys_mlp is not None and nodes.shape[1] == n_query:
            log_tf = trunk_y[..., -1].reshape(trunk_y.shape[0], n_query, n_freq)
            pooled = F.adaptive_avg_pool1d(log_tf, 32)
            nodes = nodes + self.phys_mlp(pooled)
        return self.gno(nodes)

    def forward(
        self,
        fields: torch.Tensor | None,
        stoch: torch.Tensor | None,
        trunk_y: torch.Tensor,
        geom_flags: torch.Tensor | None = None,
        query_x: torch.Tensor | None = None,
        support_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert fields is not None and stoch is not None
        n_q = trunk_y.shape[1]
        if query_x is not None:
            n_query = int(query_x.shape[-1])
        else:
            n_query = int(fields.shape[-1])
        n_freq = n_q // n_query
        if self.stoch_mlp is None:
            s = stoch.unsqueeze(1).expand(-1, n_query, -1)
        else:
            s = self.stoch_mlp(stoch).unsqueeze(1).expand(-1, n_query, -1)

        def _fuse(nodes_i: torch.Tensor, fuse_mod: nn.Module) -> torch.Tensor:
            if self.fuse_kind == "add":
                return nodes_i + s
            return fuse_mod(torch.cat([nodes_i, s], dim=-1))

        enc_kw = dict(
            query_x=query_x,
            support_x=support_x,
        )
        if self.n_mscale_branch <= 1:
            nodes = self._encode_nodes(
                fields, trunk_y, geom_flags, n_query, n_freq, **enc_kw
            )
            p = _fuse(nodes, self.fuse)
        else:
            p = None
            nodes = None
            fuse_mods = (
                list(self.fuses)
                if self.fuses is not None
                else [self.fuse] * self.n_mscale_branch
            )
            for fuse, alpha in zip(fuse_mods, self.branch_alphas):
                scaled = fields if abs(alpha - 1.0) < 1e-12 else fields * alpha
                nodes_i = self._encode_nodes(
                    scaled, trunk_y, geom_flags, n_query, n_freq, **enc_kw
                )
                if abs(alpha - 1.0) < 1e-12:
                    nodes = nodes_i
                contrib = _fuse(nodes_i, fuse)
                p = contrib if p is None else p + contrib
            if nodes is None:
                nodes = nodes_i
        self.last_nodes = nodes
        p_q = (
            p.unsqueeze(2)
            .expand(-1, n_query, n_freq, -1)
            .reshape(trunk_y.shape[0], n_q, self.latent_dim)
        )
        bq = self.trunk(trunk_y.reshape(-1, trunk_y.shape[-1])).reshape(
            trunk_y.shape[0], n_q, self.latent_dim
        )
        self.last_branch = p
        self.last_trunk = bq
        if self.tf1d_head is not None:
            self.last_tf1d_hat = self.tf1d_head(p_q).squeeze(-1)
        else:
            self.last_tf1d_hat = None
        return (p_q * bq).sum(dim=-1) + self.bias


def gno_core(module: nn.Module) -> nn.Module:
    """Unwrap FNO / gate / POD / boost wrappers down to RecorderGNODeepONet when present."""
    inner = module
    for _ in range(6):
        nxt = (
            getattr(inner, "base", None)
            or getattr(inner, "inner", None)
            or getattr(inner, "booster", None)
        )
        if nxt is None:
            break
        inner = nxt
    return inner


def _spatial_kwargs(
    module: nn.Module,
    *,
    geom_flags: torch.Tensor | None = None,
    query_x: torch.Tensor | None = None,
    support_x: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {}
    if geom_flags is not None and getattr(module, "accepts_geom_flags", False):
        kwargs["geom_flags"] = geom_flags
    if getattr(module, "accepts_query_x", False):
        if query_x is not None:
            kwargs["query_x"] = query_x
        if support_x is not None:
            kwargs["support_x"] = support_x
    return kwargs


def freeze_gno_encoder(module: nn.Module) -> int:
    """Freeze column encoder + chain GNO. Leftover head / fuse / trunk stay trainable.

    Kernel GNO is a new mixer: ``--freeze-gno`` still freezes ``col_enc`` but
    leaves the distance kernel trainable until it has its own scale run.
    """
    core = gno_core(module)
    n = 0
    names = ["col_enc"]
    gno = getattr(core, "gno", None)
    if gno is not None and not isinstance(gno, KernelGNO):
        names.append("gno")
    for name in names:
        sub = getattr(core, name, None)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad = False
            n += int(p.numel())
    return n


def freeze_fno_head(module: nn.Module) -> int:
    """Freeze DeepONetFNO leftover head; col_enc / GNO / fuse / trunk stay trainable."""
    n = 0
    for m in module.modules():
        if not isinstance(m, DeepONetFNO):
            continue
        for name in (
            "lift",
            "proj",
            "proj_high",
            "fno",
            "fno_low",
            "fno_high",
            "blocks",
            "local",
            "loglo",
            "axial",
        ):
            sub = getattr(m, name, None)
            if not isinstance(sub, nn.Module):
                continue
            for p in sub.parameters():
                p.requires_grad = False
                n += int(p.numel())
    return n


class GatedResidual(nn.Module):
    """Fail-soft gate: TF = TF1D + σ(g) ⊙ R. Gate is per-recorder from GNO nodes."""

    def __init__(self, inner: nn.Module, *, n_rec: int, latent_dim: int):
        super().__init__()
        self.inner = inner
        self.n_rec = int(n_rec)
        self.gate_head = nn.Linear(int(latent_dim), 1)
        self.accepts_geom_flags = bool(getattr(inner, "accepts_geom_flags", False))
        self.accepts_query_x = bool(getattr(inner, "accepts_query_x", False))
        self.last_gate: torch.Tensor | None = None

    def forward(
        self,
        fields: torch.Tensor | None,
        stoch: torch.Tensor | None,
        trunk_y: torch.Tensor,
        geom_flags: torch.Tensor | None = None,
        query_x: torch.Tensor | None = None,
        support_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs: dict[str, torch.Tensor] = {}
        if geom_flags is not None and getattr(self.inner, "accepts_geom_flags", False):
            kwargs["geom_flags"] = geom_flags
        if query_x is not None and getattr(self.inner, "accepts_query_x", False):
            kwargs["query_x"] = query_x
        if support_x is not None and getattr(self.inner, "accepts_query_x", False):
            kwargs["support_x"] = support_x
        r = self.inner(fields, stoch, trunk_y, **kwargs)
        core = gno_core(self.inner)
        nodes = getattr(core, "last_nodes", None)
        if nodes is None:
            self.last_gate = None
            return r
        gate = torch.sigmoid(self.gate_head(nodes))
        self.last_gate = gate
        b, n_q = r.shape
        n_freq = n_q // self.n_rec
        g_q = (
            gate.expand(-1, self.n_rec, n_freq).reshape(b, n_q)
            if gate.shape[1] == self.n_rec
            else gate.reshape(b, -1)
        )
        if g_q.shape != r.shape:
            g_q = gate.mean(dim=1, keepdim=True).expand_as(r)
        return r * g_q


def _run_fno_layers(fno: nn.Module, x: torch.Tensor) -> torch.Tensor:
    n_layers = int(getattr(fno, "n_layers", 1))
    for i in range(n_layers):
        x = fno(x, index=i)
    return x


class _Spectral1d(nn.Module):
    """Complex multiply on the leading rFFT modes along one spatial axis."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = int(modes)
        scale = 1.0 / max(channels, 1)
        self.weight = nn.Parameter(scale * torch.randn(channels, channels, modes, 2))

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        x = x.movedim(dim, -1)
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, x_ft.shape[-1])
        w = torch.view_as_complex(self.weight[..., :m, :].contiguous())
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., :m] = torch.einsum("bchw,oiw->bohw", x_ft[..., :m], w)
        y = torch.fft.irfft(out_ft, n=n, dim=-1)
        return y.movedim(-1, dim)


class FactorizedFNOLayer(nn.Module):
    """F-FNO layer (Tran et al. 2023): 1D spectral mixing on each axis + local skip."""

    def __init__(self, channels: int, n_modes: tuple[int, int]):
        super().__init__()
        self.spec_h = _Spectral1d(channels, n_modes[0])
        self.spec_w = _Spectral1d(channels, n_modes[1])
        self.local = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spec_h(x, dim=2) + self.spec_w(x, dim=3)
        return F.gelu(y + self.local(x))


class FreqFNOLayer(nn.Module):
    """1D FNO along frequency only (per-recorder). Oscillatory leftover in f."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.spec = _Spectral1d(channels, modes)
        self.local = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x, dim=3) + self.local(x))


class AFNOLayer(nn.Module):
    """AFNO token mixer (Guibas et al. 2022): shared channel MLP in Fourier space."""

    def __init__(self, channels: int):
        super().__init__()
        hid = 2 * channels
        self.mlp = nn.Sequential(
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
        )
        self.local = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        ri = torch.view_as_real(ft.permute(0, 2, 3, 1).contiguous())
        wf = ri.shape[2]
        ri = self.mlp(ri.reshape(b, h, wf, c * 2))
        ft = torch.view_as_complex(ri.reshape(b, h, wf, c, 2).contiguous())
        ft = ft.permute(0, 3, 1, 2)
        y = torch.fft.irfft2(ft, s=(h, w), dim=(-2, -1), norm="ortho")
        return F.gelu(y + self.local(x))


class HaarWNOLayer(nn.Module):
    """WNO-lite (Tripura & Chakraborty 2023): 1-level Haar DWT on freq + local conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.mix = nn.Conv2d(channels, channels, 3, padding=1)
        self.s2 = 2.0**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.shape[-1]
        if w % 2:
            x = F.pad(x, (0, 1), mode="replicate")
        even, odd = x[..., 0::2], x[..., 1::2]
        s = (even + odd) * self.s2
        d = (even - odd) * self.s2
        z = F.gelu(self.mix(torch.cat([s, d], dim=-1)))
        n = z.shape[-1] // 2
        s, d = z[..., :n], z[..., n:]
        even = (s + d) * self.s2
        odd = (s - d) * self.s2
        y = torch.stack([even, odd], dim=-1).flatten(-2)
        return y[..., :w] + x[..., :w]


def _load_dual_path_loglo():
    path = Path(__file__).resolve().parent.parent / "GIFNO" / "spectral_layers.py"
    spec = importlib.util.spec_from_file_location("_gifno_spectral_layers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DualPathLOGLOStack from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DualPathLOGLOStack


def _pad_to_multiple(
    x: torch.Tensor, ph: int, pw: int
) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (ph - h % ph) % ph
    pad_w = (pw - w % pw) % pw
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, h, w


class ResidualPODReadout(nn.Module):
    """Replace DeepONet query product with a POD reconstruction of leftover R."""

    def __init__(
        self,
        inner: nn.Module,
        *,
        n_rec: int,
        latent_dim: int,
        pod_modes: np.ndarray,
        pod_mean: np.ndarray,
    ):
        super().__init__()
        self.inner = inner
        self.n_rec = int(n_rec)
        self.accepts_geom_flags = bool(getattr(inner, "accepts_geom_flags", False))
        self.accepts_query_x = bool(getattr(inner, "accepts_query_x", False))
        modes = np.asarray(pod_modes, dtype=np.float32)
        mean = np.asarray(pod_mean, dtype=np.float32)
        if modes.ndim != 3:
            raise ValueError(f"pod_modes must be (R,K,F), got {modes.shape}")
        self.n_modes = int(modes.shape[1])
        self.register_buffer("_pod_modes", torch.as_tensor(modes))
        self.register_buffer("_pod_mean", torch.as_tensor(mean))
        hidden = max(int(latent_dim), self.n_modes * 2)
        self.branch = nn.Sequential(
            nn.Linear(int(latent_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, self.n_modes),
        )

    def _slice_basis(self, n_freq: int) -> tuple[torch.Tensor, torch.Tensor]:
        modes = self._pod_modes
        mean = self._pod_mean
        full = int(modes.shape[-1])
        if n_freq == full:
            return modes, mean
        if n_freq < full:
            idx = torch.linspace(0, full - 1, n_freq, device=modes.device)
            idx = idx.round().long().clamp(0, full - 1)
            return modes[..., idx], mean[..., idx]
        pad = n_freq - full
        return (
            F.pad(modes, (0, pad), mode="replicate"),
            F.pad(mean, (0, pad), mode="replicate"),
        )

    def forward(
        self,
        fields: torch.Tensor | None,
        stoch: torch.Tensor | None,
        trunk_y: torch.Tensor,
        geom_flags: torch.Tensor | None = None,
        query_x: torch.Tensor | None = None,
        support_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r_inner = self.inner(
            fields,
            stoch,
            trunk_y,
            **_spatial_kwargs(
                self.inner,
                geom_flags=geom_flags,
                query_x=query_x,
                support_x=support_x,
            ),
        )
        core = gno_core(self.inner)
        nodes = getattr(core, "last_nodes", None)
        b, n_q = r_inner.shape
        n_freq = n_q // self.n_rec
        if nodes is None:
            return r_inner
        coeff = self.branch(nodes)
        modes, mean = self._slice_basis(n_freq)
        r_pod = mean.unsqueeze(0) + torch.einsum("brk,rkf->brf", coeff, modes)
        return r_pod.reshape(b, n_q)


class FrozenBoost(nn.Module):
    """Operator boosting: frozen prior leftover + shrink × tiny booster."""

    def __init__(self, frozen: nn.Module, booster: nn.Module, *, shrink: float = 0.5):
        super().__init__()
        self.frozen = frozen
        self.booster = booster
        for p in self.frozen.parameters():
            p.requires_grad = False
        self.shrink = nn.Parameter(torch.tensor(float(shrink)))
        self.accepts_geom_flags = bool(
            getattr(booster, "accepts_geom_flags", False)
        ) or bool(getattr(frozen, "accepts_geom_flags", False))
        self.accepts_query_x = bool(getattr(booster, "accepts_query_x", False)) or bool(
            getattr(frozen, "accepts_query_x", False)
        )
        self.last_gate: torch.Tensor | None = None

    def forward(
        self,
        fields: torch.Tensor | None,
        stoch: torch.Tensor | None,
        trunk_y: torch.Tensor,
        geom_flags: torch.Tensor | None = None,
        query_x: torch.Tensor | None = None,
        support_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs_f = _spatial_kwargs(
            self.frozen, geom_flags=geom_flags, query_x=query_x, support_x=support_x
        )
        kwargs_b = _spatial_kwargs(
            self.booster, geom_flags=geom_flags, query_x=query_x, support_x=support_x
        )
        with torch.no_grad():
            r0 = self.frozen(fields, stoch, trunk_y, **kwargs_f)
        delta = self.booster(fields, stoch, trunk_y, **kwargs_b)
        self.last_gate = getattr(self.booster, "last_gate", None)
        s = self.shrink.clamp(0.0, 1.0)
        return r0 + s * delta

    def select_shrink_from_val(self, value: float) -> None:
        with torch.no_grad():
            self.shrink.copy_(torch.tensor(float(value), device=self.shrink.device))


class AxialLeftoverTF(nn.Module):
    """Bidirectional axial transformer on leftover (B, C, n_rec, n_freq).

    Attends along frequency then recorders of R̂ — not Vs-column station GNOT.
    Spectra are not causal, so this is an encoder stack, not a GPT decoder.
    """

    def __init__(self, channels: int, n_layers: int = 4):
        super().__init__()
        nhead = _n_heads(channels)
        ff = max(4 * channels, 32)

        def _layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=nhead,
                dim_feedforward=ff,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        self.freq_layers = nn.ModuleList([_layer() for _ in range(n_layers)])
        self.rec_layers = nn.ModuleList([_layer() for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        for freq_l, rec_l in zip(self.freq_layers, self.rec_layers):
            xf = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
            xf = freq_l(xf)
            x = xf.reshape(b, h, w, c).permute(0, 3, 1, 2)
            xr = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
            xr = rec_l(xr)
            x = xr.reshape(b, w, h, c).permute(0, 3, 2, 1)
        return x


def hz_band_masks(
    freq_hz: torch.Tensor, *, split_hz: float = 2.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard frequency masks: low [0.1, split), high [split, 10]. Shape (1, 1, 1, n_f)."""
    f = freq_hz.reshape(1, 1, 1, -1)
    high = f >= float(split_hz)
    return ~high, high


class DeepONetFNO(nn.Module):
    """DeepFNOnet-style: DeepONet residual plus an FNO family head on (recorder × freq).

    ``kind``:
      vanilla — neuralop FNOBlocks (Li et al. 2021)
      ufno — FNO + local 3×3 conv each layer (Wen et al. U-FNO 2022)
      ffno — factorized 1D spectral conv per axis (Tran et al. F-FNO 2023)
      afno — adaptive Fourier MLP mixer (Guibas et al. 2022)
      wno — Haar wavelet mixing on frequency (Tripura & Chakraborty 2023)
      fno1d — spectral conv along frequency only
      loglo — DualPathLOGLO + HFP on the leftover grid (all local Fourier modes)
      tf — axial transformer on leftover frequency then recorders
      band2 — two vanilla FNOs with a hard 2 Hz mask (low [0.1, 2), high [2, 10])
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        n_rec: int,
        width: int = 32,
        n_modes: tuple[int, int] = (8, 16),
        n_layers: int = 4,
        kind: FNOKind = "vanilla",
        loglo_patch: tuple[int, int] = (3, 8),
    ):
        super().__init__()
        self.base = base
        self.n_rec = int(n_rec)
        self.accepts_geom_flags = bool(getattr(base, "accepts_geom_flags", False))
        self.accepts_query_x = bool(getattr(base, "accepts_query_x", False))
        self.kind: FNOKind = kind
        self.loglo_patch = (int(loglo_patch[0]), int(loglo_patch[1]))
        self.lift = nn.Conv2d(1, width, kernel_size=1)
        self.proj = nn.Conv2d(width, 1, kernel_size=1)
        self.proj_high: nn.Conv2d | None = None
        self.local: nn.ModuleList | None = None
        self.blocks: nn.ModuleList | None = None
        self.fno: nn.Module | None = None
        self.fno_low: nn.Module | None = None
        self.fno_high: nn.Module | None = None
        self.loglo: nn.Module | None = None
        self.axial: nn.Module | None = None
        self.band_split_hz = float(config.BAND2_SPLIT_HZ)
        self.register_buffer("_query_freq", torch.empty(0), persistent=False)
        self.last_band_low: torch.Tensor | None = None
        self.last_band_high: torch.Tensor | None = None
        if kind == "band2":
            from neuralop.layers.fno_block import FNOBlocks

            kw = dict(
                n_modes=n_modes,
                in_channels=width,
                out_channels=width,
                n_layers=n_layers,
                non_linearity=F.gelu,
            )
            self.fno_low = FNOBlocks(**kw)
            self.fno_high = FNOBlocks(**kw)
            self.proj_high = nn.Conv2d(width, 1, kernel_size=1)
        elif kind == "tf":
            self.axial = AxialLeftoverTF(width, n_layers=n_layers)
        elif kind == "ffno":
            self.blocks = nn.ModuleList(
                [FactorizedFNOLayer(width, n_modes) for _ in range(n_layers)]
            )
        elif kind == "afno":
            self.blocks = nn.ModuleList([AFNOLayer(width) for _ in range(n_layers)])
        elif kind == "wno":
            self.blocks = nn.ModuleList([HaarWNOLayer(width) for _ in range(n_layers)])
        elif kind == "fno1d":
            self.blocks = nn.ModuleList(
                [FreqFNOLayer(width, n_modes[1]) for _ in range(n_layers)]
            )
        elif kind == "loglo":
            DualPath = _load_dual_path_loglo()
            self.loglo = DualPath(
                n_modes=n_modes,
                channels=width,
                n_layers=n_layers,
                patch_size=self.loglo_patch,
                hf_noise_alpha=0.0,
            )
        else:
            from neuralop.layers.fno_block import FNOBlocks

            self.fno = FNOBlocks(
                n_modes=n_modes,
                in_channels=width,
                out_channels=width,
                n_layers=n_layers,
                non_linearity=F.gelu,
            )
            if kind == "ufno":
                self.local = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(width, width, 3, padding=1),
                            nn.GELU(),
                            nn.Conv2d(width, width, 3, padding=1),
                        )
                        for _ in range(n_layers)
                    ]
                )

    def set_query_freq(self, freq: np.ndarray | torch.Tensor | None) -> None:
        if freq is None:
            self._query_freq = torch.empty(0, device=self._query_freq.device)
            return
        arr = (
            np.asarray(freq, dtype=np.float32).ravel()
            if not torch.is_tensor(freq)
            else freq
        )
        t = torch.as_tensor(
            arr, dtype=torch.float32, device=self._query_freq.device
        ).ravel()
        self._query_freq = t

    def _grid_freq_hz(
        self, n_freq: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        buf = self._query_freq
        if buf.numel() == n_freq:
            return buf.to(device=device, dtype=dtype)
        return torch.logspace(
            math.log10(float(config.FREQ_START_HZ)),
            math.log10(float(config.FREQ_END_HZ)),
            n_freq,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _run_vanilla_fno(fno: nn.Module, x: torch.Tensor) -> torch.Tensor:
        n_layers = int(getattr(fno, "n_layers", 1))
        for i in range(n_layers):
            x = fno(x, index=i)
        return x

    def forward(
        self,
        fields: torch.Tensor | None,
        stoch: torch.Tensor | None,
        trunk_y: torch.Tensor,
        geom_flags: torch.Tensor | None = None,
        query_x: torch.Tensor | None = None,
        support_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = _spatial_kwargs(
            self.base, geom_flags=geom_flags, query_x=query_x, support_x=support_x
        )
        r = self.base(fields, stoch, trunk_y, **kwargs)
        b, n_q = r.shape
        n_rec = int(query_x.shape[-1]) if query_x is not None else self.n_rec
        n_freq = n_q // n_rec
        x = self.lift(r.view(b, 1, n_rec, n_freq))
        self.last_lift = x
        if (
            self.fno_low is not None
            and self.fno_high is not None
            and self.proj_high is not None
        ):
            x_l = self._run_vanilla_fno(self.fno_low, x)
            x_h = self._run_vanilla_fno(self.fno_high, x)
            r_l = self.proj(x_l)
            r_h = self.proj_high(x_h)
            freq = self._grid_freq_hz(n_freq, x.device, x.dtype)
            m_l, m_h = hz_band_masks(freq, split_hz=self.band_split_hz)
            m_l = m_l.to(device=x.device, dtype=x.dtype)
            m_h = m_h.to(device=x.device, dtype=x.dtype)
            self.last_band_low = m_l
            self.last_band_high = m_h
            self.last_fno = x_l + x_h
            return r + (r_l * m_l + r_h * m_h).reshape(b, n_q)
        if self.loglo is not None:
            ph, pw = self.loglo_patch
            x, h0, w0 = _pad_to_multiple(x, ph, pw)
            x_g, x_l = self.loglo(x)
            x = (x_g + x_l)[..., :h0, :w0]
        elif self.axial is not None:
            x = self.axial(x)
        elif self.blocks is not None:
            for layer in self.blocks:
                x = layer(x)
        else:
            assert self.fno is not None
            n_layers = int(getattr(self.fno, "n_layers", 1))
            for i in range(n_layers):
                x = self.fno(x, index=i)
                if self.local is not None:
                    x = x + self.local[i](x)
        self.last_fno = x
        return r + self.proj(x).reshape(b, n_q)


def interp_along_x(
    values: torch.Tensor,
    x_src: torch.Tensor,
    x_dst: torch.Tensor,
) -> torch.Tensor:
    """Linear interpolate ``values`` (B, Nsrc, F) from ``x_src`` (Nsrc,) to ``x_dst`` (Ndst,)."""
    xs = x_src.reshape(-1).to(dtype=values.dtype, device=values.device)
    xd = x_dst.reshape(-1).to(dtype=values.dtype, device=values.device)
    order = torch.argsort(xs)
    xs = xs[order]
    vs = values[:, order]
    xd_c = xd.clamp(xs[0], xs[-1])
    idx = torch.searchsorted(xs, xd_c).clamp(1, xs.numel() - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    w = ((xd_c - x0) / (x1 - x0).clamp_min(1e-8)).view(1, -1, 1)
    return vs[:, idx - 1] * (1.0 - w) + vs[:, idx] * w


class LatentGridFNO(nn.Module):
    """FNO on a fixed x-grid, then decode back to query locations.

    The leftover is interpolated from the labeled queries onto ``n_lat``
    uniform stations, mixed with vanilla FNOBlocks, and interpolated back.
    Domain length is still the query span (the 500 m strip), not a new box.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        n_lat: int = N_LATENT_X,
        width: int = 32,
        n_modes: tuple[int, int] = (8, 16),
        n_layers: int = 4,
    ):
        super().__init__()
        from neuralop.layers.fno_block import FNOBlocks

        self.base = base
        self.n_lat = max(4, int(n_lat))
        self.accepts_geom_flags = bool(getattr(base, "accepts_geom_flags", False))
        self.accepts_query_x = True
        modes_x = min(int(n_modes[0]), max(1, self.n_lat // 2 - 1))
        self.lift = nn.Conv2d(1, width, kernel_size=1)
        self.proj = nn.Conv2d(width, 1, kernel_size=1)
        self.fno = FNOBlocks(
            n_modes=(modes_x, int(n_modes[1])),
            in_channels=width,
            out_channels=width,
            n_layers=n_layers,
            non_linearity=F.gelu,
        )

    def forward(
        self,
        fields: torch.Tensor | None,
        stoch: torch.Tensor | None,
        trunk_y: torch.Tensor,
        geom_flags: torch.Tensor | None = None,
        query_x: torch.Tensor | None = None,
        support_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = _spatial_kwargs(
            self.base, geom_flags=geom_flags, query_x=query_x, support_x=support_x
        )
        r = self.base(fields, stoch, trunk_y, **kwargs)
        if query_x is None:
            return r
        b, n_q = r.shape
        n_query = int(query_x.shape[-1])
        n_freq = n_q // n_query
        r2 = r.view(b, n_query, n_freq)
        qx = query_x[0] if query_x.ndim == 2 else query_x
        x_lat = torch.linspace(
            float(qx.min().item()),
            float(qx.max().item()),
            self.n_lat,
            device=r.device,
            dtype=r.dtype,
        )
        r_lat = interp_along_x(r2, qx, x_lat)
        x = self.lift(r_lat.unsqueeze(1))
        n_layers = int(getattr(self.fno, "n_layers", 1))
        for i in range(n_layers):
            x = self.fno(x, index=i)
        self.last_fno = x
        delta_lat = self.proj(x).squeeze(1)
        delta = interp_along_x(delta_lat, x_lat, qx)
        return r + delta.reshape(b, n_q)


def apply_query_freq(module: nn.Module, freq: np.ndarray | torch.Tensor | None) -> None:
    """Push the current query-grid Hz onto any DeepONetFNO (train 200 vs eval 1000)."""
    for m in module.modules():
        if isinstance(m, DeepONetFNO):
            m.set_query_freq(freq)


BranchMode = Literal["single", "multi", "stoch_only", "fields_only"]


def build_model(
    mode: BranchMode,
    *,
    field_channels: int,
    stoch_dim: int,
    trunk_dim: int,
    latent_dim: int = 64,
    field_hidden: int = 32,
    branch_hidden: int = 128,
    trunk_hidden: int = 128,
    trunk_layers: int = 4,
    field_encoder: FieldEncoderKind = "conv",
    residual_fno: bool = False,
    n_rec: int = 21,
    fno_width: int = 32,
    fno_n_modes: tuple[int, int] = (8, 16),
    fno_n_layers: int = 4,
    n_gno_layers: int = 3,
    fno_kind: FNOKind = "vanilla",
    physics_tokens: bool = False,
    geom_flag_dim: int = 0,
    learned_1d: bool = False,
    gated: bool = False,
    pod_readout: bool = False,
    pod_modes: np.ndarray | None = None,
    pod_mean: np.ndarray | None = None,
    pod_n_modes: int = 32,
    loglo_patch: tuple[int, int] = (3, 8),
    log_residual: bool = False,
    col_enc_depth_tokens: int = 1,
    boost: bool = False,
    boost_fno_kind: FNOKind = "loglo",
    boost_width: int | None = None,
    boost_shrink: float = 0.5,
    n_mscale_trunk: int = 1,
    n_mscale_branch: int = 1,
    mscale_coord_dims: tuple[int, ...] = FULL_TRUNK_COORD_DIMS,
    col_enc: ColEncKind = "conv",
    stoch_inject: StochInjectKind = "mlp",
    fuse_kind: FuseKind = "mlp",
    gno_rh_dilate: bool = False,
    kernel_k: int = KERNEL_K,
    latent_fno: bool = False,
    n_latent: int = N_LATENT_X,
) -> nn.Module:
    def _core(
        *,
        fno_kind_local: FNOKind,
        width_local: int,
        gated_local: bool,
        pod_local: bool,
    ) -> nn.Module:
        if field_encoder in ("gno", "attn", "gat", "identity", "kernel"):
            if mode != "single":
                raise ValueError(
                    "GNO/attn/gat/kernel encoder is only implemented for single-branch DeepONet"
                )
            mixer = (
                "attn"
                if field_encoder == "attn"
                else (
                    "gat"
                    if field_encoder == "gat"
                    else (
                        "identity"
                        if field_encoder == "identity"
                        else ("kernel" if field_encoder == "kernel" else "gno")
                    )
                )
            )
            inner: nn.Module = RecorderGNODeepONet(
                field_channels=field_channels,
                stoch_dim=stoch_dim,
                trunk_dim=trunk_dim,
                latent_dim=latent_dim,
                field_hidden=field_hidden,
                branch_hidden=branch_hidden,
                trunk_hidden=trunk_hidden,
                trunk_layers=trunk_layers,
                n_gno_layers=n_gno_layers,
                node_mixer=mixer,
                physics_tokens=physics_tokens,
                geom_flag_dim=geom_flag_dim,
                learned_1d=learned_1d,
                col_enc_depth_tokens=col_enc_depth_tokens,
                n_mscale_trunk=n_mscale_trunk,
                n_mscale_branch=n_mscale_branch,
                mscale_coord_dims=mscale_coord_dims,
                col_enc=col_enc,
                stoch_inject=stoch_inject,
                fuse_kind=fuse_kind,
                gno_rh_dilate=gno_rh_dilate,
                kernel_k=kernel_k,
            )
        elif mode == "multi":
            inner = MultiBranchDeepONet(
                n_field_channels=field_channels,
                stoch_dim=stoch_dim,
                trunk_dim=trunk_dim,
                latent_dim=latent_dim,
                field_hidden=field_hidden,
                branch_hidden=branch_hidden,
                trunk_hidden=trunk_hidden,
                trunk_layers=trunk_layers,
                field_encoder=field_encoder,
            )
        else:
            use_fields = mode in ("single", "fields_only")
            use_stoch = mode in ("single", "stoch_only")
            inner = SingleBranchDeepONet(
                field_channels=field_channels,
                stoch_dim=stoch_dim,
                trunk_dim=trunk_dim,
                latent_dim=latent_dim,
                field_hidden=field_hidden,
                branch_hidden=branch_hidden,
                trunk_hidden=trunk_hidden,
                trunk_layers=trunk_layers,
                use_fields=use_fields,
                use_stoch=use_stoch,
                field_encoder=field_encoder,
                n_mscale_trunk=n_mscale_trunk,
                n_mscale_branch=n_mscale_branch,
                mscale_coord_dims=mscale_coord_dims,
            )
        if pod_local:
            modes = pod_modes
            mean = pod_mean
            if modes is None or mean is None:
                modes = np.zeros((n_rec, int(pod_n_modes), 1000), dtype=np.float32)
                mean = np.zeros((n_rec, 1000), dtype=np.float32)
            inner = ResidualPODReadout(
                inner,
                n_rec=n_rec,
                latent_dim=latent_dim,
                pod_modes=modes,
                pod_mean=mean,
            )
        # FNO-on-R needs a fixed (n_rec × n_f) lattice. Kernel queries are not
        # that grid unless Phase 1b lifts them onto a latent x-mesh.
        use_r_fno = bool(residual_fno) and not (
            field_encoder == "kernel" and not latent_fno
        )
        if use_r_fno:
            inner = DeepONetFNO(
                inner,
                n_rec=n_rec,
                width=width_local,
                n_modes=fno_n_modes,
                n_layers=fno_n_layers,
                kind=fno_kind_local,
                loglo_patch=loglo_patch,
            )
        if latent_fno:
            inner = LatentGridFNO(
                inner,
                n_lat=n_latent,
                width=width_local,
                n_modes=fno_n_modes,
                n_layers=fno_n_layers,
            )
        if gated_local:
            inner = GatedResidual(inner, n_rec=n_rec, latent_dim=latent_dim)
        return inner

    if boost:
        frozen = _core(
            fno_kind_local=fno_kind,
            width_local=fno_width,
            gated_local=False,
            pod_local=False,
        )
        booster = _core(
            fno_kind_local=boost_fno_kind,
            width_local=int(boost_width)
            if boost_width is not None
            else max(8, fno_width // 2),
            gated_local=gated,
            pod_local=pod_readout,
        )
        net: nn.Module = FrozenBoost(frozen, booster, shrink=boost_shrink)
    else:
        net = _core(
            fno_kind_local=fno_kind,
            width_local=fno_width,
            gated_local=gated,
            pod_local=pod_readout,
        )
    net.log_residual = bool(log_residual)  # type: ignore[attr-defined]
    return net


def _arch_from_src(
    src: dict[str, Any],
    blob: dict[str, Any],
    *,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Architecture kwargs for ``build_model`` from a checkpoint (or nested) dict."""
    fno_width = int(
        src.get("fno_width", blob.get("fno_width", defaults.get("fno_width", 32)))
    )
    patch = src.get("loglo_patch", blob.get("loglo_patch", (3, 8)))
    modes = src.get("pod_modes", blob.get("pod_modes"))
    mean = src.get("pod_mean", blob.get("pod_mean"))
    n_modes = src.get("fno_n_modes", blob.get("fno_n_modes", (8, 16)))
    return {
        "field_encoder": src.get(
            "field_encoder",
            blob.get("field_encoder", defaults.get("field_encoder", "conv")),
        ),
        "residual_fno": bool(src.get("residual_fno", blob.get("residual_fno", False))),
        "n_rec": int(src.get("n_rec", blob.get("n_rec", defaults.get("n_rec", 21)))),
        "fno_width": fno_width,
        "fno_n_modes": tuple(n_modes),
        "fno_n_layers": int(
            src.get(
                "fno_n_layers",
                blob.get("fno_n_layers", defaults.get("fno_n_layers", 4)),
            )
        ),
        "n_gno_layers": int(
            src.get(
                "n_gno_layers",
                blob.get("n_gno_layers", defaults.get("n_gno_layers", 3)),
            )
        ),
        "fno_kind": src.get("fno_kind", blob.get("fno_kind", "vanilla")),
        "physics_tokens": bool(
            src.get("physics_tokens", blob.get("physics_tokens", False))
        ),
        "geom_flag_dim": int(src.get("geom_flag_dim", blob.get("geom_flag_dim", 0))),
        "learned_1d": bool(src.get("learned_1d", blob.get("learned_1d", False))),
        "gated": bool(src.get("gated", blob.get("gated", False))),
        "pod_readout": bool(src.get("pod_readout", blob.get("pod_readout", False))),
        "pod_modes": None if modes is None else np.asarray(modes),
        "pod_mean": None if mean is None else np.asarray(mean),
        "pod_n_modes": int(src.get("pod_n_modes", blob.get("pod_n_modes", 32))),
        "loglo_patch": (int(patch[0]), int(patch[1])),
        "log_residual": bool(src.get("log_residual", blob.get("log_residual", False))),
        "col_enc_depth_tokens": int(
            src.get(
                "col_enc_depth_tokens",
                blob.get(
                    "col_enc_depth_tokens", defaults.get("col_enc_depth_tokens", 1)
                ),
            )
        ),
        "boost": False,
        "boost_fno_kind": src.get(
            "boost_fno_kind", blob.get("boost_fno_kind", "loglo")
        ),
        "boost_width": int(
            src.get(
                "boost_width",
                blob.get("boost_width", max(8, fno_width // 2)),
            )
        ),
        "boost_shrink": float(src.get("boost_shrink", blob.get("boost_shrink", 0.5))),
        "n_mscale_trunk": int(src.get("n_mscale_trunk", blob.get("n_mscale_trunk", 1))),
        "n_mscale_branch": int(
            src.get("n_mscale_branch", blob.get("n_mscale_branch", 1))
        ),
        "mscale_coord_dims": tuple(
            src.get(
                "mscale_coord_dims",
                blob.get("mscale_coord_dims", FULL_TRUNK_COORD_DIMS),
            )
        ),
        "col_enc": src.get(
            "col_enc", blob.get("col_enc", defaults.get("col_enc", "conv"))
        ),
        "stoch_inject": src.get(
            "stoch_inject",
            blob.get("stoch_inject", defaults.get("stoch_inject", "mlp")),
        ),
        "fuse_kind": src.get(
            "fuse_kind", blob.get("fuse_kind", defaults.get("fuse_kind", "mlp"))
        ),
        "gno_rh_dilate": bool(
            src.get(
                "gno_rh_dilate",
                blob.get("gno_rh_dilate", defaults.get("gno_rh_dilate", False)),
            )
        ),
        "kernel_k": int(
            src.get(
                "kernel_k", blob.get("kernel_k", defaults.get("kernel_k", KERNEL_K))
            )
        ),
        "latent_fno": bool(
            src.get(
                "latent_fno", blob.get("latent_fno", defaults.get("latent_fno", False))
            )
        ),
        "n_latent": int(
            src.get(
                "n_latent", blob.get("n_latent", defaults.get("n_latent", N_LATENT_X))
            )
        ),
    }


def build_from_checkpoint_blob(
    blob: dict[str, Any],
    *,
    field_channels: int,
    stoch_dim: int,
    trunk_dim: int,
    latent_dim: int = 128,
    field_hidden: int = 48,
    branch_hidden: int = 256,
    trunk_hidden: int = 256,
    trunk_layers: int = 5,
) -> nn.Module:
    """Rebuild a leftover model (including FrozenBoost) from a train.py checkpoint."""
    common = dict(
        field_channels=field_channels,
        stoch_dim=stoch_dim,
        trunk_dim=trunk_dim,
        latent_dim=latent_dim,
        field_hidden=field_hidden,
        branch_hidden=branch_hidden,
        trunk_hidden=trunk_hidden,
        trunk_layers=trunk_layers,
    )
    mode = blob.get("branch_mode", "single")
    if blob.get("boost"):
        if blob.get("frozen_arch"):
            frozen_src = blob["frozen_arch"]
        else:
            frozen_src = {
                **blob,
                "fno_kind": blob.get("fno_kind", "vanilla"),
                "gated": False,
                "pod_readout": False,
            }
        booster_src = blob.get("booster_arch") or {
            **blob,
            "fno_kind": blob.get("boost_fno_kind", "loglo"),
            "fno_width": blob.get(
                "boost_width", max(8, int(blob.get("fno_width", 32)) // 2)
            ),
            "gated": bool(blob.get("gated", False)),
            "pod_readout": bool(blob.get("pod_readout", False)),
        }
        frozen = build_model(
            mode, **common, **_arch_from_src(frozen_src, blob, defaults={})
        )
        booster = build_model(
            mode, **common, **_arch_from_src(booster_src, blob, defaults={})
        )
        net: nn.Module = FrozenBoost(
            frozen,
            booster,
            shrink=float(blob.get("boost_shrink", 0.5)),
        )
        net.log_residual = bool(blob.get("log_residual", False))  # type: ignore[attr-defined]
        return net
    return build_model(mode, **common, **_arch_from_src(blob, blob, defaults={}))
