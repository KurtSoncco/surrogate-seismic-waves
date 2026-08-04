"""Single- vs multi-branch DeepONet for signed residual R(x, f)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        e1_p = F.pad(e1, (0, pad_w1, 0, pad_h1), mode="replicate") if (pad_h1 or pad_w1) else e1
        e2 = self.down2(e1_p)
        b = self.bottleneck(e2)
        d1 = self.up1(b, e1)
        d0 = self.up2(d1, e0)
        return self.head(d0)


FieldEncoderKind = Literal["conv", "resunet"]


def build_field_encoder(
    kind: FieldEncoderKind,
    *,
    in_channels: int,
    hidden: int,
    out_dim: int,
) -> nn.Module:
    if kind == "resunet":
        return ResUNetFieldEncoder(in_channels=in_channels, base=hidden, out_dim=out_dim)
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
    ):
        super().__init__()
        if not use_fields and not use_stoch:
            raise ValueError("Need at least fields or stochastic inputs")
        self.use_fields = use_fields
        self.use_stoch = use_stoch
        self.latent_dim = latent_dim
        self.field_encoder_kind = field_encoder

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
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, branch_hidden),
            nn.GELU(),
            nn.Linear(branch_hidden, branch_hidden),
            nn.GELU(),
            nn.Linear(branch_hidden, latent_dim),
        )
        self.trunk = TrunkMLP(trunk_dim, latent_dim, trunk_hidden, trunk_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def branch(
        self,
        fields: torch.Tensor | None,
        stoch: torch.Tensor | None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.use_fields:
            assert fields is not None
            parts.append(self.field_enc(fields))
        if self.use_stoch:
            assert stoch is not None
            parts.append(stoch)
        return self.fuse(torch.cat(parts, dim=-1))

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
) -> nn.Module:
    if mode == "multi":
        return MultiBranchDeepONet(
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
    use_fields = mode in ("single", "fields_only")
    use_stoch = mode in ("single", "stoch_only")
    return SingleBranchDeepONet(
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
    )
