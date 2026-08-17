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


FieldEncoderKind = Literal["conv", "resunet", "gno", "attn", "gat"]
FNOKind = Literal["vanilla", "ufno", "ffno", "afno", "wno", "fno1d"]


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


class _ColumnEncoder(nn.Module):
    """Per-recorder 1D conv down the depth: (B, C, Nz, Nr) → (B, Nr, out_dim)."""

    def __init__(self, in_channels: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden * 2, out_dim)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        b, c, nz, nr = fields.shape
        x = fields.permute(0, 3, 1, 2).reshape(b * nr, c, nz)
        h = self.net(x).squeeze(-1)
        return self.proj(h).view(b, nr, -1)


class _ChainGNO(nn.Module):
    """kNN=2 message passing along the recorder line (no periodic wrap)."""

    def __init__(self, dim: int, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * dim, dim),
                    nn.GELU(),
                    nn.Linear(dim, dim),
                )
                for _ in range(n_layers)
            ]
        )

    @staticmethod
    def _neighbors(x: torch.Tensor) -> torch.Tensor:
        left = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        right = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
        return 0.5 * (left + right)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            msg = self._neighbors(x)
            x = x + layer(torch.cat([x, msg], dim=-1))
        return x


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
    """DeepONet whose branch is per-recorder after chain GNO (lateral leftover)."""

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
        node_mixer: Literal["gno", "attn", "gat"] = "gno",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.field_encoder_kind = (
            "attn" if node_mixer == "attn" else ("gat" if node_mixer == "gat" else "gno")
        )
        self.col_enc = _ColumnEncoder(field_channels, field_hidden, latent_dim)
        if node_mixer == "attn":
            self.gno = _RecorderAttn(latent_dim, n_layers=n_gno_layers)
        elif node_mixer == "gat":
            self.gno = _RecorderGAT(latent_dim, n_layers=n_gno_layers)
        else:
            self.gno = _ChainGNO(latent_dim, n_layers=n_gno_layers)
        self.stoch_mlp = nn.Sequential(nn.Linear(stoch_dim, latent_dim), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(2 * latent_dim, branch_hidden),
            nn.GELU(),
            nn.Linear(branch_hidden, latent_dim),
        )
        self.trunk = TrunkMLP(trunk_dim, latent_dim, trunk_hidden, trunk_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        fields: torch.Tensor | None,
        stoch: torch.Tensor | None,
        trunk_y: torch.Tensor,
    ) -> torch.Tensor:
        assert fields is not None and stoch is not None
        nodes = self.gno(self.col_enc(fields))
        s = self.stoch_mlp(stoch).unsqueeze(1).expand(-1, nodes.shape[1], -1)
        p = self.fuse(torch.cat([nodes, s], dim=-1))
        n_rec = p.shape[1]
        n_q = trunk_y.shape[1]
        n_freq = n_q // n_rec
        p_q = (
            p.unsqueeze(2)
            .expand(-1, n_rec, n_freq, -1)
            .reshape(trunk_y.shape[0], n_q, self.latent_dim)
        )
        bq = self.trunk(trunk_y.reshape(-1, trunk_y.shape[-1])).reshape(
            trunk_y.shape[0], n_q, self.latent_dim
        )
        return (p_q * bq).sum(dim=-1) + self.bias


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


class DeepONetFNO(nn.Module):
    """DeepFNOnet-style: DeepONet residual plus an FNO family head on (recorder × freq).

    ``kind``:
      vanilla — neuralop FNOBlocks (Li et al. 2021)
      ufno — FNO + local 3×3 conv each layer (Wen et al. U-FNO 2022)
      ffno — factorized 1D spectral conv per axis (Tran et al. F-FNO 2023)
      afno — adaptive Fourier MLP mixer (Guibas et al. 2022)
      wno — Haar wavelet mixing on frequency (Tripura & Chakraborty 2023)
      fno1d — spectral conv along frequency only
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
    ):
        super().__init__()
        self.base = base
        self.n_rec = int(n_rec)
        self.kind: FNOKind = kind
        self.lift = nn.Conv2d(1, width, kernel_size=1)
        self.proj = nn.Conv2d(width, 1, kernel_size=1)
        self.local: nn.ModuleList | None = None
        self.blocks: nn.ModuleList | None = None
        self.fno: nn.Module | None = None
        if kind == "ffno":
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

    def forward(
        self,
        fields: torch.Tensor | None,
        stoch: torch.Tensor | None,
        trunk_y: torch.Tensor,
    ) -> torch.Tensor:
        r = self.base(fields, stoch, trunk_y)
        b, n_q = r.shape
        n_freq = n_q // self.n_rec
        x = self.lift(r.view(b, 1, self.n_rec, n_freq))
        if self.blocks is not None:
            for layer in self.blocks:
                x = layer(x)
        else:
            assert self.fno is not None
            n_layers = int(getattr(self.fno, "n_layers", 1))
            for i in range(n_layers):
                x = self.fno(x, index=i)
                if self.local is not None:
                    x = x + self.local[i](x)
        return r + self.proj(x).reshape(b, n_q)


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
) -> nn.Module:
    if field_encoder in ("gno", "attn", "gat"):
        if mode != "single":
            raise ValueError("GNO/attn/gat encoder is only implemented for single-branch DeepONet")
        mixer = "attn" if field_encoder == "attn" else ("gat" if field_encoder == "gat" else "gno")
        net: nn.Module = RecorderGNODeepONet(
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
        )
    elif mode == "multi":
        net = MultiBranchDeepONet(
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
        net = SingleBranchDeepONet(
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
    if residual_fno:
        net = DeepONetFNO(
            net,
            n_rec=n_rec,
            width=fno_width,
            n_modes=fno_n_modes,
            n_layers=fno_n_layers,
            kind=fno_kind,
        )
    return net
