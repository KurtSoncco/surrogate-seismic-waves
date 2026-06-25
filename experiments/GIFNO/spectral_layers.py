# spectral_layers.py
"""Shared spectral-bias modules for GIFNO-FDO-XT refinement experiments."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.fno_block import FNOBlocks


class DepthwiseLocalConv2d(nn.Module):
    """Depthwise-separable 3x3 local branch (H-FNO style)."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.pointwise.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class FNOEncoderLoop(nn.Module):
    """Run all FNOBlocks layers (neuralop defaults to index=0 only)."""

    def __init__(
        self,
        n_modes: Tuple[int, int],
        in_channels: int,
        out_channels: int,
        n_layers: int,
    ):
        super().__init__()
        self.fno = FNOBlocks(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            non_linearity=nn.functional.gelu,
        )

    @property
    def n_layers(self) -> int:
        return self.fno.n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.fno.n_layers):
            x = self.fno(x, index=i)
        return x


class HybridFNOStack(nn.Module):
    """
    H-FNO encoder: global FNO layer + parallel local conv per depth.

    Each step: x <- FNO_layer(x) + LocalConv(x_in), where x_in is the layer input.
    """

    def __init__(
        self,
        n_modes: Tuple[int, int],
        channels: int,
        n_layers: int,
    ):
        super().__init__()
        self.fno = FNOBlocks(
            n_modes=n_modes,
            in_channels=channels,
            out_channels=channels,
            n_layers=n_layers,
            non_linearity=nn.functional.gelu,
        )
        self.local_convs = nn.ModuleList(
            [DepthwiseLocalConv2d(channels) for _ in range(n_layers)]
        )

    @property
    def n_layers(self) -> int:
        return self.fno.n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.fno.n_layers):
            x_in = x
            x = self.fno(x, index=i)
            x = x + self.local_convs[i](x_in)
        return x


class HFSModule(nn.Module):
    """
    High-frequency scaling on latent feature maps (arXiv:2503.13695, Eq. 4-6).

    Splits spatial dims into non-overlapping patches; scales DC and HFC separately.
    """

    def __init__(self, channels: int, patch_size: int = 4):
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.channels = channels
        self.patch_size = patch_size
        self.lambda_dc = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.lambda_hfc = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[-2:]
        b, c, h, w = x.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            pad_h = (p - h % p) % p
            pad_w = (p - w % p) % p
            x = F.pad(x, (0, pad_w, 0, pad_h))
            h, w = x.shape[-2], x.shape[-1]

        n_h = h // p
        n_w = w // p
        n_patches = n_h * n_w

        patches = (
            x.reshape(b, c, n_h, p, n_w, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b, n_patches, c, p, p)
        )
        dc = patches.mean(dim=1, keepdim=True)
        hfc = patches - dc
        scaled = patches + self.lambda_dc * dc + self.lambda_hfc * hfc
        out = (
            scaled.reshape(b, n_h, n_w, c, p, p)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(b, c, h, w)
        )
        return out[..., :orig_h, :orig_w]


class LocalPatchSpectralConv2d(nn.Module):
    """LOGLO-style local spectral conv on non-overlapping patches (all rfft modes)."""

    def __init__(self, channels: int, patch_size: Tuple[int, int]):
        super().__init__()
        ph, pw = patch_size
        if ph < 1 or pw < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        self.ph = ph
        self.pw = pw
        mh, mw = ph, pw // 2 + 1
        scale = 1.0 / max(channels * channels, 1)
        weight = scale * torch.randn(channels, channels, mh, mw, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ph, pw = self.ph, self.pw
        if h % ph != 0 or w % pw != 0:
            raise ValueError(
                f"Spatial dims ({h}, {w}) must be divisible by patch ({ph}, {pw})"
            )
        in_dtype = x.dtype
        n_h, n_w = h // ph, w // pw
        n_p = n_h * n_w
        # Non-overlapping patches: reshape/permute (avoids unfold/fold col2im copies).
        patches = (
            x.reshape(b, c, n_h, ph, n_w, pw)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b * n_p, c, ph, pw)
        )
        # FFTs require float32 (cuFFT has no half support); guard for AMP.
        x_ft = torch.fft.rfft2(patches.float(), norm="ortho")
        out_ft = torch.einsum("bixy,ijxy->bjxy", x_ft, self.weight)
        out = torch.fft.irfft2(out_ft, s=(ph, pw), norm="ortho")
        out = (
            out.reshape(b, n_h, n_w, c, ph, pw)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(b, c, h, w)
        )
        return out.to(in_dtype)


class HighFrequencyPropagation(nn.Module):
    """HFP: X - upsample(avgpool(X)) per LOGLO-FNO Eq. 4."""

    def __init__(self, kernel_size: int = 4, stride: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(x, self.kernel_size, stride=self.stride)
        up = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x - up


def _hf_adaptive_noise(x_hf: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-sample HF adaptive Gaussian noise (LOGLO training augmentation)."""
    flat = x_hf.reshape(x_hf.shape[0], -1)
    mu = flat.mean(dim=1, keepdim=True)
    sigma = flat.std(dim=1, keepdim=True).clamp_min(1e-8)
    noise = torch.randn_like(x_hf)
    return (mu.view(-1, 1, 1, 1) + alpha * sigma.view(-1, 1, 1, 1) * noise).to(
        dtype=x_hf.dtype
    )


class DualPathLOGLOStack(nn.Module):
    """
    Dual-path LOGLO encoder: separate global FNO and local patch-spectral streams.

    Returns (x_global, x_local) after L layers for concat at the readout branch.
    """

    def __init__(
        self,
        n_modes: Tuple[int, int],
        channels: int,
        n_layers: int,
        patch_size: Tuple[int, int] = (16, 20),
        hfp_kernel: int = 4,
        hfp_stride: int = 4,
        hf_noise_alpha: float = 0.025,
    ):
        super().__init__()
        self.channels = channels
        self.n_layers = n_layers
        self.hf_noise_alpha = hf_noise_alpha
        self.fno = FNOBlocks(
            n_modes=n_modes,
            in_channels=channels,
            out_channels=channels,
            n_layers=n_layers,
            non_linearity=nn.functional.gelu,
        )
        self.local_layers = nn.ModuleList(
            [LocalPatchSpectralConv2d(channels, patch_size) for _ in range(n_layers)]
        )
        self.hfp = HighFrequencyPropagation(hfp_kernel, hfp_stride)
        self.hf_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=1),
                )
                for _ in range(n_layers)
            ]
        )
        self.skips = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=1) for _ in range(n_layers)]
        )
        self.gates = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=1) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_global = x
        x_local = x
        use_noise = self.training and self.hf_noise_alpha > 0
        for i in range(self.n_layers):
            z = x_global + x_local
            x_hf = self.hfp(z)
            if use_noise:
                noise = _hf_adaptive_noise(x_hf, self.hf_noise_alpha)
                g_in = x_global + noise
                l_in = x_local + noise
            else:
                g_in = x_global
                l_in = x_local

            # FNO/patch spectral convs use FFTs (no half support) -> force fp32.
            with torch.autocast(device_type=g_in.device.type, enabled=False):
                x_g = self.fno(g_in.float(), index=i)
            x_l = self.local_layers[i](l_in)
            hf_feat = self.hf_mlps[i](x_hf)
            mix = x_g + x_l + hf_feat + self.skips[i](z)
            gate = torch.sigmoid(self.gates[i](mix))
            x_global = F.gelu(x_g + gate * (mix - x_g))
            x_local = F.gelu(x_l + (1.0 - gate) * (mix - x_l))
        return x_global, x_local
