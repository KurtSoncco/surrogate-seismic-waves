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
