# model.py
"""Dual-path LOGLO encoder + POD-DeepONet readout."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

import config
from pod_readout import RecorderPODDeepONetHeadXT, load_pod_basis
from spectral_layers import DualPathLOGLOStack
from xt_readout import ChannelLift


class DepthCollapse(nn.Module):
    """Collapse depth (B,C,H,W) -> (B,C,1,W) via a per-channel weighted sum.

    Mathematically identical to a depthwise (H,1) conv, but expressed as an
    einsum so the backward is matmul-based (the large-kernel depthwise conv
    backward is very slow on GPU).
    """

    def __init__(self, channels: int, depth: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels, depth) / depth**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bchw,ch->bcw", x, self.weight).unsqueeze(2)


class GIFNOXTLOGLOPODModel(nn.Module):
    """ChannelLift -> DualPathLOGLOStack -> RecorderPODDeepONetHeadXT."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        n_freq: int,
        nx: int,
        fno_modes: Tuple[int, int],
        num_fno_layers: int,
        pod_modes: np.ndarray,
        pod_mean: np.ndarray,
        patch_size: Tuple[int, int] = (16, 20),
        hfp_kernel: int = 4,
        hfp_stride: int = 4,
        hf_noise_alpha: float = 0.025,
        depth_stride: int = 1,
        branch_hidden: int | None = None,
        branch_mode: str = "surface",
        recorder_x: np.ndarray | None = None,
    ):
        super().__init__()
        if recorder_x is None:
            recorder_x = config.recorder_x_indices()
        self.lift = ChannelLift(in_channels, latent_channels)
        # Depth downsampling stem: the head only reads the surface, so run the
        # (cost ~ H*W) LOGLO layers at reduced depth. depth_stride=k reduces
        # NZ_MAX -> NZ_MAX/k; k>=NZ_MAX collapses depth to 1 (encoder is then a
        # 1D-along-x operator). Depthwise so the stem stays cheap.
        in_depth = config.NZ_MAX
        k = max(1, min(depth_stride, in_depth))
        if k >= in_depth:
            # Full collapse to a single surface row -> 1D-along-x encoder.
            self.depth_pool = DepthCollapse(latent_channels, in_depth)
            enc_depth = 1
        elif k > 1:
            self.depth_pool = nn.Conv2d(
                latent_channels,
                latent_channels,
                kernel_size=(k, 1),
                stride=(k, 1),
                groups=latent_channels,
                bias=False,
            )
            enc_depth = in_depth // k
        else:
            self.depth_pool = nn.Identity()
            enc_depth = in_depth
        # Clamp depth-axis hyperparameters to the (possibly tiny) encoder depth
        # so the 2D LOGLO stack degrades gracefully to 1D when enc_depth == 1.
        enc_patch = (max(1, min(patch_size[0], enc_depth)), patch_size[1])
        enc_modes = (max(1, min(fno_modes[0], enc_depth)), fno_modes[1])
        self.encoder = DualPathLOGLOStack(
            n_modes=enc_modes,
            channels=latent_channels,
            n_layers=num_fno_layers,
            patch_size=enc_patch,
            hfp_kernel=hfp_kernel,
            hfp_stride=hfp_stride,
            hf_noise_alpha=hf_noise_alpha,
        )
        self.head = RecorderPODDeepONetHeadXT(
            branch_in_channels=2 * latent_channels,
            nx=nx,
            n_freq=n_freq,
            recorder_x=recorder_x,
            pod_modes=pod_modes,
            pod_mean=pod_mean,
            branch_hidden=branch_hidden,
            branch_mode=branch_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        x = self.depth_pool(x)
        x_global, x_local = self.encoder(x)
        return self.head(x_global, x_local)


def create_model(
    in_channels: int = config.IN_CHANNELS,
    latent_channels: int = config.LATENT_CHANNELS,
    n_freq: int | None = None,
    nx: int = config.NX,
    fno_modes: Tuple[int, int] = config.FNO_MODES,
    num_fno_layers: int = config.NUM_FNO_LAYERS,
    branch_mode: str = config.BRANCH_MODE,
) -> GIFNOXTLOGLOPODModel:
    pod_modes, pod_mean = load_pod_basis(config.POD_MODES_PATH, config.POD_MEAN_PATH)
    if n_freq is None:
        n_freq = int(pod_modes.shape[-1])
    elif n_freq != pod_modes.shape[-1]:
        raise ValueError(
            f"n_freq={n_freq} does not match POD basis F={pod_modes.shape[-1]}"
        )
    return GIFNOXTLOGLOPODModel(
        in_channels=in_channels,
        latent_channels=latent_channels,
        n_freq=n_freq,
        nx=nx,
        fno_modes=fno_modes,
        num_fno_layers=num_fno_layers,
        pod_modes=pod_modes,
        pod_mean=pod_mean,
        patch_size=config.LOGLO_PATCH_SIZE,
        hfp_kernel=config.LOGLO_HFP_KERNEL,
        hfp_stride=config.LOGLO_HFP_STRIDE,
        hf_noise_alpha=config.LOGLO_HF_NOISE_ALPHA,
        depth_stride=config.LOGLO_DEPTH_STRIDE,
        branch_hidden=config.POD_BRANCH_HIDDEN,
        branch_mode=branch_mode,
    )
