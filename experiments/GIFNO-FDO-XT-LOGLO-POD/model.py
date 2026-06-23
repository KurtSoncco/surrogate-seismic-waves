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
        branch_hidden: int | None = None,
        branch_mode: str = "surface",
        recorder_x: np.ndarray | None = None,
    ):
        super().__init__()
        if recorder_x is None:
            recorder_x = config.recorder_x_indices()
        self.lift = ChannelLift(in_channels, latent_channels)
        self.encoder = DualPathLOGLOStack(
            n_modes=fno_modes,
            channels=latent_channels,
            n_layers=num_fno_layers,
            patch_size=patch_size,
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
        branch_hidden=config.POD_BRANCH_HIDDEN,
        branch_mode=branch_mode,
    )
