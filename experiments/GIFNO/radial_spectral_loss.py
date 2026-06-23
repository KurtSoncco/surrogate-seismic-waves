# radial_spectral_loss.py
"""Radial binned spectral loss (LOGLO-FNO) adapted for recorder TF grids."""

from __future__ import annotations

import torch
import torch.nn as nn


class RadialBinnedSpectralLoss(nn.Module):
    """
    Penalize mid + high radial frequency bands of the 2D FFT error.

    pred, target: (B, R, F) gathered recorder transfer functions.
    """

    def __init__(
        self,
        i_low: int = 4,
        i_high: int = 12,
        mid_high_only: bool = True,
    ):
        super().__init__()
        self.i_low = i_low
        self.i_high = i_high
        self.mid_high_only = mid_high_only

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = (pred - target).unsqueeze(1)
        b, _, nx, ny = err.shape
        err_fft = torch.fft.fft2(err, dim=(-2, -1))
        err_sq = err_fft.abs().square()
        err_sq_h = err_sq[..., : nx // 2, : ny // 2]

        x = torch.arange(nx // 2, device=err.device)
        y = torch.arange(ny // 2, device=err.device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        radii = torch.sqrt(xx.float().square() + yy.float().square()).floor().long()
        max_radius = int(radii.max().item())
        radii_flat = radii.reshape(-1)

        flat = err_sq_h.reshape(b, -1, radii_flat.numel())
        err_bins = err.new_zeros(b, max_radius + 1)
        valid = radii_flat <= max_radius
        for bi in range(b):
            err_bins[bi].index_add_(
                0,
                radii_flat[valid],
                flat[bi, 0, valid],
            )

        nrm = float(nx * ny)
        err_f = torch.sqrt(err_bins.mean(dim=0)) / nrm

        if self.mid_high_only:
            mid = err_f[self.i_low : self.i_high].mean()
            high = err_f[self.i_high :].mean()
            return mid + high
        low = err_f[: self.i_low].mean()
        mid = err_f[self.i_low : self.i_high].mean()
        high = err_f[self.i_high :].mean()
        return low + mid + high
