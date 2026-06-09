# losses.py
"""Masked relative L2 loss for spatial transfer function fields."""

import torch
import torch.nn as nn
from neuralop.losses import LpLoss


class MaskedLpLoss(nn.Module):
    """
    Relative L2 loss on recorder positions only.

    pred, target: (B, Nx, N_freq)
    mask: (B, Nx) with 1.0 at recorder x-indices
    """

    def __init__(self, d: int = 2, p: int = 2):
        super().__init__()
        self.lp = LpLoss(d=d, p=p, reduction="mean")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # LpLoss expects (B, C, *spatial); add channel dim and apply mask
        m = mask.unsqueeze(1).unsqueeze(-1)
        pred_m = pred.unsqueeze(1) * m
        target_m = target.unsqueeze(1) * m
        return self.lp(pred_m, target_m)
