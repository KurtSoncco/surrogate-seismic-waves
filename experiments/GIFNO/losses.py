# losses.py
"""Masked losses for spatial transfer-function fields."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from neuralop.losses import LpLoss

import config
from metrics import gather_recorder_tf_torch

_EPS = 1e-8


class MaskedLpLoss(nn.Module):
    """
    Relative Lp loss on recorder positions only.

    pred, target: (B, Nx, N_freq)
    mask: (B, Nx) with 1.0 at recorder x-indices
    """

    def __init__(self, d: int = 2, p: int = 2):
        super().__init__()
        self.d = d
        self.p = p
        self.lp = LpLoss(d=d, p=p, reduction="mean")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        m = mask.unsqueeze(1).unsqueeze(-1)
        pred_m = pred.unsqueeze(1) * m
        target_m = target.unsqueeze(1) * m
        if not return_per_sample:
            return self.lp(pred_m, target_m)

        per_sample = self._relative_lp_per_sample(pred_m, target_m)
        return per_sample.mean(), per_sample

    def _relative_lp_per_sample(
        self, pred_m: torch.Tensor, target_m: torch.Tensor
    ) -> torch.Tensor:
        """Relative Lp per batch item on masked (B, 1, Nx, F) tensors."""
        diff = torch.flatten(pred_m, start_dim=-self.d) - torch.flatten(
            target_m, start_dim=-self.d
        )
        y = torch.flatten(target_m, start_dim=-self.d)
        p = self.p
        if p == 1:
            num = torch.sum(torch.abs(diff), dim=-1)
            den = torch.sum(torch.abs(y), dim=-1)
        elif p % 2 == 0:
            num = torch.sum(diff**p, dim=-1) ** (1.0 / p)
            den = torch.sum(y**p, dim=-1) ** (1.0 / p)
        else:
            num = torch.sum(torch.abs(diff) ** p, dim=-1) ** (1.0 / p)
            den = torch.sum(torch.abs(y) ** p, dim=-1) ** (1.0 / p)
        return num / (den + _EPS)


class MaskedCompositeLoss(nn.Module):
    """
    Composite masked loss: relative Lp + H1(log f) + frequency-domain term.

    Optional hard mining reweights batch items with high per-sample loss.
    """

    def __init__(
        self,
        rel_weight: float = 1.0,
        h1_weight: float = 0.0,
        freq_weight: float = 0.0,
        p: int = 2,
        hard_mining: bool = False,
        hard_mining_power: float = 1.0,
        freq_loss_log_weight: bool = True,
        freq: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.rel_weight = rel_weight
        self.h1_weight = h1_weight
        self.freq_weight = freq_weight
        self.p = p
        self.hard_mining = hard_mining
        self.hard_mining_power = hard_mining_power
        self.freq_loss_log_weight = freq_loss_log_weight
        self.rel_loss = MaskedLpLoss(d=2, p=p)

        if freq is None:
            freq = np.load(config.TF_FREQ_PATH)
        log_f = np.log(np.maximum(freq, _EPS))
        self.register_buffer(
            "_log_f",
            torch.from_numpy(log_f.astype(np.float32)),
            persistent=False,
        )
        if freq_loss_log_weight:
            w = np.log(np.maximum(freq / freq.min(), _EPS)) + 1.0
        else:
            w = np.ones_like(freq)
        self.register_buffer(
            "_freq_weights",
            torch.from_numpy(w.astype(np.float32)),
            persistent=False,
        )

    def _relative_lp_on_recorder(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Relative Lp per batch item on gathered recorder curves (B, R, F)."""
        diff = (pred - target).reshape(pred.shape[0], -1)
        y = target.reshape(target.shape[0], -1)
        p = self.p
        if p == 1:
            num = torch.sum(torch.abs(diff), dim=1)
            den = torch.sum(torch.abs(y), dim=1)
        elif p % 2 == 0:
            num = torch.sum(diff**p, dim=1) ** (1.0 / p)
            den = torch.sum(y**p, dim=1) ** (1.0 / p)
        else:
            num = torch.sum(torch.abs(diff) ** p, dim=1) ** (1.0 / p)
            den = torch.sum(torch.abs(y) ** p, dim=1) ** (1.0 / p)
        return num / (den + _EPS)

    def _h1_log_freq_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Relative L2 on values + relative L2 on d/d(log f) per sample."""
        log_f = self._log_f.to(pred.device)
        d_log = log_f[1:] - log_f[:-1]
        dp = (pred[..., 1:] - pred[..., :-1]) / d_log.clamp_min(_EPS)
        dt = (target[..., 1:] - target[..., :-1]) / d_log.clamp_min(_EPS)
        rel_val = self._relative_lp_on_recorder(pred, target)
        rel_deriv = self._relative_lp_on_recorder(dp, dt)
        return rel_val + rel_deriv

    def _freq_domain_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Weighted relative L2 on rFFT coefficients along frequency."""
        w = self._freq_weights.to(pred.device)
        pw = pred * w
        tw = target * w
        pf = torch.fft.rfft(pw, dim=-1)
        tf = torch.fft.rfft(tw, dim=-1)
        diff = torch.abs(pf - tf).reshape(pred.shape[0], -1)
        den = torch.abs(tf).reshape(target.shape[0], -1).norm(dim=1)
        num = diff.norm(dim=1)
        return num / (den + _EPS)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        p_rec = gather_recorder_tf_torch(pred)
        t_rec = gather_recorder_tf_torch(target)

        if self.rel_weight > 0:
            rel_ps = self._relative_lp_on_recorder(p_rec, t_rec)
        else:
            rel_ps = torch.zeros(pred.shape[0], device=pred.device)

        total = self.rel_weight * rel_ps

        if self.h1_weight > 0:
            h1 = self._h1_log_freq_loss(p_rec, t_rec)
            total = total + self.h1_weight * h1

        if self.freq_weight > 0:
            freq_loss = self._freq_domain_loss(p_rec, t_rec)
            total = total + self.freq_weight * freq_loss

        if self.hard_mining and total.numel() > 0:
            mean = total.mean().clamp_min(_EPS)
            weights = (total / mean) ** self.hard_mining_power
            weights = weights / weights.mean().clamp_min(_EPS)
            return (weights * total).mean()
        return total.mean()


def build_training_loss() -> nn.Module:
    """Factory using weights from config."""
    if (
        config.LOSS_H1_WEIGHT == 0.0
        and config.LOSS_FREQ_WEIGHT == 0.0
        and not config.HARD_MINING
    ):
        return MaskedLpLoss(d=2, p=config.LOSS_P)
    return MaskedCompositeLoss(
        rel_weight=config.LOSS_REL_WEIGHT,
        h1_weight=config.LOSS_H1_WEIGHT,
        freq_weight=config.LOSS_FREQ_WEIGHT,
        p=config.LOSS_P,
        hard_mining=config.HARD_MINING,
        hard_mining_power=config.HARD_MINING_POWER,
        freq_loss_log_weight=config.FREQ_LOSS_LOG_WEIGHT,
    )
