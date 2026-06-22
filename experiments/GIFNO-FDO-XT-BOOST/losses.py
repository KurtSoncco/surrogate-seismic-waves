# losses.py
"""Band-weighted masked loss for BOOST residual training."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import config
from metrics import gather_recorder_tf_torch

_EPS = 1e-8


def _freq_band_weights(freq: np.ndarray, device: torch.device) -> torch.Tensor:
    """Per-frequency weights (F,) emphasizing mid/high bands."""
    w = torch.ones(len(freq), device=device, dtype=torch.float32)
    bands = (
        (config.FREQ_BAND_LOW, config.BOOST_BAND_WEIGHT_LOW),
        (config.FREQ_BAND_MID, config.BOOST_BAND_WEIGHT_MID),
        (config.FREQ_BAND_HIGH, config.BOOST_BAND_WEIGHT_HIGH),
    )
    for (lo, hi), weight in bands:
        mask = (freq >= lo) & (freq <= hi)
        w[torch.as_tensor(mask, device=device)] = float(weight)
    return w


class BandWeightedMaskedLoss(nn.Module):
    """
    Relative Lp + H1 on recorder TFs with per-frequency band weights.

    Used on final prediction TF = base + eta * delta.
    """

    def __init__(
        self,
        p: int = 1,
        h1_weight: float = 0.25,
        freq: np.ndarray | None = None,
    ):
        super().__init__()
        self.p = p
        self.h1_weight = h1_weight
        if freq is None:
            freq = np.load(config.TF_FREQ_PATH)
        log_f = np.log(np.maximum(freq, _EPS).astype(np.float32))
        self.register_buffer(
            "_band_w",
            _freq_band_weights(freq, torch.device("cpu")),
            persistent=False,
        )
        self.register_buffer(
            "_log_f",
            torch.from_numpy(log_f),
            persistent=False,
        )

    def _weighted_relative_lp(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        freq_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """pred, target: (B, R, F) -> (B,)"""
        w = self._band_w.to(pred.device) if freq_weights is None else freq_weights
        diff = (pred - target) * w
        y = target * w
        p = self.p
        if p == 1:
            num = torch.sum(torch.abs(diff), dim=(1, 2))
            den = torch.sum(torch.abs(y), dim=(1, 2))
        elif p % 2 == 0:
            num = torch.sum(diff**p, dim=(1, 2)) ** (1.0 / p)
            den = torch.sum(y**p, dim=(1, 2)) ** (1.0 / p)
        else:
            num = torch.sum(torch.abs(diff) ** p, dim=(1, 2)) ** (1.0 / p)
            den = torch.sum(torch.abs(y) ** p, dim=(1, 2)) ** (1.0 / p)
        return num / (den + _EPS)

    def _h1_log_freq(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_f = self._log_f.to(pred.device)
        d_log = log_f[1:] - log_f[:-1]
        dp = (pred[..., 1:] - pred[..., :-1]) / d_log.clamp_min(_EPS)
        dt = (target[..., 1:] - target[..., :-1]) / d_log.clamp_min(_EPS)
        w = self._band_w.to(pred.device)[1:]
        return self._weighted_relative_lp(dp, dt, freq_weights=w)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        del mask
        p_rec = gather_recorder_tf_torch(pred)
        t_rec = gather_recorder_tf_torch(target)
        rel = self._weighted_relative_lp(p_rec, t_rec)
        if self.h1_weight > 0:
            h1 = self._h1_log_freq(p_rec, t_rec)
            rel = rel + self.h1_weight * h1
        return rel.mean()


def build_training_loss() -> nn.Module:
    return BandWeightedMaskedLoss(p=config.LOSS_P, h1_weight=config.LOSS_H1_WEIGHT)
