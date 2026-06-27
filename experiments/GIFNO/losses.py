# losses.py
"""Masked losses for spatial transfer-function fields."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from neuralop.losses import LpLoss

import config
from metrics import gather_recorder_tf_torch, valley_mask_from_log_target_torch
from radial_spectral_loss import RadialBinnedSpectralLoss

_EPS = 1e-8


def band_curriculum_weights(
    epoch: int,
    num_epochs: int,
    floor: float = 0.25,
    mid_start: float = 0.33,
    high_start: float = 0.66,
    ramp: float = 0.15,
) -> Tuple[float, float, float]:
    """
    Per-band scalar weights for the frequency-band loss curriculum.

    Emphasize low log-f early; ramp mid up from ``mid_start``; ramp high up from
    ``high_start``. Each ramp spans a fraction ``ramp`` of training. Bands never
    drop below ``floor`` so no band starves. Returns (w_low, w_mid, w_high) in
    [floor, 1]; the per-frequency expansion and mean-1 normalization happen in
    ``MaskedCompositeLoss.set_band_weights``.
    """
    t = epoch / max(num_epochs - 1, 1)
    span = max(ramp, _EPS)

    def ramp_up(start: float) -> float:
        return min(1.0, max(0.0, (t - start) / span))

    w_low = 1.0
    w_mid = floor + (1.0 - floor) * ramp_up(mid_start)
    w_high = floor + (1.0 - floor) * ramp_up(high_start)
    return w_low, w_mid, w_high


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
        linf_weight: float = 0.0,
        p: int = 2,
        hard_mining: bool = False,
        hard_mining_power: float = 1.0,
        freq_loss_log_weight: bool = True,
        log_tf_loss: bool = False,
        valley_loss_weight: float = 0.0,
        valley_percentile: float = 20.0,
        radial_weight: float = 0.0,
        radial_i_low: int = 4,
        radial_i_high: int = 12,
        band_balanced_weight: float = 0.0,
        freq: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.rel_weight = rel_weight
        self.h1_weight = h1_weight
        self.freq_weight = freq_weight
        self.linf_weight = linf_weight
        self.radial_weight = radial_weight
        self.p = p
        self.hard_mining = hard_mining
        self.hard_mining_power = hard_mining_power
        self.freq_loss_log_weight = freq_loss_log_weight
        self.log_tf_loss = log_tf_loss
        self.valley_loss_weight = valley_loss_weight
        self.valley_percentile = valley_percentile
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
        self.radial_loss = (
            RadialBinnedSpectralLoss(
                i_low=radial_i_low, i_high=radial_i_high, mid_high_only=True
            )
            if radial_weight > 0
            else None
        )

        # Frequency-band curriculum weights. None -> neutral (no reweighting),
        # which reproduces the un-weighted relative loss exactly.
        self._band_freq = np.asarray(freq, dtype=np.float64)
        self._band_w: Optional[torch.Tensor] = None

        # Band-balanced term (Tier 2): each band's relative L2 normalized by its
        # OWN target energy, so the low-energy high band is not suppressed. The
        # per-band scalar weights are curriculum-controlled via
        # set_band_balanced_weights; None -> equal weighting.
        self.band_balanced_weight = band_balanced_weight
        self._bb_slices = self._compute_band_slices()
        self._bb_weights: Optional[Tuple[float, float, float]] = None

    def _compute_band_slices(self) -> list[slice]:
        freq = self._band_freq
        bands = (
            getattr(config, "FREQ_BAND_LOW", (0.1, 0.5)),
            getattr(config, "FREQ_BAND_MID", (0.5, 2.0)),
            getattr(config, "FREQ_BAND_HIGH", (2.0, 10.0)),
        )
        slices: list[slice] = []
        for lo, hi in bands:
            i_lo = int(np.searchsorted(freq, lo, side="left"))
            i_hi = int(np.searchsorted(freq, hi, side="right"))
            slices.append(slice(i_lo, max(i_lo + 1, i_hi)))
        return slices

    def set_band_balanced_weights(
        self,
        w_low: Optional[float] = None,
        w_mid: Optional[float] = None,
        w_high: Optional[float] = None,
    ) -> None:
        """Set per-band scalar weights for the band-balanced term.

        All ``None`` -> equal weighting across bands.
        """
        if w_low is None and w_mid is None and w_high is None:
            self._bb_weights = None
            return
        self._bb_weights = (
            float(w_low if w_low is not None else 1.0),
            float(w_mid if w_mid is not None else 1.0),
            float(w_high if w_high is not None else 1.0),
        )

    def _band_balanced_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample weighted mean of per-band self-normalized relative L2.

        pred, target: (B, R, F) recorder curves. Each band is normalized by its
        own target energy so band amplitude does not dictate its gradient share.
        Dividing by the weight sum keeps the term's scale stable as the
        curriculum changes the per-band weights.
        """
        b = pred.shape[0]
        weights = self._bb_weights if self._bb_weights is not None else (1.0, 1.0, 1.0)
        total = pred.new_zeros(b)
        wsum = 0.0
        for slc, w in zip(self._bb_slices, weights):
            if w == 0.0:
                wsum += w
                continue
            p = pred[..., slc].reshape(b, -1)
            t = target[..., slc].reshape(b, -1)
            num = torch.linalg.norm(p - t, dim=1)
            den = torch.linalg.norm(t, dim=1).clamp_min(_EPS)
            total = total + w * (num / den)
            wsum += w
        if wsum <= _EPS:
            return total
        return total / wsum

    def set_band_weights(
        self,
        w_low: Optional[float] = None,
        w_mid: Optional[float] = None,
        w_high: Optional[float] = None,
    ) -> None:
        """
        Set per-frequency loss weights from per-band scalars.

        Passing all ``None`` (or calling with no args) clears the weighting and
        restores the neutral relative loss. Otherwise a piecewise-constant
        per-frequency vector is built over the configured bands and normalized to
        mean 1 so the overall loss scale is preserved across epochs.
        """
        if w_low is None and w_mid is None and w_high is None:
            self._band_w = None
            return

        freq = self._band_freq
        w = np.ones_like(freq, dtype=np.float64)
        bands = {
            "low": (getattr(config, "FREQ_BAND_LOW", (0.1, 0.5)), w_low),
            "mid": (getattr(config, "FREQ_BAND_MID", (0.5, 2.0)), w_mid),
            "high": (getattr(config, "FREQ_BAND_HIGH", (2.0, 10.0)), w_high),
        }
        for (lo, hi), weight in bands.values():
            if weight is None:
                continue
            i_lo = int(np.searchsorted(freq, lo, side="left"))
            i_hi = int(np.searchsorted(freq, hi, side="right"))
            i_hi = max(i_lo + 1, i_hi)
            w[i_lo:i_hi] = float(weight)
        mean = float(w.mean())
        if mean > _EPS:
            w = w / mean
        self._band_w = torch.from_numpy(w.astype(np.float32))

    def _apply_band_weights(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale pred/target by per-frequency weights when set and shape-matched."""
        if self._band_w is None or pred.shape[-1] != self._band_w.numel():
            return pred, target
        w = self._band_w.to(pred.device, pred.dtype)
        return pred * w, target * w

    def _transform_for_rel_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.log_tf_loss:
            return pred, target
        return (
            torch.log(pred.abs().clamp_min(_EPS)),
            torch.log(target.abs().clamp_min(_EPS)),
        )

    def _relative_lp_on_recorder(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Relative Lp per batch item on gathered recorder curves (B, R, F)."""
        pred, target = self._transform_for_rel_loss(pred, target)
        pred, target = self._apply_band_weights(pred, target)
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

    def _valley_weighted_relative_lp(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Relative L2 with extra weight on valley bins from the target curve."""
        pred_t, target_t = self._transform_for_rel_loss(pred, target)
        valley = valley_mask_from_log_target_torch(
            target, percentile=self.valley_percentile
        ).float()
        weights = 1.0 + self.valley_loss_weight * valley
        diff = (pred_t - target_t) * weights
        y = target_t * weights
        num = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1)
        den = torch.linalg.norm(y.reshape(y.shape[0], -1), dim=1).clamp_min(_EPS)
        return num / den

    def _relative_linf_on_recorder(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Mean per-sample relative max frequency error."""
        pred_t, target_t = self._transform_for_rel_loss(pred, target)
        rel_point = torch.abs(pred_t - target_t) / (
            target_t.abs().clamp_min(_EPS)
            if self.log_tf_loss
            else target.abs().clamp_min(_EPS)
        )
        return rel_point.amax(dim=(1, 2))

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
            if self.valley_loss_weight > 0:
                rel_ps = self._valley_weighted_relative_lp(p_rec, t_rec)
            else:
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

        if self.linf_weight > 0:
            linf = self._relative_linf_on_recorder(p_rec, t_rec)
            total = total + self.linf_weight * linf

        if self.radial_weight > 0 and self.radial_loss is not None:
            radial = self.radial_loss(p_rec, t_rec)
            total = total + self.radial_weight * radial

        if self.band_balanced_weight > 0:
            bb = self._band_balanced_loss(p_rec, t_rec)
            total = total + self.band_balanced_weight * bb

        if self.hard_mining and total.numel() > 0:
            mean = total.mean().clamp_min(_EPS)
            weights = (total / mean) ** self.hard_mining_power
            weights = weights / weights.mean().clamp_min(_EPS)
            return (weights * total).mean()
        return total.mean()


def build_training_loss() -> nn.Module:
    """Factory using weights from config."""
    radial_weight = getattr(config, "LOSS_RADIAL_WEIGHT", 0.0)
    radial_i_low = getattr(config, "RADIAL_I_LOW", 4)
    radial_i_high = getattr(config, "RADIAL_I_HIGH", 12)
    band_balanced_weight = getattr(config, "LOSS_BAND_BALANCED_WEIGHT", 0.0)
    band_curriculum = getattr(config, "BAND_CURRICULUM", False)
    curriculum_mode = getattr(config, "BAND_CURRICULUM_MODE", "time")
    # Convergence-mode curriculum drives the band-balanced term's per-band
    # weights, so it needs a non-zero band-balanced weight to have a lever.
    if (
        band_curriculum
        and curriculum_mode == "convergence"
        and band_balanced_weight <= 0
    ):
        band_balanced_weight = 0.5
    use_composite = (
        config.LOSS_H1_WEIGHT != 0.0
        or config.LOSS_FREQ_WEIGHT != 0.0
        or config.LOSS_LINF_WEIGHT != 0.0
        or config.VALLEY_LOSS_WEIGHT != 0.0
        or radial_weight != 0.0
        or band_balanced_weight != 0.0
        or config.LOG_TF_LOSS
        or config.HARD_MINING
        or band_curriculum
    )
    if not use_composite:
        return MaskedLpLoss(d=2, p=config.LOSS_P)
    return MaskedCompositeLoss(
        rel_weight=config.LOSS_REL_WEIGHT,
        h1_weight=config.LOSS_H1_WEIGHT,
        freq_weight=config.LOSS_FREQ_WEIGHT,
        linf_weight=config.LOSS_LINF_WEIGHT,
        p=config.LOSS_P,
        hard_mining=config.HARD_MINING,
        hard_mining_power=config.HARD_MINING_POWER,
        freq_loss_log_weight=config.FREQ_LOSS_LOG_WEIGHT,
        log_tf_loss=config.LOG_TF_LOSS,
        valley_loss_weight=config.VALLEY_LOSS_WEIGHT,
        valley_percentile=config.VALLEY_PERCENTILE,
        radial_weight=radial_weight,
        radial_i_low=radial_i_low,
        radial_i_high=radial_i_high,
        band_balanced_weight=band_balanced_weight,
    )
