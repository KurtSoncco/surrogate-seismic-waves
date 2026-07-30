"""TH-FNO losses — one amplitude map A(·) for all terms.

Default: **linear** |TF| (raw amplitude). Set ``AMPLITUDE_DOMAIN=log`` for
``ln(max(|TF|, EPS))`` instead.

Session N+1 C3: each term is divided by a running magnitude estimate BEFORE λ,
so λ_spec / λ_peak mean what they say (∂/∂logf was ~95% of unnormalized loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

_EPS = float(getattr(config, "TF_LOG_EPS", 1e-3))


def amplitude_map(tf: torch.Tensor) -> torch.Tensor:
    """A(|TF|) — linear raw amplitude, or log with floor."""
    if config.AMPLITUDE_DOMAIN == "log":
        return torch.log(tf.clamp_min(_EPS))
    return tf


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float | None = None,
) -> torch.Tensor:
    beta = config.SMOOTH_L1_BETA if beta is None else beta
    m = mask.unsqueeze(-1)
    pa = amplitude_map(pred) * m
    ta = amplitude_map(target) * m
    n = m.sum() * pred.shape[-1]
    n = n.clamp_min(1.0)
    return F.smooth_l1_loss(pa, ta, beta=beta, reduction="sum") / n


def peak_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """SmoothL1 on peak amplitude in the same A(·) domain."""
    m = mask > 0.5
    losses = []
    for b in range(pred.shape[0]):
        cols = torch.where(m[b])[0]
        for c in cols:
            ap = amplitude_map(pred[b, c].max().unsqueeze(0))
            at = amplitude_map(target[b, c].max().unsqueeze(0))
            losses.append(F.smooth_l1_loss(ap, at, beta=config.SMOOTH_L1_BETA))
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def spectral_deriv_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    log_f: torch.Tensor,
) -> torch.Tensor:
    """SmoothL1 on ∂/∂logf of A(|TF|)."""
    m = mask.unsqueeze(-1)
    pa = amplitude_map(pred) * m
    ta = amplitude_map(target) * m
    dlog = (log_f[1:] - log_f[:-1]).clamp_min(1e-8)
    dp = (pa[..., 1:] - pa[..., :-1]) / dlog
    dt = (ta[..., 1:] - ta[..., :-1]) / dlog
    n = m[..., :-1].sum().clamp_min(1.0) * 1.0
    return F.smooth_l1_loss(dp, dt, beta=config.SMOOTH_L1_BETA, reduction="sum") / (
        n * pred.shape[-1]
    )


class RunningTermNorm(nn.Module):
    """EMA of |term| so λ weights terms on a comparable scale (Session N+1 C3)."""

    def __init__(
        self,
        momentum: float | None = None,
        eps: float | None = None,
        enabled: bool | None = None,
    ):
        super().__init__()
        self.momentum = float(
            config.LOSS_TERM_NORM_MOMENTUM if momentum is None else momentum
        )
        self.eps = float(config.LOSS_TERM_NORM_EPS if eps is None else eps)
        self.enabled = bool(
            config.LOSS_TERM_NORM if enabled is None else enabled
        )
        self.register_buffer("_mag", torch.zeros(()), persistent=True)
        self.register_buffer("_initialized", torch.zeros((), dtype=torch.bool), persistent=True)

    def forward(self, term: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return term
        # Detach magnitude tracking — normalization scale is not a learnable path.
        with torch.no_grad():
            m = term.detach().abs()
            if not bool(self._initialized.item()):
                self._mag.copy_(m.clamp_min(self.eps))
                self._initialized.fill_(True)
            else:
                self._mag.mul_(self.momentum).add_(m, alpha=1.0 - self.momentum)
                self._mag.clamp_min_(self.eps)
        return term / self._mag


class THFNOLoss(nn.Module):
    def __init__(self, freq: torch.Tensor | None = None):
        super().__init__()
        if freq is None:
            freq = torch.logspace(-1, 1, config.N_FREQ)
        self.register_buffer("log_f", torch.log(freq.clamp_min(1e-8)), persistent=False)
        self.norm_base = RunningTermNorm()
        self.norm_peak = RunningTermNorm()
        self.norm_spec = RunningTermNorm()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raw_base = masked_smooth_l1(pred, target, mask)
        raw_peak = (
            peak_smooth_l1(pred, target, mask)
            if config.LOSS_PEAK_WEIGHT > 0
            else pred.new_zeros(())
        )
        raw_spec = (
            spectral_deriv_smooth_l1(pred, target, mask, self.log_f)
            if config.LOSS_SPEC_WEIGHT > 0
            else pred.new_zeros(())
        )
        # Normalize BEFORE λ so weights are commensurable.
        base = config.LOSS_SMOOTH_L1_WEIGHT * self.norm_base(raw_base)
        peak = (
            config.LOSS_PEAK_WEIGHT * self.norm_peak(raw_peak)
            if config.LOSS_PEAK_WEIGHT > 0
            else pred.new_zeros(())
        )
        spec = (
            config.LOSS_SPEC_WEIGHT * self.norm_spec(raw_spec)
            if config.LOSS_SPEC_WEIGHT > 0
            else pred.new_zeros(())
        )
        total = base + peak + spec
        parts = {
            "loss_smooth_l1": float(base.detach()),
            "loss_peak": float(peak.detach()),
            "loss_spec": float(spec.detach()),
            "loss_smooth_l1_raw": float(raw_base.detach()),
            "loss_peak_raw": float(raw_peak.detach()),
            "loss_spec_raw": float(raw_spec.detach()),
            "loss_norm_mag_base": float(self.norm_base._mag.detach()),
            "loss_norm_mag_peak": float(self.norm_peak._mag.detach()),
            "loss_norm_mag_spec": float(self.norm_spec._mag.detach()),
        }
        return total, parts
