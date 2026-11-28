"""Losses on the log residual and on the linear outcrop |TF|.

``log_l1`` is the training default. ``rel_l1`` scores linear amplitude, so the
fundamental and the second peak dominate it. ``hybrid_rel_log`` adds a ±25%
window around those two peaks and the LOGLO band-energy and slope terms.
The network predicts a residual; the column prior is added before the linear
amplitude is formed.
"""

from __future__ import annotations

import numpy as np
import torch

BANDS = {"low": (0.1, 0.4), "mid": (0.4, 2.5), "high": (2.5, 10.0)}


def linear_tf(pred: torch.Tensor, prior: torch.Tensor, *, residual: bool) -> torch.Tensor:
    """Linear |TF| from a residual prediction or from a direct log|TF| prediction."""
    log_tf = pred + prior if residual else pred
    return torch.exp(log_tf.clamp(-20.0, 20.0))


def rel_l1(pred_tf: torch.Tensor, true_tf: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """sum |TF_pred - TF_true| / sum |TF_true| over valid stations and frequencies."""
    mask = valid.to(pred_tf.dtype)[:, :, None]
    num = (pred_tf - true_tf).abs() * mask
    den = true_tf.abs() * mask
    return num.sum() / den.sum().clamp_min(1e-6)


def _mode_windows(true_tf: torch.Tensor, freq: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(B, K, F) weight, 1 outside the modes and 4 inside ±25% of each true peak."""
    weight = torch.ones(true_tf.shape, device=true_tf.device, dtype=true_tf.dtype)
    freq_np = freq.detach().float().cpu().numpy()
    amp = true_tf.detach().float().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    low = (freq_np >= 0.15) & (freq_np <= 4.0)
    if not np.any(low):
        return weight
    low_idx = np.flatnonzero(low)
    for b in range(amp.shape[0]):
        for k in range(amp.shape[1]):
            if not bool(valid_np[b, k]):
                continue
            m1 = int(low_idx[np.argmax(amp[b, k, low_idx])])
            f1 = float(freq_np[m1])
            hi = np.flatnonzero((freq_np >= 1.5 * f1) & (freq_np <= min(10.0, 4.0 * f1)))
            m2 = int(hi[np.argmax(amp[b, k, hi])]) if hi.size else m1
            for centre in (float(freq_np[m1]), float(freq_np[m2])):
                band = (freq >= 0.75 * centre) & (freq <= 1.25 * centre)
                weight[b, k, band] = 4.0
    return weight


def modal_peak_l1(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, freq: torch.Tensor, mode_w: torch.Tensor,
) -> torch.Tensor:
    """Log-L1 inside the mode windows. ``mode_w`` is 4 there and 1 elsewhere; the extra 3 is the weight."""
    extra = (mode_w - 1.0).clamp_min(0.0)
    mask = valid.to(pred.dtype)[:, :, None] * extra
    err = (pred - target).abs() * mask
    return err.sum() / mask.sum().clamp_min(1.0)


def spectral_binned_loss(pred: torch.Tensor, target: torch.Tensor, freq: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean absolute band energy error plus |d g / d log f| slope error.

    Bands are 0.1–0.4, 0.4–2.5 and 2.5–10 Hz. The slope is the first difference
    along frequency, divided by the log-frequency step, on the log residual.
    """
    mask = valid.to(pred.dtype)[:, :, None].expand_as(pred)
    pieces = []
    for lo, hi in BANDS.values():
        band = (freq >= lo) & (freq <= hi)
        if int(band.sum()) == 0:
            continue
        num = ((pred - target).abs() * mask)[:, :, band]
        den = mask[:, :, band].sum().clamp_min(1.0)
        pieces.append(num.sum() / den)
    energy = torch.stack(pieces).mean() if pieces else pred.new_zeros(())
    dlog = torch.diff(torch.log(freq.clamp_min(1e-6)))
    slope_p = torch.diff(pred, dim=-1) / dlog.view(1, 1, -1)
    slope_t = torch.diff(target, dim=-1) / dlog.view(1, 1, -1)
    smask = mask[..., :-1]
    slope = ((slope_p - slope_t).abs() * smask).sum() / smask.sum().clamp_min(1.0)
    return energy + slope


def hybrid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    prior: torch.Tensor,
    valid: torch.Tensor,
    freq: torch.Tensor,
    width: torch.Tensor,
    freq_weight: torch.Tensor | None,
    *,
    residual: bool,
    alpha: float = 1.0,
    beta: float = 1.0,
    spectral: float = 0.5,
    log_l1_fn,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """log-L1 + Rel-L1 + modal log-L1 + LOGLO spectral loss. Returns the sum and the parts."""
    log_term = log_l1_fn(pred, target, valid, freq, width, freq_weight)
    pred_tf = linear_tf(pred, prior, residual=residual)
    true_tf = linear_tf(target, prior, residual=residual)
    rel = rel_l1(pred_tf, true_tf, valid)
    modes = _mode_windows(true_tf, freq, valid)
    modal = modal_peak_l1(pred, target, valid, freq, modes)
    spec = spectral_binned_loss(pred, target, freq, valid)
    total = log_term + float(alpha) * rel + float(beta) * modal + float(spectral) * spec
    return total, {"log_l1": log_term, "rel_l1": rel, "modal": modal, "spectral": spec}
