"""Wave A spectral aux losses: trough-windowed logspec and DCT-below-cutoff on R."""

from __future__ import annotations

import math

import torch

_EPS = 1e-12
TROUGH_FRAC = 0.05
DCT_CUTOFF = 16


def trough_keep_mask(
    tf: torch.Tensor, *, floor_frac: float = TROUGH_FRAC
) -> torch.Tensor:
    """Keep bins whose |TF| is at least ``floor_frac`` of the per-row peak."""
    amp = tf.abs()
    peak = amp.amax(dim=-1, keepdim=True).clamp_min(_EPS)
    return amp >= (float(floor_frac) * peak)


def trough_safe_logspec_loss(
    tf_hat: torch.Tensor,
    tf_true: torch.Tensor,
    *,
    floor_frac: float = TROUGH_FRAC,
) -> torch.Tensor:
    """Relative L2 on log|TF|, dropping near-zero trough bins of the target."""
    keep = trough_keep_mask(tf_true, floor_frac=floor_frac)
    lp = tf_hat.abs().clamp_min(_EPS).log()
    lt = tf_true.abs().clamp_min(_EPS).log()
    w = keep.to(dtype=lp.dtype)
    diff = (lp - lt) * w
    num = torch.linalg.vector_norm(diff, dim=-1)
    den = torch.linalg.vector_norm(lt * w, dim=-1).clamp_min(_EPS)
    return (num / den).mean()


def logspec_full_loss(
    tf_hat: torch.Tensor,
    tf_true: torch.Tensor,
    *,
    floor_frac: float = 0.02,
) -> torch.Tensor:
    """Mean-square log-amplitude error over *all* bins, troughs retained.

    `trough_safe_logspec_loss` drops low-amplitude bins for numerical safety, but
    the geotechnical misfit score is an RMS of ln(pred/true) over every bin, so
    the dropped bins are exactly the ones driving the reported shortfall. Here the
    troughs are kept and the log is taken after clamping both spectra to a common
    relative floor, which bounds the gradient without excluding the bin.
    """
    floor = (float(floor_frac) * tf_true.abs().amax(dim=-1, keepdim=True)).clamp_min(
        _EPS
    )
    lp = tf_hat.abs().clamp_min(floor).log()
    lt = tf_true.abs().clamp_min(floor).log()
    return ((lp - lt) ** 2).mean()


def band_normalized_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    band_masks: torch.Tensor,
    band_scales: torch.Tensor,
) -> torch.Tensor:
    """Mean over bands of that band's MSE divided by that band's own target scale.

    The plain objective is dominated by the high-amplitude mid band, so the low
    band -- where the residual is small in absolute terms but the surrogate's
    error is large *relative* to it -- contributes almost nothing to the gradient.
    Dividing each band's squared error by that band's target variance makes the
    three bands carry equal weight without clamping the low band toward zero.

    band_masks: (n_bands, n_freq) boolean. band_scales: (n_bands,) target RMS.
    """
    diff2 = (pred - true) ** 2
    flat = diff2.reshape(diff2.shape[0], -1, band_masks.shape[-1])
    per_band = []
    for b in range(band_masks.shape[0]):
        m = band_masks[b]
        if not bool(m.any()):
            continue
        per_band.append(flat[..., m].mean() / band_scales[b].clamp_min(_EPS) ** 2)
    if not per_band:
        return pred.sum() * 0.0
    return torch.stack(per_band).mean()


def _dct2_ortho_matrix(
    n: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=dtype)
    i = torch.arange(n, device=device, dtype=dtype)
    mat = torch.cos(math.pi / n * (i + 0.5).unsqueeze(0) * k.unsqueeze(1))
    mat = mat * math.sqrt(2.0 / n)
    mat[0] = mat[0] * math.sqrt(0.5)
    return mat


def dct_along_freq(x: torch.Tensor) -> torch.Tensor:
    """Ortho type-II DCT along the last axis (frequency)."""
    n = int(x.shape[-1])
    mat = _dct2_ortho_matrix(n, device=x.device, dtype=x.dtype)
    return torch.matmul(mat, x.unsqueeze(-1)).squeeze(-1)


def dct_below_cutoff_loss(
    r_hat: torch.Tensor,
    r_true: torch.Tensor,
    *,
    n_rec: int,
    cutoff: int = DCT_CUTOFF,
) -> torch.Tensor:
    """L1 on |DCT_k(R)| for frequency modes k < cutoff (FNO n_modes[1] axis)."""
    b = r_hat.shape[0]
    n_freq = r_hat.shape[-1] // int(n_rec)
    hat = r_hat.reshape(b, n_rec, n_freq)
    true = r_true.reshape(b, n_rec, n_freq)
    k_max = min(int(cutoff), n_freq)
    amp_h = dct_along_freq(hat)[..., :k_max].abs()
    amp_t = dct_along_freq(true)[..., :k_max].abs()
    return (amp_h - amp_t).abs().mean()
