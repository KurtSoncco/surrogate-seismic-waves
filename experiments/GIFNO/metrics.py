# metrics.py
"""Per-sample masked metrics for GIFNO transfer-function evaluation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

import config

_EPS = 1e-12


def recorder_x_indices_from_mask(mask: np.ndarray) -> np.ndarray:
    """Sorted recorder x-indices from a single sample mask."""
    return np.where(mask > 0.5)[0]


def gather_recorder_tf_numpy(
    field: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """(N, Nx, F) -> (N, R, F) at fixed lateral recorder columns."""
    if recorder_x is None:
        recorder_x = config.recorder_x_indices()
    return field[:, recorder_x, :]


def gather_recorder_tf_torch(
    field: torch.Tensor,
    recorder_x: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """(B, Nx, F) -> (B, R, F) at fixed lateral recorder columns."""
    if recorder_x is None:
        recorder_x = config.recorder_x_indices()
    idx = torch.as_tensor(recorder_x, device=field.device, dtype=torch.long)
    return field.index_select(1, idx)


def pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 1-D arrays; NaN if undefined."""
    if a.size < 2:
        return float("nan")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < _EPS:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _log_freq_derivative(values: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Central difference d(values)/d(log f) along last axis."""
    log_f = np.log(np.maximum(freq, _EPS))
    d_log = np.diff(log_f)
    d_val = np.diff(values, axis=-1)
    deriv = d_val / np.maximum(d_log, _EPS)
    return deriv


def _log_freq_derivative_torch(
    values: torch.Tensor, log_f: torch.Tensor
) -> torch.Tensor:
    """Central difference d(values)/d(log f) along last axis."""
    d_log = log_f[1:] - log_f[:-1]
    d_val = values[..., 1:] - values[..., :-1]
    return d_val / d_log.clamp_min(_EPS)


def per_sample_rel_l2_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Relative L2 per sample on masked recorder×freq values. Shape (N,)."""
    p = gather_recorder_tf_numpy(pred, recorder_x).reshape(len(pred), -1)
    t = gather_recorder_tf_numpy(target, recorder_x).reshape(len(target), -1)
    num = np.linalg.norm(p - t, axis=1)
    den = np.linalg.norm(t, axis=1) + _EPS
    return num / den


def valley_mask_from_log_target(
    target: np.ndarray,
    percentile: float = config.VALLEY_PERCENTILE,
) -> np.ndarray:
    """
    Boolean mask (..., F) for TF valleys: local minima in log(|TF|) and bottom
    ``percentile`` of log(|TF|) per recorder curve.
    """
    log_t = np.log(np.maximum(np.abs(target), _EPS))
    thresh = np.percentile(log_t, percentile, axis=-1, keepdims=True)
    below = log_t <= thresh
    local_min = np.zeros_like(below, dtype=bool)
    if log_t.shape[-1] >= 3:
        mid = log_t[..., 1:-1]
        local_min[..., 1:-1] = (mid < log_t[..., :-2]) & (mid < log_t[..., 2:])
    if log_t.shape[-1] >= 2:
        local_min[..., 0] |= log_t[..., 0] < log_t[..., 1]
        local_min[..., -1] |= log_t[..., -1] < log_t[..., -2]
    return below | local_min


def valley_mask_from_log_target_torch(
    target: torch.Tensor,
    percentile: float = config.VALLEY_PERCENTILE,
) -> torch.Tensor:
    """Torch version of ``valley_mask_from_log_target`` for (B, R, F) tensors."""
    log_t = torch.log(target.abs().clamp_min(_EPS))
    flat = log_t.reshape(-1, log_t.shape[-1])
    q = max(0.0, min(100.0, percentile)) / 100.0
    thresh = torch.quantile(flat, q, dim=1, keepdim=True)
    thresh = thresh.view(*log_t.shape[:-1], 1)
    below = log_t <= thresh
    local_min = torch.zeros_like(below)
    if log_t.shape[-1] >= 3:
        mid = log_t[..., 1:-1]
        local_min[..., 1:-1] = (mid < log_t[..., :-2]) & (mid < log_t[..., 2:])
    if log_t.shape[-1] >= 2:
        local_min[..., 0] = local_min[..., 0] | (log_t[..., 0] < log_t[..., 1])
        local_min[..., -1] = local_min[..., -1] | (log_t[..., -1] < log_t[..., -2])
    return below | local_min.bool()


def per_sample_linf_split_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
    percentile: float = config.VALLEY_PERCENTILE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample max |error| at valley vs non-valley bins. Shapes (N,), (N,)."""
    p = gather_recorder_tf_numpy(pred, recorder_x)
    t = gather_recorder_tf_numpy(target, recorder_x)
    err = np.abs(p - t)
    n = len(pred)
    linf_valley = np.zeros(n, dtype=np.float64)
    linf_peak = np.zeros(n, dtype=np.float64)
    for i in range(n):
        vmask = valley_mask_from_log_target(t[i], percentile=percentile)
        if np.any(vmask):
            linf_valley[i] = float(np.max(err[i][vmask]))
        if np.any(~vmask):
            linf_peak[i] = float(np.max(err[i][~vmask]))
    return linf_valley, linf_peak


def per_sample_rel_l2_valley_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
    percentile: float = config.VALLEY_PERCENTILE,
) -> np.ndarray:
    """Relative L2 on valley bins only. Shape (N,)."""
    p = gather_recorder_tf_numpy(pred, recorder_x)
    t = gather_recorder_tf_numpy(target, recorder_x)
    n = len(pred)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        vmask = valley_mask_from_log_target(t[i], percentile=percentile)
        if not np.any(vmask):
            continue
        pv = p[i][vmask].reshape(-1)
        tv = t[i][vmask].reshape(-1)
        out[i] = np.linalg.norm(pv - tv) / (np.linalg.norm(tv) + _EPS)
    return out


def per_sample_linf_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Max absolute error per sample. Shape (N,)."""
    p = gather_recorder_tf_numpy(pred, recorder_x)
    t = gather_recorder_tf_numpy(target, recorder_x)
    return np.max(np.abs(p - t), axis=(1, 2))


def per_sample_pearson_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mean Pearson r across recorders (|TF| along frequency). Shape (N,)."""
    p = gather_recorder_tf_numpy(np.abs(pred), recorder_x)
    t = gather_recorder_tf_numpy(np.abs(target), recorder_x)
    n, n_rec, _ = p.shape
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        rs = [pearson_1d(p[i, r], t[i, r]) for r in range(n_rec)]
        rs = [r for r in rs if np.isfinite(r)]
        out[i] = float(np.mean(rs)) if rs else float("nan")
    return out


def _freq_band_slice(freq: np.ndarray, f_lo: float, f_hi: float) -> slice:
    """Index slice for frequencies in [f_lo, f_hi]."""
    lo = int(np.searchsorted(freq, f_lo, side="left"))
    hi = int(np.searchsorted(freq, f_hi, side="right"))
    return slice(lo, max(lo + 1, hi))


def _rel_l2_on_slice(
    pred_rf: np.ndarray, target_rf: np.ndarray, f_slice: slice
) -> np.ndarray:
    """Relative L2 per sample on recorder×freq subset. pred/target: (N, R, F)."""
    p = pred_rf[..., f_slice].reshape(len(pred_rf), -1)
    t = target_rf[..., f_slice].reshape(len(target_rf), -1)
    num = np.linalg.norm(p - t, axis=1)
    den = np.linalg.norm(t, axis=1) + _EPS
    return num / den


def per_sample_peak_freq_err_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    freq: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mean |Δlog f| of dominant peaks across recorders. Shape (N,)."""
    p = gather_recorder_tf_numpy(np.abs(pred), recorder_x)
    t = gather_recorder_tf_numpy(np.abs(target), recorder_x)
    log_f = np.log(np.maximum(freq, _EPS))
    n, n_rec, _ = p.shape
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        errs: list[float] = []
        for r in range(n_rec):
            t_idx = int(np.argmax(np.log(t[i, r] + _EPS)))
            p_idx = int(np.argmax(np.log(p[i, r] + _EPS)))
            errs.append(float(abs(log_f[p_idx] - log_f[t_idx])))
        out[i] = float(np.mean(errs)) if errs else float("nan")
    return out


def per_sample_peak_amp_err_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    freq: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mean relative amplitude error at target peak frequency. Shape (N,)."""
    del freq  # peak index comes from target curve
    p = gather_recorder_tf_numpy(np.abs(pred), recorder_x)
    t = gather_recorder_tf_numpy(np.abs(target), recorder_x)
    n, n_rec, _ = p.shape
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        errs: list[float] = []
        for r in range(n_rec):
            t_idx = int(np.argmax(np.log(t[i, r] + _EPS)))
            tv = float(t[i, r, t_idx])
            if tv < _EPS:
                continue
            errs.append(float(abs(p[i, r, t_idx] - tv) / tv))
        out[i] = float(np.mean(errs)) if errs else float("nan")
    return out


def per_sample_bandwise_rel_l2_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    freq: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Relative L2 in low/mid/high frequency bands. Each array shape (N,)."""
    p = gather_recorder_tf_numpy(pred, recorder_x)
    t = gather_recorder_tf_numpy(target, recorder_x)
    bands = {
        "low": getattr(config, "FREQ_BAND_LOW", (0.1, 0.5)),
        "mid": getattr(config, "FREQ_BAND_MID", (0.5, 2.0)),
        "high": getattr(config, "FREQ_BAND_HIGH", (2.0, 10.0)),
    }
    return {
        name: _rel_l2_on_slice(p, t, _freq_band_slice(freq, lo, hi))
        for name, (lo, hi) in bands.items()
    }


def per_sample_logspec_rel_l2_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Relative L2 on log(|TF|) over recorders×freq. Shape (N,)."""
    p = gather_recorder_tf_numpy(np.abs(pred), recorder_x)
    t = gather_recorder_tf_numpy(np.abs(target), recorder_x)
    lp = np.log(np.maximum(p, _EPS)).reshape(len(pred), -1)
    lt = np.log(np.maximum(t, _EPS)).reshape(len(target), -1)
    num = np.linalg.norm(lp - lt, axis=1)
    den = np.linalg.norm(lt, axis=1) + _EPS
    return num / den


def _central_recorder_index(recorder_x: np.ndarray) -> int:
    """Recorder slot closest to the lateral domain center."""
    center = config.NX // 2
    return int(np.argmin(np.abs(recorder_x - center)))


def per_recorder_lsd_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Log-Spectral Distance (dB) per sample at each recorder. Shape (N, R).

    LSD = sqrt(mean_f (20*log10|T| - 20*log10|P|)^2), the standard spectral
    distortion in decibels on the magnitude transfer function.
    """
    p = np.abs(gather_recorder_tf_numpy(pred, recorder_x)).astype(np.float64)
    t = np.abs(gather_recorder_tf_numpy(target, recorder_x)).astype(np.float64)
    db_p = 20.0 * np.log10(np.maximum(p, _EPS))
    db_t = 20.0 * np.log10(np.maximum(t, _EPS))
    return np.sqrt(np.mean((db_t - db_p) ** 2, axis=-1))


def per_recorder_log_mae_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mean absolute error on natural log|TF| per recorder. Shape (N, R)."""
    p = np.abs(gather_recorder_tf_numpy(pred, recorder_x)).astype(np.float64)
    t = np.abs(gather_recorder_tf_numpy(target, recorder_x)).astype(np.float64)
    lp = np.log(np.maximum(p, _EPS))
    lt = np.log(np.maximum(t, _EPS))
    return np.mean(np.abs(lp - lt), axis=-1)


def per_recorder_log_rmse_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Root mean squared error on natural log|TF| per recorder. Shape (N, R)."""
    p = np.abs(gather_recorder_tf_numpy(pred, recorder_x)).astype(np.float64)
    t = np.abs(gather_recorder_tf_numpy(target, recorder_x)).astype(np.float64)
    lp = np.log(np.maximum(p, _EPS))
    lt = np.log(np.maximum(t, _EPS))
    return np.sqrt(np.mean((lp - lt) ** 2, axis=-1))


def per_recorder_spectral_cosine_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Cosine similarity between |TF| spectra per recorder. Shape (N, R).

    Unlike Pearson, spectra are not mean-subtracted, so this measures shape
    agreement of the raw magnitude spectrum (1.0 == identical direction).
    """
    p = np.abs(gather_recorder_tf_numpy(pred, recorder_x)).astype(np.float64)
    t = np.abs(gather_recorder_tf_numpy(target, recorder_x)).astype(np.float64)
    dot = np.sum(p * t, axis=-1)
    denom = np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1)
    return dot / np.maximum(denom, _EPS)


def central_and_allrec_summary(
    values_nr: np.ndarray,
    recorder_x: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    """Summaries for a (N, R) metric: central recorder + all-recorder mean.

    Emits distribution summaries (mean/p10/p50/p90) for the central recorder
    curve and for the per-sample mean across recorders, plus a per-recorder
    tail summary (min/mean of per-recorder p10/p50).
    """
    if values_nr.size == 0:
        return {}
    out: Dict[str, float] = {}
    r_center = _central_recorder_index(recorder_x)
    out.update(distribution_summary(values_nr[:, r_center], f"{prefix}_central"))
    out.update(distribution_summary(np.nanmean(values_nr, axis=1), f"{prefix}_allrec"))
    out.update(per_recorder_tail_summary(values_nr, f"{prefix}_rec"))
    return out


def per_sample_h1_freq_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    freq: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Relative H1-style metric along log-frequency per sample:
    rel_l2(values) + rel_l2(d/d(log f)).
    """
    p = gather_recorder_tf_numpy(pred, recorder_x)
    t = gather_recorder_tf_numpy(target, recorder_x)
    n = len(pred)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        pv = p[i].reshape(-1)
        tv = t[i].reshape(-1)
        rel_val = np.linalg.norm(pv - tv) / (np.linalg.norm(tv) + _EPS)
        pd = _log_freq_derivative(p[i], freq).reshape(-1)
        td = _log_freq_derivative(t[i], freq).reshape(-1)
        rel_deriv = np.linalg.norm(pd - td) / (np.linalg.norm(td) + _EPS)
        out[i] = rel_val + rel_deriv
    return out


def per_sample_rel_l2_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    recorder_x: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Relative L2 per batch item. Shape (B,)."""
    p = gather_recorder_tf_torch(pred, recorder_x).reshape(pred.shape[0], -1)
    t = gather_recorder_tf_torch(target, recorder_x).reshape(target.shape[0], -1)
    num = torch.linalg.norm(p - t, dim=1)
    den = torch.linalg.norm(t, dim=1).clamp_min(_EPS)
    return num / den


def per_sample_linf_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    recorder_x: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Max absolute error per batch item. Shape (B,)."""
    p = gather_recorder_tf_torch(pred, recorder_x)
    t = gather_recorder_tf_torch(target, recorder_x)
    return torch.amax(torch.abs(p - t), dim=(1, 2))


def per_recorder_rel_l2_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Relative L2 per sample at each recorder. Shape (N, R)."""
    p = gather_recorder_tf_numpy(pred, recorder_x)
    t = gather_recorder_tf_numpy(target, recorder_x)
    num = np.linalg.norm(p - t, axis=2)
    den = np.linalg.norm(t, axis=2) + _EPS
    return num / den


def per_recorder_pearson_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    recorder_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pearson r per sample at each recorder (|TF| vs freq). Shape (N, R)."""
    p = gather_recorder_tf_numpy(np.abs(pred), recorder_x)
    t = gather_recorder_tf_numpy(np.abs(target), recorder_x)
    n, n_rec, _ = p.shape
    out = np.full((n, n_rec), np.nan, dtype=np.float64)
    for i in range(n):
        for r in range(n_rec):
            out[i, r] = pearson_1d(p[i, r], t[i, r])
    return out


def per_recorder_rel_l2_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    recorder_x: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Relative L2 per batch item at each recorder. Shape (B, R)."""
    p = gather_recorder_tf_torch(pred, recorder_x)
    t = gather_recorder_tf_torch(target, recorder_x)
    num = torch.linalg.norm(p - t, dim=2)
    den = torch.linalg.norm(t, dim=2).clamp_min(_EPS)
    return num / den


def per_recorder_tail_summary(values_nr: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Summarize (N, R) per-sample metrics per recorder.

    For each recorder, compute p10/p50 across samples, then report min across
    recorders (worst lateral site tail) plus mean of per-recorder p10.
    """
    if values_nr.size == 0:
        return {}
    n_rec = values_nr.shape[1]
    p10_per_rec = np.full(n_rec, np.nan, dtype=np.float64)
    p50_per_rec = np.full(n_rec, np.nan, dtype=np.float64)
    for j in range(n_rec):
        col = values_nr[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            p10_per_rec[j] = np.percentile(col, 10)
            p50_per_rec[j] = np.percentile(col, 50)
    out: Dict[str, float] = {}
    if np.any(np.isfinite(p10_per_rec)):
        out[f"{prefix}_min_p10"] = float(np.nanmin(p10_per_rec))
        out[f"{prefix}_mean_p10"] = float(np.nanmean(p10_per_rec))
    if np.any(np.isfinite(p50_per_rec)):
        out[f"{prefix}_min_p50"] = float(np.nanmin(p50_per_rec))
        out[f"{prefix}_mean_p50"] = float(np.nanmean(p50_per_rec))
    return out


def distribution_summary(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """Mean and percentile summaries for a 1-D metric array."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
        }
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    out = {
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_p10": float(p10),
        f"{prefix}_p50": float(p50),
        f"{prefix}_p90": float(p90),
    }
    if "linf" in prefix:
        out[f"{prefix}_max"] = float(np.max(v))
    return out


def compute_per_sample_metrics_numpy(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    freq: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute per-sample metric vectors."""
    recorder_x = recorder_x_indices_from_mask(masks[0])
    linf_valley, linf_peak = per_sample_linf_split_numpy(
        predictions, targets, recorder_x
    )
    rel_l2_valley = per_sample_rel_l2_valley_numpy(predictions, targets, recorder_x)
    band_rel = per_sample_bandwise_rel_l2_numpy(predictions, targets, freq, recorder_x)
    return {
        "rel_l2": per_sample_rel_l2_numpy(predictions, targets, recorder_x),
        "linf": per_sample_linf_numpy(predictions, targets, recorder_x),
        "linf_valley": linf_valley,
        "linf_peak": linf_peak,
        "rel_l2_valley": rel_l2_valley,
        "pearson": per_sample_pearson_numpy(predictions, targets, recorder_x),
        "h1_freq": per_sample_h1_freq_numpy(predictions, targets, freq, recorder_x),
        "peak_freq_err": per_sample_peak_freq_err_numpy(
            predictions, targets, freq, recorder_x
        ),
        "peak_amp_err": per_sample_peak_amp_err_numpy(
            predictions, targets, freq, recorder_x
        ),
        "rel_l2_band_low": band_rel["low"],
        "rel_l2_band_mid": band_rel["mid"],
        "rel_l2_band_high": band_rel["high"],
        "logspec_rel_l2": per_sample_logspec_rel_l2_numpy(
            predictions, targets, recorder_x
        ),
    }


def aggregate_test_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    freq: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Global pooled metrics plus per-sample distribution summaries.

    Returns
    -------
    summary : flat dict for W&B logging
    per_sample : metric name -> (N,) arrays
    """
    per_sample = compute_per_sample_metrics_numpy(predictions, targets, masks, freq)

    pred_flat = gather_recorder_tf_numpy(predictions).reshape(-1)
    target_flat = gather_recorder_tf_numpy(targets).reshape(-1)

    summary: Dict[str, float] = {
        "test_mse": float(np.mean((target_flat - pred_flat) ** 2)),
        "test_mae": float(np.mean(np.abs(target_flat - pred_flat))),
        "test_rel_l2": float(
            np.linalg.norm(target_flat - pred_flat)
            / (np.linalg.norm(target_flat) + _EPS)
        ),
        "test_pearson": float(
            pearson_1d(np.abs(pred_flat), np.abs(target_flat))
            if np.isfinite(pearson_1d(np.abs(pred_flat), np.abs(target_flat)))
            else 0.0
        ),
    }

    summary.update(distribution_summary(per_sample["rel_l2"], "test_rel_l2"))
    summary.update(distribution_summary(per_sample["linf"], "test_linf"))
    summary.update(distribution_summary(per_sample["linf_valley"], "test_linf_valley"))
    summary.update(distribution_summary(per_sample["linf_peak"], "test_linf_peak"))
    summary.update(
        distribution_summary(per_sample["rel_l2_valley"], "test_rel_l2_valley")
    )
    summary.update(distribution_summary(per_sample["pearson"], "test_pearson"))
    summary.update(distribution_summary(per_sample["h1_freq"], "test_h1_freq"))
    summary.update(
        distribution_summary(per_sample["peak_freq_err"], "test_peak_freq_err")
    )
    summary.update(
        distribution_summary(per_sample["peak_amp_err"], "test_peak_amp_err")
    )
    summary.update(
        distribution_summary(per_sample["rel_l2_band_low"], "test_rel_l2_band_low")
    )
    summary.update(
        distribution_summary(per_sample["rel_l2_band_mid"], "test_rel_l2_band_mid")
    )
    summary.update(
        distribution_summary(per_sample["rel_l2_band_high"], "test_rel_l2_band_high")
    )
    summary.update(
        distribution_summary(per_sample["logspec_rel_l2"], "test_logspec_rel_l2")
    )

    recorder_x = recorder_x_indices_from_mask(masks[0])
    rec_rel_l2 = per_recorder_rel_l2_numpy(predictions, targets, recorder_x)
    rec_pearson = per_recorder_pearson_numpy(predictions, targets, recorder_x)
    summary.update(per_recorder_tail_summary(rec_rel_l2, "test_rec_rel_l2"))
    summary.update(per_recorder_tail_summary(rec_pearson, "test_rec_pearson"))

    rec_lsd = per_recorder_lsd_numpy(predictions, targets, recorder_x)
    rec_log_mae = per_recorder_log_mae_numpy(predictions, targets, recorder_x)
    rec_log_rmse = per_recorder_log_rmse_numpy(predictions, targets, recorder_x)
    rec_spec_cos = per_recorder_spectral_cosine_numpy(predictions, targets, recorder_x)
    summary.update(central_and_allrec_summary(rec_lsd, recorder_x, "test_lsd"))
    summary.update(central_and_allrec_summary(rec_log_mae, recorder_x, "test_log_mae"))
    summary.update(
        central_and_allrec_summary(rec_log_rmse, recorder_x, "test_log_rmse")
    )
    summary.update(
        central_and_allrec_summary(rec_spec_cos, recorder_x, "test_spec_cos")
    )

    return summary, per_sample


def compute_val_tail_metrics_torch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    freq: np.ndarray,
) -> Dict[str, float]:
    """Validation pass: pooled and per-recorder tail stats."""
    model.eval()
    rel_all: list[np.ndarray] = []
    linf_all: list[np.ndarray] = []
    rec_rel_all: list[np.ndarray] = []
    rec_pearson_all: list[np.ndarray] = []
    peak_freq_all: list[np.ndarray] = []
    peak_amp_all: list[np.ndarray] = []
    logspec_all: list[np.ndarray] = []
    band_low_all: list[np.ndarray] = []
    band_mid_all: list[np.ndarray] = []
    band_high_all: list[np.ndarray] = []
    pearson_all: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets, _masks in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            rel_all.append(per_sample_rel_l2_torch(outputs, targets).cpu().numpy())
            linf_all.append(per_sample_linf_torch(outputs, targets).cpu().numpy())
            rec_rel_all.append(
                per_recorder_rel_l2_torch(outputs, targets).cpu().numpy()
            )
            pred_np = outputs.cpu().numpy()
            tgt_np = targets.cpu().numpy()
            rec_pearson_all.append(per_recorder_pearson_numpy(pred_np, tgt_np))
            peak_freq_all.append(per_sample_peak_freq_err_numpy(pred_np, tgt_np, freq))
            peak_amp_all.append(per_sample_peak_amp_err_numpy(pred_np, tgt_np, freq))
            logspec_all.append(per_sample_logspec_rel_l2_numpy(pred_np, tgt_np))
            bands = per_sample_bandwise_rel_l2_numpy(pred_np, tgt_np, freq)
            band_low_all.append(bands["low"])
            band_mid_all.append(bands["mid"])
            band_high_all.append(bands["high"])
            pearson_all.append(per_sample_pearson_numpy(pred_np, tgt_np))

    rel = np.concatenate(rel_all) if rel_all else np.array([])
    linf = np.concatenate(linf_all) if linf_all else np.array([])
    rec_rel = np.concatenate(rec_rel_all) if rec_rel_all else np.array([]).reshape(0, 0)
    rec_pearson = (
        np.concatenate(rec_pearson_all)
        if rec_pearson_all
        else np.array([]).reshape(0, 0)
    )

    out: Dict[str, float] = {}
    if rel.size:
        out.update(distribution_summary(rel, "val_rel_l2"))
    if linf.size:
        out.update(distribution_summary(linf, "val_linf"))
    if rec_rel.size:
        out.update(per_recorder_tail_summary(rec_rel, "val_rec_rel_l2"))
    if rec_pearson.size:
        out.update(per_recorder_tail_summary(rec_pearson, "val_rec_pearson"))
    if peak_freq_all:
        out.update(
            distribution_summary(np.concatenate(peak_freq_all), "val_peak_freq_err")
        )
    if peak_amp_all:
        out.update(
            distribution_summary(np.concatenate(peak_amp_all), "val_peak_amp_err")
        )
    if logspec_all:
        out.update(
            distribution_summary(np.concatenate(logspec_all), "val_logspec_rel_l2")
        )
    if band_low_all:
        out.update(
            distribution_summary(np.concatenate(band_low_all), "val_rel_l2_band_low")
        )
    if band_mid_all:
        out.update(
            distribution_summary(np.concatenate(band_mid_all), "val_rel_l2_band_mid")
        )
    if band_high_all:
        out.update(
            distribution_summary(np.concatenate(band_high_all), "val_rel_l2_band_high")
        )
    if pearson_all:
        out.update(distribution_summary(np.concatenate(pearson_all), "val_pearson"))
    return out
