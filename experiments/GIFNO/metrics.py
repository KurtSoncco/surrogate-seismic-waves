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


def _log_freq_derivative_torch(values: torch.Tensor, log_f: torch.Tensor) -> torch.Tensor:
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
        local_min[..., -1] = local_min[..., -1] | (
            log_t[..., -1] < log_t[..., -2]
        )
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
    return {
        "rel_l2": per_sample_rel_l2_numpy(predictions, targets, recorder_x),
        "linf": per_sample_linf_numpy(predictions, targets, recorder_x),
        "linf_valley": linf_valley,
        "linf_peak": linf_peak,
        "rel_l2_valley": rel_l2_valley,
        "pearson": per_sample_pearson_numpy(predictions, targets, recorder_x),
        "h1_freq": per_sample_h1_freq_numpy(predictions, targets, freq, recorder_x),
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
    summary.update(distribution_summary(per_sample["rel_l2_valley"], "test_rel_l2_valley"))
    summary.update(distribution_summary(per_sample["pearson"], "test_pearson"))
    summary.update(distribution_summary(per_sample["h1_freq"], "test_h1_freq"))

    return summary, per_sample


def compute_val_tail_metrics_torch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    freq: np.ndarray,
) -> Dict[str, float]:
    """Lightweight validation pass: per-sample rel_l2 and linf tail stats."""
    model.eval()
    rel_all: list[np.ndarray] = []
    linf_all: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets, _masks in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            rel_all.append(per_sample_rel_l2_torch(outputs, targets).cpu().numpy())
            linf_all.append(per_sample_linf_torch(outputs, targets).cpu().numpy())

    rel = np.concatenate(rel_all) if rel_all else np.array([])
    linf = np.concatenate(linf_all) if linf_all else np.array([])

    out: Dict[str, float] = {}
    if rel.size:
        out.update(
            {
                "val_rel_l2_mean": float(np.mean(rel)),
                "val_rel_l2_p10": float(np.percentile(rel, 10)),
                "val_rel_l2_p90": float(np.percentile(rel, 90)),
            }
        )
    if linf.size:
        out.update(
            {
                "val_linf_mean": float(np.mean(linf)),
                "val_linf_p90": float(np.percentile(linf, 90)),
                "val_linf_max": float(np.max(linf)),
            }
        )
    return out
