"""Harm-rate / fail-soft / peak-shift for residual vs frozen 1D Haskell.

A residual *harms* 1D when ||TF_hat - TF2D|| > ||TF1D - TF2D||.
Fail-soft: |TF_hat - TF1D| should be small on files whose true leftover is in
the bottom quantile. Peak ΔlnA / Δf0: tallest scipy.find_peaks |TF| in a
log-symmetric ±20% window around soil travel-time f0_calc = 1/(4T), central
recorder. Ground truth is the 2D-TF peak in that window, not a global argmax.

LOGLO-style extras (E0+): logspec rel L2, peak Δln A, band rel L2, harm rate
stratified by leftover quantile.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks

_EPS = 1e-12
F0_LOG_REL = 0.20  # log-symmetric window: f ∈ [f0/1.2, 1.2 f0]

FREQ_BAND_LOW = (0.1, 0.5)
FREQ_BAND_MID = (0.5, 2.0)
FREQ_BAND_HIGH = (2.0, 10.0)


def _as_samples(arr: np.ndarray, n_rec: int, n_freq: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    q = int(n_rec) * int(n_freq)
    if a.ndim == 3:
        return a.reshape(a.shape[0], q)
    if a.ndim == 2 and a.shape[-1] == q:
        return a
    if a.ndim == 1:
        if a.size % q != 0:
            raise ValueError(f"flat size {a.size} not divisible by n_rec*n_freq={q}")
        return a.reshape(-1, q)
    raise ValueError(f"unexpected array shape {a.shape}")


def tf_from_residual(
    tf1d: np.ndarray,
    r_hat: np.ndarray,
    *,
    log_residual: bool = False,
) -> np.ndarray:
    """Reconstruct TF from a leftover prediction (additive or log-multiplicative)."""
    tf1d = np.asarray(tf1d, dtype=np.float64)
    r_hat = np.asarray(r_hat, dtype=np.float64)
    if log_residual:
        return tf1d * np.exp(np.clip(r_hat, -20.0, 20.0))
    return tf1d + r_hat


def rel_l2_rows(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    num = np.linalg.norm(pred - true, axis=-1)
    den = np.linalg.norm(true, axis=-1)
    return num / np.maximum(den, _EPS)


def rel_l1_rows(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Σ|pred-true| / Σ|true| along the last axis (sharp TF; less peak-blind than L2)."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    num = np.sum(np.abs(pred - true), axis=-1)
    den = np.sum(np.abs(true), axis=-1)
    return num / np.maximum(den, _EPS)


def residual_tv_rows(r: np.ndarray) -> np.ndarray:
    """Total variation Σ_f |R(f+Δf)−R(f)| along the last axis (wiggliness)."""
    r = np.asarray(r, dtype=np.float64)
    if r.shape[-1] < 2:
        return np.zeros(r.shape[:-1], dtype=np.float64)
    return np.sum(np.abs(np.diff(r, axis=-1)), axis=-1)


def leftover_rel_l2(r: np.ndarray, tf1d: np.ndarray) -> np.ndarray:
    """||R|| / ||TF1D|| per sample (how large the 2D leftover is)."""
    r = np.asarray(r, dtype=np.float64)
    tf1d = np.asarray(tf1d, dtype=np.float64)
    return np.linalg.norm(r, axis=-1) / np.maximum(np.linalg.norm(tf1d, axis=-1), _EPS)


def harm_mask(
    tf1d: np.ndarray,
    r_hat: np.ndarray,
    tf2d: np.ndarray,
    *,
    log_residual: bool = False,
    tf_hat: np.ndarray | None = None,
) -> np.ndarray:
    """True where TF_hat is worse than Haskell-only."""
    if tf_hat is None:
        tf_hat = tf_from_residual(tf1d, r_hat, log_residual=log_residual)
    err_hat = np.linalg.norm(tf_hat - tf2d, axis=-1)
    err_1d = np.linalg.norm(tf1d - tf2d, axis=-1)
    return err_hat > err_1d + 1e-15


def log_f0_window(f0_calc: float, *, rel: float = F0_LOG_REL) -> tuple[float, float]:
    """Log-symmetric ±rel window: log f ∈ [log f0 − log(1+rel), log f0 + log(1+rel)]."""
    f0 = float(f0_calc)
    if not np.isfinite(f0) or f0 <= 0.0:
        return float("nan"), float("nan")
    factor = 1.0 + max(float(rel), 0.0)
    return f0 / factor, f0 * factor


def max_peak_near_f0_calc(
    freq: np.ndarray,
    af: np.ndarray,
    f0_calc: float,
    *,
    rel: float = F0_LOG_REL,
) -> tuple[float, float]:
    """Tallest scipy ``find_peaks`` |TF| in ±20% logspace around ``1/(4T)``.

    Same detector for 2D-TF ground truth **and** the predicted TF (f_hat).
    """
    f = np.asarray(freq, dtype=np.float64).ravel()
    a = np.abs(np.asarray(af, dtype=np.float64).ravel())
    n = min(f.size, a.size)
    f, a = f[:n], a[:n]
    lo, hi = log_f0_window(f0_calc, rel=rel)
    if not np.isfinite(lo):
        return float("nan"), float("nan")
    mask = (f >= lo) & (f <= hi) & np.isfinite(a)
    if not np.any(mask) or float(np.nanmax(a[mask])) <= _EPS:
        return float("nan"), float("nan")
    f_w = f[mask]
    a_w = a[mask]
    peaks, _props = find_peaks(a_w, prominence=0.0)
    if peaks.size == 0:
        i = int(np.argmax(a_w))
    else:
        i = int(peaks[np.argmax(a_w[peaks])])
    return float(f_w[i]), float(a_w[i])


def f0_2d_from_tf(
    freq: np.ndarray,
    tf_2d: np.ndarray,
    f0_calc: np.ndarray,
    *,
    recorder: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth f0: max 2D |TF| peak in the 1/(4T) log window."""
    y = np.asarray(tf_2d, dtype=np.float64)
    n = y.shape[0]
    f0c = np.asarray(f0_calc, dtype=np.float64).ravel()
    if f0c.size < n:
        f0c = np.pad(f0c, (0, n - f0c.size), constant_values=np.nan)
    f0_true = np.full(n, np.nan)
    amp = np.full(n, np.nan)
    for i in range(n):
        ref = y[i, recorder] if y.ndim == 3 else y[i]
        f0_true[i], amp[i] = max_peak_near_f0_calc(freq, ref, float(f0c[i]))
    return f0_true, amp


def peak_dlnA_df0_per_sample(
    freq: np.ndarray,
    tf_true: np.ndarray,
    tf_pred: np.ndarray,
    *,
    recorder: int = 10,
    f0_calc: np.ndarray | None = None,
    f0_2d: np.ndarray | None = None,
    a_2d: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ΔlnA / Δf0 vs 2D-TF peak ground truth (not vs 1/(4T)).

    Search both curves with the same ``max_peak_near_f0_calc`` (scipy
    find_peaks, log ±20% of f0_calc). Ground truth is the tallest 2D peak;
    f_hat is the tallest predicted peak in that same window.

    Returns ``(dlnA, |f_hat-f0_2d|, |f0_2d-f0_calc|, |f_hat-f0_calc|)``.
    """
    y = np.asarray(tf_true, dtype=np.float64)
    p = np.asarray(tf_pred, dtype=np.float64)
    n = y.shape[0]
    f0c = (
        np.asarray(f0_calc, dtype=np.float64).ravel()
        if f0_calc is not None
        else np.full(n, np.nan)
    )
    if f0c.size < n:
        f0c = np.pad(f0c, (0, n - f0c.size), constant_values=np.nan)
    if f0_2d is None or a_2d is None:
        f0_true, a_true = f0_2d_from_tf(freq, y, f0c, recorder=recorder)
    else:
        f0_true = np.asarray(f0_2d, dtype=np.float64).ravel()
        a_true = np.asarray(a_2d, dtype=np.float64).ravel()
        if f0_true.size < n:
            f0_true = np.pad(f0_true, (0, n - f0_true.size), constant_values=np.nan)
        if a_true.size < n:
            a_true = np.pad(a_true, (0, n - a_true.size), constant_values=np.nan)
    dln = np.full(n, np.nan)
    df0 = np.full(n, np.nan)
    df0_true_vs_calc = np.full(n, np.nan)
    df0_hat_vs_calc = np.full(n, np.nan)
    for i in range(n):
        cand = p[i, recorder] if p.ndim == 3 else p[i]
        f_hat, a_hat = max_peak_near_f0_calc(freq, cand, float(f0c[i]))
        f_ref = float(f0_true[i])
        a_ref = float(a_true[i])
        if np.isfinite(a_hat) and np.isfinite(a_ref):
            dln[i] = abs(np.log(max(a_hat, _EPS)) - np.log(max(a_ref, _EPS)))
        if np.isfinite(f_hat) and np.isfinite(f_ref):
            df0[i] = abs(f_hat - f_ref)
        if np.isfinite(f_ref) and np.isfinite(f0c[i]):
            df0_true_vs_calc[i] = abs(f_ref - float(f0c[i]))
        if np.isfinite(f_hat) and np.isfinite(f0c[i]):
            df0_hat_vs_calc[i] = abs(f_hat - float(f0c[i]))
    return dln, df0, df0_true_vs_calc, df0_hat_vs_calc


def flat_r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def flat_rel_l2(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.linalg.norm(y - p) / max(np.linalg.norm(y), 1e-12))


def flat_pearson(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.float64).ravel()
    p = p.astype(np.float64).ravel()
    if y.size < 2 or y.std() < 1e-12 or p.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def pearson_across_freq(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_rec: int,
    n_freq: int,
) -> float:
    """Mean Pearson correlation of spectra along frequency for each (sample, recorder)."""
    y = y.astype(np.float64).ravel()
    p = p.astype(np.float64).ravel()
    q = n_rec * n_freq
    if y.size % q != 0:
        # fallback: global pearson if layout unexpected
        return flat_pearson(y, p)
    n_s = y.size // q
    Y = y.reshape(n_s, n_rec, n_freq)
    P = p.reshape(n_s, n_rec, n_freq)
    cors: list[float] = []
    for i in range(n_s):
        for r in range(n_rec):
            a, b = Y[i, r], P[i, r]
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            cors.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(cors)) if cors else 0.0


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    out = np.zeros(a.shape[0], dtype=np.float64)
    ok = den > _EPS
    out[ok] = num[ok] / den[ok]
    return out


def logspec_rel_l2_rows(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    lp = np.log(np.maximum(np.abs(pred), _EPS))
    lt = np.log(np.maximum(np.abs(true), _EPS))
    return rel_l2_rows(lp, lt)


def _band_slice(
    pred: np.ndarray,
    true: np.ndarray,
    freq: np.ndarray,
    n_rec: int,
    n_freq: int,
    band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    f = np.asarray(freq, dtype=np.float64).ravel()
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        return None
    p = np.asarray(pred, dtype=np.float64).reshape(-1, n_rec, n_freq)[..., mask]
    t = np.asarray(true, dtype=np.float64).reshape(-1, n_rec, n_freq)[..., mask]
    return p.reshape(p.shape[0], -1), t.reshape(t.shape[0], -1)


def band_rel_l2_mean(
    pred: np.ndarray,
    true: np.ndarray,
    freq: np.ndarray,
    n_rec: int,
    n_freq: int,
    band: tuple[float, float],
) -> float:
    sl = _band_slice(pred, true, freq, n_rec, n_freq, band)
    if sl is None:
        return float("nan")
    p, t = sl
    return float(np.mean(rel_l2_rows(p, t)))


def band_rel_l1_mean(
    pred: np.ndarray,
    true: np.ndarray,
    freq: np.ndarray,
    n_rec: int,
    n_freq: int,
    band: tuple[float, float],
) -> float:
    sl = _band_slice(pred, true, freq, n_rec, n_freq, band)
    if sl is None:
        return float("nan")
    p, t = sl
    return float(np.mean(rel_l1_rows(p, t)))


def score_leftover_batch(
    *,
    tf1d: np.ndarray,
    r_true: np.ndarray,
    r_hat: np.ndarray,
    tf2d: np.ndarray,
    freq: np.ndarray | None = None,
    n_rec: int = 21,
    n_freq: int | None = None,
    fail_soft_q: float = 0.2,
    leftover_tau: float = 0.15,
    gate: np.ndarray | None = None,
    log_residual: bool = False,
    tf_hat: np.ndarray | None = None,
    f0_calc: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-batch leftover diagnostics. Arrays are (n_s, n_rec*n_freq) or flat."""
    if n_freq is None:
        n_freq = int(np.asarray(freq).ravel().size) if freq is not None else 1000
    tf1d = _as_samples(tf1d, n_rec, n_freq)
    r_true = _as_samples(r_true, n_rec, n_freq)
    r_hat = _as_samples(r_hat, n_rec, n_freq)
    tf2d = _as_samples(tf2d, n_rec, n_freq)
    if tf_hat is None:
        tf_hat = tf_from_residual(tf1d, r_hat, log_residual=log_residual)
    else:
        tf_hat = _as_samples(tf_hat, n_rec, n_freq)
    n_s = tf1d.shape[0]
    harmed = harm_mask(tf1d, r_hat, tf2d, log_residual=log_residual, tf_hat=tf_hat)
    lev = leftover_rel_l2(r_true, tf1d)
    q = float(np.quantile(lev, fail_soft_q)) if n_s else 0.0
    small = lev <= q if n_s else np.zeros(0, dtype=bool)
    rhat_lin = np.mean(np.abs(tf_hat - tf1d), axis=-1)
    rel_hat = rel_l2_rows(tf_hat, tf2d) if n_s else np.zeros(0)
    rel_1d = rel_l2_rows(tf1d, tf2d) if n_s else np.zeros(0)
    rel_l1_hat = rel_l1_rows(tf_hat, tf2d) if n_s else np.zeros(0)
    rel_l1_1d = rel_l1_rows(tf1d, tf2d) if n_s else np.zeros(0)
    r_hat_tv = residual_tv_rows(r_hat) if n_s else np.zeros(0)
    r_true_tv = residual_tv_rows(r_true) if n_s else np.zeros(0)
    out: dict[str, Any] = {
        "n": int(n_s),
        "harm_rate": float(np.mean(harmed)) if n_s else 0.0,
        "n_harmed": int(np.sum(harmed)),
        "rel_l2_tf_hat": float(np.mean(rel_hat)) if n_s else 0.0,
        "rel_l2_tf_1d": float(np.mean(rel_1d)) if n_s else 0.0,
        "rel_l1_tf_hat": float(np.mean(rel_l1_hat)) if n_s else 0.0,
        "rel_l1_tf_1d": float(np.mean(rel_l1_1d)) if n_s else 0.0,
        "residual_tv_hat": float(np.mean(r_hat_tv)) if n_s else 0.0,
        "residual_tv_true": float(np.mean(r_true_tv)) if n_s else 0.0,
        "logspec_rel_l2_tf_hat": float(np.mean(logspec_rel_l2_rows(tf_hat, tf2d)))
        if n_s
        else 0.0,
        "logspec_rel_l2_tf_1d": float(np.mean(logspec_rel_l2_rows(tf1d, tf2d)))
        if n_s
        else 0.0,
        "fail_soft_q": float(fail_soft_q),
        "fail_soft_leftover_q": float(q),
        "fail_soft_n": int(np.sum(small)),
        "fail_soft_mean_abs_rhat": float(np.mean(rhat_lin[small]))
        if np.any(small)
        else 0.0,
        "mean_abs_rhat": float(np.mean(rhat_lin)) if n_s else 0.0,
        "frac_small_leftover": float(np.mean(lev < leftover_tau)) if n_s else 0.0,
        "rel_l2_tf_hat_per_sample": rel_hat.tolist() if n_s else [],
        "rel_l2_tf_1d_per_sample": rel_1d.tolist() if n_s else [],
        "rel_l1_tf_hat_per_sample": rel_l1_hat.tolist() if n_s else [],
        "rel_l1_tf_1d_per_sample": rel_l1_1d.tolist() if n_s else [],
    }
    if n_s:
        q20 = float(np.quantile(lev, 0.2))
        q80 = float(np.quantile(lev, 0.8))
        lo = lev <= q20
        mid = (lev > q20) & (lev <= q80)
        hi = lev > q80
        out["harm_rate_small_leftover"] = float(np.mean(harmed[lo])) if np.any(lo) else 0.0
        out["harm_rate_mid_leftover"] = float(np.mean(harmed[mid])) if np.any(mid) else 0.0
        out["harm_rate_large_leftover"] = float(np.mean(harmed[hi])) if np.any(hi) else 0.0
    if freq is not None and n_s:
        f = np.asarray(freq, dtype=np.float64).ravel()
        f0c = (
            np.asarray(f0_calc, dtype=np.float64).ravel()
            if f0_calc is not None
            else np.zeros(0)
        )
        if f0c.size == n_s:
            rec = int(n_rec) // 2
            tf_hat_3 = tf_hat.reshape(n_s, n_rec, n_freq)
            tf2d_3 = tf2d.reshape(n_s, n_rec, n_freq)
            tf1d_3 = tf1d.reshape(n_s, n_rec, n_freq)
            dln, df0, _, _ = peak_dlnA_df0_per_sample(
                f, tf2d_3, tf_hat_3, recorder=rec, f0_calc=f0c
            )
            dln_1d, _, _, _ = peak_dlnA_df0_per_sample(
                f, tf2d_3, tf1d_3, recorder=rec, f0_calc=f0c
            )
            out["peak_dlnA_vs_2d_mean"] = float(np.nanmean(dln))
            out["peak_dlnA_1d_vs_2d_mean"] = float(np.nanmean(dln_1d))
            out["peak_df0_hz_vs_2d_mean"] = float(np.nanmean(df0))
            out["peak_shift_hz_vs_2d_mean"] = out["peak_df0_hz_vs_2d_mean"]
        for name, band in (
            ("low", FREQ_BAND_LOW),
            ("mid", FREQ_BAND_MID),
            ("high", FREQ_BAND_HIGH),
        ):
            out[f"rel_l2_band_{name}_hat"] = band_rel_l2_mean(
                tf_hat, tf2d, f, n_rec, n_freq, band
            )
            out[f"rel_l2_band_{name}_1d"] = band_rel_l2_mean(
                tf1d, tf2d, f, n_rec, n_freq, band
            )
            out[f"rel_l1_band_{name}_hat"] = band_rel_l1_mean(
                tf_hat, tf2d, f, n_rec, n_freq, band
            )
            out[f"rel_l1_band_{name}_1d"] = band_rel_l1_mean(
                tf1d, tf2d, f, n_rec, n_freq, band
            )
    if gate is not None and n_s:
        g = np.asarray(gate, dtype=np.float64)
        if g.ndim > 1:
            g = g.reshape(n_s, -1).mean(axis=-1)
        else:
            g = g.reshape(n_s)
        out["gate_mean"] = float(np.mean(g))
        out["gate_mean_small_leftover"] = (
            float(np.mean(g[small])) if np.any(small) else 0.0
        )
    return out


def win_rate(rel_a: list[float] | np.ndarray, rel_b: list[float] | np.ndarray) -> float:
    a = np.asarray(rel_a, dtype=np.float64).ravel()
    b = np.asarray(rel_b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    return float(np.mean(a[:n] < b[:n]))
