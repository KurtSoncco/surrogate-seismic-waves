"""Transfer-function metrics matching seiskit Response_Variability."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12
FREQ_BANDS: dict[str, tuple[float, float]] = {
    "low": (0.1, 0.5),
    "mid": (0.5, 2.0),
    "high": (2.0, 10.0),
    "all": (0.1, 10.0),
}


def theoretical_f0(vs1: float, H: float) -> float:
    """Quarter-wave site frequency f0 = Vs1 / (4 H)."""
    if H <= 0 or vs1 <= 0:
        return float("nan")
    return float(vs1) / (4.0 * float(H))


def peak_af(
    freq: np.ndarray,
    af: np.ndarray,
    *,
    fmin: float = 0.1,
    fmax: float = 10.0,
) -> tuple[float, float]:
    """Peak |TF| within the comparison band (default 0.1–10 Hz)."""
    freq = np.asarray(freq, dtype=float).ravel()
    af = np.asarray(af, dtype=float).ravel()
    mask = (freq >= fmin) & (freq <= fmax) & np.isfinite(af)
    if not np.any(mask):
        if not np.any(np.isfinite(af)):
            return float("nan"), float("nan")
        i = int(np.nanargmax(af))
        return float(freq[i]), float(af[i])
    i = int(np.argmax(af[mask]))
    return float(freq[mask][i]), float(af[mask][i])


def anderson_frequency_domain(
    freq: np.ndarray,
    ref_af: np.ndarray,
    cand_af: np.ndarray,
    *,
    f_weight_center: float | None = None,
    f_weight_width: float = 1.5,
) -> float:
    """Weighted L1 norm of ln(|TF|) residuals (Anderson-style; lower is better)."""
    f = np.asarray(freq, dtype=float).ravel()
    r = np.log(np.clip(np.asarray(ref_af, float).ravel(), _EPS, None))
    c = np.log(np.clip(np.asarray(cand_af, float).ravel(), _EPS, None))
    n = min(len(f), len(r), len(c))
    f, r, c = f[:n], r[:n], c[:n]
    if f_weight_center is not None:
        w = np.exp(-0.5 * ((f - f_weight_center) / max(f_weight_width, 1e-6)) ** 2)
    else:
        w = np.ones_like(f)
    w = w / max(np.sum(w), _EPS)
    return float(np.sum(w * np.abs(r - c)))


def log_residual_bias(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean ln(candidate / reference) for positive arrays."""
    r = np.asarray(reference, dtype=float).ravel()
    c = np.asarray(candidate, dtype=float).ravel()
    mask = (r > 0) & (c > 0) & np.isfinite(r) & np.isfinite(c)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.log(c[mask] / r[mask])))


def sigma_ln(values: np.ndarray) -> float:
    """Sample standard deviation of ln(x) over strictly positive finite values."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 2:
        return 0.0
    return float(np.std(np.log(x), ddof=1))


def spatial_sigma_ln(af_spatial: np.ndarray) -> np.ndarray:
    """σ_ln across recorders at each frequency. ``af_spatial`` is (n_rec, n_freq)."""
    stack = np.asarray(af_spatial, dtype=float)
    if stack.ndim != 2:
        raise ValueError(f"expected (n_rec, n_freq), got {stack.shape}")
    return np.array([sigma_ln(stack[:, j]) for j in range(stack.shape[1])])


def spatial_percentiles(
    af_spatial: np.ndarray, *, q: tuple[float, float, float] = (16.0, 50.0, 84.0)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(p16, median, p84) along the recorder axis."""
    stack = np.asarray(af_spatial, dtype=float)
    p16, med, p84 = np.nanpercentile(stack, q, axis=0)
    return (
        np.asarray(p16, dtype=float),
        np.asarray(med, dtype=float),
        np.asarray(p84, dtype=float),
    )


def rel_l2(pred: np.ndarray, true: np.ndarray, *, mask: np.ndarray | None = None) -> float:
    p = np.asarray(pred, dtype=float).ravel()
    t = np.asarray(true, dtype=float).ravel()
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        p, t = p[m], t[m]
    finite = np.isfinite(p) & np.isfinite(t)
    p, t = p[finite], t[finite]
    den = float(np.linalg.norm(t))
    if den < _EPS:
        return float("nan")
    return float(np.linalg.norm(p - t) / den)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def band_mask(freq: np.ndarray, lo: float, hi: float) -> np.ndarray:
    f = np.asarray(freq, dtype=float).ravel()
    return (f >= lo) & (f <= hi)


def band_rel_l2(
    pred: np.ndarray,
    true: np.ndarray,
    freq: np.ndarray,
    *,
    lo: float,
    hi: float,
) -> float:
    """Relative L2 on a frequency band; ``pred``/``true`` are (n_freq,) or (n_rec, n_freq)."""
    f = np.asarray(freq, dtype=float).ravel()
    m = band_mask(f, lo, hi)
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    if p.ndim == 1:
        return rel_l2(p, t, mask=m)
    return rel_l2(p[:, m], t[:, m])


def method_vs_reference(
    *,
    freq: np.ndarray,
    af_ref: np.ndarray,
    af_cand: np.ndarray,
    af_ref_spatial: np.ndarray | None = None,
    af_cand_spatial: np.ndarray | None = None,
) -> dict[str, float]:
    """Peak, GOF, bias, and optional spatial σ_ln vs a reference spectrum."""
    f_ref, a_ref = peak_af(freq, af_ref)
    f_c, a_c = peak_af(freq, af_cand)
    out = {
        "f_peak": f_c,
        "A_peak": a_c,
        "delta_f_peak": f_c - f_ref,
        "delta_ln_A_peak": float(np.log(max(a_c, _EPS) / max(a_ref, _EPS))),
        "delta_mu_ln_af": log_residual_bias(af_ref, af_cand),
        "gof_af": anderson_frequency_domain(
            freq, af_ref, af_cand, f_weight_center=f_ref, f_weight_width=1.5
        ),
        "rel_l2": rel_l2(af_cand, af_ref),
        "pearson": pearson(af_cand, af_ref),
        "ref_f_peak": f_ref,
        "ref_A_peak": a_ref,
    }
    if af_ref_spatial is not None and af_cand_spatial is not None:
        sig_r = spatial_sigma_ln(af_ref_spatial)
        sig_c = spatial_sigma_ln(af_cand_spatial)
        out["sigma_ln_spatial_mean"] = float(np.mean(sig_c))
        out["delta_sigma_ln_spatial_mean"] = float(np.mean(sig_c - sig_r))
        out["rel_l2_spatial"] = rel_l2(af_cand_spatial, af_ref_spatial)
    return out
