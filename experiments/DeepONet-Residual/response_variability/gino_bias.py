"""Wave 0 inductive-bias diagnostics on nested test packs (no checkpoint).

Competence is reported against a *scalar-optimal* 4D cell-mean yardstick, not
an irreducible floor: GINO is field-conditioned, so Var(TF | θ_4D) is what a
scalar predictor cannot beat. R²_scalar ≈ 1 is ambiguous until collapse_ratio
says whether the field was used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.fft import dct

from response_variability.metrics import (
    band_mask,
    peak_af,
    rel_l2,
    spatial_sigma_ln,
    theoretical_f0,
)

_EPS = 1e-12
TROUGH_FRAC = 0.05
N_BOOT = 400
LATENT_DIM = 128
FNO_FREQ_MODES = 16
FNO_SPATIAL_MODES = 8
ENERGY_RANK = 0.99
NEIGHBOR_K = 8


def _as_3d(tf: np.ndarray) -> np.ndarray:
    x = np.asarray(tf, dtype=np.float64)
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim != 3:
        raise ValueError(f"expected (n, n_rec, n_freq) or (n, n_freq), got {x.shape}")
    return x


def central_slice(tf: np.ndarray) -> np.ndarray:
    x = _as_3d(tf)
    return x[:, x.shape[1] // 2, :]


def broadcast_tf(tf1d: np.ndarray, like: np.ndarray) -> np.ndarray:
    y = _as_3d(like)
    b = np.asarray(tf1d, dtype=np.float64)
    if b.ndim == 2:
        b = b[:, None, :]
    if b.ndim != 3:
        raise ValueError(f"tf1d shape {b.shape}")
    if b.shape[1] == 1 and y.shape[1] > 1:
        b = np.broadcast_to(b, y.shape)
    elif b.shape != y.shape:
        b = np.broadcast_to(b, y.shape)
    return b


def leftover(tf: np.ndarray, tf1d: np.ndarray) -> np.ndarray:
    y = _as_3d(tf)
    return y - broadcast_tf(tf1d, y)


def trough_keep_mask(af: np.ndarray, *, floor_frac: float = TROUGH_FRAC) -> np.ndarray:
    """Keep bins at least ``floor_frac`` of that sample's peak |TF|."""
    a = np.asarray(af, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    peak = np.nanmax(np.abs(a), axis=1, keepdims=True)
    peak = np.maximum(peak, _EPS)
    return np.abs(a) >= (float(floor_frac) * peak)


def ols_ab(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = int(x.size)
    if n < 4:
        return {"a": float("nan"), "b": float("nan"), "r2": float("nan"), "n": float(n)}
    xc = x - x.mean()
    yc = y - y.mean()
    den = float(np.dot(xc, xc))
    if den < _EPS:
        return {"a": float(y.mean()), "b": float("nan"), "r2": float("nan"), "n": float(n)}
    b = float(np.dot(xc, yc) / den)
    a = float(y.mean() - b * x.mean())
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum(yc**2))
    r2 = float("nan") if ss_tot < _EPS else 1.0 - ss_res / ss_tot
    return {"a": a, "b": b, "r2": r2, "n": float(n)}


def _masked_pairs(
    r_hat: np.ndarray,
    r: np.ndarray,
    keep: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    yh = np.asarray(r_hat, dtype=np.float64)
    yt = np.asarray(r, dtype=np.float64)
    m = np.asarray(keep, dtype=bool)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in range(yh.shape[0]):
        mi = m[i]
        xi = yt[i, mi]
        yi = yh[i, mi]
        finite = np.isfinite(xi) & np.isfinite(yi)
        xs.append(xi[finite])
        ys.append(yi[finite])
    return xs, ys


def cluster_bootstrap_slope(
    x_per_sample: list[np.ndarray],
    y_per_sample: list[np.ndarray],
    *,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict[str, float]:
    n = len(x_per_sample)
    if n == 0:
        return {"b_lo": float("nan"), "b_hi": float("nan"), "n_boot": 0.0}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for t in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x = np.concatenate([x_per_sample[i] for i in idx])
        y = np.concatenate([y_per_sample[i] for i in idx])
        boots[t] = ols_ab(x, y)["b"]
    finite = np.isfinite(boots)
    if not np.any(finite):
        return {"b_lo": float("nan"), "b_hi": float("nan"), "n_boot": float(n_boot)}
    lo, hi = np.quantile(boots[finite], [0.025, 0.975])
    return {"b_lo": float(lo), "b_hi": float(hi), "n_boot": float(n_boot)}


def rhat_on_r_slope(
    tf_gino: np.ndarray,
    tf_ops: np.ndarray,
    tf_1d: np.ndarray,
    freq: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict[str, Any]:
    """OLS R̂ = a + b R on the central recorder (trough-windowed)."""
    r_hat = leftover(tf_gino, tf_1d)
    r = leftover(tf_ops, tf_1d)
    rh = central_slice(r_hat)
    rt = central_slice(r)
    ops_c = central_slice(tf_ops)
    fmask = band_mask(freq)
    keep = trough_keep_mask(ops_c) & fmask[None, :]
    xs, ys = _masked_pairs(rh, rt, keep)
    x_all = np.concatenate(xs) if xs else np.array([])
    y_all = np.concatenate(ys) if ys else np.array([])
    pooled = ols_ab(x_all, y_all)
    boot = cluster_bootstrap_slope(xs, ys, n_boot=n_boot, seed=seed)
    per = np.array([ols_ab(x, y)["b"] for x, y in zip(xs, ys)], dtype=np.float64)
    finite = np.isfinite(per)
    return {
        "a": pooled["a"],
        "b": pooled["b"],
        "r2": pooled["r2"],
        "n_pairs": pooled["n"],
        "n_samples": float(len(xs)),
        "b_lo": boot["b_lo"],
        "b_hi": boot["b_hi"],
        "n_boot": boot["n_boot"],
        "b_median": float(np.nanmedian(per)) if np.any(finite) else float("nan"),
        "b_p16": float(np.nanpercentile(per[finite], 16)) if np.any(finite) else float("nan"),
        "b_p84": float(np.nanpercentile(per[finite], 84)) if np.any(finite) else float("nan"),
        "reading": _slope_reading(pooled["b"], boot["b_lo"], boot["b_hi"]),
        "per_sample_b": per,
    }


def _slope_reading(b: float, lo: float, hi: float) -> str:
    if not np.isfinite(b):
        return "undefined"
    if np.isfinite(hi) and hi < 0.9:
        return "under_correction"
    if np.isfinite(lo) and lo > 1.1:
        return "over_correction"
    if np.isfinite(lo) and lo <= 1.0 <= (hi if np.isfinite(hi) else 1.0):
        return "consistent_with_one"
    if b < 0.9:
        return "under_correction"
    if b > 1.1:
        return "over_correction"
    return "near_one"


def mean_leftover(tf_ops: np.ndarray, tf_1d: np.ndarray) -> dict[str, float]:
    r = leftover(tf_ops, tf_1d)
    rc = central_slice(r)
    return {
        "mean_R_central": float(np.nanmean(rc)),
        "mean_abs_R_central": float(np.nanmean(np.abs(rc))),
        "rms_R_over_rms_1d": float(
            np.linalg.norm(rc) / max(np.linalg.norm(central_slice(tf_1d)), _EPS)
        ),
    }


def per_recorder_rel_l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = _as_3d(pred)
    t = _as_3d(true)
    n, n_rec, _ = t.shape
    out = np.full((n, n_rec), np.nan, dtype=np.float64)
    for i in range(n):
        for r in range(n_rec):
            out[i, r] = rel_l2(p[i, r], t[i, r])
    return out


def wraparound_report(tf_gino: np.ndarray, tf_ops: np.ndarray) -> dict[str, float]:
    rec = per_recorder_rel_l2(tf_gino, tf_ops)
    mean_rec = np.nanmean(rec, axis=0)
    n_rec = mean_rec.size
    edge = float(np.nanmean(mean_rec[[0, n_rec - 1]]))
    mid0, mid1 = max(n_rec // 2 - 1, 0), min(n_rec // 2 + 2, n_rec)
    center = float(np.nanmean(mean_rec[mid0:mid1]))
    resid = _as_3d(tf_gino) - _as_3d(tf_ops)
    e0 = resid[:, 0, :].ravel()
    eN = resid[:, n_rec - 1, :].ravel()
    e1 = resid[:, 1, :].ravel()
    return {
        "n_rec": float(n_rec),
        "rel_l2_edge": edge,
        "rel_l2_center": center,
        "edge_minus_center": edge - center,
        "corr_ends": _safe_corr(e0, eN),
        "corr_neighbors": _safe_corr(e0, e1),
        "ends_minus_neighbors": _safe_corr(e0, eN) - _safe_corr(e0, e1),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 8 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def effective_rank(matrix: np.ndarray, *, energy: float = ENERGY_RANK) -> dict[str, float]:
    x = np.asarray(matrix, dtype=np.float64)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = np.nan_to_num(x, nan=0.0)
    if x.size == 0 or min(x.shape) < 2:
        return {"rank_energy": float("nan"), "n_modes": 0.0, "n_samples": float(x.shape[0])}
    s = np.linalg.svd(x, compute_uv=False)
    energy_s = s**2
    tot = float(np.sum(energy_s))
    if tot < _EPS:
        return {"rank_energy": 0.0, "n_modes": float(s.size), "n_samples": float(x.shape[0])}
    cdf = np.cumsum(energy_s) / tot
    k = int(np.searchsorted(cdf, energy) + 1)
    return {
        "rank_energy": float(k),
        "n_modes": float(s.size),
        "n_samples": float(x.shape[0]),
        "energy": float(energy),
        "frac_at_latent": float(cdf[min(LATENT_DIM, s.size) - 1]) if s.size else float("nan"),
    }


def svd_targets(
    tf_ops: np.ndarray,
    tf_1d: np.ndarray,
    *,
    latent_dim: int = LATENT_DIM,
) -> dict[str, Any]:
    ops_c = central_slice(tf_ops)
    r_c = central_slice(leftover(tf_ops, tf_1d))
    tf_rank = effective_rank(ops_c)
    r_rank = effective_rank(r_c)
    return {
        "latent_dim": float(latent_dim),
        "tf_rank_99": tf_rank["rank_energy"],
        "r_rank_99": r_rank["rank_energy"],
        "tf_energy_at_p": tf_rank["frac_at_latent"],
        "r_energy_at_p": r_rank["frac_at_latent"],
        "tf_ceiling": bool(tf_rank["rank_energy"] > latent_dim + 1e-9),
        "r_ceiling": bool(r_rank["rank_energy"] > latent_dim + 1e-9),
    }


def param_4d(pack: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(pack["vs1"], dtype=float),
            np.asarray(pack["H"], dtype=float),
            np.asarray(pack["cov"], dtype=float),
            np.asarray(pack["vs2"], dtype=float),
        ]
    )


def cell_ids_4d(x4: np.ndarray, *, decimals: int = 8) -> np.ndarray:
    rows = np.round(np.asarray(x4, dtype=float), decimals)
    _, inv = np.unique(rows, axis=0, return_inverse=True)
    return inv.astype(int)


def collapse_and_r2_scalar(
    tf_gino: np.ndarray,
    tf_ops: np.ndarray,
    x4: np.ndarray,
    freq: np.ndarray,
) -> dict[str, Any]:
    """Within-cell prediction collapse vs OpenSees, and R² vs scalar cell mean.

    Uses log|TF| on the central recorder in 0.1–10 Hz. Neighborhood-pooled
    variance for singleton cells is an *upper bound* on the scalar-optimal MSE.
    """
    g = np.log(np.clip(np.abs(central_slice(tf_gino)), _EPS, None))
    o = np.log(np.clip(np.abs(central_slice(tf_ops)), _EPS, None))
    m = band_mask(freq)
    g, o = g[:, m], o[:, m]
    ids = cell_ids_4d(x4)
    exact = _cell_stats(g, o, ids)
    z = (x4 - x4.mean(axis=0)) / np.clip(x4.std(axis=0), 1e-8, None)
    neigh = _neighborhood_r2(g, o, z, k=NEIGHBOR_K)
    return {
        "n_unique_4d": float(len(np.unique(ids))),
        "n_exact_cells_ge2": exact["n_cells_ge2"],
        "n_exact_replicates": exact["n_replicates"],
        "r2_scalar_exact": exact["r2"],
        "collapse_ratio": exact["collapse"],
        "mse_gino_exact": exact["mse_gino"],
        "var_ops_exact": exact["var_ops"],
        "var_gino_exact": exact["var_gino"],
        "r2_scalar_neighborhood_ub": neigh["r2"],
        "note": (
            "R2_scalar = 1 - MSE(GINO)/Var(TF|θ_4D). Equals 1 if the field is "
            "read (MSE≈0); equals 0 if GINO matches the scalar cell mean. "
            "collapse_ratio << 1 with R2≈0 is field-blind pooling, not a ceiling."
        ),
    }


def _cell_stats(g: np.ndarray, o: np.ndarray, ids: np.ndarray) -> dict[str, float]:
    mse_g = []
    var_o = []
    var_g = []
    n_rep = 0
    n_cells = 0
    for cid in np.unique(ids):
        sel = ids == cid
        n = int(sel.sum())
        if n < 2:
            continue
        n_cells += 1
        n_rep += n
        oc, gc = o[sel], g[sel]
        mu = oc.mean(axis=0)
        var_o.append(float(np.mean((oc - mu) ** 2)))
        var_g.append(float(np.mean((gc - gc.mean(axis=0)) ** 2)))
        mse_g.append(float(np.mean((gc - oc) ** 2)))
    if not mse_g:
        return {
            "n_cells_ge2": 0.0,
            "n_replicates": 0.0,
            "r2": float("nan"),
            "collapse": float("nan"),
            "mse_gino": float("nan"),
            "var_ops": float("nan"),
            "var_gino": float("nan"),
        }
    mse = float(np.mean(mse_g))
    vo = float(np.mean(var_o))
    vg = float(np.mean(var_g))
    r2 = float("nan") if vo < _EPS else 1.0 - mse / vo
    collapse = float("nan") if vo < _EPS else vg / vo
    return {
        "n_cells_ge2": float(n_cells),
        "n_replicates": float(n_rep),
        "r2": r2,
        "collapse": collapse,
        "mse_gino": mse,
        "var_ops": vo,
        "var_gino": vg,
    }


def _neighborhood_r2(
    g: np.ndarray, o: np.ndarray, z: np.ndarray, *, k: int
) -> dict[str, float]:
    n = z.shape[0]
    kk = min(max(int(k), 2), n)
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    mse = []
    var = []
    for i in range(n):
        nb = np.argpartition(d[i], kk - 1)[:kk]
        oc = o[nb]
        mu = oc.mean(axis=0)
        var.append(float(np.mean((oc - mu) ** 2)))
        mse.append(float(np.mean((g[i] - o[i]) ** 2)))
    vo = float(np.mean(var))
    mse_m = float(np.mean(mse))
    r2 = float("nan") if vo < _EPS else 1.0 - mse_m / vo
    return {"r2": r2, "var_ub": vo, "mse_gino": mse_m}


def quartile_bins(
    values: np.ndarray,
    metric: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> list[dict[str, float]]:
    v = np.asarray(values, dtype=float)
    m = np.asarray(metric, dtype=float)
    finite = np.isfinite(v) & np.isfinite(m)
    v, m = v[finite], m[finite]
    if v.size < 8:
        return []
    edges = np.quantile(v, [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(4):
        sel = (v > edges[q]) & (v <= edges[q + 1])
        mm = m[sel]
        n = int(mm.size)
        if n == 0:
            continue
        lo, hi = _boot_mean_ci(mm, rng, n_boot)
        rows.append(
            {
                "quartile": float(q + 1),
                "n": float(n),
                "lo": float(edges[q]),
                "hi": float(edges[q + 1]),
                "mean": float(np.mean(mm)),
                "median": float(np.median(mm)),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return rows


def _boot_mean_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    n = x.size
    if n < 2:
        return float("nan"), float("nan")
    boots = np.empty(n_boot)
    for t in range(n_boot):
        boots[t] = float(np.mean(x[rng.integers(0, n, size=n)]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(lo), float(hi)


def trough_safe_log_bias(tf_gino: np.ndarray, tf_ops: np.ndarray) -> np.ndarray:
    g = central_slice(tf_gino)
    o = central_slice(tf_ops)
    keep = trough_keep_mask(o)
    out = np.full(g.shape[0], np.nan, dtype=np.float64)
    for i in range(g.shape[0]):
        gi, oi, mi = g[i], o[i], keep[i]
        sel = mi & (oi > 0) & (gi > 0) & np.isfinite(gi) & np.isfinite(oi)
        if np.count_nonzero(sel) < 4:
            continue
        out[i] = float(np.mean(np.log(gi[sel] / oi[sel])))
    return out


def peak_signed_bias(tf_gino: np.ndarray, tf_ops: np.ndarray, freq: np.ndarray) -> dict[str, np.ndarray]:
    g = central_slice(tf_gino)
    o = central_slice(tf_ops)
    n = g.shape[0]
    df = np.full(n, np.nan)
    dln = np.full(n, np.nan)
    for i in range(n):
        f_r, a_r = peak_af(freq, o[i])
        f_c, a_c = peak_af(freq, g[i])
        df[i] = f_c - f_r
        dln[i] = float(np.log(max(a_c, _EPS) / max(a_r, _EPS)))
    return {"delta_f_peak": df, "delta_ln_A_peak": dln}


def spatial_dispersion(tf_gino: np.ndarray, tf_ops: np.ndarray, cov: np.ndarray) -> dict[str, Any]:
    g = _as_3d(tf_gino)
    o = _as_3d(tf_ops)
    n = g.shape[0]
    sig_g = np.array([float(np.mean(spatial_sigma_ln(g[i]))) for i in range(n)])
    sig_o = np.array([float(np.mean(spatial_sigma_ln(o[i]))) for i in range(n)])
    ratio = sig_g / np.clip(sig_o, _EPS, None)
    c = np.asarray(cov, dtype=float)
    return {
        "sigma_ln_gino_mean": float(np.nanmean(sig_g)),
        "sigma_ln_ops_mean": float(np.nanmean(sig_o)),
        "ratio_mean": float(np.nanmean(ratio)),
        "spearman_ratio_cov": _spearman(c, ratio),
        "spearman_ops_cov": _spearman(c, sig_o),
        "spearman_gino_cov": _spearman(c, sig_g),
        "per_sample_ratio": ratio,
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 8:
        return float("nan")
    ra = np.argsort(np.argsort(a[finite]))
    rb = np.argsort(np.argsort(b[finite]))
    return float(np.corrcoef(ra, rb)[0, 1])


def offline_rebaseline(
    tf_gino: np.ndarray,
    tf_ops: np.ndarray,
    tf_nom: np.ndarray,
    tf_alt: np.ndarray,
) -> dict[str, float]:
    """TF_alt = TF_1D_alt + (GINO − TF_nom). No training."""
    ops = _as_3d(tf_ops)
    r_hat = leftover(tf_gino, tf_nom)
    alt_tf = broadcast_tf(tf_alt, ops)
    nom = broadcast_tf(tf_nom, ops)
    recon = alt_tf + r_hat
    n = ops.shape[0]
    l2_g = np.array([rel_l2(_as_3d(tf_gino)[i], ops[i]) for i in range(n)])
    l2_a = np.array([rel_l2(recon[i], ops[i]) for i in range(n)])
    l2_nom = np.array([rel_l2(nom[i], ops[i]) for i in range(n)])
    l2_alt = np.array([rel_l2(alt_tf[i], ops[i]) for i in range(n)])
    return {
        "rel_l2_gino": float(np.nanmean(l2_g)),
        "rel_l2_alt_recon": float(np.nanmean(l2_a)),
        "delta_rel_l2": float(np.nanmean(l2_a) - np.nanmean(l2_g)),
        "rel_l2_nom_backbone": float(np.nanmean(l2_nom)),
        "rel_l2_alt_backbone": float(np.nanmean(l2_alt)),
        "demand_nom": float(np.nanmean(l2_nom)),
        "demand_alt": float(np.nanmean(l2_alt)),
    }


def dct_transfer_ratio(
    r_hat: np.ndarray,
    r: np.ndarray,
    *,
    cutoff: int = FNO_FREQ_MODES,
) -> dict[str, Any]:
    """Per-k DCT amplitude ratio on central leftover (ortho type-II)."""
    yh = dct(central_slice(r_hat), type=2, norm="ortho", axis=-1)
    yt = dct(central_slice(r), type=2, norm="ortho", axis=-1)
    amp_h = np.mean(np.abs(yh), axis=0)
    amp_t = np.mean(np.abs(yt), axis=0)
    ratio = amp_h / np.clip(amp_t, _EPS, None)
    n_f = ratio.size
    below = ratio[: min(cutoff, n_f)]
    above = ratio[min(cutoff, n_f) :]
    return {
        "cutoff": float(cutoff),
        "n_freq": float(n_f),
        "ratio_mean_below": float(np.mean(below)) if below.size else float("nan"),
        "ratio_mean_above": float(np.mean(above)) if above.size else float("nan"),
        "ratio": ratio,
        "note": (
            "Learned spectral bias lives below cutoff; above-cutoff energy is "
            "architecture (skip/pointwise only). Residual R is high-pass vs TF."
        ),
    }


def per_sample_rel_l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = _as_3d(pred)
    t = _as_3d(true)
    return np.array([rel_l2(p[i], t[i]) for i in range(t.shape[0])], dtype=np.float64)


def f0_impedance(pack: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    vs1 = np.asarray(pack["vs1"], dtype=float)
    H = np.asarray(pack["H"], dtype=float)
    vs2 = np.asarray(pack["vs2"], dtype=float)
    f0 = np.array([theoretical_f0(v, h) for v, h in zip(vs1, H)])
    z = vs2 / np.clip(vs1, _EPS, None)
    return f0, z


def analyze_pack(
    pack: dict[str, np.ndarray],
    *,
    domain: str,
    n_boot: int = N_BOOT,
    seed: int = 42,
    inside_hull: np.ndarray | None = None,
) -> dict[str, Any]:
    ops = pack["tf_opensees"]
    gino = pack["tf_gino"]
    nom = pack["tf_haskell_nominal"]
    freq = np.asarray(pack["freq"], dtype=float)
    x4 = param_4d(pack)
    slope = rhat_on_r_slope(gino, ops, nom, freq, n_boot=n_boot, seed=seed)
    per_b = slope.pop("per_sample_b")
    wrap = wraparound_report(gino, ops)
    svd = svd_targets(ops, nom)
    floor = collapse_and_r2_scalar(gino, ops, x4, freq)
    disp = spatial_dispersion(gino, ops, pack["cov"])
    disp.pop("per_sample_ratio", None)
    peaks = peak_signed_bias(gino, ops, freq)
    logb = trough_safe_log_bias(gino, ops)
    l2 = per_sample_rel_l2(gino, ops)
    from response_variability.covariates import (
        attach_extracted_f0,
        pearson_anderson_per_sample,
        present_covariates,
    )

    pack_f0 = pack if "f0_calc" in pack else attach_extracted_f0(pack)
    pack_f0 = dict(pack_f0)
    f0 = np.asarray(pack_f0["f0"], dtype=float)
    zimp = np.asarray(
        pack_f0.get("impedance", pack["vs2"] / np.clip(pack["vs1"], _EPS, None))
    )
    pack_f0["impedance"] = zimp
    pack_f0["f0"] = f0
    pa = pearson_anderson_per_sample(gino, ops, freq, f0_extracted=f0)
    r_hat = leftover(gino, nom)
    r = leftover(ops, nom)
    tr = dct_transfer_ratio(r_hat, r)
    tr_ratio = tr.pop("ratio")
    alt = None
    if "tf_pretell" in pack:
        alt = offline_rebaseline(gino, ops, nom, pack["tf_pretell"])
    elif "tf_haskell_column" in pack:
        alt = offline_rebaseline(gino, ops, nom, pack["tf_haskell_column"])
    hull = None
    if inside_hull is not None:
        mask = np.asarray(inside_hull, dtype=bool)
        if mask.shape[0] == l2.shape[0] and np.any(mask):
            hull_slope = rhat_on_r_slope(
                gino[mask],
                ops[mask],
                _as_3d(nom)[mask],
                freq,
                n_boot=max(n_boot // 2, 80),
                seed=seed,
            )
            hull_slope.pop("per_sample_b", None)
            hull = {"n": float(int(mask.sum())), "slope": hull_slope}
    bins = {
        "Vs1": quartile_bins(pack["vs1"], l2, n_boot=n_boot, seed=seed),
        "H": quartile_bins(pack["H"], l2, n_boot=n_boot, seed=seed),
        "CoV": quartile_bins(pack["cov"], l2, n_boot=n_boot, seed=seed),
        "Vs2": quartile_bins(pack["vs2"], l2, n_boot=n_boot, seed=seed),
        "f0": quartile_bins(f0, l2, n_boot=n_boot, seed=seed),
        "impedance": quartile_bins(zimp, l2, n_boot=n_boot, seed=seed),
        "CoV_delta_lnA": quartile_bins(
            pack["cov"], peaks["delta_ln_A_peak"], n_boot=n_boot, seed=seed
        ),
        "CoV_logbias": quartile_bins(pack["cov"], logb, n_boot=n_boot, seed=seed),
    }
    quartile_pearson: dict[str, list] = {}
    quartile_gof: dict[str, list] = {}
    for key in present_covariates(pack_f0, domain):
        vals = np.asarray(pack_f0[key], dtype=float)
        quartile_pearson[key] = quartile_bins(
            vals, pa["pearson"], n_boot=n_boot, seed=seed
        )
        quartile_gof[key] = quartile_bins(vals, pa["gof_af"], n_boot=n_boot, seed=seed)
        for band in ("low", "mid", "high", "all"):
            quartile_pearson[f"{key}_{band}"] = quartile_bins(
                vals, pa[f"pearson_{band}"], n_boot=n_boot, seed=seed
            )
            quartile_gof[f"{key}_{band}"] = quartile_bins(
                vals, pa[f"gof_{band}"], n_boot=n_boot, seed=seed
            )
    mean_r = mean_leftover(ops, nom)
    return {
        "domain": domain,
        "n": float(ops.shape[0]),
        "slope": slope,
        "mean_R": mean_r,
        "wraparound": wrap,
        "svd": svd,
        "scalar_floor": floor,
        "spatial": disp,
        "offline_1d": alt,
        "dct": tr,
        "rel_l2_mean": float(np.nanmean(l2)),
        "pearson_mean": float(np.nanmean(pa["pearson"])),
        "gof_af_mean": float(np.nanmean(pa["gof_af"])),
        "delta_ln_A_median": float(np.nanmedian(peaks["delta_ln_A_peak"])),
        "delta_f_peak_median": float(np.nanmedian(peaks["delta_f_peak"])),
        "log_bias_trough_safe_median": float(np.nanmedian(logb)),
        "quartile_rel_l2": bins,
        "quartile_pearson": quartile_pearson,
        "quartile_gof": quartile_gof,
        "in_hull_iid": hull,
        "per_sample": {
            "rel_l2": l2,
            "pearson": pa["pearson"],
            "gof_af": pa["gof_af"],
            "pearson_low": pa["pearson_low"],
            "pearson_mid": pa["pearson_mid"],
            "pearson_high": pa["pearson_high"],
            "pearson_all": pa["pearson_all"],
            "gof_low": pa["gof_low"],
            "gof_mid": pa["gof_mid"],
            "gof_high": pa["gof_high"],
            "gof_all": pa["gof_all"],
            "delta_ln_A_peak": peaks["delta_ln_A_peak"],
            "log_bias_trough_safe": logb,
            "slope_b": per_b,
            "f0": f0,
        },
        "dct_ratio": tr_ratio,
    }


def crossed_f0_cov(
    f0: np.ndarray,
    cov: np.ndarray,
    metric: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Pre-registered 2-way table; cells with tiny n are still printed with n."""
    f0 = np.asarray(f0, dtype=float)
    cov = np.asarray(cov, dtype=float)
    metric = np.asarray(metric, dtype=float)
    finite = np.isfinite(f0) & np.isfinite(cov) & np.isfinite(metric)
    f0, cov, metric = f0[finite], cov[finite], metric[finite]
    if f0.size < 16:
        return []
    fe = np.quantile(f0, [0.0, 0.5, 1.0])
    ce = np.quantile(cov, [0.0, 0.5, 1.0])
    fe[0] -= 1e-12
    ce[0] -= 1e-12
    fe[-1] += 1e-12
    ce[-1] += 1e-12
    rng = np.random.default_rng(seed)
    rows = []
    for i, flab in enumerate(("f0_low", "f0_high")):
        for j, clab in enumerate(("cov_low", "cov_high")):
            sel = (
                (f0 > fe[i])
                & (f0 <= fe[i + 1])
                & (cov > ce[j])
                & (cov <= ce[j + 1])
            )
            mm = metric[sel]
            n = int(mm.size)
            lo, hi = _boot_mean_ci(mm, rng, n_boot) if n else (float("nan"), float("nan"))
            rows.append(
                {
                    "cell": f"{flab}|{clab}",
                    "n": float(n),
                    "mean_rel_l2": float(np.mean(mm)) if n else float("nan"),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    return rows


def phase3_protocol(summary: dict[str, Any]) -> dict[str, Any]:
    """One primary from Wave 0; formulation tests are offline, not smokes."""
    domains = summary.get("domains", {})
    tl = domains.get("three_layer", {})
    iid = domains.get("iid", {})
    reasons: list[str] = []
    primary = "wait_wave1_spectral"
    kind = "future_retrain"

    tl_off = (tl.get("offline_1d") or {}) if tl else {}
    if tl_off and float(tl_off.get("delta_rel_l2", 0.0)) < -0.02:
        primary = "offline_1d_backbone"
        kind = "offline_wave0"
        reasons.append(
            "three-layer rel L2 dropped when R̂ was added to Pretell/column "
            f"({tl_off.get('rel_l2_gino'):.3f} → {tl_off.get('rel_l2_alt_recon'):.3f})"
        )
    else:
        bs = []
        for name, blob in domains.items():
            b = float((blob.get("slope") or {}).get("b", float("nan")))
            bs.append((name, b))
            if np.isfinite(b) and b < 0.9:
                primary = "undercorrection_slope"
                kind = "observational_wave0"
                reasons.append(f"{name} R̂-on-R slope b={b:.3f} < 0.9")
                break
        wrap = float((iid.get("wraparound") or {}).get("edge_minus_center", 0.0))
        if primary == "wait_wave1_spectral" and wrap > 0.02:
            primary = "fno_wraparound"
            kind = "observational_wave0"
            reasons.append(f"IID edge−center rel L2 = {wrap:.3f}")
        coll = float((iid.get("scalar_floor") or {}).get("collapse_ratio", float("nan")))
        if primary == "wait_wave1_spectral" and np.isfinite(coll) and coll < 0.5:
            primary = "field_blind_collapse"
            kind = "future_retrain"
            reasons.append(f"IID collapse_ratio={coll:.3f} << 1 (field discarded)")
        ratio_b = float((iid.get("dct") or {}).get("ratio_mean_below", float("nan")))
        if primary == "wait_wave1_spectral" and np.isfinite(ratio_b) and ratio_b < 0.85:
            primary = "spectral_below_cutoff"
            kind = "future_retrain"
            reasons.append(f"IID DCT transfer ratio below k=16 is {ratio_b:.3f}")

    return {
        "primary": primary,
        "kind": kind,
        "confirmatory_only": True,
        "reasons": reasons or ["no Wave 0 locus crossed a pre-registered threshold"],
        "estimand": "Delta(on) - Delta(off) vs ship; never intervention-on vs ship",
        "no_op": {
            "required_if_trained": True,
            "placebo": "same recipe, same seed budget, knob off",
            "wobble": "second seed of the control; off-target = below control wobble",
        },
        "cannot_smoke": [
            "FNO mode count",
            "FiLM / CBN",
            "high-k or peak-amp losses",
        ],
        "offline_done": "TF_alt = TF_1D_alt + R̂ on presentation packs",
        "smoke": "wiring tests only; no scientific claim",
        "single_seed": "exploratory diagnostic of this checkpoint",
    }


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items() if k != "per_sample"}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return to_jsonable(obj.item())
        if obj.size > 64 and obj.ndim == 1:
            return {
                "mean": to_jsonable(float(np.nanmean(obj))),
                "median": to_jsonable(float(np.nanmedian(obj))),
                "n": int(obj.size),
            }
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj
