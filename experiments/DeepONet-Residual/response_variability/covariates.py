"""H5 covariates and window-extracted f0 for nested-test bias tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from response_variability.metrics import (
    FREQ_BANDS,
    anderson_frequency_domain,
    band_anderson,
    band_pearson,
    central_recorder,
    theoretical_f0,
)

_EPS = 1e-12
CENTRAL_REC = 10

# Continuous H5 / derived axes (rf_seed and f0_effective are excluded).
SHARED_COVARIATES = (
    "vs1",
    "H",
    "cov",
    "vs2",
    "rH",
    "aHV",
    "f0",
    "impedance",
    "xi_mean",
    "field_cov",
)
DIPPING_COVARIATES = ("dip_angle_deg", "dip_span", "bedrock_H", "dip_direction")
THREE_LAYER_COVARIATES = ("H1", "H2", "vs_mid", "vs_contrast")

COVARIATE_LABELS = {
    "vs1": r"$V_{s1}$ (m s$^{-1}$)",
    "H": r"$H$ (m)",
    "cov": r"CoV",
    "vs2": r"$V_{s2}$ (m s$^{-1}$)",
    "rH": r"$r_H$ (m)",
    "aHV": r"$a_{HV}$",
    "f0": r"$f_0$ (Hz, window)",
    "impedance": r"$V_{s2}/V_{s1}$",
    "xi_mean": r"mean $\zeta$",
    "field_cov": r"field CoV",
    "dip_angle_deg": r"dip ($^\circ$)",
    "dip_span": r"dip span (m)",
    "bedrock_H": r"$H_{\mathrm{bedrock}}$ (m)",
    "dip_direction": r"dip direction",
    "H1": r"$H_1$ (m)",
    "H2": r"$H_2$ (m)",
    "vs_mid": r"$V_{s,\mathrm{mid}}$ (m s$^{-1}$)",
    "vs_contrast": r"$\ln(V_{s,\mathrm{mid}}/V_{s1})$",
}


def f0_window_center(pack: dict[str, np.ndarray], i: int) -> float:
    """Travel-time 1/(4T) used only as the search-window center."""
    vs1 = float(pack["vs1"][i])
    h1 = float(pack["H1"][i]) if "H1" in pack else float("nan")
    h2 = float(pack["H2"][i]) if "H2" in pack else float("nan")
    vs_mid = float(pack["vs_mid"][i]) if "vs_mid" in pack else float("nan")
    if (
        np.isfinite(h1)
        and np.isfinite(h2)
        and np.isfinite(vs_mid)
        and h1 > 0.0
        and h2 > 0.0
        and vs_mid > 0.0
        and vs1 > 0.0
    ):
        t = h1 / vs1 + h2 / vs_mid
        return 1.0 / (4.0 * max(t, _EPS))
    return theoretical_f0(vs1, float(pack["H"][i]))


def extracted_f0_from_tf(pack: dict[str, np.ndarray]) -> np.ndarray:
    """OpenSees |TF| peak in the log ±20% window around 1/(4T)."""
    from unified_metrics import max_peak_near_f0_calc

    freq = np.asarray(pack["freq"], dtype=float)
    ops = np.asarray(pack["tf_opensees"])
    n = ops.shape[0]
    rec = ops.shape[1] // 2 if ops.ndim == 3 else 0
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        af = ops[i, rec] if ops.ndim == 3 else ops[i]
        f0c = f0_window_center(pack, i)
        f_pk, _amp = max_peak_near_f0_calc(freq, af, f0c)
        out[i] = f_pk
    return out


def attach_extracted_f0(pack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Set ``f0`` to the window-extracted OpenSees peak; keep ``f0_calc`` as 1/(4T)."""
    out = dict(pack)
    n = int(np.asarray(out["tf_opensees"]).shape[0])
    f0c = np.array([f0_window_center(out, i) for i in range(n)], dtype=float)
    out["f0_calc"] = f0c
    out["f0"] = extracted_f0_from_tf(out)
    return out


def impedance(pack: dict[str, np.ndarray]) -> np.ndarray:
    vs1 = np.asarray(pack["vs1"], dtype=float)
    vs2 = np.asarray(pack["vs2"], dtype=float)
    return vs2 / np.clip(vs1, _EPS, None)


def _attr_float(params: dict[str, Any], *keys: str, default: float = float("nan")) -> float:
    for k in keys:
        if k in params and params[k] is not None:
            try:
                return float(params[k])
            except (TypeError, ValueError):
                continue
    return float(default)


def field_cov(vs_strip: np.ndarray, soil_nz: int) -> float:
    n = max(1, min(int(soil_nz), vs_strip.shape[0]))
    soil = np.asarray(vs_strip[:n], dtype=float)
    soil = soil[np.isfinite(soil) & (soil > 0)]
    if soil.size < 4:
        return float("nan")
    mu = float(np.mean(soil))
    if mu < _EPS:
        return float("nan")
    return float(np.std(soil, ddof=1) / mu)


def mean_zeta(zeta_strip: np.ndarray, soil_nz: int) -> float:
    n = max(1, min(int(soil_nz), zeta_strip.shape[0]))
    z = np.asarray(zeta_strip[:n], dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return float("nan")
    return float(np.mean(z))


def attach_h5_covariates(
    pack: dict[str, np.ndarray],
    *,
    domain: str = "iid",
) -> dict[str, np.ndarray]:
    """Fill rH/aHV/dip/three-layer/field stats from H5 when paths resolve."""
    from ood_io import (
        crop_variability,
        nominal_layer_params,
        read_h5_sample,
        soil_nz_from_params,
    )
    from residual_signed import resolve_h5_path

    n = int(np.asarray(pack["tf_opensees"]).shape[0])
    try:
        from response_variability.plots.plot_presentation import attach_stoch_from_cache

        pack = attach_stoch_from_cache(pack, domain)
    except Exception:
        pass
    out = dict(pack)
    for key in ("vs1", "H", "vs2", "cov", "rH", "aHV", "H1", "H2", "vs_mid"):
        if key in out:
            out[key] = np.asarray(out[key], dtype=float).copy()
    for key, default in (
        ("rH", np.nan),
        ("aHV", np.nan),
        ("xi_mean", np.nan),
        ("field_cov", np.nan),
        ("dip_angle_deg", np.nan),
        ("dip_span", np.nan),
        ("bedrock_H", np.nan),
        ("dip_direction", np.nan),
        ("H1", np.nan),
        ("H2", np.nan),
        ("vs_mid", np.nan),
        ("vs_contrast", np.nan),
    ):
        if key not in out:
            out[key] = np.full(n, default, dtype=float)

    paths = out.get("h5_path")
    if paths is None:
        _fill_field_cov_from_pack(out)
        out["impedance"] = impedance(out)
        return attach_extracted_f0(out)

    for i in range(n):
        raw = str(paths[i])
        h5 = Path(raw)
        if not h5.is_file():
            cand = resolve_h5_path(raw)
            h5 = cand if cand.is_file() else h5
        if not h5.is_file():
            continue
        try:
            vs, zeta, params, _extra = read_h5_sample(h5)
        except OSError:
            continue
        vs_c = crop_variability(vs)
        zeta_c = crop_variability(zeta)
        nom = nominal_layer_params(params)
        soil_nz = soil_nz_from_params(params, vs_c.shape[0])
        out["vs1"][i] = float(nom["vs1"])
        out["H"][i] = float(nom["H"])
        out["vs2"][i] = float(nom["vs2"])
        if nom.get("H1") is not None:
            out["H1"][i] = float(nom["H1"])
        if nom.get("H2") is not None:
            out["H2"][i] = float(nom["H2"])
        if nom.get("vs_mid") is not None and nom["vs_mid"] is not None:
            out["vs_mid"][i] = float(nom["vs_mid"])
        out["rH"][i] = _attr_float(params, "rH")
        out["aHV"][i] = _attr_float(params, "aHV")
        out["cov"][i] = _attr_float(params, "CoV", "cov", default=float(out["cov"][i]))
        out["xi_mean"][i] = mean_zeta(zeta_c, soil_nz)
        out["field_cov"][i] = field_cov(vs_c, soil_nz)
        out["dip_angle_deg"][i] = _attr_float(params, "dip_angle_deg")
        out["dip_span"][i] = _attr_float(params, "dip_span")
        out["bedrock_H"][i] = _attr_float(
            params, "H_bedrock", "bedrock_thickness", "H_bedrock_discretized"
        )
        out["dip_direction"][i] = _attr_float(params, "dip_direction")
    _fill_field_cov_from_pack(out)
    vs_mid = np.asarray(out["vs_mid"], dtype=float)
    vs1 = np.asarray(out["vs1"], dtype=float)
    contrast = np.full(n, np.nan)
    ok = np.isfinite(vs_mid) & np.isfinite(vs1) & (vs1 > 0) & (vs_mid > 0)
    contrast[ok] = np.log(vs_mid[ok] / vs1[ok])
    out["vs_contrast"] = contrast
    out["impedance"] = impedance(out)
    return attach_extracted_f0(out)


def _fill_field_cov_from_pack(out: dict[str, np.ndarray]) -> None:
    """Use cropped ``vs_2d`` when H5 field CoV is missing."""
    if "vs_2d" not in out or "field_cov" not in out:
        return
    n = int(out["field_cov"].shape[0])
    vs2d = np.asarray(out["vs_2d"])
    for i in range(n):
        if np.isfinite(out["field_cov"][i]):
            continue
        soil_nz = int(out["soil_nz"][i]) if "soil_nz" in out else vs2d.shape[1]
        out["field_cov"][i] = field_cov(vs2d[i], soil_nz)


def covariates_for_domain(domain: str) -> tuple[str, ...]:
    keys = list(SHARED_COVARIATES)
    if domain == "dipping":
        keys.extend(DIPPING_COVARIATES)
    if domain == "three_layer":
        keys.extend(THREE_LAYER_COVARIATES)
    return tuple(keys)


def present_covariates(pack: dict[str, np.ndarray], domain: str) -> list[str]:
    keys = []
    for k in covariates_for_domain(domain):
        if k not in pack:
            continue
        v = np.asarray(pack[k], dtype=float)
        if np.isfinite(v).sum() >= 8 and float(np.nanstd(v)) > 1e-12:
            keys.append(k)
    return keys


def pearson_anderson_per_sample(
    pred: np.ndarray,
    true: np.ndarray,
    freq: np.ndarray,
    *,
    f0_extracted: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Primary metrics per sample: Pearson and Anderson, all-band and bands."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    n = t.shape[0]
    if p.ndim == 2 and t.ndim == 3:
        p = np.broadcast_to(p[:, None, :], t.shape)
    pearson_all = np.full(n, np.nan)
    gof_all = np.full(n, np.nan)
    band_p = {b: np.full(n, np.nan) for b in FREQ_BANDS}
    band_g = {b: np.full(n, np.nan) for b in FREQ_BANDS}
    f0 = (
        np.asarray(f0_extracted, dtype=float)
        if f0_extracted is not None
        else np.full(n, np.nan)
    )
    for i in range(n):
        pi = p[i]
        ti = t[i]
        pc, tc = central_recorder(pi), central_recorder(ti)
        pearson_all[i] = band_pearson(pc, tc, freq, lo=0.1, hi=10.0)
        center = float(f0[i]) if i < f0.size and np.isfinite(f0[i]) else None
        gof_all[i] = anderson_frequency_domain(
            freq, tc, pc, f_weight_center=center, f_weight_width=1.5
        )
        for band, (lo, hi) in FREQ_BANDS.items():
            band_p[band][i] = band_pearson(pc, tc, freq, lo=lo, hi=hi)
            band_g[band][i] = band_anderson(pc, tc, freq, lo=lo, hi=hi)
    out = {"pearson": pearson_all, "gof_af": gof_all}
    for band in FREQ_BANDS:
        out[f"pearson_{band}"] = band_p[band]
        out[f"gof_{band}"] = band_g[band]
    return out
