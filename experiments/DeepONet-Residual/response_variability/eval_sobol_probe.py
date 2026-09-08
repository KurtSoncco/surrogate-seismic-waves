#!/usr/bin/env python3
"""Sobol covering + frequency probes for residual GINO (no retrain).

Answers two questions on the shipped checkpoint using nested test packs
and mix-train caches:

1. Adding Sobol points (IID / OOD): unique 4D/6D covering vs file count,
   nested unique-ID fill distance, and error vs 1-NN distance in parameter space.
2. Extrapolation: axis-aligned hull of IID train vs seiskit RV 64 (fixed
   rH=10 m, aHV=50) and frequency train-query bins vs the other 800 eval bins.

    uv run python experiments/DeepONet-Residual/response_variability/eval_sobol_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from mix_ladder import mix_train_parts  # noqa: E402
from response_variability.metrics import (  # noqa: E402
    FREQ_BANDS,
    band_pearson,
    band_rel_l2,
    pearson,
    rel_l2,
    theoretical_f0,
)
from response_variability.plot_presentation import (  # noqa: E402
    DOMAIN_SPECS,
    attach_stoch_from_cache,
    load_pack,
    pack_path,
)
from response_variability.sobol_design import (  # noqa: E402
    AHV_FIXED,
    DEFAULT_RV_COUNT,
    PARAM_4D,
    PARAM_6D,
    RH_FIXED,
    aabb_inside,
    cases_matrix,
    fill_distance,
    freq_train_mask,
    generate_rv_base_cases,
    knn_distance,
    nested_permutation_prefixes,
    physical_to_unit_6d,
    standardize,
    unique_rows,
)

PACK_DIR = config.RESULTS_DIR / "presentation"
OUT_DIR = config.RESULTS_DIR / "response_variability" / "sobol_probe"
COVER_SIZES = (16, 32, 64, 128, 256)
MIX_TAGS = ("M700", "M1400", "M2100")

# Held-out pooled rel L2 from the experiment README (seed-42 nested tests).
PUBLISHED_REL_L2 = {
    "M700 GINO": {"iid": 0.371, "dipping": 0.335, "three_layer": 0.533},
    "Unweighted M7680": {"iid": 0.310, "dipping": 0.333, "three_layer": 0.538},
    "Rebal FT (ship)": {"iid": 0.353, "dipping": 0.322, "three_layer": 0.525},
}
PUBLISHED_PEARSON = {
    "M700 GINO": {"iid": 0.915, "dipping": 0.896, "three_layer": 0.866},
    "Rebal FT (ship)": {"iid": 0.926, "dipping": 0.903, "three_layer": 0.869},
}

PACK_4D = ("vs1", "H", "cov", "vs2")
PACK_6D = ("vs1", "H", "cov", "rH", "aHV", "vs2")


def _meta_matrix(meta: dict[str, Any], idx: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
    loc = np.asarray(idx, dtype=int)
    return np.column_stack([np.asarray(meta[key], dtype=float)[loc] for key in keys])


def _pack_matrix(pack: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([np.asarray(pack[k], dtype=float) for k in keys])


def _rel_l2_per_sample(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    if p.ndim == 2:
        p = np.broadcast_to(p[:, None, :], t.shape)
    out = np.full(t.shape[0], np.nan, dtype=float)
    for i in range(t.shape[0]):
        out[i] = rel_l2(p[i], t[i])
    return out


def _rel_l2_mask_per_sample(
    pred: np.ndarray, true: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if p.ndim == 2:
        p = np.broadcast_to(p[:, None, :], t.shape)
    out = np.full(t.shape[0], np.nan, dtype=float)
    for i in range(t.shape[0]):
        pi, ti = p[i], t[i]
        if pi.ndim == 1:
            out[i] = rel_l2(pi, ti, mask=m)
        else:
            out[i] = rel_l2(pi[:, m], ti[:, m])
    return out


def _pearson_mask_per_sample(
    pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    if p.ndim == 2:
        p = np.broadcast_to(p[:, None, :], t.shape)
    m = None if mask is None else np.asarray(mask, dtype=bool)
    out = np.full(t.shape[0], np.nan, dtype=float)
    for i in range(t.shape[0]):
        pi, ti = p[i], t[i]
        if pi.ndim == 1:
            out[i] = pearson(pi, ti, mask=m)
            continue
        cors = []
        for r in range(pi.shape[0]):
            if m is None:
                cors.append(pearson(pi[r], ti[r]))
            else:
                cors.append(pearson(pi[r, m], ti[r, m]))
        finite = np.isfinite(cors)
        if np.any(finite):
            out[i] = float(np.mean(np.asarray(cors)[finite]))
    return out


def _abs_log_stack(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(pred, dtype=float), 1e-12, None)
    t = np.clip(np.asarray(true, dtype=float), 1e-12, None)
    if p.ndim == 2:
        p = np.broadcast_to(p[:, None, :], t.shape)
    return np.abs(np.log(p / t))


def _mean_abs_log_ratio(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Mean over samples and recorders of |ln(pred/true)| at each frequency."""
    return np.nanmean(_abs_log_stack(pred, true), axis=(0, 1))


def _abs_log_per_sample_freq(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Mean over recorders: shape (n_sample, n_freq)."""
    err = _abs_log_stack(pred, true)
    if err.ndim == 2:
        return err
    return np.nanmean(err, axis=1)


def load_train_cloud(mix_tag: str) -> dict[str, np.ndarray]:
    parts = mix_train_parts(mix_tag)
    blocks_4d: list[np.ndarray] = []
    blocks_6d: list[np.ndarray] = []
    names: list[np.ndarray] = []
    for name, cache_dir, idx in parts:
        meta = dict(np.load(Path(cache_dir) / "meta.npz", allow_pickle=True))
        idx = np.asarray(idx, dtype=int)
        blocks_4d.append(_meta_matrix(meta, idx, PARAM_4D))
        blocks_6d.append(_meta_matrix(meta, idx, PARAM_6D))
        names.append(np.full(len(idx), name))
    x4 = np.vstack(blocks_4d)
    x6 = np.vstack(blocks_6d)
    iid_mask = np.array([str(d).startswith("iid") for d in np.concatenate(names)])
    return {
        "x4": x4,
        "x6": x6,
        "iid_mask": iid_mask,
        "n_files": np.array([len(x4)]),
        "n_unique_4d": np.array([len(unique_rows(x4))]),
        "n_unique_6d": np.array([len(unique_rows(x6))]),
        "n_unique_4d_iid": np.array([len(unique_rows(x4[iid_mask]))]),
        "n_unique_6d_iid": np.array([len(unique_rows(x6[iid_mask]))]),
    }


def load_ood_train(domain_key: str) -> dict[str, np.ndarray]:
    hit = [p for p in mix_train_parts("M700") if p[0] == domain_key]
    if not hit:
        raise KeyError(domain_key)
    _name, cache_dir, idx = hit[0]
    meta = dict(np.load(Path(cache_dir) / "meta.npz", allow_pickle=True))
    idx = np.asarray(idx, dtype=int)
    x4 = _meta_matrix(meta, idx, PARAM_4D)
    x6 = _meta_matrix(meta, idx, PARAM_6D)
    return {
        "x4": x4,
        "x6": x6,
        "n_files": len(idx),
        "n_unique_4d": len(unique_rows(x4)),
    }


def load_domain_pack(domain: str, pack_dir: Path) -> dict[str, np.ndarray]:
    pack = load_pack(pack_path(pack_dir, domain))
    pack = attach_stoch_from_cache(pack, domain)
    ops = pack["tf_opensees"]
    gino = pack["tf_gino"]
    pack["rel_l2_gino"] = _rel_l2_per_sample(gino, ops)
    pack["rel_l2_1d"] = _rel_l2_per_sample(pack["tf_haskell_nominal"], ops)
    pack["pearson_gino"] = _pearson_mask_per_sample(gino, ops)
    pack["pearson_1d"] = _pearson_mask_per_sample(pack["tf_haskell_nominal"], ops)
    if "tf_pretell" in pack:
        pack["rel_l2_pretell"] = _rel_l2_per_sample(pack["tf_pretell"], ops)
        pack["pearson_pretell"] = _pearson_mask_per_sample(pack["tf_pretell"], ops)
    from response_variability.seiskit_arms import attach_pretell_p84

    pack = attach_pretell_p84(pack)
    if "tf_pretell_p84" in pack:
        pack["rel_l2_pretell_p84"] = _rel_l2_per_sample(pack["tf_pretell_p84"], ops)
        pack["pearson_pretell_p84"] = _pearson_mask_per_sample(pack["tf_pretell_p84"], ops)
    vs1 = np.asarray(pack["vs1"], dtype=float)
    H = np.asarray(pack["H"], dtype=float)
    pack["f0"] = np.array([theoretical_f0(v, h) for v, h in zip(vs1, H)], dtype=float)
    return pack


def covering_table(clouds: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for tag, cloud in clouds.items():
        rows.append(
            {
                "mix": tag,
                "n_files": int(np.asarray(cloud["n_files"]).ravel()[0]),
                "n_unique_4d": int(np.asarray(cloud["n_unique_4d"]).ravel()[0]),
                "n_unique_6d": int(np.asarray(cloud["n_unique_6d"]).ravel()[0]),
                "n_unique_6d_iid": int(np.asarray(cloud["n_unique_6d_iid"]).ravel()[0]),
            }
        )
    return rows


def nested_fill_curve(
    train_unique: np.ndarray,
    test: np.ndarray,
    sizes: tuple[int, ...] = COVER_SIZES,
) -> dict[str, list[float]]:
    n = len(train_unique)
    prefixes = nested_permutation_prefixes(n, sizes)
    ns: list[int] = []
    gaps: list[float] = []
    t_z, q_z = standardize(train_unique, test)
    seen: set[int] = set()
    for n_use, idx in sorted(prefixes.items()):
        n_use = int(n_use)
        if n_use in seen:
            continue
        seen.add(n_use)
        ns.append(n_use)
        gaps.append(fill_distance(t_z[idx], q_z))
    if int(n) not in seen:
        ns.append(int(n))
        gaps.append(fill_distance(t_z, q_z))
    return {"n_unique": ns, "fill_distance": gaps}


def domain_distance_rows(
    domain: str,
    pack: dict[str, np.ndarray],
    train_4d: np.ndarray,
    train_6d: np.ndarray,
) -> dict[str, np.ndarray]:
    q4 = _pack_matrix(pack, PACK_4D)
    t4, q4z = standardize(train_4d, q4)
    d4 = knn_distance(t4, q4z, k=1)
    out: dict[str, np.ndarray] = {
        "domain": np.array(domain),
        "rel_l2_gino": pack["rel_l2_gino"],
        "rel_l2_1d": pack["rel_l2_1d"],
        "pearson_gino": pack["pearson_gino"],
        "pearson_1d": pack["pearson_1d"],
        "knn_4d": d4,
        "inside_aabb_4d": aabb_inside(train_4d, q4).astype(int),
        "vs1": pack["vs1"],
        "H": pack["H"],
        "cov": pack["cov"],
        "vs2": pack["vs2"],
        "f0": pack["f0"],
    }
    rho4, p4 = spearmanr(d4, pack["rel_l2_gino"], nan_policy="omit")
    out["spearman_knn4d_l2"] = np.array([float(rho4), float(p4)])
    rp, pp = spearmanr(d4, pack["pearson_gino"], nan_policy="omit")
    out["spearman_knn4d_pearson"] = np.array([float(rp), float(pp)])
    has_geo = all(k in pack and np.isfinite(pack[k]).any() for k in ("rH", "aHV"))
    if has_geo:
        q6 = _pack_matrix(pack, PACK_6D)
        t6, q6z = standardize(train_6d, q6)
        out["knn_6d"] = knn_distance(t6, q6z, k=1)
        out["inside_aabb_6d"] = aabb_inside(train_6d, q6).astype(int)
        out["rH"] = pack["rH"]
        out["aHV"] = pack["aHV"]
        rho, pval = spearmanr(out["knn_6d"], out["rel_l2_gino"], nan_policy="omit")
        out["spearman_knn6d_l2"] = np.array([float(rho), float(pval)])
        rp6, pp6 = spearmanr(out["knn_6d"], out["pearson_gino"], nan_policy="omit")
        out["spearman_knn6d_pearson"] = np.array([float(rp6), float(pp6)])
    return out


def rv_overlay(train_6d: np.ndarray, n: int = DEFAULT_RV_COUNT) -> dict[str, Any]:
    cases = generate_rv_base_cases(n)
    rv6 = cases_matrix(cases, kind="6d")
    rv4 = cases_matrix(cases, kind="4d")
    unit = physical_to_unit_6d(rv6)
    t4 = train_6d[:, [0, 1, 2, 5]]
    t6z, q6z = standardize(train_6d, rv6)
    t4z, q4z = standardize(t4, rv4)
    rh_range = (float(train_6d[:, 3].min()), float(train_6d[:, 3].max()))
    ahv_range = (float(train_6d[:, 4].min()), float(train_6d[:, 4].max()))
    inside_6 = aabb_inside(train_6d, rv6)
    return {
        "cases": cases,
        "rv6": rv6,
        "rv4": rv4,
        "unit_6d": unit,
        "knn_4d": knn_distance(t4z, q4z, k=1),
        "knn_6d": knn_distance(t6z, q6z, k=1),
        "frac_inside_aabb_4d": float(np.mean(aabb_inside(t4, rv4))),
        "frac_inside_aabb_6d": float(np.mean(inside_6)),
        "n_outside_aabb_6d": int(np.sum(~inside_6)),
        "train_rH_range": rh_range,
        "train_aHV_range": ahv_range,
        "rv_rH": RH_FIXED,
        "rv_aHV": AHV_FIXED,
        "rH_below_train_min": bool(RH_FIXED < rh_range[0]),
        "aHV_above_train_max": bool(AHV_FIXED > ahv_range[1]),
        "unit_rH_mean": float(np.mean(unit[:, 3])),
        "unit_aHV_mean": float(np.mean(unit[:, 4])),
    }


def freq_probe(pack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    freq = np.asarray(pack["freq"], dtype=float)
    mask_tr = freq_train_mask(freq, config.N_FREQ_TRAIN)
    ops = pack["tf_opensees"]
    gino = pack["tf_gino"]
    one = pack["tf_haskell_nominal"]
    out: dict[str, np.ndarray] = {
        "freq": freq,
        "mask_train": mask_tr,
        "rel_l2_gino_train_bins": _rel_l2_mask_per_sample(gino, ops, mask_tr),
        "rel_l2_gino_heldout_bins": _rel_l2_mask_per_sample(gino, ops, ~mask_tr),
        "rel_l2_1d_train_bins": _rel_l2_mask_per_sample(one, ops, mask_tr),
        "rel_l2_1d_heldout_bins": _rel_l2_mask_per_sample(one, ops, ~mask_tr),
        "pearson_gino_train_bins": _pearson_mask_per_sample(gino, ops, mask_tr),
        "pearson_gino_heldout_bins": _pearson_mask_per_sample(gino, ops, ~mask_tr),
        "pearson_1d_train_bins": _pearson_mask_per_sample(one, ops, mask_tr),
        "pearson_1d_heldout_bins": _pearson_mask_per_sample(one, ops, ~mask_tr),
        "abs_log_gino": _mean_abs_log_ratio(gino, ops),
        "abs_log_1d": _mean_abs_log_ratio(one, ops),
        "abs_log_gino_sf": _abs_log_per_sample_freq(gino, ops),
        "f0": pack["f0"],
    }
    for band, (lo, hi) in FREQ_BANDS.items():
        g_rows = []
        d_rows = []
        gp = []
        dp = []
        for i in range(ops.shape[0]):
            g_rows.append(band_rel_l2(gino[i], ops[i], freq, lo=lo, hi=hi))
            d_rows.append(band_rel_l2(one[i], ops[i], freq, lo=lo, hi=hi))
            gp.append(band_pearson(gino[i], ops[i], freq, lo=lo, hi=hi))
            dp.append(band_pearson(one[i], ops[i], freq, lo=lo, hi=hi))
        out[f"rel_l2_gino_{band}"] = np.asarray(g_rows, dtype=float)
        out[f"rel_l2_1d_{band}"] = np.asarray(d_rows, dtype=float)
        out[f"pearson_gino_{band}"] = np.asarray(gp, dtype=float)
        out[f"pearson_1d_{band}"] = np.asarray(dp, dtype=float)
    return out


def nearest_iid_proxy(
    pack_iid: dict[str, np.ndarray], rv4: np.ndarray
) -> dict[str, np.ndarray]:
    """Nearest nested-IID test sample per RV 4D case (proxy; not OpenSees RV)."""
    iid4 = _pack_matrix(pack_iid, PACK_4D)
    t_z, q_z = standardize(iid4, rv4)
    d_rv = knn_distance(t_z, q_z, k=1)
    dist_mat = np.linalg.norm(q_z[:, None, :] - t_z[None, :, :], axis=-1)
    nn = np.argmin(dist_mat, axis=1)
    return {
        "nn_local": nn.astype(int),
        "nn_dist_4d": d_rv,
        "nn_rel_l2_gino": pack_iid["rel_l2_gino"][nn],
        "nn_rel_l2_1d": pack_iid["rel_l2_1d"][nn],
        "nn_pearson_gino": pack_iid["pearson_gino"][nn],
        "nn_pearson_1d": pack_iid["pearson_1d"][nn],
        "nn_vs1": pack_iid["vs1"][nn],
        "nn_H": pack_iid["H"][nn],
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return _jsonable(obj.item())
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def compute(pack_dir: Path) -> dict[str, Any]:
    clouds: dict[str, dict[str, np.ndarray]] = {}
    for tag in MIX_TAGS:
        try:
            clouds[tag] = load_train_cloud(tag)
        except FileNotFoundError as exc:
            print(f"[sobol-probe] skip {tag}: {exc}", flush=True)
    if "M700" not in clouds:
        raise FileNotFoundError("need at least M700 mix caches (n1000 + OOD signed)")

    train6_iid = clouds["M700"]["x6"][clouds["M700"]["iid_mask"]]
    train4_iid = clouds["M700"]["x4"][clouds["M700"]["iid_mask"]]
    packs = {d: load_domain_pack(d, pack_dir) for d in DOMAIN_SPECS}
    dist = {
        d: domain_distance_rows(d, packs[d], train4_iid, train6_iid) for d in packs
    }

    fill: dict[str, dict[str, list[float]]] = {
        "iid_4d": nested_fill_curve(
            unique_rows(train4_iid), _pack_matrix(packs["iid"], PACK_4D)
        )
    }
    if all(k in packs["iid"] for k in PACK_6D):
        fill["iid_6d"] = nested_fill_curve(
            unique_rows(train6_iid), _pack_matrix(packs["iid"], PACK_6D)
        )
    for ood_key, dname in (("ood_dipping", "dipping"), ("ood_three_layer", "three_layer")):
        try:
            ood_tr = load_ood_train(ood_key)
        except (KeyError, FileNotFoundError):
            continue
        fill[dname] = nested_fill_curve(
            unique_rows(ood_tr["x4"]), _pack_matrix(packs[dname], PACK_4D)
        )
        fill[f"{dname}_vs_iid"] = nested_fill_curve(
            unique_rows(train4_iid), _pack_matrix(packs[dname], PACK_4D)
        )

    rv = rv_overlay(train6_iid)
    proxy = nearest_iid_proxy(packs["iid"], rv["rv4"])
    freq = {d: freq_probe(packs[d]) for d in packs}

    summary: dict[str, Any] = {
        "covering": covering_table(clouds),
        "published_rel_l2": PUBLISHED_REL_L2,
        "published_pearson": PUBLISHED_PEARSON,
        "n_freq_train": int(config.N_FREQ_TRAIN),
        "n_freq_eval": int(config.N_FREQ_EVAL),
        "rv": {
            "n": DEFAULT_RV_COUNT,
            "frac_inside_aabb_4d": rv["frac_inside_aabb_4d"],
            "frac_inside_aabb_6d": rv["frac_inside_aabb_6d"],
            "n_outside_aabb_6d": rv["n_outside_aabb_6d"],
            "train_rH_range": rv["train_rH_range"],
            "train_aHV_range": rv["train_aHV_range"],
            "rH_below_train_min": rv["rH_below_train_min"],
            "aHV_above_train_max": rv["aHV_above_train_max"],
            "unit_rH_mean": rv["unit_rH_mean"],
            "unit_aHV_mean": rv["unit_aHV_mean"],
            "knn_6d_mean": float(np.mean(rv["knn_6d"])),
            "proxy_nn_rel_l2_gino_mean": float(np.mean(proxy["nn_rel_l2_gino"])),
            "proxy_nn_rel_l2_1d_mean": float(np.mean(proxy["nn_rel_l2_1d"])),
            "proxy_nn_pearson_gino_mean": float(np.mean(proxy["nn_pearson_gino"])),
            "proxy_nn_pearson_1d_mean": float(np.mean(proxy["nn_pearson_1d"])),
        },
        "spearman": {
            d: {
                "knn4d": [
                    float(dist[d]["spearman_knn4d_l2"][0]),
                    float(dist[d]["spearman_knn4d_l2"][1]),
                ],
                **(
                    {
                        "knn4d_pearson": [
                            float(dist[d]["spearman_knn4d_pearson"][0]),
                            float(dist[d]["spearman_knn4d_pearson"][1]),
                        ]
                    }
                    if "spearman_knn4d_pearson" in dist[d]
                    else {}
                ),
                **(
                    {
                        "knn6d": [
                            float(dist[d]["spearman_knn6d_l2"][0]),
                            float(dist[d]["spearman_knn6d_l2"][1]),
                        ]
                    }
                    if "spearman_knn6d_l2" in dist[d]
                    else {}
                ),
                **(
                    {
                        "knn6d_pearson": [
                            float(dist[d]["spearman_knn6d_pearson"][0]),
                            float(dist[d]["spearman_knn6d_pearson"][1]),
                        ]
                    }
                    if "spearman_knn6d_pearson" in dist[d]
                    else {}
                ),
            }
            for d in dist
        },
        "aabb_frac_inside": {
            d: {
                "4d": float(np.mean(dist[d]["inside_aabb_4d"])),
                **(
                    {"6d": float(np.mean(dist[d]["inside_aabb_6d"]))}
                    if "inside_aabb_6d" in dist[d]
                    else {}
                ),
            }
            for d in dist
        },
        "freq": {
            d: {
                "gino_train_bins_mean": float(np.nanmean(freq[d]["rel_l2_gino_train_bins"])),
                "gino_heldout_bins_mean": float(
                    np.nanmean(freq[d]["rel_l2_gino_heldout_bins"])
                ),
                "gino_low_mean": float(np.nanmean(freq[d]["rel_l2_gino_low"])),
                "gino_mid_mean": float(np.nanmean(freq[d]["rel_l2_gino_mid"])),
                "gino_high_mean": float(np.nanmean(freq[d]["rel_l2_gino_high"])),
                "haskell_high_mean": float(np.nanmean(freq[d]["rel_l2_1d_high"])),
                "gino_train_bins_pearson": float(
                    np.nanmean(freq[d]["pearson_gino_train_bins"])
                ),
                "gino_heldout_bins_pearson": float(
                    np.nanmean(freq[d]["pearson_gino_heldout_bins"])
                ),
                "gino_low_pearson": float(np.nanmean(freq[d]["pearson_gino_low"])),
                "gino_mid_pearson": float(np.nanmean(freq[d]["pearson_gino_mid"])),
                "gino_high_pearson": float(np.nanmean(freq[d]["pearson_gino_high"])),
                "haskell_high_pearson": float(np.nanmean(freq[d]["pearson_1d_high"])),
            }
            for d in freq
        },
        "fill": fill,
        "note": (
            "n1000 already covers ~all GIFNO Sobol IDs; extra mix files are mostly "
            "RF replicates. RV 64 is a different 4D sequence with (rH, aHV) at the "
            "6D cube corner. Frequency held-out bins are in-band interpolation "
            "(0.1–10 Hz), not f outside the training band."
        ),
    }
    return {
        "summary": summary,
        "clouds": clouds,
        "packs": packs,
        "dist": dist,
        "rv": rv,
        "proxy": proxy,
        "freq": freq,
        "fill": fill,
        "train6_iid": train6_iid,
        "train4_iid": train4_iid,
    }


def write_outputs(blob: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(blob["summary"]), indent=2))
    rv = blob["rv"]
    proxy = blob["proxy"]
    lines = [
        "sobol_id,Vs1,H,CoV,Vs2,rH,aHV,knn_4d,knn_6d,nn_rel_l2_gino,nn_rel_l2_1d,nn_pearson_gino,nn_pearson_1d\n"
    ]
    for i, case in enumerate(rv["cases"]):
        lines.append(
            f"{case.sobol_id},{case.vs1:.6f},{case.H:.6f},{case.cov:.6f},{case.vs2:.6f},"
            f"{case.rH:.1f},{case.aHV:.1f},{rv['knn_4d'][i]:.6f},{rv['knn_6d'][i]:.6f},"
            f"{proxy['nn_rel_l2_gino'][i]:.6f},{proxy['nn_rel_l2_1d'][i]:.6f},"
            f"{proxy['nn_pearson_gino'][i]:.6f},{proxy['nn_pearson_1d'][i]:.6f}\n"
        )
    (out_dir / "rv_sobol_base_cases.csv").write_text("".join(lines))

    dlines = [
        "domain,sample,rel_l2_gino,rel_l2_1d,pearson_gino,pearson_1d,knn_4d,inside_aabb_4d,Vs1,H,CoV,Vs2\n"
    ]
    for d, rec in blob["dist"].items():
        for i in range(len(rec["rel_l2_gino"])):
            dlines.append(
                f"{d},{i},{rec['rel_l2_gino'][i]:.6f},{rec['rel_l2_1d'][i]:.6f},"
                f"{rec['pearson_gino'][i]:.6f},{rec['pearson_1d'][i]:.6f},"
                f"{rec['knn_4d'][i]:.6f},{int(rec['inside_aabb_4d'][i])},"
                f"{rec['vs1'][i]:.6f},{rec['H'][i]:.6f},{rec['cov'][i]:.6f},"
                f"{rec['vs2'][i]:.6f}\n"
            )
    (out_dir / "test_error_vs_knn.csv").write_text("".join(dlines))
    return summary_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    missing = [d for d in DOMAIN_SPECS if not pack_path(args.pack_dir, d).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing presentation packs {missing}. Run plot_presentation.py first."
        )
    blob = compute(args.pack_dir)
    path = write_outputs(blob, args.out_dir)
    print(f"Wrote {path}", flush=True)
    rv = blob["summary"]["rv"]
    print(
        f"RV 64: 4D AABB inside={rv['frac_inside_aabb_4d']:.2f}  "
        f"6D inside={rv['frac_inside_aabb_6d']:.2f}  "
        f"rH<train_min={rv['rH_below_train_min']}  "
        f"aHV>train_max={rv['aHV_above_train_max']}",
        flush=True,
    )
    for d, freq in blob["summary"]["freq"].items():
        print(
            f"{d}: GINO train-bins L2={freq['gino_train_bins_mean']:.3f}  "
            f"Pearson={freq['gino_train_bins_pearson']:.3f}  "
            f"held-out L2={freq['gino_heldout_bins_mean']:.3f}  "
            f"Pearson={freq['gino_heldout_bins_pearson']:.3f}  "
            f"high-band L2={freq['gino_high_mean']:.3f}  "
            f"Pearson={freq['gino_high_pearson']:.3f}",
            flush=True,
        )
    if not args.skip_plots:
        from response_variability.plot_sobol_probe import plot_all

        for pth in plot_all(blob, args.out_dir):
            print(f"  {pth}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
