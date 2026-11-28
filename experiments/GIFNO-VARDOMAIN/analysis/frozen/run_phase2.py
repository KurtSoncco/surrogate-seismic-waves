#!/usr/bin/env python3
"""Phase 2 frozen P128n128 tests.

Scores go through ``score_harness``. The reported model is loaded only with
``diag_forward.load_fp32``. Float32, no training autocast. Writes only under
``analysis/frozen/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FROZEN = Path(__file__).resolve().parent
ANALYSIS = FROZEN.parent
VARDOMAIN = ANALYSIS.parent
REPO = VARDOMAIN.parents[1]
CASE7A = REPO / "experiments" / "case7a"
GEOTECH = VARDOMAIN / "paper" / "geotech_v2"
for _path in (FROZEN, ANALYSIS, VARDOMAIN, CASE7A, GEOTECH, ANALYSIS / "audit"):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from gates import (  # noqa: E402
    G2_POSITIVE_BANDS,
    MIRROR_L1_FRACTION,
    MONOTONE_SIGN_AGREEMENT,
    NOISE_FLOOR_PLACEHOLDER_LOG_L1,
)
from score_harness import (  # noqa: E402
    BAND_NAMES,
    aggregate,
    assert_comparable,
    bootstrap_independent,
    bootstrap_moving_block,
    g_log_l1,
    git_commit,
    harm_rate,
    prior_log_l1,
    sample_id_hash,
    score_curves,
    skill_of_pooled,
    write_provenance,
)

AUDIT = ANALYSIS / "audit"
CACHE = REPO / "data" / "vardomain" / "cache"
CKPT = REPO / "checkpoints" / "vardomain" / "v2_ablations" / "P128n128"
FULL_QVS = REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5" / "full_qvs.h5"
ENSEMBLE_DIR = VARDOMAIN / "paper" / "dump"
TREASURE_FIELDS = VARDOMAIN / "paper" / "geotech_v2" / "05_treasure_island_full" / "fields.npz"
TREASURE_G = VARDOMAIN / "paper" / "geotech_v2" / "07_one_model" / "treasure_g.npz"
PROPOSAL = float(NOISE_FLOOR_PLACEHOLDER_LOG_L1)
N_PRETELL = 9
CENTER_STATION = 10
RNG_SEED = 0
CLIP = 1e-12


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def column_subset(n_col: int) -> np.ndarray:
    """Center column plus a fixed linspace. At most ``N_PRETELL`` columns."""

    n_col = int(n_col)
    center = n_col // 2
    cols = np.linspace(0, n_col - 1, N_PRETELL).round().astype(int)
    cols[len(cols) // 2] = center
    seen: list[int] = []
    for col in cols:
        col = int(np.clip(col, 0, n_col - 1))
        if col not in seen:
            seen.append(col)
    return np.asarray(seen, dtype=int)


def _layer_xi(freq: np.ndarray, zeta_col: np.ndarray, f1: float, f2: float) -> np.ndarray:
    from section_1d import rayleigh_xi_target

    return np.stack([rayleigh_xi_target(freq, float(zeta), f1, f2) for zeta in np.asarray(zeta_col)])


def center_outcrop(freq, vs, zeta, f1, f2) -> np.ndarray:
    from matched_physics import outcrop_many

    center = int(vs.shape[1] // 2)
    return np.asarray(outcrop_many(freq, vs[:, center], zeta[:, center], f1, f2, dz=1.0), dtype=np.float64)


def pretell_geomean(freq, vs, zeta, f1, f2, cols: np.ndarray) -> np.ndarray:
    from section_1d import surface_over_outcrop

    rows = []
    for col in cols:
        xi = _layer_xi(freq, zeta[:, int(col)], f1, f2)
        rows.append(surface_over_outcrop(freq, vs[:, int(col)], dz=1.0, xi=xi))
    stack = np.vstack(rows)
    return np.exp(np.mean(np.log(np.clip(stack, CLIP, None)), axis=0))


def _log(curve: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(curve, dtype=np.float64), CLIP, None))


def _empty_lists():
    return {key: [] for key in ("pred", "true", "prior", "sid", "station", "cluster", "x", "center", "pretell")}


def _as_pack(bucket: dict) -> dict:
    return {
        "log_pred": np.stack(bucket["pred"]),
        "log_true": np.stack(bucket["true"]),
        "log_prior": np.stack(bucket["prior"]),
        "log_center": np.stack(bucket["center"]),
        "log_pretell": np.stack(bucket["pretell"]),
        "sample_id": np.asarray(bucket["sid"]),
        "station": np.asarray(bucket["station"], dtype=int),
        "cluster_id": np.asarray(bucket["cluster"]),
        "x_m": np.asarray(bucket["x"], dtype=np.float64),
    }


def _append_station(bucket, *, g_row, true, prior, sid, station, cluster, x_m, log_center, log_pretell):
    bucket["pred"].append(np.asarray(prior, dtype=np.float64) + np.asarray(g_row, dtype=np.float64))
    bucket["true"].append(np.asarray(true, dtype=np.float64))
    bucket["prior"].append(np.asarray(prior, dtype=np.float64))
    bucket["center"].append(log_center)
    bucket["pretell"].append(log_pretell)
    bucket["sid"].append(sid)
    bucket["station"].append(int(station))
    bucket["cluster"].append(str(cluster))
    bucket["x"].append(float(x_m))


def _score_pack(pack: dict, freq: np.ndarray) -> pd.DataFrame:
    frames = []
    common = dict(
        sample_id=pack["sample_id"],
        station=pack["station"],
        cluster_id=pack["cluster_id"],
        freq=freq,
        log_true=pack["log_true"],
        log_prior=pack["log_prior"],
        x_m=pack["x_m"],
    )
    curves = {
        "P128n128": pack["log_pred"],
        "g0": pack["log_prior"],
        "center_column": pack["log_center"],
        "pretell_geomean": pack["log_pretell"],
    }
    for method, log_pred in curves.items():
        frames.append(score_curves(**common, log_pred=log_pred, method=method))
    return pd.concat(frames, ignore_index=True)


def _forward_runs(model, cache, family: str, device, freq, width: int = 500) -> dict:
    from data import IID_MESH_M, KIND_RUN, make_window

    flats = [int(flat) for flat in cache.runs("test", (family,))]
    start = int(IID_MESH_M // 2 - width // 2)
    bucket = _empty_lists()
    center_rows = []
    subset_note = None
    for step, chunk in enumerate(_batches(flats, 4)):
        windows = [make_window(cache, "metric", KIND_RUN, flat, width, start, 0, False) for flat in chunk]
        x = torch.stack([item[0] for item in windows])
        query = torch.stack([item[1] for item in windows])
        with torch.inference_mode():
            g = model(x.to(device), query.to(device)).float().cpu().numpy()
        for local, flat in enumerate(chunk):
            camp, j = cache.campaign(flat)
            n_active = int(camp.n_active[j])
            vs = np.asarray(camp.vs[j, :n_active], dtype=np.float64)
            zeta = np.asarray(camp.zeta[j, :n_active], dtype=np.float64)
            f1, f2 = float(camp.f1[j]), float(camp.f2[j])
            cols = column_subset(vs.shape[1])
            if subset_note is None:
                subset_note = {"n_columns": int(vs.shape[1]), "columns": cols.tolist(), "center": int(vs.shape[1] // 2)}
            log_c = _log(center_outcrop(freq, vs, zeta, f1, f2))
            log_p = _log(pretell_geomean(freq, vs, zeta, f1, f2, cols))
            prior = windows[local][3].numpy()
            true = windows[local][2].numpy()
            valid = windows[local][4].numpy().astype(bool)
            x_abs = windows[local][6].numpy()
            sid = f"{family}:{int(j)}"
            cluster = str(int(camp.group[j]))
            keep = np.flatnonzero(valid)
            for station in keep:
                _append_station(
                    bucket, g_row=g[local, int(station)], true=true[int(station)], prior=prior[int(station)],
                    sid=sid, station=int(station), cluster=cluster, x_m=float(x_abs[int(station)]),
                    log_center=log_c, log_pretell=log_p,
                )
            if width == 500 and keep.size > CENTER_STATION:
                center_rows.append({
                    "run": int(j),
                    "sample_id": sid,
                    "cluster_id": cluster,
                    "station": CENTER_STATION,
                    "x_m": float(x_abs[CENTER_STATION]),
                    "log_pred": np.array(bucket["pred"][-len(keep) + CENTER_STATION], copy=True),
                    "log_true": np.asarray(true[CENTER_STATION], dtype=np.float64).copy(),
                    "log_prior": np.asarray(prior[CENTER_STATION], dtype=np.float64).copy(),
                })
        if step % 20 == 0:
            print(f"  {family} w{width} windows {min((step + 1) * 4, len(flats))}/{len(flats)}", flush=True)
    pack = _as_pack(bucket)
    pack["subset"] = subset_note
    pack["n_records"] = len(flats)
    pack["center_rows"] = center_rows
    pack["width_m"] = width
    return pack


def _forward_sections(model, cache, device, freq) -> dict:
    from data import KIND_SECTION, make_window

    bucket = _empty_lists()
    meta = []
    subset_note = None
    indices = [i for i, sec in enumerate(cache.sections) if sec.split in ("val", "blind")]
    for count, index in enumerate(indices):
        sec = cache.sections[index]
        width = min(int(sec.lx), 2000)
        window = make_window(cache, "metric", KIND_SECTION, index, width, 0, 0, False)
        with torch.inference_mode():
            g = model(window[0].unsqueeze(0).to(device), window[1].unsqueeze(0).to(device)).float().cpu().numpy()[0]
        vs = np.asarray(sec.vs, dtype=np.float64)
        zeta = np.asarray(sec.zeta, dtype=np.float64)
        f1, f2 = float(sec.f1_hz), float(sec.f2_hz)
        cols = column_subset(vs.shape[1])
        if subset_note is None:
            subset_note = {"example": sec.name, "columns_rule": "linspace of 9 with the middle index forced to n//2"}
        log_c = _log(center_outcrop(freq, vs, zeta, f1, f2))
        log_p = _log(pretell_geomean(freq, vs, zeta, f1, f2, cols))
        prior = window[3].numpy()
        true = window[2].numpy()
        valid = window[4].numpy().astype(bool)
        x_abs = window[6].numpy()
        sid = sec.name.split("__")[-1]
        for station in np.flatnonzero(valid):
            _append_station(
                bucket, g_row=g[int(station)], true=true[int(station)], prior=prior[int(station)],
                sid=sid, station=int(station), cluster=str(sec.group_id), x_m=float(x_abs[int(station)]),
                log_center=log_c, log_pretell=log_p,
            )
        meta.append({
            "sample_id": sid, "split": sec.split, "width_m": int(sec.lx), "damping": sec.damping,
            "n_columns_scored": int(cols.size), "columns": cols.tolist(), "unpadded": False,
            "domain_flag": "pad_cropped_cache",
        })
        if count % 5 == 0:
            print(f"  manifold {count + 1}/{len(indices)} {sid}", flush=True)
    pack = _as_pack(bucket)
    pack["meta"] = pd.DataFrame(meta)
    pack["subset"] = subset_note
    pack["n_records"] = len(indices)
    return pack


def _metric_frame(long: pd.DataFrame, method: str, band: str, metric: str) -> pd.DataFrame:
    part = long[(long["method"] == method) & (long["band"] == band) & (long["metric"] == metric)]
    return part[["sample_id", "station", "cluster_id", "x_m", "value"]].copy()


def _pooled(long: pd.DataFrame, method: str, band: str) -> dict:
    model = _metric_frame(long, method, band, "g_log_l1")
    prior = _metric_frame(long, method, band, "prior_log_l1")
    merged = model.merge(prior, on=["sample_id", "station", "cluster_id", "x_m"], suffixes=("_model", "_prior"))
    values = merged["value_model"].to_numpy()
    clusters = merged["cluster_id"].to_numpy()
    skill = skill_of_pooled(values, merged["value_prior"].to_numpy())
    ci = bootstrap_independent(
        values, clusters, unit="master", statistic=np.mean, rng=np.random.default_rng(RNG_SEED),
    )
    return {
        "g_log_l1": float(ci["point"]),
        "ci_low": float(ci["low"]),
        "ci_high": float(ci["high"]),
        "ci_statistic": "mean",
        "bootstrap_unit": "master",
        "n_draw": int(ci["n_draw"]),
        "n_masters": int(ci["n_records"]),
        "n_values": int(values.size),
        "pooled_skill": np.nan if skill is None else float(skill),
        "skill_status": "pooled" if skill is not None else "suppressed_by_proposal_floor",
        "harm_rate": float(harm_rate(long, band, method)),
        "n_records": int(merged["sample_id"].nunique()),
        "n_stations": int(len(merged)),
    }


def _treasure_band(long: pd.DataFrame, method: str, band: str) -> dict:
    model = _metric_frame(long, method, band, "g_log_l1").sort_values("x_m")
    prior = _metric_frame(long, method, band, "prior_log_l1")
    merged = model.merge(prior, on=["sample_id", "station", "cluster_id", "x_m"], suffixes=("_model", "_prior"))
    block = bootstrap_moving_block(
        merged["value_model"].to_numpy(), merged["x_m"].to_numpy(), rng=np.random.default_rng(RNG_SEED),
    )
    skill = skill_of_pooled(merged["value_model"].to_numpy(), merged["value_prior"].to_numpy())
    return {
        "g_log_l1": float(np.mean(merged["value_model"])),
        "g_log_l1_median": float(block["point"]),
        "ci_low": float(block["low"]),
        "ci_high": float(block["high"]),
        "ci_statistic": "median",
        "bootstrap_unit": "moving block along x",
        "block_m": float(block["block_m"]),
        "n_blocks": int(block["n_blocks"]),
        "n_draw": int(block["n_draw"]),
        "n_masters": 1,
        "n_values": int(len(merged)),
        "pooled_skill": np.nan if skill is None else float(skill),
        "skill_status": "pooled" if skill is not None else "suppressed_by_proposal_floor",
        "harm_rate": float(harm_rate(long, band, method)),
        "n_records": 1,
        "n_stations": int(len(merged)),
    }


def _lookup_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    real = pd.read_csv(AUDIT / "realization.csv")
    real = real[real["table"] == "realization"]
    pod = pd.read_csv(AUDIT / "representation.csv")
    pod = pod[pod["table"] == "pod_floor"]
    noise = json.loads((AUDIT / "noise_floor.json").read_text())
    return real, pod, noise


def _realization_median(real: pd.DataFrame, family: str, band: str) -> float | None:
    part = real[(real["family"] == family) & (real["band"] == band)]
    if part.empty:
        return None
    return float(part["value"].median())


def _pod_value(pod: pd.DataFrame, pool: str, band: str) -> float | None:
    part = pod[(pod["pool"] == pool) & (pod["band"] == band)]
    if part.empty or "pod_floor_l1" not in part.columns:
        return None
    return float(part["pod_floor_l1"].iloc[0])


def _ratio(prior: float, model: float, floors: list[float]) -> tuple[float | None, str]:
    finite = [float(item) for item in floors if item is not None and np.isfinite(item)]
    if not finite:
        return None, "omitted_no_measured_floor"
    cap = max(finite)
    gap = float(prior) - cap
    if gap <= 0:
        return None, "omitted_nonpositive_gap_fe_unmeasured"
    return (float(prior) - float(model)) / gap, "provisional_fe_unmeasured"


def _provenance(path: Path, **fields) -> None:
    payload = {
        "label_campaign": fields["label_campaign"],
        "oned_engine": fields["oned_engine"],
        "outcrop_convention": fields["outcrop_convention"],
        "damping_source": fields["damping_source"],
        "fft_window_s": fields.get("fft_window_s"),
        "checkpoint": fields.get("checkpoint"),
        "epoch": fields.get("epoch"),
        "dtype": fields["dtype"],
        "bands": list(BAND_NAMES),
        "aggregation_unit": fields["aggregation_unit"],
        "sample_id_hash": sample_id_hash(fields["sample_ids"]),
        "git_commit": git_commit(),
        "definition_difference": fields.get("definition_difference", ""),
    }
    payload.update({key: value for key, value in fields.items() if key not in payload and key != "sample_ids"})
    write_provenance(path, payload)


def _annotate(frame: pd.DataFrame, **columns) -> pd.DataFrame:
    out = frame.copy()
    for key, value in columns.items():
        out[key] = value
    return out


def _save_center(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    np.savez_compressed(
        path,
        run=np.asarray([row["run"] for row in rows], dtype=np.int64),
        sample_id=np.asarray([row["sample_id"] for row in rows]),
        cluster_id=np.asarray([row["cluster_id"] for row in rows]),
        station=np.asarray([row["station"] for row in rows], dtype=np.int64),
        x_m=np.asarray([row["x_m"] for row in rows], dtype=np.float64),
        log_pred=np.stack([row["log_pred"] for row in rows]),
        log_true=np.stack([row["log_true"] for row in rows]),
        log_prior=np.stack([row["log_prior"] for row in rows]),
    )


def _rung_rows(long: pd.DataFrame, *, rung: str, order: int, family: str, split: str, width_m,
               unpadded: bool, domain_flag: str, accuracy: str, statistic_fn) -> list[dict]:
    rows = []
    for band in BAND_NAMES:
        summary = statistic_fn(long, "P128n128", band)
        claim = accuracy
        status = "scored"
        if family == "treasure_island" and band == "low":
            claim = "not_an_accuracy_result"
            status = "not_an_accuracy_result"
        elif family == "treasure_island" and band == "overall":
            claim = "not_an_accuracy_result"
            status = "includes_unscored_low_band"
        elif split == "val":
            claim = "selection_split"
            status = "selection_split"
        rows.append({
            "order": order,
            "rung": rung,
            "family": family,
            "split": split,
            "width_m": width_m,
            "band": band,
            "aggregation_unit": "master" if family != "treasure_island" else "moving block along x",
            "skill_reduction": "pooled",
            "unpadded": unpadded,
            "domain_flag": domain_flag,
            "accuracy_claim": claim,
            "status": status,
            **summary,
        })
    return rows


def _baseline_rows(long, rung, family, split, width_m, real, pod, pod_pool, real_family) -> list[dict]:
    rows = []
    for band in BAND_NAMES:
        for method, kind in (
            ("g0", "prediction"),
            ("center_column", "prediction"),
            ("pretell_geomean", "prediction"),
        ):
            if family == "treasure_island":
                summary = _treasure_band(long, method, band)
            else:
                summary = _pooled(long, method, band)
            rows.append({
                "rung": rung, "family": family, "split": split, "width_m": width_m, "band": band,
                "baseline": method, "kind": kind, "status": "scored",
                "g_log_l1": summary["g_log_l1"], "ci_low": summary["ci_low"], "ci_high": summary["ci_high"],
                "ci_statistic": summary["ci_statistic"], "bootstrap_unit": summary["bootstrap_unit"],
                "aggregation_unit": summary["bootstrap_unit"] if family == "treasure_island" else "master",
            })
        real_value = _realization_median(real, real_family, band) if real_family else None
        rows.append({
            "rung": rung, "family": family, "split": split, "width_m": width_m, "band": band,
            "baseline": "realization_floor", "kind": "floor",
            "status": "measured" if real_value is not None else "not_available",
            "g_log_l1": real_value, "ci_low": np.nan, "ci_high": np.nan,
            "ci_statistic": "", "bootstrap_unit": "",
            "aggregation_unit": "master" if family != "treasure_island" else "moving block along x",
            "note": "Median across Sobol groups of the Phase 1.5 pairwise |g| spread." if real_value is not None else "No seed replicates for this rung.",
        })
        if pod_pool is None:
            pod_status, pod_value, note = "not_stratified_by_width", None, "representation.csv is not stratified by width. The split pool is on the family error-budget row."
        else:
            pod_value = _pod_value(pod, pod_pool, band)
            pod_status = "measured" if pod_value is not None else "not_available"
            note = f"Phase 1.6 pool {pod_pool}. Not recomputed."
        rows.append({
            "rung": rung, "family": family, "split": split, "width_m": width_m, "band": band,
            "baseline": "pod_floor", "kind": "floor", "status": pod_status,
            "g_log_l1": pod_value, "ci_low": np.nan, "ci_high": np.nan,
            "ci_statistic": "", "bootstrap_unit": "",
            "aggregation_unit": "master" if family != "treasure_island" else "moving block along x",
            "note": note,
        })
        rows.append({
            "rung": rung, "family": family, "split": split, "width_m": width_m, "band": band,
            "baseline": "fe_floor", "kind": "floor", "status": "unmeasured",
            "g_log_l1": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "ci_statistic": "", "bootstrap_unit": "",
            "aggregation_unit": "master" if family != "treasure_island" else "moving block along x",
            "note": "Phase 1.3 did not measure a mesh-refinement or repeated-solve log-L1. 0.000228 is a lower bound on max(FE, 1D), not this floor.",
        })
    return rows


def _treasure_pack(freq: np.ndarray) -> dict:
    import h5py
    import hdf5plugin  # noqa: F401
    from applied_damping import applied_zeta

    fields = np.load(TREASURE_FIELDS)
    one = np.load(TREASURE_G)
    if not np.allclose(np.asarray(one["x"]), np.asarray(fields["x"])):
        raise RuntimeError("treasure_g.npz stations do not match fields.npz")
    column = np.asarray(fields["column"], dtype=np.float64)
    truth = np.asarray(fields["truth"], dtype=np.float64)
    g = np.asarray(one["g"], dtype=np.float64)
    x = np.asarray(fields["x"], dtype=np.float64)
    log_prior = _log(column)
    log_true = _log(truth)
    log_pred = log_prior + g
    with h5py.File(FULL_QVS, "r") as handle:
        vs = np.asarray(handle["Vs_realization_2D"][:], dtype=np.float64)
        zeta = applied_zeta(handle)
        attrs = handle["params"].attrs
        f1 = float(attrs["damping_freq_first"])
        f2 = float(attrs["damping_freq_second"])
    cols = column_subset(vs.shape[1])
    log_center = np.repeat(_log(center_outcrop(freq, vs, zeta, f1, f2))[None, :], x.size, axis=0)
    log_pretell = np.repeat(_log(pretell_geomean(freq, vs, zeta, f1, f2, cols))[None, :], x.size, axis=0)
    n = x.size
    return {
        "log_pred": log_pred,
        "log_true": log_true,
        "log_prior": log_prior,
        "log_center": log_center,
        "log_pretell": log_pretell,
        "sample_id": np.array(["treasure_island"] * n),
        "station": np.arange(n),
        "cluster_id": np.array(["treasure_island"] * n),
        "x_m": x,
        "subset": {"n_columns": int(vs.shape[1]), "columns": cols.tolist(), "center": int(vs.shape[1] // 2), "source": "full_qvs.h5"},
        "f1": f1,
        "f2": f2,
        "recomputed_forward": False,
    }


def _g_stack_l1(g: np.ndarray, freq: np.ndarray, band: str) -> float:
    return float(g_log_l1(g, np.zeros_like(g), freq, band))


def _homogeneous(model, device, freq) -> list[dict]:
    from data import K_MAX
    from encoding import encode, query_columns

    cases = []
    for vs_value in (200.0, 400.0, 800.0):
        n_layer, width = 80, 500
        vs = np.full((n_layer, width), vs_value, dtype=np.float64)
        zeta = np.full((n_layer, width), 0.02, dtype=np.float64)
        encoded = encode("metric", vs, zeta, n_layer, f1_hz=1.0, f2_hz=3.0)
        query = np.zeros(K_MAX, dtype=np.float32)
        query[:1] = query_columns("metric", np.array([width / 2.0]), width).astype(np.float32)
        with torch.inference_mode():
            g = model(
                torch.from_numpy(np.ascontiguousarray(encoded)).unsqueeze(0).to(device),
                torch.from_numpy(query).unsqueeze(0).to(device),
            ).float().cpu().numpy()[0, 0]
        cases.append(g)
        print(f"  homogeneous Vs {vs_value}", flush=True)
    rows = []
    for band in BAND_NAMES:
        values = np.array([_g_stack_l1(g, freq, band) for g in cases])
        median = float(np.median(values))
        rows.append({
            "test": "homogeneous_column",
            "family": "synthetic_column",
            "width_m": 500,
            "band": band,
            "n": int(values.size),
            "value": median,
            "reference": PROPOSAL,
            "reference_name": "gates.NOISE_FLOOR_PLACEHOLDER_LOG_L1",
            "passed": bool(median < PROPOSAL),
            "status": "scored",
            "note": "Median over Vs 200, 400, 800 m/s, zeta 0.02, one center query. L1(|g_pred|) against a zero residual. No new OpenSees label.",
        })
    return rows


def _cov0(model, cache, device, freq) -> list[dict]:
    from data import make_cov0_window

    flats = [int(flat) for flat in cache.runs("test", ("iid",))]
    model_l1 = {band: [] for band in BAND_NAMES}
    zero_l1 = {band: [] for band in BAND_NAMES}
    for step, chunk in enumerate(_batches(flats, 4)):
        windows = [make_cov0_window(cache, "metric", flat, width=500) for flat in chunk]
        x = torch.stack([item[0] for item in windows])
        query = torch.stack([item[1] for item in windows])
        with torch.inference_mode():
            g = model(x.to(device), query.to(device)).float().cpu().numpy()
        for local in range(len(chunk)):
            valid = windows[local][4].numpy().astype(bool)
            keep = np.flatnonzero(valid)
            # Every station is the same column. The center station is the probe.
            station = CENTER_STATION if CENTER_STATION in set(keep.tolist()) else int(keep[len(keep) // 2])
            row = g[local, station]
            zeros = np.zeros_like(row)
            for band in BAND_NAMES:
                model_l1[band].append(g_log_l1(row, zeros, freq, band))
                zero_l1[band].append(prior_log_l1(zeros, freq, band))
        if step % 20 == 0:
            print(f"  cov0 {min((step + 1) * 4, len(flats))}/{len(flats)}", flush=True)
    rows = []
    for band in BAND_NAMES:
        model_mean = float(np.mean(model_l1[band]))
        zero_mean = float(np.mean(zero_l1[band]))
        rows.append({
            "test": "cov0",
            "family": "iid",
            "width_m": 500,
            "band": band,
            "n": len(flats),
            "value": model_mean,
            "reference": zero_mean,
            "reference_name": "g0_log_l1",
            "passed": bool(model_mean <= zero_mean),
            "status": "scored",
            "note": "Every IID test record via data.make_cov0_window. The label equals the prior, so g=0 has log-L1 0. Center station of the repeated column. Not a new solve.",
        })
    return rows


def _mirror(model, cache, device, freq) -> list[dict]:
    from data import IID_MESH_M, KIND_RUN, make_window

    flats = [int(flat) for flat in cache.runs("test", ("iid",))]
    width = 500
    start = int(IID_MESH_M // 2 - width // 2)
    per_band = {band: [] for band in BAND_NAMES}
    typical = {band: [] for band in BAND_NAMES}
    matched = 0
    for step, chunk in enumerate(_batches(flats, 4)):
        plain = [make_window(cache, "metric", KIND_RUN, flat, width, start, 0, False) for flat in chunk]
        mirrored = [make_window(cache, "metric", KIND_RUN, flat, width, start, 0, True) for flat in chunk]
        with torch.inference_mode():
            g_plain = model(torch.stack([item[0] for item in plain]).to(device), torch.stack([item[1] for item in plain]).to(device)).float().cpu().numpy()
            g_mirror = model(torch.stack([item[0] for item in mirrored]).to(device), torch.stack([item[1] for item in mirrored]).to(device)).float().cpu().numpy()
        for local in range(len(chunk)):
            # make_window keeps the station slot and queries width - x on the reversed field,
            # so slot i is the image of the original station i.
            valid = plain[local][4].numpy().astype(bool)
            true = plain[local][2].numpy()
            prior = plain[local][3].numpy()
            for station in np.flatnonzero(valid):
                matched += 1
                g_true = np.asarray(true[station] - prior[station], dtype=np.float64)
                for band in BAND_NAMES:
                    per_band[band].append(g_log_l1(g_plain[local, station], g_mirror[local, station], freq, band))
                    typical[band].append(prior_log_l1(g_true, freq, band))
        if step % 20 == 0:
            print(f"  mirror {min((step + 1) * 4, len(flats))}/{len(flats)}", flush=True)
    rows = []
    for band in BAND_NAMES:
        discrepancy = float(np.median(per_band[band]))
        scale = float(np.median(typical[band]))
        cutoff = float(MIRROR_L1_FRACTION) * scale
        rows.append({
            "test": "mirror",
            "family": "iid",
            "width_m": 500,
            "band": band,
            "n": int(len(per_band[band])),
            "value": discrepancy,
            "reference": cutoff,
            "reference_name": "0.1 * median L1(g_true)",
            "passed": bool(discrepancy <= cutoff),
            "status": "scored",
            "note": f"Same packed station slot on the reversed field, which make_window queries at width - x. Median L1(|g - g_mirror|) over {matched} stations. Cutoff is gates.MIRROR_L1_FRACTION.",
        })
    return rows


def _context(model, cache, device, freq) -> list[dict]:
    from data import IID_MESH_M, KIND_RUN, make_window

    rows = []
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        flats = [int(flat) for flat in cache.runs("test", (family,))]
        for width in (500, 1000, 1500):
            start = int(IID_MESH_M // 2 - width // 2)
            totals = {band: [] for band in BAND_NAMES}
            for chunk in _batches(flats, 4):
                windows = [make_window(cache, "metric", KIND_RUN, flat, width, start, 0, False) for flat in chunk]
                with torch.inference_mode():
                    g = model(
                        torch.stack([item[0] for item in windows]).to(device),
                        torch.stack([item[1] for item in windows]).to(device),
                    ).float().cpu().numpy()
                for local in range(len(chunk)):
                    true = windows[local][2].numpy()
                    prior = windows[local][3].numpy()
                    valid = windows[local][4].numpy().astype(bool)
                    for station in np.flatnonzero(valid):
                        residual = np.asarray(true[station] - prior[station], dtype=np.float64)
                        pred = np.asarray(g[local, station], dtype=np.float64)
                        for band in BAND_NAMES:
                            totals[band].append(g_log_l1(pred, residual, freq, band))
            for band in BAND_NAMES:
                rows.append({
                    "test": "context_invariance",
                    "family": family,
                    "width_m": width,
                    "band": band,
                    "n": len(flats),
                    "value": float(np.mean(totals[band])),
                    "reference": np.nan,
                    "reference_name": "",
                    "passed": "",
                    "status": "compatibility",
                    "note": "Same test records and the same 21 labeled stations. The cache Vs strip is 500 columns; a wider window repeats the edge column. This is context invariance, not a lateral-accuracy result.",
                })
            print(f"  context {family} {width}", flush=True)
    return rows


def _epoch_table(freq: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    notes = []
    missing = []
    for family, path in (
        ("iid", FROZEN / "center_iid.npz"),
        ("ood_dipping", FROZEN / "center_ood_dipping.npz"),
    ):
        blob = np.load(path, allow_pickle=True)
        runs = np.asarray(blob["run"]).astype(int)
        long_model = score_curves(
            sample_id=np.asarray(blob["sample_id"]),
            station=np.asarray(blob["station"]),
            cluster_id=np.asarray(blob["cluster_id"]),
            freq=freq,
            log_pred=np.asarray(blob["log_pred"]),
            log_true=np.asarray(blob["log_true"]),
            log_prior=np.asarray(blob["log_prior"]),
            method="P128n128",
            x_m=np.asarray(blob["x_m"]),
        )
        ensemble_path = ENSEMBLE_DIR / f"ensemble_center_tf_{family}.npz"
        ens = np.load(ensemble_path)
        order = {int(run): i for i, run in enumerate(np.asarray(ens["run"]))}
        take = [order[int(run)] for run in runs]
        if len(take) != len(runs):
            raise RuntimeError(f"ensemble center runs do not cover {family}")
        tf = np.asarray(ens["tf"], dtype=np.float64)[take]
        long_ens = score_curves(
            sample_id=np.asarray(blob["sample_id"]),
            station=np.asarray(blob["station"]),
            cluster_id=np.asarray(blob["cluster_id"]),
            freq=freq,
            log_pred=_log(tf),
            log_true=np.asarray(blob["log_true"]),
            log_prior=np.asarray(blob["log_prior"]),
            method="ensemble_mean_g",
            x_m=np.asarray(blob["x_m"]),
        )
        both = pd.concat([long_model, long_ens], ignore_index=True)
        left = aggregate(long_model, "master")
        right = aggregate(long_ens, "master")
        assert_comparable(left, right)
        both.insert(0, "family", family)
        frames.append(both)
        for method in ("P128n128", "ensemble_mean_g"):
            for band in BAND_NAMES:
                summary = _pooled(both, method, band)
                summary.update({
                    "family": family,
                    "method": method,
                    "band": band,
                    "station": "center index 10",
                    "aggregation_unit": "master",
                    "skill_reduction": "pooled",
                    "status": "scored",
                    "checkpoint": str(CKPT) if method == "P128n128" else "stored ensemble_center_tf; _mean_g was not re-run",
                    "epoch": 46 if method == "P128n128" else "P128 58, P128s1 74, P128n128 46, P128c 30",
                })
                notes.append(summary)
    present = sorted(path.name for path in CKPT.glob("*"))
    for epoch in (43, 44, 45, 47, 48, 49):
        missing.append({
            "family": "iid,ood_dipping",
            "method": f"P128n128_epoch_{epoch}",
            "band": "",
            "status": "not_on_disk",
            "aggregation_unit": "master",
            "skill_reduction": "pooled",
            "note": f"P128n128 contains {present}. Epoch {epoch} was not trained for this phase.",
        })
    for family in ("ood_three_layer", "manifold", "treasure_island"):
        missing.append({
            "family": family,
            "method": "ensemble_mean_g",
            "band": "",
            "status": "spectra_not_on_disk",
            "aggregation_unit": "master",
            "skill_reduction": "pooled",
            "note": "Stored four-checkpoint center curves exist for IID and dipping only. The four models were not re-run.",
        })
    missing.append({
        "family": "stage1",
        "method": "stage1",
        "band": "",
        "status": "no_spectra_not_invented",
        "aggregation_unit": "",
        "skill_reduction": "mean_of_unit_skills_in_eval_test_json_only",
        "note": "Stage 1 has no spectra on disk. eval_test.json is mean-of-skills and is not copied into this table.",
    })
    detail = pd.concat(frames, ignore_index=True)
    return pd.DataFrame(notes + missing), detail


def _error_budget(ladder: pd.DataFrame, baselines: pd.DataFrame, pod: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rungs = [
        ("iid", "iid:test", "iid"),
        ("ood_dipping", "ood_dipping:test", "ood_dipping"),
        ("ood_three_layer", "ood_three_layer:test", "ood_three_layer"),
        ("manifold_blind", "manifold:blind", None),
        ("treasure_island", "treasure_island", None),
    ]
    for family, pool, real_family in rungs:
        part = ladder[ladder["family"] == family]
        base = baselines[baselines["family"] == family]
        for band in BAND_NAMES:
            model_row = part[part["band"] == band]
            if model_row.empty:
                continue
            model = model_row.iloc[0]
            def _base(name: str):
                hit = base[(base["band"] == band) & (base["baseline"] == name)]
                if family == "manifold_blind":
                    hit = hit[hit["rung"] == "manifold_blind"]
                if hit.empty:
                    return None, "not_available"
                value = hit.iloc[0]["g_log_l1"]
                status = str(hit.iloc[0]["status"])
                if pd.isna(value):
                    return None, status
                return float(value), status

            prior, prior_status = _base("g0")
            pod_value = _pod_value(pod, pool, band)
            real_value, real_status = _base("realization_floor")
            floors = []
            floor_note = []
            if real_value is not None:
                floors.append(real_value)
            else:
                floor_note.append(real_status or "realization_not_available")
            if pod_value is not None:
                floors.append(pod_value)
            else:
                floor_note.append("pod_not_available")
            ratio, ratio_status = (None, "omitted_no_model")
            if prior is not None:
                ratio, ratio_status = _ratio(prior, float(model["g_log_l1"]), floors)
            claim = str(model["accuracy_claim"])
            rows.append({
                "family": family,
                "band": band,
                "prior_g_log_l1": prior,
                "model_g_log_l1": float(model["g_log_l1"]),
                "model_ci_low": float(model["ci_low"]),
                "model_ci_high": float(model["ci_high"]),
                "realization_floor": real_value,
                "realization_status": real_status,
                "pod_floor": pod_value,
                "pod_pool": pool,
                "fe_floor": np.nan,
                "fe_status": "unmeasured",
                "ratio": ratio,
                "ratio_status": ratio_status,
                "floor_notes": ";".join(floor_note),
                "accuracy_claim": claim,
                "aggregation_unit": model["aggregation_unit"],
                "skill_reduction": "pooled",
            })
    return pd.DataFrame(rows)


def _figure(budget: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    families = ["iid", "ood_dipping", "ood_three_layer", "manifold_blind", "treasure_island"]
    labels = ["IID", "Dipping", "Three-layer", "Manifold blind", "Treasure Island"]
    bands = list(BAND_NAMES)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharey=False)
    series = (
        ("prior_g_log_l1", "1D prior"),
        ("model_g_log_l1", "P128n128"),
        ("realization_floor", "Realization floor"),
        ("pod_floor", "POD floor"),
    )
    for ax, band in zip(axes.ravel(), bands):
        part = budget[budget["band"] == band].set_index("family")
        x = np.arange(len(families))
        width = 0.18
        for i, (column, label) in enumerate(series):
            vals = []
            for family in families:
                if family not in part.index or pd.isna(part.loc[family, column]):
                    vals.append(np.nan)
                else:
                    vals.append(float(part.loc[family, column]))
            ax.bar(x + (i - 1.5) * width, vals, width=width, label=label if band == "low" else None)
        title = band
        if band == "low":
            title = "low (Treasure Island is not an accuracy result)"
        if band == "overall":
            title = "overall (Treasure Island includes the unscored low band)"
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("log-L1")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Error budget. FE floor is unmeasured and is not a bar. Ratios in the CSV are provisional.", y=1.02)
    fig.tight_layout()
    fig.savefig(FROZEN / "error_budget.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _beats(ladder: pd.DataFrame, baselines: pd.DataFrame, family: str, band: str) -> dict:
    model = ladder[(ladder["family"] == family) & (ladder["band"] == band) & (ladder["split"] == "test")]
    if model.empty:
        model = ladder[(ladder["family"] == family) & (ladder["band"] == band)]
    model = model.iloc[0]
    out = {"family": family, "band": band, "model": float(model["g_log_l1"]), "ci_high": float(model["ci_high"])}
    for name in ("g0", "center_column", "pretell_geomean"):
        hit = baselines[(baselines["family"] == family) & (baselines["band"] == band) & (baselines["baseline"] == name) & (baselines["split"] == model["split"])]
        value = float(hit.iloc[0]["g_log_l1"])
        out[name] = value
        out[f"point_below_{name}"] = bool(out["model"] < value)
        out[f"ci_below_{name}"] = bool(out["ci_high"] < value)
    out["beats_b_point"] = out["point_below_center_column"] and out["point_below_pretell_geomean"]
    out["beats_b_ci"] = out["ci_below_center_column"] and out["ci_below_pretell_geomean"]
    return out


def _write_g2(ladder, baselines, budget, consistency, epoch, noise, decision: dict) -> None:
    lines = [
        "# G2",
        "",
        decision["paragraph"],
        "",
        "## Rule",
        "",
        "A positive headline requires the P128n128 pooled mean g log-L1 on the IID test and on the dipping test to sit below both baseline (b) components, the center-column outcrop and the Pretell geometric-mean outcrop, on the mid band (0.4–2.5 Hz) and the high band (2.5–10 Hz). The master-bootstrap upper end is reported beside that comparison. The low band is stated separately and is not part of the positive test. Lateral accuracy is outside this gate. Treasure Island below 0.4 Hz is outside this gate. `gates.G2_POSITIVE_BANDS` is mid and high.",
        "",
        f"Decision code: `{decision['code']}`.",
        "",
        "## Ladder",
        "",
        "Aggregation is one reduction per file. `ladder.csv` pools stations and resamples Sobol groups or section ids (`bootstrap_unit=master`, skill of the pooled mean). `ladder_treasure.csv` is one section. Its interval is `bootstrap_moving_block`, which returns the median, and the pooled mean is a separate column. Skill uses `gates.noise_floor_log_l1()` (the proposal 0.0136). Absolute log-L1 is reported where that floor suppresses skill.",
        "",
        f"IID, dipping, and three-layer test counts: {decision['counts']}. Every test record is scored at every labeled station (21 stations, 15 m). Manifold val and blind sections that the cache loads are scored at every station inside the margin. Troughs are absent from that cache ({decision['dropped_troughs']} dropped). Manifold interiors are pad-cropped. Round-0 unpadded 2000 m sections are a status row and were not scored. `full_svlab.h5` was not scored.",
        "",
        "## Baselines",
        "",
        "Baseline (a) is g=0, the stored per-station prior. Baseline (b) is the center-column elastic outcrop from `matched_physics.outcrop_many` and a 9-column geometric mean from `section_1d.surface_over_outcrop`. The column index list is in the sidecar. Rigid-base arms were not read. Realization and POD numbers are the Phase 1 tables. The FE floor is unmeasured on every row. The headline ratio uses max of the measured floors only and is marked provisional because an unmeasured FE floor could change it. The ratio is omitted when that gap is not positive.",
        "",
        "## What the low band and the widths are",
        "",
        f"The measured 1D-convention floor in the audit is {noise['measured_noise_floor']} on the high band, a lower bound on max(FE, 1D) because the FE floor was not measured. Halving the 15 s Treasure Island record moves 0.1–0.4 Hz by {noise['record_length_leakage']['low']['full_qvs.h5@7.490000000223517']:.3f}. Treasure Island low-band and overall rows are labeled not an accuracy result. Flat width rows and the synthetic 500/1000/1500 m context probe repeat the edge of a 500 m strip or describe pad-cropped sections. They are compatibility.",
        "",
        "## Epoch 46",
        "",
        "Epoch 46 is the stored `best_model.pt` (`val_score` 0.04987). The selection that stored it averages IID and manifold validation log-L1. Blind was not in that score. `history.csv` is absent, so this phase does not say epoch 46 is the minimum of a curve. Neighboring epochs 43–45 and 47–49 are not on disk and were not trained. The four-checkpoint mean is the stored IID and dipping center curves. Those curves were not recomputed. Three-layer, manifold, and Treasure Island have no stored ensemble spectra. Stage 1 has no spectra. No validation horse-race among the four checkpoints is stated.",
        "",
        "## Consistency",
        "",
        "Cutoffs are the proposals in `gates.py`: homogeneous median L1(|g_pred|) against 0.0136, mirror station-matched L1 against 0.1 times the median L1(g_true), monotone sign agreement 0.80. Monotone pairs were not run: existing records do not hold a pair that changes only CoV or only dip. CoV=0 uses `make_cov0_window` on every IID test record. Context invariance is the compatibility block.",
        "",
    ]
    (FROZEN / "G2.md").write_text("\n".join(lines))


def _decide(ladder, baselines, noise) -> dict:
    checks = []
    for family in ("iid", "ood_dipping"):
        for band in G2_POSITIVE_BANDS:
            checks.append(_beats(ladder, baselines, family, band))
    point_ok = all(row["beats_b_point"] for row in checks)
    ci_ok = all(row["beats_b_ci"] for row in checks)
    if point_ok and ci_ok:
        code = "positive_in_family_0p4_10"
        paragraph = (
            "Paper 2's headline is positive on the in-family 500 m tests and is not a lateral-accuracy result. "
            "On both the IID test and the dipping test, the P128n128 pooled mean g log-L1 and its master-bootstrap upper end "
            "are below the center-column outcrop and below the 9-column Pretell geometric-mean outcrop in 0.4–2.5 Hz and in 2.5–10 Hz. "
            "The 0.1–0.4 Hz band is a separate absolute-error report and is not part of that claim. "
            "Treasure Island below 0.4 Hz is not an accuracy result. Manifold widths and the synthetic wider windows are compatibility: "
            "the training strip is 500 m, and those wider windows do not establish lateral accuracy."
        )
    elif point_ok:
        code = "compatibility_interval_overlaps_baseline_b"
        paragraph = (
            "Paper 2's headline is scoped to compatibility. The P128n128 point estimate on the IID and dipping 500 m tests "
            "is below both baseline (b) components in 0.4–2.5 Hz and in 2.5–10 Hz, and the master-bootstrap interval still reaches "
            "at least one of those baselines, so the positive gate is not met. The 0.1–0.4 Hz band is reported separately. "
            "Treasure Island below 0.4 Hz is not an accuracy result. Width rows are compatibility, not lateral accuracy."
        )
    else:
        code = "compatibility"
        paragraph = (
            "Paper 2's headline is scoped to compatibility. On the IID and dipping 500 m tests the frozen P128n128 model does not "
            "beat both baseline (b) components, the center-column outcrop and the Pretell geometric-mean outcrop, on both "
            "0.4–2.5 Hz and 2.5–10 Hz. The 0.1–0.4 Hz band is reported as absolute error and is not an accuracy claim for Treasure Island. "
            "Manifold widths, round-0 sections, and the synthetic wider windows are not a lateral-accuracy result."
        )
    return {"code": code, "paragraph": paragraph, "checks": checks, "leakage_low": noise["record_length_leakage"]["low"]["full_qvs.h5@7.490000000223517"]}


def _pool_manifold(long: pd.DataFrame, meta: pd.DataFrame, split: str) -> pd.DataFrame:
    ids = set(meta.loc[meta["split"] == split, "sample_id"])
    return long[long["sample_id"].isin(ids)].copy()


def main() -> None:
    from data import Cache

    FROZEN.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"device {device}", flush=True)
    from diag_forward import load_fp32

    model, stamp = load_fp32(CKPT, device)
    model.eval()
    epoch = int(stamp["epoch"])
    if epoch != 46:
        raise RuntimeError(f"expected epoch 46, found {epoch}")
    cache = Cache(CACHE)
    freq = np.asarray(cache.freq, dtype=np.float64)
    real, pod, noise = _lookup_tables()
    print(f"epoch {epoch} sections {len(cache.sections)} dropped troughs {len(cache.dropped_troughs)}", flush=True)

    run_long = {}
    subsets = {}
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        print(f"forward {family}", flush=True)
        pack = _forward_runs(model, cache, family, device, freq, 500)
        long = _score_pack(pack, freq)
        long.to_csv(FROZEN / f"stations_{family}.csv", index=False)
        run_long[family] = long
        subsets[family] = pack["subset"]
        if pack["center_rows"] and family in ("iid", "ood_dipping"):
            _save_center(FROZEN / f"center_{family}.npz", pack["center_rows"])
        print(f"  {family} records {pack['n_records']} stations {len(pack['sample_id'])}", flush=True)

    print("forward manifold", flush=True)
    sec_pack = _forward_sections(model, cache, device, freq)
    sec_long = _score_pack(sec_pack, freq)
    meta = sec_pack["meta"]
    sec_long = sec_long.merge(meta[["sample_id", "split", "width_m", "damping"]], on="sample_id", how="left")
    sec_long.to_csv(FROZEN / "stations_manifold.csv", index=False)
    meta.to_csv(FROZEN / "manifold_sections.csv", index=False)
    subsets["manifold"] = sec_pack["subset"]

    print("treasure from stored g", flush=True)
    treasure = _treasure_pack(freq)
    treasure_long = _score_pack(treasure, freq)
    treasure_long.to_csv(FROZEN / "stations_treasure.csv", index=False)
    subsets["treasure_island"] = treasure["subset"]

    ladder_rows = []
    base_rows = []
    specs = [
        ("iid", "iid_test", 1, "iid", "test", 500, False, "solved_500m_strip", "in_family", "iid:test", "iid"),
        ("ood_dipping", "dipping_test", 2, "ood_dipping", "test", 500, False, "solved_500m_strip", "in_family", "ood_dipping:test", "ood_dipping"),
        ("ood_three_layer", "three_layer_test", 3, "ood_three_layer", "test", 500, False, "solved_500m_strip", "in_family", "ood_three_layer:test", "ood_three_layer"),
    ]
    for key, rung, order, family, split, width, unpadded, flag, claim, pool, real_family in specs:
        ladder_rows.extend(_rung_rows(
            run_long[key], rung=rung, order=order, family=family, split=split, width_m=width,
            unpadded=unpadded, domain_flag=flag, accuracy=claim, statistic_fn=_pooled,
        ))
        base_rows.extend(_baseline_rows(run_long[key], rung, family, split, width, real, pod, pool, real_family))

    order = 4
    for split in ("blind", "val"):
        for width in (500, 1000, 1500, 2000):
            ids = set(meta.loc[(meta["split"] == split) & (meta["width_m"] == width), "sample_id"])
            part = sec_long[sec_long["sample_id"].isin(ids)]
            rung = f"manifold_{split}_w{width}"
            if part.empty:
                ladder_rows.append({
                    "order": order, "rung": rung, "family": "manifold", "split": split, "width_m": width,
                    "band": "", "aggregation_unit": "master", "skill_reduction": "pooled",
                    "unpadded": False, "domain_flag": "pad_cropped_cache", "accuracy_claim": "not_scored",
                    "status": "no_sections_in_cache", "g_log_l1": np.nan,
                })
            else:
                ladder_rows.extend(_rung_rows(
                    part, rung=rung, order=order, family="manifold", split=split, width_m=width,
                    unpadded=False, domain_flag="pad_cropped_cache",
                    accuracy="small_n_compatibility" if split == "blind" else "selection_split",
                    statistic_fn=_pooled,
                ))
                base_rows.extend(_baseline_rows(part, rung, "manifold", split, width, real, pod, None, None))
            order += 1

    for split, pool in (("blind", "manifold:blind"), ("val", "manifold:val")):
        pooled = _pool_manifold(sec_long, meta, split)
        rung = f"manifold_{split}"
        ladder_rows.extend(_rung_rows(
            pooled, rung=rung, order=order, family="manifold_blind" if split == "blind" else "manifold_val",
            split=split, width_m="pooled", unpadded=False, domain_flag="pad_cropped_cache",
            accuracy="small_n_compatibility" if split == "blind" else "selection_split",
            statistic_fn=_pooled,
        ))
        base_rows.extend(_baseline_rows(pooled, rung, "manifold_blind" if split == "blind" else "manifold_val", split, "pooled", real, pod, pool, None))
        order += 1

    ladder_rows.append({
        "order": order,
        "rung": "round0_2000",
        "family": "round0",
        "split": "",
        "width_m": 2000,
        "band": "",
        "aggregation_unit": "master",
        "skill_reduction": "pooled",
        "unpadded": True,
        "domain_flag": "unpadded_absorbing_boundary_inside_window",
        "accuracy_claim": "not_a_clean_domain",
        "status": "not_scored_unpadded",
        "g_log_l1": np.nan,
        "note": "Round-0 2000 m sections have no free-field pad. They were not scored.",
    })
    ladder_rows.append({
        "order": order + 1,
        "rung": "full_svlab",
        "family": "full_svlab",
        "split": "",
        "width_m": "",
        "band": "",
        "aggregation_unit": "",
        "skill_reduction": "pooled",
        "unpadded": "",
        "domain_flag": "same_surface_motion_as_full_qvs",
        "accuracy_claim": "not_a_second_solve",
        "status": "not_scored",
        "g_log_l1": np.nan,
        "note": "full_svlab.h5 is not a second solve and was not scored.",
    })

    treasure_rows = _rung_rows(
        treasure_long, rung="treasure_island", order=1, family="treasure_island", split="blind",
        width_m=4000, unpadded=False, domain_flag="one_15s_section_scored_from_0p4_Hz",
        accuracy="scored_band", statistic_fn=_treasure_band,
    )
    base_rows.extend(_baseline_rows(
        treasure_long, "treasure_island", "treasure_island", "blind", 4000, real, pod, "treasure_island", None,
    ))

    ladder = pd.DataFrame(ladder_rows).sort_values(["order", "band"], kind="mergesort")
    treasure_table = pd.DataFrame(treasure_rows)
    baselines = pd.DataFrame(base_rows)
    ladder.to_csv(FROZEN / "ladder.csv", index=False)
    treasure_table.to_csv(FROZEN / "ladder_treasure.csv", index=False)
    baselines.to_csv(FROZEN / "baselines.csv", index=False)

    print("error budget", flush=True)
    # The family error-budget rows use the family ladder rows. Manifold blind pooled is included.
    budget_ladder = pd.concat([
        ladder[ladder["family"].isin(["iid", "ood_dipping", "ood_three_layer", "manifold_blind"])],
        treasure_table,
    ], ignore_index=True)
    budget = _error_budget(budget_ladder, baselines, pod)
    budget.to_csv(FROZEN / "error_budget.csv", index=False)
    _figure(budget)

    print("consistency", flush=True)
    consistency = []
    consistency.extend(_homogeneous(model, device, freq))
    consistency.extend(_cov0(model, cache, device, freq))
    consistency.extend(_mirror(model, cache, device, freq))
    consistency.append({
        "test": "monotone_pairs",
        "family": "",
        "width_m": "",
        "band": "",
        "n": 0,
        "value": np.nan,
        "reference": float(MONOTONE_SIGN_AGREEMENT),
        "reference_name": "gates.MONOTONE_SIGN_AGREEMENT",
        "passed": "",
        "status": "not_run",
        "note": "No existing record pair changes only CoV or only dip. Seeds inside a Sobol group share theta. A new finite-element pair was not solved.",
    })
    consistency.extend(_context(model, cache, device, freq))
    consistency_frame = pd.DataFrame(consistency)
    consistency_frame.to_csv(FROZEN / "consistency_tests.csv", index=False)

    print("epoch neighborhood", flush=True)
    epoch_frame, epoch_long = _epoch_table(freq)
    epoch_frame.to_csv(FROZEN / "epoch_neighborhood.csv", index=False)
    epoch_long.to_csv(FROZEN / "epoch_center_long.csv", index=False)

    sample_ids = sorted(set(run_long["iid"]["sample_id"]) | set(run_long["ood_dipping"]["sample_id"]) | set(run_long["ood_three_layer"]["sample_id"]) | set(sec_long["sample_id"]))
    common_diff = (
        "Float32 load_fp32 of P128n128 epoch 46. No evaluate.load_model autocast. "
        "IID, dipping, and three-layer: every test record, all 21 labeled stations. "
        f"Pretell subset is {N_PRETELL} columns (linspace, middle index forced to the center column), geometric mean of surface_over_outcrop. "
        "Center column is outcrop_many. g=0 is the stored log_tf1d. "
        "cache/meta.json fixed is empty, so the synthetic FFT window is null. "
        "Skill suppression uses the proposal 0.0136 because gates.py was not edited. "
        "Round-0 was not scored. full_svlab.h5 was not scored. Stage 1 curves were not invented."
    )
    _provenance(
        FROZEN / "ladder.provenance.json",
        label_campaign="iid, ood_dipping, ood_three_layer test; manifold val and blind non-trough; round-0 flagged and not scored",
        oned_engine="stored cache log_tf1d; cache/meta.json fixed is empty, so the engine version is not in that file",
        outcrop_convention="stored per-station log_tf1d is the prior; baseline b recomputes an elastic half-space outcrop",
        damping_source="qvs on the three run families; manifold damping is the section field (qvs or explicit) in manifold_sections.csv",
        fft_window_s=None,
        checkpoint=str(CKPT),
        epoch=epoch,
        dtype="float32",
        aggregation_unit="master",
        sample_ids=sample_ids,
        definition_difference=common_diff,
        bootstrap_unit="master",
        skill_reduction="pooled",
        pretell_columns=subsets,
        n_iid=int(run_long["iid"]["sample_id"].nunique()) if "sample_id" in run_long["iid"] else None,
        dropped_troughs=int(len(cache.dropped_troughs)),
        station_set="all labeled stations; no record subsample",
    )
    # n_iid fix: sample_id is a column
    _provenance(
        FROZEN / "ladder_treasure.provenance.json",
        label_campaign="case7a full_qvs, one 4000 m section",
        oned_engine="section_1d.surface_over_outcrop via the stored fields.npz column; half-space outcrop",
        outcrop_convention="surface over elastic outcrop",
        damping_source="qvs",
        fft_window_s=1310.72,
        checkpoint=str(CKPT),
        epoch=46,
        dtype="float32",
        aggregation_unit="moving block along x",
        sample_ids=["treasure_island"],
        definition_difference=(
            "P128n128 g is 07_one_model/treasure_g.npz. The forward was not rerun. "
            "fields.npz pred is not the curve used here. "
            "The moving-block interval is the median of station g log-L1. The pooled mean is a separate column. "
            "Low band and overall, which includes 0.1–0.4 Hz, are not accuracy results. "
            f"Pretell columns {subsets['treasure_island']}. full_svlab.h5 was not read."
        ),
        bootstrap_unit="moving block along x",
        skill_reduction="pooled",
        record_s=15.0,
        recomputed_forward=False,
    )
    _provenance(
        FROZEN / "baselines.provenance.json",
        label_campaign="same records as the ladder",
        oned_engine="matched_physics.outcrop_many for the center column; section_1d.surface_over_outcrop for the Pretell mean; stored log_tf1d for g=0",
        outcrop_convention="elastic half-space outcrop, Rayleigh at the record (f1, f2)",
        damping_source="qvs on run families and Treasure Island; manifold per section",
        fft_window_s=None,
        checkpoint=None,
        epoch=None,
        dtype="float64 outcrop",
        aggregation_unit="master",
        sample_ids=sample_ids + ["treasure_island"],
        definition_difference=(
            "Prediction baselines use the same station set and the same bootstrap as the matching ladder file. "
            "Treasure Island rows in this file use the moving-block median interval. "
            "Realization and POD values are copied from the Phase 1 CSVs. FE is unmeasured. "
            "Cached rigid-base arms were not opened. DeepONet-Residual was not written."
        ),
        bootstrap_unit="master, except Treasure Island moving block along x",
        skill_reduction="pooled absolute g log-L1; this table does not report mean-of-skills",
        pretell_n_columns=N_PRETELL,
    )
    _provenance(
        FROZEN / "error_budget.provenance.json",
        label_campaign="ladder families, manifold blind pooled, Treasure Island",
        oned_engine="same as the ladder",
        outcrop_convention="same as the ladder",
        damping_source="same as the ladder",
        fft_window_s=None,
        checkpoint=str(CKPT),
        epoch=epoch,
        dtype="float32",
        aggregation_unit="master",
        sample_ids=sample_ids + ["treasure_island"],
        definition_difference=(
            "Ratio (1D error - model error) / (1D error - max(measured floors)). "
            "FE is unmeasured, so every finite ratio is provisional. "
            "A nonpositive gap is omitted. Treasure Island low and overall are not accuracy claims."
        ),
        bootstrap_unit="master for the model interval; Treasure Island median moving block is copied from ladder_treasure.csv",
        skill_reduction="pooled",
    )
    _provenance(
        FROZEN / "consistency_tests.provenance.json",
        label_campaign="iid test for cov0, mirror, and context; dipping and three-layer test for context; three homogeneous columns",
        oned_engine="make_cov0_window repeats the stored center column; homogeneous columns have no OpenSees label",
        outcrop_convention="not a scored outcrop; g_pred is compared with zero or with its mirror",
        damping_source="qvs",
        fft_window_s=None,
        checkpoint=str(CKPT),
        epoch=epoch,
        dtype="float32",
        aggregation_unit="record",
        sample_ids=[f"iid_test_n={len(cache.runs('test', ('iid',)))}"],
        definition_difference=(
            "Homogeneous cutoff is the proposal 0.0136. Mirror cutoff is 0.1 times median L1(g_true). "
            "Monotone status is not_run. Context rows are compatibility because wider windows repeat the 500-column edge. "
            "CoV=0 covers every IID test record, center station of the repeated column."
        ),
        bootstrap_unit="not resampled; the reported value is a mean or a median as named in the note",
        skill_reduction="not a skill table",
    )
    _provenance(
        FROZEN / "epoch_neighborhood.provenance.json",
        label_campaign="iid and ood_dipping test, center station index 10",
        oned_engine="stored cache log_tf1d",
        outcrop_convention="stored log_tf1d",
        damping_source="qvs",
        fft_window_s=None,
        checkpoint=str(CKPT),
        epoch=46,
        dtype="float32",
        aggregation_unit="master",
        sample_ids=list(np.load(FROZEN / "center_iid.npz", allow_pickle=True)["sample_id"]) + list(np.load(FROZEN / "center_ood_dipping.npz", allow_pickle=True)["sample_id"]),
        definition_difference=(
            "Epoch 46 center stations are the fresh load_fp32 forward. "
            "The four-checkpoint mean is the stored ensemble_center_tf npz from ensemble_update._mean_g and was not recomputed. "
            "assert_comparable was called on the two master-pooled headlines. "
            "Epochs ±3 are not on disk. No best-of-four validation ranking is in this file. "
            "history.csv is absent."
        ),
        bootstrap_unit="master",
        skill_reduction="pooled",
        neighbor_epochs_on_disk=[],
    )

    decision = _decide(ladder, baselines, noise)
    decision["counts"] = {
        "iid": int(run_long["iid"].loc[run_long["iid"]["method"] == "P128n128", "sample_id"].nunique()),
        "ood_dipping": int(run_long["ood_dipping"].loc[run_long["ood_dipping"]["method"] == "P128n128", "sample_id"].nunique()),
        "ood_three_layer": int(run_long["ood_three_layer"].loc[run_long["ood_three_layer"]["method"] == "P128n128", "sample_id"].nunique()),
        "manifold": int(sec_long.loc[sec_long["method"] == "P128n128", "sample_id"].nunique()),
    }
    decision["dropped_troughs"] = int(len(cache.dropped_troughs))
    _write_g2(ladder, baselines, budget, consistency_frame, epoch_frame, noise, decision)
    (FROZEN / "g2_checks.json").write_text(json.dumps({"code": decision["code"], "checks": decision["checks"]}, indent=2) + "\n")
    print(decision["code"], flush=True)
    print(decision["paragraph"], flush=True)


if __name__ == "__main__":
    main()
