#!/usr/bin/env python3
"""Phase 1.3 label floors, 1.4 g-signal, and 1.5 realization spread.

The finite-element floor is left unmeasured. No mesh refinement and no new
OpenSees solve are launched. gates.py is not rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    AUDIT, BAND_NAMES, BOX, CACHE, PROPOSAL_FLOOR, REPO,
    band_l1, bootstrap_independent, mean_pairwise_abs, prior_log_l1,
    provenance, save_fig, use_agg,
)
from score_harness import score_curves

LABELS = REPO / "data" / "vardomain" / "section_labels"
ROUND0_H5 = REPO / "data" / "vardomain" / "case7_sections_round0_h5"


def _probe_dx(path: Path) -> float:
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(path, "r") as handle:
        return float(handle["grid"].attrs["dx"])


def _vs_low(path: Path, n_active_path: Path | None) -> dict:
    vs = np.load(path, mmap_mode="r")
    taken = []
    step = max(int(vs.shape[0] // 400), 1)
    if n_active_path is not None and n_active_path.is_file():
        n_active = np.load(n_active_path)
        for i in range(0, vs.shape[0], step):
            block = np.asarray(vs[i], dtype=np.float32)
            n = int(n_active[i])
            taken.append(block[:n])
    else:
        for i in range(0, vs.shape[0], step):
            taken.append(np.asarray(vs[i], dtype=np.float32))
    flat = np.concatenate([a.ravel() for a in taken])
    flat = flat[np.isfinite(flat) & (flat > 1.0)]
    if flat.size == 0:
        return {"vs_min": np.nan, "vs_p01": np.nan, "n_samples": 0}
    return {
        "vs_min": float(np.min(flat)),
        "vs_p01": float(np.quantile(flat, 0.01)),
        "n_samples": int(flat.size),
    }


def _ppw_rows() -> list[dict]:
    specs = [
        ("iid", CACHE / "iid" / "vs.npy", CACHE / "iid" / "n_active.npy", BOX / "h5" / "run_0.h5"),
        ("ood_dipping", CACHE / "ood_dipping" / "vs.npy", CACHE / "ood_dipping" / "n_active.npy",
         next((BOX / "ood_dipping" / "h5").glob("*.h5"), None) if (BOX / "ood_dipping" / "h5").is_dir() else None),
        ("ood_three_layer", CACHE / "ood_three_layer" / "vs.npy", CACHE / "ood_three_layer" / "n_active.npy",
         next((BOX / "ood_three_layer" / "h5").glob("*.h5"), None) if (BOX / "ood_three_layer" / "h5").is_dir() else None),
        ("manifold", None, None, next((BOX / "vardomain" / "manifold150" / "h5").glob("*.h5"), None)
         if (BOX / "vardomain" / "manifold150" / "h5").is_dir() else None),
        ("round0", None, None, next(ROUND0_H5.glob("*.h5"), None)),
        ("treasure_island", None, None, REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5" / "full_qvs.h5"),
    ]
    rows = []
    for name, vs_path, n_path, h5 in specs:
        row = {"table": "points_per_wavelength", "campaign": name, "frequency_hz": 10.0}
        try:
            if h5 is None or not Path(h5).is_file():
                raise FileNotFoundError(f"no mesh sample for {name}")
            dx = _probe_dx(Path(h5))
            if name == "manifold":
                sample = next((CACHE / "sections").glob("*.npz"))
                field = np.asarray(np.load(sample)["vs"], dtype=np.float32)
                stats = {"vs_min": float(field[field > 1].min()), "vs_p01": float(np.quantile(field[field > 1], 0.01))}
                # The finest pair is the minimum over sections, not one file.
                mins = []
                for path in (CACHE / "sections").glob("*.npz"):
                    field = np.asarray(np.load(path)["vs"], dtype=np.float32)
                    positive = field[field > 1]
                    if positive.size:
                        mins.append(float(positive.min()))
                stats["vs_min"] = float(np.min(mins)) if mins else np.nan
                stats["vs_p01"] = float(np.quantile(mins, 0.01)) if mins else np.nan
            elif name == "round0":
                field = None
                import h5py
                import hdf5plugin  # noqa: F401
                mins = []
                for path in ROUND0_H5.glob("*.h5"):
                    with h5py.File(path, "r") as handle:
                        field = np.asarray(handle["Vs_realization_2D"][:], dtype=np.float32)
                    positive = field[field > 1]
                    if positive.size:
                        mins.append(float(positive.min()))
                stats = {"vs_min": float(np.min(mins)), "vs_p01": float(np.quantile(mins, 0.01))}
            elif name == "treasure_island":
                import h5py
                import hdf5plugin  # noqa: F401
                with h5py.File(h5, "r") as handle:
                    field = np.asarray(handle["Vs_realization_2D"][:], dtype=np.float32)
                positive = field[field > 1]
                stats = {"vs_min": float(positive.min()), "vs_p01": float(np.quantile(positive, 0.01))}
            else:
                stats = _vs_low(vs_path, n_path)
            row.update({
                "read_ok": True,
                "dx_m": dx,
                "vs_min_mps": stats["vs_min"],
                "vs_p01_mps": stats["vs_p01"],
                "ppw_at_vs_min": stats["vs_min"] / (10.0 * dx),
                "ppw_at_vs_p01": stats["vs_p01"] / (10.0 * dx),
                "mesh_sample": str(h5),
                "note": "Points per wavelength at 10 Hz. The finest pair is the lowest Vs on that campaign's mesh spacing. No refined mesh was solved.",
            })
        except Exception as exc:
            row.update({"read_ok": False, "note": f"{type(exc).__name__}: {exc}", "ppw_at_vs_min": np.nan})
        rows.append(row)
    return rows


def _deferred_rows() -> list[dict]:
    rows = []
    for band in BAND_NAMES:
        rows.append({
            "table": "fe_floor", "campaign": "all", "band": band, "value": np.nan,
            "status": "unmeasured",
            "note": "Requires one mesh refinement and one repeated OpenSees solve on 3–5 masters. Those reruns were not launched.",
        })
    rows.append({
        "table": "pad_contamination", "campaign": "all", "band": "all", "value": np.nan,
        "status": "deferred",
        "note": "Pad contamination waits for the Phase 4 pilot. It is not estimated from the unpadded round-0 meshes.",
    })
    rows.append({
        "table": "round0_flag", "campaign": "round0", "band": "all", "value": np.nan,
        "status": "flagged",
        "note": "Unpadded 2000 m sections include the absorbing boundary. They are not scored as clean domains.",
    })
    rows.append({
        "table": "full_svlab", "campaign": "treasure_island", "band": "all", "value": np.nan,
        "status": "not_a_second_solve",
        "note": "full_svlab.h5 is not a second OpenSees solve.",
    })
    return rows


def _leakage_file(path: Path, durations: list[float], ttf) -> list[dict]:
    import h5py
    import hdf5plugin  # noqa: F401

    from labels import LABEL_WINDOW_S, seiskit_input_acceleration, tf_over_input

    with h5py.File(path, "r") as handle:
        time = np.asarray(handle["recorders/accel/time"][:], dtype=np.float64)
        if "recorders/accel/surface" in handle:
            acc = handle["recorders/accel/surface"]
            n_x = int(acc.shape[1])
            cols = np.linspace(0, n_x - 1, num=min(7, n_x)).round().astype(int)
            cols = np.unique(cols)
            surface = np.asarray(acc[:, cols], dtype=np.float64)
        elif "recorders/accel/data" in handle:
            # IID seiskit layout: [base | surface], 21 and 21, time by channel.
            data = handle["recorders/accel/data"]
            n_chan = int(data.shape[1])
            surf = data[:, n_chan // 2 :]
            cols = np.linspace(0, surf.shape[1] - 1, num=min(7, surf.shape[1])).round().astype(int)
            cols = np.unique(cols)
            surface = np.asarray(surf[:, cols], dtype=np.float64)
        else:
            raise KeyError(f"{path} has no surface acceleration dataset")
        params = handle["params"].attrs
        motion = float(handle["grid"].attrs["motion_freq"])
        ricker = params.get("ricker_is", "acceleration")
        if isinstance(ricker, bytes):
            ricker = ricker.decode()
    dt = float(time[1] - time[0])
    full_s = float(time[-1] - time[0])
    logs = {}
    freq = None
    for dur in durations:
        n = min(int(time.size), max(int(round(float(dur) / dt)), 2))
        if n >= time.size:
            n = int(time.size)
        freq_i, mag = tf_over_input(
            surface[:n],
            seiskit_input_acceleration(motion, 0.5, time[:n], str(ricker)),
            dt, ttf, window_s=LABEL_WINDOW_S,
        )
        logs[float(time[n - 1] - time[0])] = np.log(np.maximum(mag, 1e-6))
        freq = np.asarray(freq_i, dtype=np.float64)
    full_key = max(logs)
    rows = []
    for key, log_short in logs.items():
        if abs(key - full_key) < 1e-6:
            continue
        diff = log_short - logs[full_key]
        scored = band_l1(diff, freq)
        for band in BAND_NAMES:
            values = np.asarray(scored[band], dtype=np.float64)
            rows.append({
                "table": "record_length_leakage",
                "campaign": path.name,
                "band": band,
                "full_s": full_s,
                "truncated_s": key,
                "n_stations": int(values.size),
                "value": float(np.mean(values)),
                "value_max_station": float(np.max(values)),
                "status": "measured",
                "note": (
                    "Mean |log|TF_truncated| − log|TF_full|| on the same stations and the 1310.72 s FFT window. "
                    "This is label leakage from the record length, not a new OpenSees solve."
                ),
            })
    if not rows:
        rows.append({
            "table": "record_length_leakage", "campaign": path.name, "band": "all",
            "full_s": full_s, "status": "no_shorter_prefix", "value": np.nan,
            "note": "The file has one duration, so a within-file truncation was not available.",
        })
    return rows


def _leakage_rows() -> list[dict]:
    try:
        from run_case7a import load_ttf_batch_fast
        ttf = load_ttf_batch_fast()
    except Exception as exc:
        return [{
            "table": "record_length_leakage", "campaign": "all", "band": "all",
            "status": "unread", "value": np.nan,
            "note": f"Transfer-function import failed: {type(exc).__name__}: {exc}",
        }]
    jobs = [
        (REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5" / "full_qvs.h5", [15.0, 7.5]),
        (REPO / "data" / "case7a" / "opensees_crops" / "record_pilot" / "h5" / "crop_west_qvs_d60.h5", [60.0, 30.0, 15.0]),
        (REPO / "data" / "case7a" / "opensees_crops" / "record_pilot" / "h5" / "crop_west_qvs_d30.h5", [30.0, 15.0]),
        (BOX / "h5" / "run_0.h5", [50.0, 15.0]),
        (next(ROUND0_H5.glob("*.h5")) if ROUND0_H5.is_dir() else Path("missing"), [15.0, 7.5]),
    ]
    rows = []
    for path, durations in jobs:
        if not path.is_file():
            rows.append({
                "table": "record_length_leakage", "campaign": path.name, "band": "all",
                "status": "missing", "value": np.nan, "note": f"{path} is not on disk",
            })
            continue
        try:
            rows.extend(_leakage_file(path, durations, ttf))
        except Exception as exc:
            rows.append({
                "table": "record_length_leakage", "campaign": path.name, "band": "all",
                "status": "failed", "value": np.nan, "note": f"{type(exc).__name__}: {exc}",
            })
    return rows


def _log_l1_pair(left: np.ndarray, right: np.ndarray, freq: np.ndarray) -> dict[str, float]:
    diff = np.log(np.maximum(left, 1e-6)) - np.log(np.maximum(right, 1e-6))
    scored = band_l1(diff, freq)
    return {band: float(np.mean(scored[band])) for band in BAND_NAMES}


def _convention_rows() -> list[dict]:
    rows = []
    pairs = (
        ("round0_vs_round1_fixed", LABELS / "round0", LABELS / "round1_fixed", "tf1d"),
        ("round0_applied_vs_round1_fixed", LABELS / "round0_applied", LABELS / "round1_fixed", "tf1d"),
        ("round0_vs_round0_applied", LABELS / "round0", LABELS / "round0_applied", "tf1d"),
        ("round0_vs_round1_fixed_tf", LABELS / "round0", LABELS / "round1_fixed", "tf"),
    )
    for name, left_dir, right_dir, key in pairs:
        files = sorted(left_dir.glob("sec_*.npz")) if left_dir.is_dir() else []
        bucket = {band: [] for band in BAND_NAMES}
        missing = 0
        freq = None
        for path in files:
            other = right_dir / path.name
            if not other.is_file():
                missing += 1
                continue
            a = np.load(path)
            b = np.load(other)
            freq = np.asarray(a["freq"], dtype=np.float64)
            scored = _log_l1_pair(np.asarray(a[key]), np.asarray(b[key]), freq)
            for band in BAND_NAMES:
                bucket[band].append(scored[band])
        for band in BAND_NAMES:
            values = np.asarray(bucket[band], dtype=np.float64)
            rows.append({
                "table": "oned_convention",
                "campaign": name,
                "band": band,
                "n_sections": int(values.size),
                "n_missing": missing,
                "value": float(np.median(values)) if values.size else np.nan,
                "value_p90": float(np.quantile(values, 0.9)) if values.size else np.nan,
                "status": "measured" if values.size else "missing",
                "note": (
                    "Median over the 64 sections of the station-mean |Δ log|. "
                    "round0 is the CSV-zeta prior. round0_applied uses applied zeta. "
                    "round1_fixed is the corrected outcrop engine. The OpenSees surface TF is compared in the _tf row."
                ),
            })
    # Cache prior versus the local fixed-window linear TF, float16 quantization included.
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        linear = REPO / "data" / "vardomain" / "outcrop_fixed_window" / family / "tf1d_outcrop_per_sample.npy"
        stored = CACHE / family / "log_tf1d.npy"
        freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
        if not linear.is_file() or not stored.is_file():
            rows.append({
                "table": "oned_convention", "campaign": f"cache_vs_fixed_window_{family}",
                "band": "all", "status": "missing", "value": np.nan,
                "note": f"missing {linear} or {stored}",
            })
            continue
        raw = np.load(linear, mmap_mode="r")
        log_stored = np.load(stored, mmap_mode="r")
        if raw.shape != log_stored.shape:
            rows.append({
                "table": "oned_convention", "campaign": f"cache_vs_fixed_window_{family}",
                "band": "all", "status": "shape_mismatch", "value": np.nan,
                "note": f"fixed {raw.shape} vs cache {log_stored.shape}",
            })
            continue
        step = max(int(raw.shape[0] // 200), 1)
        idx = np.arange(0, raw.shape[0], step)
        diff = np.asarray(log_stored[idx], dtype=np.float64) - np.log(np.maximum(np.asarray(raw[idx], dtype=np.float64), 1e-6))
        scored = band_l1(diff, freq)
        for band in BAND_NAMES:
            rows.append({
                "table": "oned_convention",
                "campaign": f"cache_vs_fixed_window_{family}",
                "band": band,
                "n_records_sampled": int(idx.size),
                "value": float(np.median(scored[band])),
                "value_p90": float(np.quantile(scored[band], 0.9)),
                "status": "measured",
                "note": "Median station |log cache − log fixed-window TF1D|. This is the live prior agreement, including float16 storage.",
            })
    rows.extend(_live_column_rows())
    return rows


def _live_column_rows() -> list[dict]:
    try:
        from section_1d import RHO, rayleigh_xi_target, surface_over_outcrop
    except Exception as exc:
        return [{
            "table": "oned_convention", "campaign": "live_column", "band": "all",
            "status": "failed", "value": np.nan, "note": f"{type(exc).__name__}: {exc}",
        }]
    freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
    rows = []
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        root = CACHE / family
        vs = np.load(root / "vs.npy", mmap_mode="r")
        zeta = np.load(root / "zeta.npy", mmap_mode="r")
        n_active = np.load(root / "n_active.npy")
        log_prior = np.load(root / "log_tf1d.npy", mmap_mode="r")
        rec_x = np.load(root / "rec_x.npy")
        f1 = np.load(root / "rayleigh_f1.npy")
        f2 = np.load(root / "rayleigh_f2.npy")
        split = np.load(root / "split.npy").astype(str)
        station = int(np.argmin(np.abs(rec_x - (rec_x[0] + rec_x[-1]) / 2)))
        col = int(rec_x[station])
        hits = np.flatnonzero(split == "test")[:2]
        for index in hits:
            try:
                n = int(n_active[index])
                column = np.asarray(vs[index, :n, col], dtype=np.float64)
                damp = np.asarray(zeta[index, :n, col], dtype=np.float64)
                xi = np.stack([rayleigh_xi_target(freq, float(z), float(f1[index]), float(f2[index])) for z in damp])
                curve = surface_over_outcrop(freq, column, dz=1.0, xi=xi, rho=RHO)
                diff = np.log(np.maximum(curve, 1e-6)) - np.asarray(log_prior[index, station], dtype=np.float64)
                scored = band_l1(diff, freq)
                for band in BAND_NAMES:
                    # band_l1 on a 1-d curve returns a scalar array
                    rows.append({
                        "table": "oned_convention",
                        "campaign": "live_column_vs_cache",
                        "family": family,
                        "sample_id": f"{family}:{int(index)}",
                        "band": band,
                        "value": float(np.asarray(scored[band]).reshape(-1)[0] if np.ndim(scored[band]) else scored[band]),
                        "status": "measured",
                        "note": "Fresh surface_over_outcrop of the center column minus the stored log_tf1d. A few test rows, not a campaign sweep.",
                    })
            except Exception as exc:
                rows.append({
                    "table": "oned_convention", "campaign": "live_column_vs_cache",
                    "family": family, "sample_id": f"{family}:{int(index)}", "band": "all",
                    "status": "failed", "value": np.nan, "note": f"{type(exc).__name__}: {exc}",
                })
    return rows


def _record_signal(log_tf, log_prior, freq, valid, split, family, width) -> list[dict]:
    rows = []
    step = 128
    center = log_tf.shape[1] // 2
    for start in range(0, log_tf.shape[0], step):
        stop = min(start + step, log_tf.shape[0])
        g = np.asarray(log_tf[start:stop], dtype=np.float64) - np.asarray(log_prior[start:stop], dtype=np.float64)
        scored = band_l1(g, freq)
        for local, index in enumerate(range(start, stop)):
            if not bool(valid[index]):
                continue
            for band in BAND_NAMES:
                station = np.asarray(scored[band][local], dtype=np.float64)
                rows.append({
                    "family": family,
                    "split": str(split[index]),
                    "sample_id": f"{family}:{index}",
                    "width_m": width,
                    "band": band,
                    "record_mean": float(np.mean(station)),
                    "center": float(station[center]),
                    "n_stations": int(station.size),
                })
    return rows


def _section_signal(freq: np.ndarray) -> list[dict]:
    rows = []
    for path in sorted((CACHE / "sections").glob("*.npz")):
        blob = np.load(path)
        g = np.asarray(blob["log_tf"], dtype=np.float64) - np.asarray(blob["log_tf1d"], dtype=np.float64)
        scored = band_l1(g, freq)
        sid = path.stem.split("__")[-1]
        width = int(np.asarray(blob["vs"]).shape[1])
        center = g.shape[0] // 2
        for band in BAND_NAMES:
            station = np.asarray(scored[band], dtype=np.float64)
            rows.append({
                "family": "manifold",
                "split": str(blob["split"]),
                "sample_id": sid,
                "width_m": width,
                "band": band,
                "record_mean": float(np.mean(station)),
                "center": float(station[center]),
                "n_stations": int(station.size),
            })
    return rows


def _summarize_signal(records: pd.DataFrame, floor_by_band: dict[str, float]) -> pd.DataFrame:
    rows = []
    for (family, split, width, band), part in records.groupby(["family", "split", "width_m", "band"], sort=False):
        values = part["record_mean"].to_numpy()
        floor = float(floor_by_band.get(band, np.nan))
        row = {
            "table": "signal",
            "family": family,
            "split": split,
            "width_m": width,
            "band": band,
            "n_records": int(values.size),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.1)),
            "p90": float(np.quantile(values, 0.9)),
            "mean": float(np.mean(values)),
            "fraction_at_or_below_floor": float(np.mean(values <= floor)) if np.isfinite(floor) else np.nan,
            "floor_compared": floor,
            "absolute_error_only": bool(np.isfinite(floor) and np.median(values) <= floor),
        }
        if split in ("test", "val", "blind") and values.size >= 4:
            boot = bootstrap_independent(
                values, part["sample_id"].to_numpy(), unit="record", statistic=np.median,
                n_draw=400, rng=np.random.default_rng(0),
            )
            row["median_low"] = boot["low"]
            row["median_high"] = boot["high"]
        rows.append(row)
    return pd.DataFrame(rows)


def _center_sample(freq: np.ndarray) -> pd.DataFrame:
    """A harness long table for the IID test center station, so the audit calls score_curves."""

    split = np.load(CACHE / "iid" / "split.npy").astype(str)
    log_tf = np.load(CACHE / "iid" / "log_tf.npy", mmap_mode="r")
    log_prior = np.load(CACHE / "iid" / "log_tf1d.npy", mmap_mode="r")
    group = np.load(CACHE / "iid" / "group.npy")
    hits = np.flatnonzero(split == "test")[:64]
    center = log_tf.shape[1] // 2
    true = np.asarray(log_tf[hits, center], dtype=np.float64)
    prior = np.asarray(log_prior[hits, center], dtype=np.float64)
    # g = 0 prediction: the long table's prior_log_l1 is L1(g_true).
    return score_curves(
        sample_id=np.array([f"iid:{int(i)}" for i in hits]),
        station=np.full(hits.shape, center),
        cluster_id=group[hits].astype(str),
        freq=freq,
        log_pred=prior,
        log_true=true,
        log_prior=prior,
        method="g_equals_0",
    )


def _realization_rows(freq: np.ndarray) -> list[dict]:
    rows = []
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        rows.extend(_realization_family(family, freq))
    return rows


def _realization_family(family: str, freq: np.ndarray) -> list[dict]:
    root = CACHE / family
    group = np.load(root / "group.npy")
    valid = np.load(root / "valid.npy").astype(bool)
    log_tf = np.load(root / "log_tf.npy", mmap_mode="r")
    log_prior = np.load(root / "log_tf1d.npy", mmap_mode="r")
    rows = []
    for gid in np.unique(group):
        index = np.flatnonzero((group == gid) & valid)
        if index.size < 2:
            continue
        g = np.asarray(log_tf[index], dtype=np.float64) - np.asarray(log_prior[index], dtype=np.float64)
        # g: (replicate, station, freq). Pairwise inside each station, then mean stations.
        band_values = {band: [] for band in BAND_NAMES}
        for station in range(g.shape[1]):
            pair = mean_pairwise_abs(g[:, station, :])
            scored = band_l1(pair, freq)
            for band in BAND_NAMES:
                band_values[band].append(float(np.asarray(scored[band]).reshape(-1)[0]))
        for band in BAND_NAMES:
            rows.append({
                "table": "realization",
                "family": family,
                "group_id": int(gid),
                "n_replicates": int(index.size),
                "band": band,
                "value": float(np.mean(band_values[band])),
                "note": "Mean over stations of the mean pairwise |g_i − g_j| among seed replicates of one Sobol group. This is the OpenSees spread if the model knew only θ.",
            })
    return rows


def _floor_from(convention: pd.DataFrame, leakage: pd.DataFrame) -> dict:
    """Live 1D agreement, with the 15 s leakage kept beside it. FE stays out."""

    live = convention[convention["campaign"].astype(str).str.startswith("cache_vs_fixed_window")]
    historical = convention[convention["campaign"] == "round0_vs_round1_fixed"]
    fresh = convention[convention["campaign"] == "live_column_vs_cache"]
    by_band = {}
    for band in BAND_NAMES:
        live_vals = live.loc[live["band"] == band, "value"].to_numpy(dtype=float)
        hist_vals = historical.loc[historical["band"] == band, "value"].to_numpy(dtype=float)
        fresh_vals = fresh.loc[fresh["band"] == band, "value"].to_numpy(dtype=float)
        live_max = float(np.nanmax(live_vals)) if live_vals.size else float("nan")
        hist = float(np.nanmax(hist_vals)) if hist_vals.size else float("nan")
        fresh_med = float(np.nanmedian(fresh_vals)) if fresh_vals.size else float("nan")
        pieces = [v for v in (live_max, fresh_med) if np.isfinite(v)]
        by_band[band] = {
            "cache_vs_fixed_window_max_median": live_max,
            "live_column_median": fresh_med,
            "historical_round0_vs_round1_median": hist,
            "oned_convention": float(np.max(pieces)) if pieces else float("nan"),
        }
    leak = {}
    measured = leakage[leakage["status"] == "measured"] if "status" in leakage.columns else leakage.iloc[0:0]
    for band in BAND_NAMES:
        part = measured[measured["band"] == band] if "band" in measured.columns else measured.iloc[0:0]
        leak[band] = {}
        for _, row in part.iterrows():
            leak[band][f"{row['campaign']}@{row.get('truncated_s', '')}"] = float(row["value"])
    # Training-label floor is the live 1D term. Short-record claims also see 15 s leakage.
    short = {}
    for band in BAND_NAMES:
        candidates = []
        for key, value in leak.get(band, {}).items():
            if "full_qvs" in key or "sec_" in key or "round0" in key:
                candidates.append(value)
            if "@7.5" in key or str(key).endswith("@7.5"):
                candidates.append(value)
        # keys look like "full_qvs.h5@7.5"
        short_vals = [v for k, v in leak.get(band, {}).items() if "full_qvs" in k or k.startswith("sec_")]
        base = by_band[band]["oned_convention"]
        extra = float(np.nanmax(short_vals)) if short_vals else float("nan")
        short[band] = float(np.nanmax([base, extra])) if np.isfinite(base) or np.isfinite(extra) else float("nan")
    headline_pieces = [by_band[band]["oned_convention"] for band in BAND_NAMES]
    headline = float(np.nanmax(headline_pieces)) if any(np.isfinite(headline_pieces)) else float("nan")
    return {
        "proposal_log_l1": PROPOSAL_FLOOR,
        "fe_floor": "unmeasured",
        "gates_py_unchanged": True,
        "definition": (
            "Measured noise floor for the stored training labels is the live 1D-convention uncertainty: "
            "max of the cache-versus-fixed-window median and the fresh column residual. "
            "The FE floor is unmeasured, so this number is a lower bound on max(FE floor, 1D-convention uncertainty). "
            "Short 15 s records add the within-file truncation leakage on top of that bound."
        ),
        "by_band": by_band,
        "record_length_leakage": leak,
        "measured_noise_floor_by_band": {band: by_band[band]["oned_convention"] for band in BAND_NAMES},
        "measured_noise_floor": headline,
        "short_record_floor_by_band": short,
        "proposal_comparison": (
            "above the 0.0136 proposal" if np.isfinite(headline) and headline > PROPOSAL_FLOOR
            else "at or below the 0.0136 proposal" if np.isfinite(headline)
            else "not measured"
        ),
    }


def plot(ppw: pd.DataFrame, signal: pd.DataFrame, floor: dict, path: Path) -> None:
    use_agg()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    part = ppw[ppw["read_ok"] == True] if "read_ok" in ppw.columns else ppw
    if len(part) and "ppw_at_vs_min" in part.columns:
        axes[0].bar(np.arange(len(part)), part["ppw_at_vs_min"].to_numpy(), color="C0")
        axes[0].set_xticks(np.arange(len(part)), part["campaign"].tolist(), rotation=20, ha="right")
        axes[0].axhline(10.0, color="0.4", ls="--", lw=1, label="10 points")
        axes[0].set_ylabel("points per wavelength at 10 Hz")
        axes[0].set_title("Finest Vs on the stored mesh")
        axes[0].legend(frameon=False)
    test = signal[(signal["split"] == "test") & (signal["family"] != "manifold")]
    families = ["iid", "ood_dipping", "ood_three_layer"]
    width = 0.18
    for i, band in enumerate(BAND_NAMES):
        vals = []
        for family in families:
            hit = test[(test["family"] == family) & (test["band"] == band)]
            vals.append(float(hit["median"].iloc[0]) if len(hit) else np.nan)
        axes[1].bar(np.arange(len(families)) + (i - 1.5) * width, vals, width=width, label=band)
    measured = floor.get("measured_noise_floor", np.nan)
    if np.isfinite(measured):
        axes[1].axhline(measured, color="C3", lw=1, label=f"measured floor {measured:.3g}")
    axes[1].axhline(PROPOSAL_FLOOR, color="0.3", ls="--", lw=1, label="proposal 0.0136")
    axes[1].set_xticks(np.arange(len(families)), families, rotation=15, ha="right")
    axes[1].set_ylabel("median L1(g_true)")
    axes[1].set_title("Test-set signal")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, path)
    plt.close(fig)


def plot_realization(frame: pd.DataFrame, signal: pd.DataFrame, path: Path) -> None:
    use_agg()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    families = ["iid", "ood_dipping", "ood_three_layer"]
    x = np.arange(len(BAND_NAMES))
    for family in families:
        vals = []
        for band in BAND_NAMES:
            part = frame[(frame["family"] == family) & (frame["band"] == band)]
            vals.append(float(part["value"].median()) if len(part) else np.nan)
        ax.plot(x, vals, marker="o", label=f"{family} realization")
        sig = []
        for band in BAND_NAMES:
            hit = signal[(signal["family"] == family) & (signal["split"] == "test") & (signal["band"] == band)]
            sig.append(float(hit["median"].iloc[0]) if len(hit) else np.nan)
        ax.plot(x, sig, marker="x", ls="--", label=f"{family} signal")
    ax.set_xticks(x, BAND_NAMES)
    ax.set_ylabel("log-L1")
    ax.set_title("Seed-to-seed OpenSees spread and L1(g_true)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    save_fig(fig, path)
    plt.close(fig)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
    label_rows = []
    label_rows.extend(_ppw_rows())
    label_rows.extend(_deferred_rows())
    print("record-length leakage", flush=True)
    leakage_rows = _leakage_rows()
    label_rows.extend(leakage_rows)
    print("1D convention", flush=True)
    convention_rows = _convention_rows()
    label_rows.extend(convention_rows)
    label = pd.DataFrame(label_rows)
    label.to_csv(AUDIT / "label_floors.csv", index=False)

    print("signal", flush=True)
    signal_records = []
    for family, width in (("iid", 500), ("ood_dipping", 500), ("ood_three_layer", 500)):
        root = CACHE / family
        signal_records.extend(_record_signal(
            np.load(root / "log_tf.npy", mmap_mode="r"),
            np.load(root / "log_tf1d.npy", mmap_mode="r"),
            freq, np.load(root / "valid.npy"), np.load(root / "split.npy"), family, width,
        ))
    signal_records.extend(_section_signal(freq))
    records = pd.DataFrame(signal_records)
    # Provisional floor from the 1D rows already in memory. Leakage is applied in the json.
    convention = label[label["table"] == "oned_convention"]
    leakage = label[label["table"] == "record_length_leakage"]
    floor = _floor_from(convention, leakage)
    (AUDIT / "noise_floor.json").write_text(json.dumps(floor, indent=2) + "\n")
    summary = _summarize_signal(records, floor["measured_noise_floor_by_band"])
    summary.to_csv(AUDIT / "signal.csv", index=False)
    sample = _center_sample(freq)
    sample.to_csv(AUDIT / "signal_center_sample.csv", index=False)
    # Touch prior_log_l1 so the sample's first row is checked against the helper.
    first = sample[(sample["band"] == "mid") & (sample["metric"] == "prior_log_l1")].iloc[0]
    check = prior_log_l1(
        np.asarray(np.load(CACHE / "iid" / "log_tf.npy", mmap_mode="r")[int(str(first["sample_id"]).split(":")[1]), int(first["station"])], dtype=np.float64)
        - np.asarray(np.load(CACHE / "iid" / "log_tf1d.npy", mmap_mode="r")[int(str(first["sample_id"]).split(":")[1]), int(first["station"])], dtype=np.float64),
        freq, "mid",
    )
    if abs(check - float(first["value"])) > 1e-9:
        raise AssertionError(f"center sample {first['value']} != prior_log_l1 {check}")

    print("realization", flush=True)
    realize = pd.DataFrame(_realization_rows(freq))
    realize.to_csv(AUDIT / "realization.csv", index=False)

    plot(label[label["table"] == "points_per_wavelength"], summary, floor, AUDIT / "label_floors.png")
    # 1.4 and 1.5 each need a figure. The signal panel is also saved on its own.
    plot(label.iloc[0:0], summary, floor, AUDIT / "signal.png")
    plot_realization(realize, summary, AUDIT / "realization.png")

    ids = summary["family"].astype(str) + ":" + summary["split"].astype(str)
    provenance(
        AUDIT / "label_floors.provenance.json",
        campaign="meshes on disk; full_qvs; west 30/60 s pilots; one IID h5; round0/round1 labels",
        engine="section_1d.surface_over_outcrop for the live columns; stored tf1d otherwise",
        outcrop="surface over elastic outcrop where recomputed; stored log_tf1d for the cache comparison",
        damping="qvs on IID, OOD, and full_qvs; round0 CSV zeta versus applied zeta versus round1_fixed",
        fft=1310.72,
        checkpoint=None,
        epoch=None,
        dtype="not a model score",
        unit="record",
        sample_ids=ids.tolist(),
        difference=(
            "FE numerical floor is unmeasured. Pad contamination is deferred. "
            "Round-0 unpadded sections are flagged. full_svlab.h5 is not a second solve. "
            f"Measured training-label noise floor is {floor['measured_noise_floor']}. Proposal remains {PROPOSAL_FLOOR}. gates.py was not edited."
        ),
        measured_noise_floor=floor["measured_noise_floor"],
        proposal=PROPOSAL_FLOOR,
    )
    provenance(
        AUDIT / "signal.provenance.json",
        campaign="iid, ood_dipping, ood_three_layer, manifold cache sections",
        engine="stored cache log_tf1d; cache/meta.json fixed is empty",
        outcrop="stored log_tf1d",
        damping="qvs for IID and OOD; manifold as stored on the section",
        fft=1310.72,
        checkpoint=None,
        epoch=None,
        dtype="not a model score",
        unit="record",
        sample_ids=records["sample_id"].astype(str).unique().tolist(),
        difference=(
            "L1(g_true) is the harness prior log-L1, the mean absolute residual of g = log|TF| − log|TF_1D|. "
            "The headline median is over records. Bootstrap resamples records. "
            "A band is absolute-error only when that median is at or below the measured training-label floor. "
            "signal_center_sample.csv is score_curves on 64 IID test center stations with prediction g = 0."
        ),
        measured_noise_floor=floor["measured_noise_floor"],
    )
    provenance(
        AUDIT / "realization.provenance.json",
        campaign="iid, ood_dipping, ood_three_layer valid replicates",
        engine="stored cache log_tf and log_tf1d",
        outcrop="stored log_tf1d",
        damping="qvs",
        fft=1310.72,
        checkpoint=None,
        epoch=None,
        dtype="not a model score",
        unit="master",
        sample_ids=[f"{r.family}:{r.group_id}" for r in realize.itertuples()],
        difference=(
            "Pairwise OpenSees spread of g inside one Sobol group. Stations are averaged after the pair reduction; "
            "they are not the bootstrap unit. Beating this floor is a positive claim because P128n128 sees the realization."
        ),
    )
    print(f"measured floor {floor['measured_noise_floor']} proposal {PROPOSAL_FLOOR} ({floor['proposal_comparison']})", flush=True)


if __name__ == "__main__":
    main()
