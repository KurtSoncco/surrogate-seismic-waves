#!/usr/bin/env python3
"""OpenSees versus the corrected elastic column on existing Case 7 H5s.

Reads the crop files already on disk. Does not start OpenSees, does not score
a network, and does not read full_svlab.h5. A 2000 m row is a window of
full_qvs.h5 slid onto the 4000 m mesh when the requested window would hang
off the domain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VARDOMAIN = HERE.parents[1]
REPO = VARDOMAIN.parents[1]
CASE7A = REPO / "experiments" / "case7a"
ANALYSIS = VARDOMAIN / "analysis"
for path in (CASE7A, VARDOMAIN, ANALYSIS):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from applied_damping import applied_zeta  # noqa: E402
from case7_crops import CROPS, span_columns  # noqa: E402
from case7a import X_LEFT_M  # noqa: E402
from labels import seiskit_input_acceleration, tf_over_input  # noqa: E402
from score_harness import git_commit, sample_id_hash, write_provenance  # noqa: E402
from section_1d import RHO, surface_over_outcrop  # noqa: E402

OUT = ANALYSIS / "frozen" / "case7_column.csv"
# Acceleration campaign scale from data/vardomain/outcrop_fixed_window.log.
# These H5s store ricker_is=acceleration. Pearson of demeaned log|TF| does
# not use the scale; mean |dlog| does.
SCALE = 1.9776999950408936
WINDOW_S = 1310.72
STRIDE_M = 5
BANDS = {"low": (0.1, 0.4), "mid": (0.4, 2.5), "trusted": (0.1, 2.5)}
CROP_500 = REPO / "data" / "case7a" / "opensees_crops" / "h5"
CROP_WIDE = REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5"
FULL_QVS = CROP_WIDE / "full_qvs.h5"


def _xi(freq: np.ndarray, zeta_col: np.ndarray, f1: float, f2: float) -> np.ndarray:
    """Rayleigh ratio at each layer and frequency, matching rayleigh_xi_target."""

    w1, w2 = 2.0 * np.pi * float(f1), 2.0 * np.pi * float(f2)
    zeta = np.asarray(zeta_col, dtype=np.float64)
    a0 = 2.0 * zeta * w1 * w2 / (w1 + w2)
    a1 = 2.0 * zeta / (w1 + w2)
    omega = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)
    return 0.5 * (a0[:, None] / omega[None, :] + a1[:, None] * omega[None, :])


def _pearson_rows(log_a: np.ndarray, log_b: np.ndarray) -> float:
    aa = log_a - log_a.mean(axis=1, keepdims=True)
    bb = log_b - log_b.mean(axis=1, keepdims=True)
    numer = np.sum(aa * bb, axis=1)
    denom = np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1))
    return float(np.mean(numer / np.maximum(denom, 1e-12)))


def _read(path: Path) -> dict:
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(path, "r") as handle:
        params = {}
        for key, value in handle["params"].attrs.items():
            if isinstance(value, bytes):
                value = value.decode()
            elif hasattr(value, "item"):
                value = value.item()
            params[key] = value
        if str(params.get("ricker_is", "acceleration")) != "acceleration":
            raise SystemExit(f"{path.name} is not an acceleration Ricker; the stored scale would not apply")
        grid = dict(handle["grid"].attrs)
        return {
            "vs": np.asarray(handle["Vs_realization_2D"][:], dtype=np.float64),
            "zeta": applied_zeta(handle),
            "acc": np.asarray(handle["recorders/accel/surface"][:], dtype=np.float64),
            "time": np.asarray(handle["recorders/accel/time"][:], dtype=np.float64),
            "f1": float(params["damping_freq_first"]),
            "f2": float(params["damping_freq_second"]),
            "motion": float(grid["motion_freq"]),
            "ricker": str(params.get("ricker_is", "acceleration")),
            "damping_applied": params.get("damping_applied"),
        }


def _ttf():
    from dense_labels import _ttf as load

    return load()


def _score_arrays(mesh: dict, ttf) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    vs, zeta = mesh["vs"], mesh["zeta"]
    cols = np.arange(0, vs.shape[1], STRIDE_M)
    time = mesh["time"]
    dt = float(time[1] - time[0])
    freq, raw = tf_over_input(
        mesh["acc"][:, cols],
        seiskit_input_acceleration(mesh["motion"], 0.5, time, mesh["ricker"]),
        dt,
        ttf,
        window_s=WINDOW_S,
    )
    truth = raw / SCALE
    tf1d = np.empty_like(truth)
    for j, col in enumerate(cols):
        tf1d[j] = surface_over_outcrop(
            freq, vs[:, col], dz=1.0, xi=_xi(freq, zeta[:, col], mesh["f1"], mesh["f2"]), rho=RHO,
        )
    record_s = float(time[-1] - time[0])
    return freq, truth, tf1d, record_s


def _rows_for(name: str, sample_id: str, width: int, freq, truth, tf1d, record_s: float, shift_m: float, domain: str) -> list[dict]:
    log_t = np.log(np.maximum(truth, 1e-12))
    log_d = np.log(np.maximum(tf1d, 1e-12))
    rows = []
    for band, (lo, hi) in BANDS.items():
        mask = (freq >= lo) & (freq <= hi)
        gap = np.mean(np.abs(log_t[:, mask] - log_d[:, mask]), axis=1)
        rows.append({
            "sample_id": sample_id,
            "name": name,
            "width_m": width,
            "band": band,
            "n_stations": int(truth.shape[0]),
            "stride_m": STRIDE_M,
            "mean_abs_log": float(np.mean(gap)),
            "mean_pearson_log": _pearson_rows(log_t[:, mask], log_d[:, mask]),
            "mesh_shift_m": float(shift_m),
            "domain_flag": domain,
            "record_s": record_s,
            "scale": SCALE,
            "fft_window_s": WINDOW_S,
            "network": "not_scored",
        })
    return rows


def main() -> None:
    if not FULL_QVS.is_file():
        raise SystemExit(f"missing {FULL_QVS}")
    ttf = _ttf()
    full = _read(FULL_QVS)
    rows: list[dict] = []
    ids: list[str] = []
    for crop in CROPS:
        name = crop["name"]
        center = float(crop["center_x_m"])
        for width in (500, 1000, 1500, 2000):
            if width == 2000:
                c0, n, shift = span_columns(center, width)
                mesh = {
                    **full,
                    "vs": full["vs"][:, c0 : c0 + n],
                    "zeta": full["zeta"][:, c0 : c0 + n],
                    "acc": full["acc"][:, c0 : c0 + n],
                }
                domain = "window_of_full_qvs_meets_domain_edge" if abs(shift) > 0 else "window_of_full_qvs_interior"
                sample_id = f"{crop['sample_id']}_w2000_of_full_qvs"
            else:
                path = CROP_500 / f"{crop['sample_id']}.h5" if width == 500 else CROP_WIDE / f"{crop['sample_id']}_w{width}.h5"
                if not path.is_file():
                    raise SystemExit(f"missing {path}")
                mesh = _read(path)
                shift = 0.0
                domain = "crop_h5"
                sample_id = path.stem
            print(f"score {name} {width} m", flush=True)
            freq, truth, tf1d, record_s = _score_arrays(mesh, ttf)
            rows.extend(_rows_for(name, sample_id, width, freq, truth, tf1d, record_s, shift, domain))
            ids.append(f"{name}:{width}")
            mid = rows[-2]
            print(
                f"  mid |dlog| {mid['mean_abs_log']:.4f} pearson {mid['mean_pearson_log']:.4f} "
                f"shift {shift:.0f} m",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    write_provenance(OUT.with_name("case7_column.provenance.json"), {
        "label_campaign": "case7a Q-Vs crops 500/1000/1500 and 2000 m windows of full_qvs.h5",
        "oned_engine": "section_1d.surface_over_outcrop",
        "outcrop_convention": "surface over elastic half-space, Rayleigh at the H5 (f1, f2)",
        "damping_source": "applied_zeta; damping_applied is absent so zeta is Q(Vs), with free-field pads restored when the stored mesh is shorter than the solved length",
        "fft_window_s": WINDOW_S,
        "checkpoint": None,
        "epoch": None,
        "dtype": "not_applicable_no_forward",
        "bands": list(BANDS),
        "aggregation_unit": "station",
        "sample_id_hash": sample_id_hash(ids),
        "git_commit": git_commit(),
        "definition_difference": (
            "OpenSees |TF| / scale versus the corrected column. "
            f"Scale {SCALE} is the acceleration value in data/vardomain/outcrop_fixed_window.log. "
            f"Stations every {STRIDE_M} m. mean_abs_log is the mean over stations of the mean |log| gap. "
            "mean_pearson_log is the mean of per-station Pearson of demeaned log|TF|. "
            "No network forward. full_svlab.h5 was not read. "
            "Pre-fix Case 7 Pearson tables and figures/stage2/case7_width.json are a different checkpoint and engine and are not this file. "
            "A 2000 m row with nonzero mesh_shift_m was slid onto the 4000 m mesh and meets the domain edge. "
            "Hallal's 600 m, 1700 m, and 1 km lengths are not computed here."
        ),
        "bootstrap_unit": "not resampled; one section per crop",
        "skill_reduction": "not a skill table",
        "network": "not_scored",
        "stride_m": STRIDE_M,
        "scale": SCALE,
        "scale_source": "data/vardomain/outcrop_fixed_window.log acceleration",
    })
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
