#!/usr/bin/env python3
"""Provenance stamps that do not rescore a network.

Writes the cache-engine spot check, the Treasure Island skill-identity gap,
the domain-rationale citations, and the dtype stamp. Numbers in the generated
narrative are read from these tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VARDOMAIN = HERE.parents[1]
REPO = VARDOMAIN.parents[1]
CASE7A = REPO / "experiments" / "case7a"
ANALYSIS = VARDOMAIN / "analysis"
for path in (CASE7A, ANALYSIS):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from score_harness import git_commit, sample_id_hash, write_provenance  # noqa: E402
from section_1d import RHO, surface_over_outcrop  # noqa: E402

FROZEN = ANALYSIS / "frozen"
AUDIT = ANALYSIS / "audit"
CACHE = REPO / "data" / "vardomain" / "cache"
FAMILIES = ("iid", "ood_dipping", "ood_three_layer")


def _xi(freq: np.ndarray, zeta_col: np.ndarray, f1: float, f2: float) -> np.ndarray:
    w1, w2 = 2.0 * np.pi * float(f1), 2.0 * np.pi * float(f2)
    zeta = np.asarray(zeta_col, dtype=np.float64)
    a0 = 2.0 * zeta * w1 * w2 / (w1 + w2)
    a1 = 2.0 * zeta / (w1 + w2)
    omega = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)
    return 0.5 * (a0[:, None] / omega[None, :] + a1[:, None] * omega[None, :])


def _one_per_group(split: np.ndarray, group: np.ndarray, which: str) -> list[int]:
    picked = []
    seen = set()
    for index in np.flatnonzero(split == which):
        key = int(group[index])
        if key in seen:
            continue
        seen.add(key)
        picked.append(int(index))
    return picked


def cache_engine() -> None:
    freq = np.load(CACHE / "freq.npy")
    rows = []
    ids = []
    for family in FAMILIES:
        root = CACHE / family
        vs = np.load(root / "vs.npy", mmap_mode="r")
        zeta = np.load(root / "zeta.npy", mmap_mode="r")
        prior = np.load(root / "log_tf1d.npy", mmap_mode="r")
        f1 = np.load(root / "rayleigh_f1.npy")
        f2 = np.load(root / "rayleigh_f2.npy")
        split = np.load(root / "split.npy")
        group = np.load(root / "group.npy")
        n_active = np.load(root / "n_active.npy")
        rec_x = np.load(root / "rec_x.npy")
        x = np.asarray(rec_x if rec_x.ndim == 1 else rec_x[0])
        center_col = int(vs.shape[-1] // 2)
        station = int(np.argmin(np.abs(x - center_col)))
        edge = int(np.argmin(np.abs(x - x.min())))
        for j in _one_per_group(split, group, "test"):
            na = int(n_active[j])
            for station_index, role in ((station, "center"), (edge, "edge")):
                col = int(np.rint(x[station_index]))
                col = int(np.clip(col, 0, vs.shape[-1] - 1))
                vs_col = np.asarray(vs[j, :na, col], dtype=np.float64)
                z_col = np.asarray(zeta[j, :na, col], dtype=np.float64)
                tf = surface_over_outcrop(freq, vs_col, dz=1.0, xi=_xi(freq, z_col, float(f1[j]), float(f2[j])), rho=RHO)
                stored = np.asarray(prior[j, station_index], dtype=np.float64)
                err = float(np.max(np.abs(np.log(np.maximum(tf, 1e-12)) - stored)))
                rows.append({
                    "family": family,
                    "local_index": j,
                    "station_role": role,
                    "max_abs_log": err,
                    "stored_dtype": "float16",
                })
                ids.append(f"{family}:{j}:{role}")
        print(f"cache {family} checked {sum(1 for row in rows if row['family']==family)}", flush=True)
    frame = pd.DataFrame(rows)
    summary = []
    for family in FAMILIES:
        part = frame[frame["family"] == family]
        summary.append({
            "family": family,
            "station_role": "all_checked",
            "local_index": -1,
            "max_abs_log": float(part["max_abs_log"].max()),
            "n_columns": int(len(part)),
            "stored_dtype": "float16",
        })
    detail = FROZEN / "cache_engine_detail.csv"
    headline = FROZEN / "cache_engine_check.csv"
    frame.to_csv(detail, index=False)
    pd.DataFrame(summary).to_csv(headline, index=False)
    worst = float(frame["max_abs_log"].max())
    write_provenance(headline.with_name("cache_engine_check.provenance.json"), {
        "label_campaign": "iid, ood_dipping, ood_three_layer test; one record per Sobol group",
        "oned_engine": "section_1d.surface_over_outcrop compared with stored cache log_tf1d",
        "outcrop_convention": "surface over elastic half-space, Rayleigh at the stored (f1, f2)",
        "damping_source": "stored cache zeta",
        "fft_window_s": None,
        "checkpoint": None,
        "epoch": None,
        "dtype": "float16",
        "bands": ["0.1-10 Hz, the stored frequency grid"],
        "aggregation_unit": "column",
        "sample_id_hash": sample_id_hash(ids),
        "git_commit": git_commit(),
        "definition_difference": (
            f"Spot check only. Maximum absolute log gap on the checked columns is {worst:.6e}. "
            "cache/meta.json does not name the engine. The gap is at float16 scale. "
            "This is not a rewrite of the cache and not a full-station audit."
        ),
        "bootstrap_unit": "not resampled",
        "skill_reduction": "not a skill table",
        "max_abs_log": worst,
    })
    print(f"wrote {headline}")


def treasure_identity() -> None:
    long_path = ANALYSIS / "treasure_long.csv"
    if not long_path.is_file():
        raise SystemExit(f"missing {long_path}")
    frame = pd.read_csv(long_path)
    mid = frame[(frame["band"] == "mid") & (frame["method"] == "P128n128")]
    model = mid[mid["metric"] == "g_log_l1"]["value"].to_numpy()
    prior = mid[mid["metric"] == "prior_log_l1"]["value"].to_numpy()
    skill = mid[mid["metric"] == "skill"]["value"].to_numpy()
    rows = pd.DataFrame([
        {"name": "median_station_g_log_l1", "value": float(np.median(model)), "n_stations": int(model.size)},
        {"name": "median_station_prior_log_l1", "value": float(np.median(prior)), "n_stations": int(prior.size)},
        {"name": "median_station_skill", "value": float(np.median(skill)), "n_stations": int(skill.size)},
        {
            "name": "one_minus_ratio_of_medians",
            "value": float(1.0 - np.median(model) / np.median(prior)),
            "n_stations": int(model.size),
        },
        {
            "name": "skill_of_mean_errors",
            "value": float(1.0 - np.mean(model) / np.mean(prior)),
            "n_stations": int(model.size),
        },
    ])
    path = FROZEN / "treasure_skill_identity.csv"
    rows.to_csv(path, index=False)
    write_provenance(path.with_name("treasure_skill_identity.provenance.json"), {
        "label_campaign": "case7a full_qvs, treasure_long.csv mid band",
        "oned_engine": "stored treasure_long prior_log_l1",
        "outcrop_convention": "same as ladder_treasure",
        "damping_source": "qvs",
        "fft_window_s": 1310.72,
        "checkpoint": "/home/kurt-/surrogate-seismic-waves/checkpoints/vardomain/v2_ablations/P128n128",
        "epoch": 46,
        "dtype": "float32",
        "bands": ["mid"],
        "aggregation_unit": "station",
        "sample_id_hash": sample_id_hash(["treasure_island"]),
        "git_commit": git_commit(),
        "definition_difference": (
            "median_station_skill is the median of per-station skills. "
            "one_minus_ratio_of_medians is 1 minus the ratio of the two median log-L1 values. "
            "They are not the same reduction. skill_of_mean_errors matches the pooled skill."
        ),
        "bootstrap_unit": "not resampled",
        "skill_reduction": "identity check, not a headline skill",
    })
    print(f"wrote {path}")


def domain_rationale() -> None:
    rows = pd.DataFrame([
        {"name": "hallal_fundamental_mode_m", "value": 600, "role": "citation_not_a_repo_result"},
        {"name": "hallal_other_basin_effects_m", "value": 1700, "role": "citation_not_a_repo_result"},
        {"name": "hallal_ground_from_sensor_m", "value": 1000, "role": "citation_not_a_repo_result"},
        {"name": "future_depth_m", "value": 250, "role": "future_work"},
        {"name": "future_record_s_short", "value": 200, "role": "future_work"},
        {"name": "future_record_s_long", "value": 400, "role": "future_work"},
        {"name": "future_pad_m", "value": 500, "role": "future_work"},
        {"name": "future_width_m_500", "value": 500, "role": "future_work"},
        {"name": "future_width_m_1000", "value": 1000, "role": "future_work"},
        {"name": "future_width_m_1500", "value": 1500, "role": "future_work"},
        {"name": "future_width_m_2000", "value": 2000, "role": "future_work"},
    ])
    path = AUDIT / "domain_rationale.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)
    write_provenance(path.with_name("domain_rationale.provenance.json"), {
        "label_campaign": "not a scored campaign",
        "oned_engine": "not used",
        "outcrop_convention": "not used",
        "damping_source": "not used",
        "fft_window_s": None,
        "checkpoint": None,
        "epoch": None,
        "dtype": "not_applicable",
        "bands": ["not a band table"],
        "aggregation_unit": "citation",
        "sample_id_hash": sample_id_hash(rows["name"].tolist()),
        "git_commit": git_commit(),
        "definition_difference": (
            "Hallal, Cox, and Vantassel (JGGE 2023) and Hallal's dissertation supply the "
            "600 m, 1700 m, and 1000 m lengths. They were not computed in this repository. "
            "The future-work lengths are the deferred padded campaign. No OpenSees job is launched."
        ),
        "bootstrap_unit": "not resampled",
        "skill_reduction": "not a skill table",
    })
    print(f"wrote {path}")


def dtype_stamp() -> None:
    source = json.loads((ANALYSIS / "dtype_comparison.json").read_text())
    path = FROZEN / "dtype_stamp.csv"
    pd.DataFrame([{
        "name": "max_abs_g_log_l1",
        "value": float(source["max_abs_g_log_l1"]),
        "device": source["device"],
        "cuda": bool(source["cuda"]),
        "autocast_engaged": bool(source["autocast_engaged"]),
        "frozen_dtype": source["frozen_dtype"],
        "width_m": int(source["width_m"]),
        "n_stations": int(source["n_stations"]),
    }]).to_csv(path, index=False)
    write_provenance(path.with_name("dtype_stamp.provenance.json"), {
        "label_campaign": "one centered 500 m IID test window",
        "oned_engine": "not used",
        "outcrop_convention": "not used",
        "damping_source": "qvs",
        "fft_window_s": None,
        "checkpoint": source["model_dir"],
        "epoch": int(source["epoch"]),
        "dtype": source["frozen_dtype"],
        "bands": ["low", "mid", "high", "overall"],
        "aggregation_unit": "station",
        "sample_id_hash": sample_id_hash([f"iid-local-{source['local_index']}"]),
        "git_commit": git_commit(),
        "definition_difference": source["autocast_note"],
        "bootstrap_unit": "not resampled; one fixed window",
        "skill_reduction": "not a skill table",
    })
    print(f"wrote {path}")


def heatmap_exemplars() -> None:
    path = FROZEN / "heatmap_exemplars.csv"
    pd.DataFrame([
        {"name": "dipping_run", "value": 217, "width_m": 500, "sample_id": "run_217"},
        {"name": "blind_section", "value": 297, "width_m": 500, "sample_id": "man_0297"},
    ]).to_csv(path, index=False)
    write_provenance(path.with_name("heatmap_exemplars.provenance.json"), {
        "label_campaign": "stored 500 m exemplars already drawn; fields not recomputed",
        "oned_engine": "not recomputed",
        "outcrop_convention": "not recomputed",
        "damping_source": "qvs on the blind section title",
        "fft_window_s": None,
        "checkpoint": "/home/kurt-/surrogate-seismic-waves/checkpoints/vardomain/v2_ablations/P128n128",
        "epoch": 46,
        "dtype": "float32",
        "bands": ["not a score table"],
        "aggregation_unit": "exemplar",
        "sample_id_hash": sample_id_hash(["run_217", "man_0297"]),
        "git_commit": git_commit(),
        "definition_difference": (
            "Dipping test run 217 and blind section man_0297 are the stored high-signal 500 m panels. "
            "They are not representative records and not wide-domain evidence. The spatial fields were not redrawn."
        ),
        "bootstrap_unit": "not resampled",
        "skill_reduction": "not a skill table",
    })
    print(f"wrote {path}")


def main() -> None:
    cache_engine()
    treasure_identity()
    domain_rationale()
    dtype_stamp()
    heatmap_exemplars()


if __name__ == "__main__":
    main()
