#!/usr/bin/env python3
"""Low-band gap between 1D and OpenSees on the 64 round-0 sections.

Uses the dense labels already in ``data/vardomain/section_labels/round0``
(surface / input and the column 1D outcrop TF) and the feature geometry in
``sections/round0.csv``. The band is 0.1–0.4 Hz, where Case 7's thinning crop
and the SVLAB west both leave 1D.

    python section_lowband.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parents[1]
LABELS = REPO / "data" / "vardomain" / "section_labels" / "round0"
MANIFEST = CASE_DIR / "sections" / "round0.csv"
OUT = REPO / "checkpoints" / "case7_generalization" / "context_gate" / "section_lowband.json"
BAND = (0.1, 0.4)
DIST_BINS = ((0.0, 0.0), (0.0, 250.0), (250.0, 500.0), (500.0, 1.0e9))
DIST_NAMES = ("inside", "0-250 m", "250-500 m", ">500 m")


def _rows() -> dict[str, dict]:
    with MANIFEST.open() as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def _spans(row: dict) -> list[tuple[float, float]]:
    if row["feature_name"] == "flat":
        return []
    lx = float(row["lx_m"])
    xc = float(row["feature_x_frac"]) * lx
    half = float(row["feature_width_m"]) / 2.0
    return [(xc - half, xc + half)]


def _distance(x: float, spans: list[tuple[float, float]]) -> float:
    if not spans:
        return float("inf")
    best = float("inf")
    for a, b in spans:
        if a <= x <= b:
            return 0.0
        best = min(best, abs(x - a), abs(x - b))
    return best


def _bin(dist: float) -> str:
    if dist == 0.0:
        return DIST_NAMES[0]
    for name, (lo, hi) in zip(DIST_NAMES[1:], DIST_BINS[1:]):
        if lo < dist <= hi:
            return name
    return DIST_NAMES[-1]


def _station_stats(tf: np.ndarray, tf1d: np.ndarray, freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = (freq >= BAND[0]) & (freq <= BAND[1])
    a = np.log(np.maximum(tf[:, m], 1e-12))
    b = np.log(np.maximum(tf1d[:, m], 1e-12))
    gap = np.mean(np.abs(a - b), axis=1)
    a0, b0 = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    pear = np.sum(a0 * b0, axis=1) / np.maximum(np.sqrt(np.sum(a0**2, axis=1) * np.sum(b0**2, axis=1)), 1e-12)
    return gap, pear


def _summarize(gap: np.ndarray, pear: np.ndarray) -> dict:
    return {
        "n": int(gap.size),
        "mean_abs_log": float(np.mean(gap)) if gap.size else None,
        "p90_abs_log": float(np.quantile(gap, 0.9)) if gap.size else None,
        "mean_pearson": float(np.mean(pear)) if pear.size else None,
        "frac_pearson_below_0.8": float(np.mean(pear < 0.8)) if pear.size else None,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rows = _rows()
    by_feature: dict[str, list] = {}
    by_dist: dict[str, list] = {name: [] for name in DIST_NAMES}
    per_section = []
    for path in sorted(args.labels.glob("sec_*.npz")):
        sid = path.stem
        row = rows[sid]
        data = np.load(path)
        gap, pear = _station_stats(data["tf"], data["tf1d"], data["freq"])
        spans = _spans(row)
        dists = np.array([_distance(float(x), spans) for x in data["x_m"]])
        per_section.append({"sample_id": sid, "feature": row["feature_name"], **_summarize(gap, pear)})
        by_feature.setdefault(row["feature_name"], []).append((gap, pear))
        if row["feature_name"] == "flat":
            continue
        for name in DIST_NAMES:
            keep = np.array([_bin(float(d)) == name for d in dists])
            if np.any(keep):
                by_dist[name].append((gap[keep], pear[keep]))

    def _cat(pairs: list) -> dict:
        if not pairs:
            return _summarize(np.zeros(0), np.zeros(0))
        return _summarize(np.concatenate([g for g, _ in pairs]), np.concatenate([p for _, p in pairs]))

    report = {
        "band_hz": list(BAND),
        "by_feature": {name: _cat(pairs) for name, pairs in sorted(by_feature.items())},
        "by_distance_nonflat": {name: _cat(by_dist[name]) for name in DIST_NAMES},
        "sections": per_section,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"band {BAND[0]}-{BAND[1]} Hz  -> {args.out}")
    print(f"{'feature':<10} {'n':>7} {'|dlog|':>8} {'p90':>8} {'pearson':>8} {'frac<0.8':>9}")
    for name, row in report["by_feature"].items():
        print(f"{name:<10} {row['n']:7d} {row['mean_abs_log']:8.3f} {row['p90_abs_log']:8.3f} {row['mean_pearson']:8.3f} {row['frac_pearson_below_0.8']:9.3f}")
    print("distance from a non-flat feature")
    for name in DIST_NAMES:
        row = report["by_distance_nonflat"][name]
        if not row["n"]:
            continue
        print(f"{name:<10} {row['n']:7d} {row['mean_abs_log']:8.3f} {row['p90_abs_log']:8.3f} {row['mean_pearson']:8.3f} {row['frac_pearson_below_0.8']:9.3f}")


if __name__ == "__main__":
    main()
