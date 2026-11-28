#!/usr/bin/env python3
"""How many residual POD modes the high-frequency wiggles actually need.

The split is the seed-0 split of the spectral autoencoder. Modes are fit on
the training curves only. Two errors are reported at each rank:

- the mean absolute error of the curve, by frequency band
- the same error after a 1/6-octave moving mean is removed, which keeps the
  peaks and drops the slow envelope

    python pod_rank.py --cache ... --out .../pod_rank
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import Cache
from pretrain_ae import _refuse_shipped, select_curves, split_rows
from screen_hifreq import BANDS
from train import _curves

RANKS = (32, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768)
OCTAVE_FRACTION = 6


def smooth_bins(curves: np.ndarray, window: int) -> np.ndarray:
    """Moving mean along frequency. ``window`` is odd; the edges are reflected."""
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    pad = window // 2
    left = curves[:, 1 : pad + 1][:, ::-1]
    right = curves[:, -pad - 1 : -1][:, ::-1]
    padded = np.concatenate([left, curves, right], axis=1)
    cumulative = np.cumsum(padded, axis=1)
    cumulative = np.concatenate([np.zeros((curves.shape[0], 1)), cumulative], axis=1)
    return (cumulative[:, window:] - cumulative[:, :-window]) / window


def oscillation_window(freq: np.ndarray, octave_fraction: int = OCTAVE_FRACTION) -> int:
    """Odd bin count spanning one part of an octave on a log-frequency grid."""
    log_f = np.log(np.asarray(freq, dtype=np.float64))
    step = float(np.median(np.diff(log_f)))
    bins = int(round(np.log(2.0) / octave_fraction / step))
    bins = max(3, bins)
    if bins % 2 == 0:
        bins += 1
    return bins


def _band_mean(freq: np.ndarray, per_bin: np.ndarray) -> dict[str, float]:
    out = {}
    for lo, hi in BANDS:
        chosen = (freq >= lo) & (freq <= hi)
        out[f"{lo:g}-{hi:g}"] = float(per_bin[chosen].mean())
    return out


def _modes_of(weighted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = weighted.T @ weighted / max(weighted.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return np.maximum(vals[order], 0.0), vecs[:, order]


def weighted_band_errors(
    train: np.ndarray, hold: np.ndarray, freq: np.ndarray, weight: np.ndarray, ranks: tuple[int, ...],
) -> list[dict]:
    """POD of w(f) times the centered residual. The weight is divided back out before the error."""
    weight = np.asarray(weight, dtype=np.float64)
    if weight.shape != (train.shape[1],) or np.any(weight <= 0):
        raise ValueError("weight must be a positive value at each frequency")
    mean = train.mean(axis=0)
    vals, vecs = _modes_of((train - mean) * weight)
    rows = []
    centered = (hold - mean) * weight
    for rank in ranks:
        taken = min(int(rank), vecs.shape[1])
        modes = vecs[:, :taken].T
        recon = mean + ((centered @ modes.T) @ modes) / weight
        err = np.abs(recon - hold).mean(axis=0)
        rows.append({"rank": taken, "mae": float(err.mean()), "bands": _band_mean(freq, err)})
    return rows


def frequency_weights(freq: np.ndarray, train: np.ndarray) -> dict[str, np.ndarray]:
    """Weights that rise toward 10 Hz, plus per-bin whitening. Each has median 1."""
    freq = np.asarray(freq, dtype=np.float64)
    span = np.log(freq / freq[0]) / np.log(freq[-1] / freq[0])
    std = np.maximum((train - train.mean(axis=0)).std(axis=0), 1e-8)
    raw = {
        "uniform": np.ones_like(freq),
        "log_x4": 1.0 + 3.0 * span,
        "log_x10": 1.0 + 9.0 * span,
        "sqrt_f": np.sqrt(freq),
        "whiten": 1.0 / std,
    }
    return {name: value / np.median(value) for name, value in raw.items()}


def rank_table(
    train: np.ndarray, hold: np.ndarray, freq: np.ndarray, ranks: tuple[int, ...],
) -> dict:
    mean = train.mean(axis=0)
    centered = train - mean
    cov = centered.T @ centered / max(train.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    total = float(vals.sum())
    window = oscillation_window(freq)
    hold_c = hold - mean
    rows = []
    for rank in ranks:
        taken = min(int(rank), vecs.shape[1])
        modes = vecs[:, :taken].T
        recon = (hold_c @ modes.T) @ modes
        err = np.abs(recon - hold_c).mean(axis=0)
        wiggle = smooth_bins(recon - hold_c, window)
        wiggle_err = np.abs((recon - hold_c) - wiggle).mean(axis=0)
        rows.append({
            "rank": taken,
            "variance_explained": float(vals[:taken].sum() / total) if total else 1.0,
            "mae": float(err.mean()),
            "bands": _band_mean(freq, err),
            "oscillation_mae": float(wiggle_err.mean()),
            "oscillation_bands": _band_mean(freq, wiggle_err),
        })
    return {"window_bins": window, "octave_fraction": OCTAVE_FRACTION, "ranks": rows, "eigenvalues": vals[: max(ranks)].tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _refuse_shipped(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    cache = Cache(args.cache)
    freq = np.asarray(cache.freq, dtype=np.float64)
    curves, _groups = _curves(cache, "residual")
    curves = select_curves(curves, 0, args.seed)
    train_idx, hold_idx = split_rows(curves.shape[0], args.seed)
    print(f"residual train {train_idx.size} hold {hold_idx.size} freq {freq.size}", flush=True)
    table = rank_table(curves[train_idx], curves[hold_idx], freq, RANKS)
    report = {
        "target": "log|TF| - log|TF_1D|, training IID, seed-0 holdout",
        "note": "Oscillation error is the curve error after a 1/6-octave moving mean is removed.",
        **table,
    }
    (args.out / "report.json").write_text(json.dumps(report))
    print(f"oscillation window {table['window_bins']} bins (1/{table['octave_fraction']} octave)", flush=True)
    print(f"{'rank':>6} {'var':>7} {'mae':>8} {'5-10':>8} {'wiggle':>8} {'wiggle5-10':>10}", flush=True)
    for row in table["ranks"]:
        print(
            f"{row['rank']:6d} {row['variance_explained']:7.4f} {row['mae']:8.4f} "
            f"{row['bands']['5-10']:8.4f} {row['oscillation_mae']:8.4f} "
            f"{row['oscillation_bands']['5-10']:10.4f}",
            flush=True,
        )
    train, hold = curves[train_idx], curves[hold_idx]
    weights = frequency_weights(freq, train)
    compared = {}
    focus = (32, 64, 96, 128, 160, 192)
    print(f"{'weight':>10} {'rank':>6} {'mae':>8} {'0.1-0.4':>8} {'5-10':>8}", flush=True)
    for name, weight in weights.items():
        rows = weighted_band_errors(train, hold, freq, weight, focus)
        compared[name] = {"weight": weight.tolist(), "ranks": rows}
        for row in rows:
            print(
                f"{name:>10} {row['rank']:6d} {row['mae']:8.4f} "
                f"{row['bands']['0.1-0.4']:8.4f} {row['bands']['5-10']:8.4f}",
                flush=True,
            )
    std = (train - train.mean(axis=0)).std(axis=0)
    report["per_bin_std_bands"] = _band_mean(freq, std)
    report["weighted"] = {
        name: {"ranks": payload["ranks"]} for name, payload in compared.items()
    }
    report["weights"] = {name: payload["weight"] for name, payload in compared.items()}
    (args.out / "report.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
