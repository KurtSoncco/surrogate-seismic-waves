#!/usr/bin/env python3
"""Quantify how much per-sample linf comes from TF valleys vs peaks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

GIFNO_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("GIFNO_DATA_ROOT", str(GIFNO_DIR / "dummy_data"))
if str(GIFNO_DIR) not in sys.path:
    sys.path.insert(0, str(GIFNO_DIR))

import config  # noqa: E402
from metrics import (  # noqa: E402
    per_sample_linf_numpy,
    per_sample_linf_split_numpy,
)


def valley_linf_fraction(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Fraction of samples where valley linf exceeds peak linf."""
    linf = per_sample_linf_numpy(predictions, targets)
    linf_valley, linf_peak = per_sample_linf_split_numpy(predictions, targets)
    valley_dominant = linf_valley >= linf_peak
    return {
        "n_samples": float(len(predictions)),
        "frac_valley_dominates_linf": float(np.mean(valley_dominant)),
        "mean_linf": float(np.mean(linf)),
        "mean_linf_valley": float(np.mean(linf_valley)),
        "mean_linf_peak": float(np.mean(linf_peak)),
        "p90_linf_valley": float(np.percentile(linf_valley, 90)),
        "p90_linf_peak": float(np.percentile(linf_peak, 90)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Valley vs peak linf diagnostics")
    parser.add_argument(
        "--pred",
        type=Path,
        help="Path to predictions .npy with shape (N, Nx, N_freq)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="Path to targets .npy with shape (N, Nx, N_freq)",
    )
    args = parser.parse_args()

    if args.pred and args.target:
        predictions = np.load(args.pred)
        targets = np.load(args.target)
    else:
        # Synthetic demo when no eval arrays are provided.
        n, nx, n_freq = 8, config.NX, 128
        rec = config.recorder_x_indices()
        freq = np.logspace(-1, 1, n_freq)
        predictions = np.zeros((n, nx, n_freq), dtype=np.float32)
        targets = np.zeros((n, nx, n_freq), dtype=np.float32)
        for i in range(n):
            for x in rec:
                base = 1.0 + 0.4 * np.sin(2 * np.pi * freq / 2.5 + i * 0.1)
                dip = 0.15 * np.exp(-((np.arange(n_freq) - (40 + i * 3)) ** 2) / 80.0)
                targets[i, x, :] = base - dip
                predictions[i, x, :] = base - 0.5 * dip

    stats = valley_linf_fraction(predictions, targets)
    for key, val in stats.items():
        print(f"{key}: {val:.4f}" if key != "n_samples" else f"{key}: {int(val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
