"""Mutual information regression screening."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

import config
from build_table import feature_columns


def run_mi(
    df: pd.DataFrame,
    target: str,
    *,
    feature_names: Sequence[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    feature_names = list(feature_names or feature_columns())
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    # subsample rows if huge
    rng = np.random.default_rng(seed)
    max_rows = 50_000
    if len(y) > max_rows:
        idx = rng.choice(len(y), size=max_rows, replace=False)
        X, y = X[idx], y[idx]
    mi = mutual_info_regression(X, y, random_state=seed, n_neighbors=5)
    out = pd.DataFrame({"feature": feature_names, "mi": mi})
    return out.sort_values("mi", ascending=False).reset_index(drop=True)


def run_mi_by_band(
    df: pd.DataFrame,
    target: str,
    *,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    bands = {
        "low": config.FREQ_BAND_LOW,
        "mid": config.FREQ_BAND_MID,
        "high": config.FREQ_BAND_HIGH,
        "all": (config.FREQ_START_HZ, config.FREQ_END_HZ),
    }
    frames = []
    for name, (lo, hi) in bands.items():
        sub = df[(df["freq_hz"] >= lo) & (df["freq_hz"] <= hi)]
        if sub.empty:
            continue
        rank = run_mi(sub, target, feature_names=feature_names)
        rank["band"] = name
        frames.append(rank)
    return pd.concat(frames, ignore_index=True)


def save_mi(df_mi: pd.DataFrame, target: str, results_dir: Path | None = None) -> Path:
    results_dir = results_dir or config.RESULTS_DIR
    path = results_dir / f"mi_{target}.csv"
    df_mi.to_csv(path, index=False)
    return path
