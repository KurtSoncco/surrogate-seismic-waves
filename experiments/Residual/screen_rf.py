"""Random Forest + permutation importance screening."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import config
from build_table import feature_columns


def run_rf_importance(
    df: pd.DataFrame,
    target: str,
    *,
    feature_names: Sequence[str] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    feature_names = list(feature_names or feature_columns())
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    max_rows = 50_000
    if len(y) > max_rows:
        idx = rng.choice(len(y), size=max_rows, replace=False)
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    rf = RandomForestRegressor(
        n_estimators=config.N_RF_ESTIMATORS,
        random_state=seed,
        n_jobs=-1,
        max_depth=20,
        min_samples_leaf=5,
    )
    rf.fit(X_train, y_train)
    train_r2 = float(rf.score(X_train, y_train))
    test_r2 = float(rf.score(X_test, y_test))

    perm = permutation_importance(
        rf,
        X_test,
        y_test,
        n_repeats=config.N_PERMUTATION_REPEATS,
        random_state=seed,
        n_jobs=-1,
        scoring="r2",
    )
    out = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "perm_importance_mean": perm.importances_mean,
                "perm_importance_std": perm.importances_std,
                "rf_gini_importance": rf.feature_importances_,
            }
        )
        .sort_values("perm_importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    metrics = {"train_r2": train_r2, "test_r2": test_r2, "n_rows": int(len(y))}
    return out, metrics


def save_rf(
    df_imp: pd.DataFrame,
    metrics: dict,
    target: str,
    results_dir: Path | None = None,
) -> Path:
    results_dir = results_dir or config.RESULTS_DIR
    path = results_dir / f"rf_perm_{target}.csv"
    df_imp.to_csv(path, index=False)
    meta_path = results_dir / f"rf_metrics_{target}.json"
    import json

    meta_path.write_text(json.dumps(metrics, indent=2))
    return path
