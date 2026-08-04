"""Compare OrbitAll vs GIFNO-XT feature sets with MI and RF only (no SHAP)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import config


def _subsample_xy(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    feats = [f for f in features if f in df.columns]
    X = df[feats].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    rng = np.random.default_rng(seed)
    if len(y) > max_rows:
        idx = rng.choice(len(y), size=max_rows, replace=False)
        X, y = X[idx], y[idx]
    return X, y, feats


def run_mi_rf(
    df: pd.DataFrame,
    target: str,
    set_name: str,
    features: Sequence[str],
    *,
    seed: int = 42,
    max_rows: int = 30_000,
) -> Dict:
    X, y, feats = _subsample_xy(df, features, target, max_rows=max_rows, seed=seed)
    print(f"  MI on {len(y)} rows × {len(feats)} features…", flush=True)
    mi = mutual_info_regression(X, y, random_state=seed, n_neighbors=5)
    mi_df = (
        pd.DataFrame({"feature": feats, "mi": mi})
        .sort_values("mi", ascending=False)
        .reset_index(drop=True)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    n_estimators = min(100, int(config.N_RF_ESTIMATORS))
    n_repeats = min(5, int(config.N_PERMUTATION_REPEATS))
    print(f"  RF fit ({n_estimators} trees)…", flush=True)
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=2,
        max_depth=16,
        min_samples_leaf=5,
    )
    rf.fit(X_train, y_train)
    metrics = {
        "train_r2": float(rf.score(X_train, y_train)),
        "test_r2": float(rf.score(X_test, y_test)),
        "n_rows": int(len(y)),
        "n_features": len(feats),
    }
    print(f"  permutation importance ({n_repeats} repeats)…", flush=True)
    perm = permutation_importance(
        rf,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=2,
        scoring="r2",
    )
    rf_df = (
        pd.DataFrame(
            {
                "feature": feats,
                "perm_importance_mean": perm.importances_mean,
                "perm_importance_std": perm.importances_std,
                "rf_gini_importance": rf.feature_importances_,
            }
        )
        .sort_values("perm_importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "set_name": set_name,
        "target": target,
        "features": feats,
        "metrics": metrics,
        "mi": mi_df,
        "rf": rf_df,
    }


def save_comparison(result: Dict, results_dir: Path | None = None) -> Path:
    results_dir = results_dir or config.RESULTS_DIR
    tag = f"{result['set_name']}_{result['target']}"
    result["mi"].to_csv(results_dir / f"compare_mi_{tag}.csv", index=False)
    result["rf"].to_csv(results_dir / f"compare_rf_{tag}.csv", index=False)
    summary = {
        "set_name": result["set_name"],
        "target": result["target"],
        "metrics": result["metrics"],
        "mi_top": result["mi"].head(8)["feature"].tolist(),
        "rf_top": result["rf"].head(8)["feature"].tolist(),
    }
    path = results_dir / f"compare_summary_{tag}.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


def plot_set_comparison(
    summaries: List[Dict],
    target: str,
    *,
    results_dir: Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    results_dir = results_dir or config.RESULTS_DIR
    rows = [s for s in summaries if s["target"] == target]
    names = [s["set_name"] for s in rows]
    test_r2 = [s["metrics"]["test_r2"] for s in rows]
    train_r2 = [s["metrics"]["train_r2"] for s in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, train_r2, w, label="train R²", color="#6c757d")
    ax.bar(x + w / 2, test_r2, w, label="test R²", color="#bc4749")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("R²")
    ax.set_title(f"Feature-set RF score → {target}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = results_dir / f"compare_r2_{target}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
