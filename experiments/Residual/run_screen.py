"""End-to-end OrbitAll feature screen on a GIFNO subsample.

Usage:
  uv run python experiments/Residual/run_screen.py --n-samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as script from repo root or experiment dir
_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

# ruff: noqa: E402
import config
from build_table import build_feature_table, feature_columns
from plots import plot_importance_bars, plot_r_central_3x3
from residual_target import (
    build_residual_cache,
    load_manifest,
    stratified_sample_indices,
)
from screen_mi import run_mi_by_band, save_mi
from screen_rf import run_rf_importance, save_rf


def _gate_summary(mi_all, rf_imp, metrics, target: str, top_k: int = 8) -> dict:
    mi_top = (
        mi_all[mi_all["band"] == "all"]
        .sort_values("mi", ascending=False)
        .head(top_k)["feature"]
        .tolist()
    )
    rf_top = rf_imp.head(top_k)["feature"].tolist()
    both = [f for f in mi_top if f in rf_top]
    return {
        "target": target,
        "rf_test_r2": metrics["test_r2"],
        "rf_train_r2": metrics["train_r2"],
        "mi_top": mi_top,
        "rf_perm_top": rf_top,
        "agreed_top": both,
        "gate_pass_hint": bool(metrics["test_r2"] > 0.15 and len(both) >= 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="OrbitAll residual feature screen")
    p.add_argument("--n-samples", type=int, default=config.N_SAMPLES)
    p.add_argument("--k-xi", type=int, default=config.K_XI)
    p.add_argument("--n-freq-screen", type=int, default=config.N_FREQ_SCREEN)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--force", action="store_true", help="Recompute residual cache")
    args = p.parse_args()

    config.K_XI = args.k_xi
    config.SEED = args.seed
    config.N_FREQ_SCREEN = args.n_freq_screen

    print(f"DATA_ROOT={config.DATA_ROOT}")
    print(f"H5_DIR={config.H5_DIR}")
    print(
        f"TF={config.TF_PER_SAMPLE_PATH.exists()} manifest={config.MANIFEST_PATH.exists()}"
    )

    manifest = load_manifest()
    indices = stratified_sample_indices(manifest, args.n_samples, seed=args.seed)
    print(f"Subsample {len(indices)} / {len(manifest)} indices")

    cache_dir = build_residual_cache(indices, force=args.force)
    print(f"Residual cache: {cache_dir}")

    plot_path = plot_r_central_3x3(
        cache_dir, n_combos=config.N_PLOT_COMBOS, seed=args.seed
    )
    print(f"Wrote {plot_path}")

    print("Building feature table…")
    df = build_feature_table(cache_dir, n_freq_screen=args.n_freq_screen)
    print(f"Table shape={df.shape} cols={list(df.columns)[:12]}…")

    feats = feature_columns(args.k_xi)
    gates = []
    for target in ("R_col", "R_nom"):
        print(f"\n=== Screening target={target} ===")
        mi = run_mi_by_band(df, target, feature_names=feats)
        mi_path = save_mi(mi, target)
        print(f"MI → {mi_path}")
        rf_imp, metrics = run_rf_importance(
            df, target, feature_names=feats, seed=args.seed
        )
        rf_path = save_rf(rf_imp, metrics, target)
        print(f"RF → {rf_path}  test_R2={metrics['test_r2']:.3f}")
        plot_importance_bars(mi_path, rf_path, target)
        gates.append(_gate_summary(mi, rf_imp, metrics, target))

    summary_path = config.RESULTS_DIR / "gate_summary.json"
    summary_path.write_text(json.dumps(gates, indent=2))
    print(f"\nGate summary → {summary_path}")
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
