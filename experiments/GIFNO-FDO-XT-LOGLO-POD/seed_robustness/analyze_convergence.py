#!/usr/bin/env python3
"""
Plot seed-number CV convergence from seed_robustness_check.py outputs.

Example:
  uv run python seed_robustness/analyze_convergence.py \\
    --sample-dir ~/surrogate-seismic-waves/checkpoints/seed_robustness/sample_0000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_cv(rows: list[dict]) -> dict[int, dict[str, float]]:
    """Mean metrics per n_seeds across subsets."""
    by_n: dict[int, list[dict]] = {}
    for row in rows:
        n = int(row["n_seeds"])
        by_n.setdefault(n, []).append(row)

    out: dict[int, dict[str, float]] = {}
    for n, group in sorted(by_n.items()):
        out[n] = {
            "sigma_ln_rmse_truth": float(
                np.mean([float(r["sigma_ln_rmse_truth"]) for r in group])
            ),
            "sigma_ln_rmse_pred": float(
                np.mean([float(r["sigma_ln_rmse_pred"]) for r in group])
            ),
            "median_af_rmse_truth": float(
                np.mean([float(r["median_af_rmse_truth"]) for r in group])
            ),
            "median_af_rmse_pred": float(
                np.mean([float(r["median_af_rmse_pred"]) for r in group])
            ),
            "sigma_ln_mean_truth": float(
                np.mean([float(r["sigma_ln_mean_truth"]) for r in group])
            ),
            "sigma_ln_mean_pred": float(
                np.mean([float(r["sigma_ln_mean_pred"]) for r in group])
            ),
        }
    return out


def plot_sigma_ln_convergence(
    sample_dir: Path, agg: dict[int, dict[str, float]]
) -> None:
    ns = sorted(agg.keys())
    truth_rmse = [agg[n]["sigma_ln_rmse_truth"] for n in ns]
    pred_rmse = [agg[n]["sigma_ln_rmse_pred"] for n in ns]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, truth_rmse, "o-", label="OpenSees σ_ln RMSE vs ref")
    ax.plot(ns, pred_rmse, "s--", label="Surrogate σ_ln RMSE vs ref")
    ax.set_xlabel("Ensemble size N")
    ax.set_ylabel("RMSE of σ_ln(f) vs full ensemble")
    ax.set_title(f"{sample_dir.name} — σ_ln convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(sample_dir / "sigma_ln_convergence.png", dpi=150)
    plt.close(fig)


def plot_median_af_bands(sample_dir: Path) -> None:
    freq = np.load(sample_dir / "freq.npy")
    truth = np.load(sample_dir / "truth_af_central_stack.npy")
    pred = np.load(sample_dir / "pred_af_central_stack.npy")

    t16, t50, t84 = np.percentile(truth, [16, 50, 84], axis=0)
    p16, p50, p84 = np.percentile(pred, [16, 50, 84], axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(freq, t50, "b-", lw=1.5, label="Truth median")
    ax.fill_between(freq, t16, t84, color="blue", alpha=0.2, label="Truth p16–p84")
    ax.semilogx(freq, p50, "r--", lw=1.5, label="Surrogate median")
    ax.fill_between(freq, p16, p84, color="red", alpha=0.15, label="Surrogate p16–p84")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Linear |TF| (central recorder)")
    ax.set_title(f"{sample_dir.name} — AF bands across seeds")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(sample_dir / "median_af_bands.png", dpi=150)
    plt.close(fig)


def plot_per_seed_errors(sample_dir: Path) -> None:
    rows = load_csv(sample_dir / "per_seed_metrics.csv")
    rep = [int(r["replicate_id"]) for r in rows]
    rel = [float(r["rel_l2_central"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(rep, rel, "o-", lw=1)
    ax.set_xlabel("replicate_id")
    ax.set_ylabel("rel L2 (central recorder)")
    ax.set_title(f"{sample_dir.name} — per-seed surrogate error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(sample_dir / "per_seed_rel_l2.png", dpi=150)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot seed-robustness convergence")
    p.add_argument("--sample-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    sample_dir = args.sample_dir
    cv_path = sample_dir / "seed_cv_metrics.csv"
    if not cv_path.is_file():
        raise SystemExit(f"Missing {cv_path}; run seed_robustness_check.py first")

    cv_rows = load_csv(cv_path)
    agg = aggregate_cv(cv_rows)
    plot_sigma_ln_convergence(sample_dir, agg)
    plot_median_af_bands(sample_dir)
    plot_per_seed_errors(sample_dir)

    # Write aggregated summary for quick inspection
    summary = {
        "sample_dir": str(sample_dir),
        "by_n_seeds": {str(k): v for k, v in agg.items()},
    }
    meta_path = sample_dir / "meta.json"
    if meta_path.is_file():
        summary["meta"] = json.loads(meta_path.read_text())
    (sample_dir / "convergence_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[analyze] Wrote plots -> {sample_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
