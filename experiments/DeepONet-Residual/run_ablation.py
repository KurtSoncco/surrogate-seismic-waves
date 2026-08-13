"""Ablation sweep: branch topology × trunk coords × R_col vs R_nom."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from train import train_one


# Default science sweep (plan S0/S1/S2/M1 + trunk + both targets)
DEFAULT_RUNS = [
    # Primary: single-branch full trunk, both targets
    ("single", "full", "R_col"),
    ("single", "full", "R_nom"),
    # Branch importance
    ("stoch_only", "full", "R_col"),
    ("fields_only", "full", "R_col"),
    ("multi", "full", "R_col"),
    # Trunk importance on R_col
    ("single", "fstar", "R_col"),
    ("single", "fstar_fourier", "R_col"),
    ("single", "xL", "R_col"),
]


def plot_summary(rows: list[dict], out_path: Path) -> None:
    """Primary bars: residual R² and Pearson-across-freq (not TF R²)."""
    names = [r["name"] for r in rows]
    r2_r = [r["test"]["r2_R"] for r in rows]
    pearson_f = [r["test"].get("pearson_R_freq", 0.0) for r in rows]
    delta_tf = [r["test"].get("delta_r2_TF", 0.0) for r in rows]
    x = np.arange(len(names))
    w = 0.28
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(names)), 4.8))
    ax.bar(x - w, r2_r, w, label="R² (signed R)")
    ax.bar(x, pearson_f, w, label="Pearson R across freq")
    ax.bar(x + w, delta_tf, w, label="ΔR² TF vs TF₁D-only")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            n.replace("n100_seed42", "n100").replace("n1000_seed42", "n1000")
            for n in names
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("score")
    ax.set_title("DeepONet-Residual: residual learning (not lazy TF R²)")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-tag", default="n100_seed42")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Only single-branch R_col vs R_nom (full trunk)",
    )
    p.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Run all epochs (ignore patience)",
    )
    args = p.parse_args()

    runs = (
        [("single", "full", "R_col"), ("single", "full", "R_nom")]
        if args.quick
        else DEFAULT_RUNS
    )

    results = []
    for branch_mode, trunk_set, target in runs:
        print(f"\n=== {branch_mode} | {trunk_set} | {target} ===", flush=True)
        results.append(
            train_one(
                cache_tag=args.cache_tag,
                target=target,  # type: ignore[arg-type]
                branch_mode=branch_mode,  # type: ignore[arg-type]
                trunk_set=trunk_set,  # type: ignore[arg-type]
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                patience=args.patience,
                no_early_stop=args.no_early_stop,
            )
        )

    out = config.RESULTS_DIR / f"ablation_{args.cache_tag}.json"
    out.write_text(json.dumps(results, indent=2))
    plot_summary(results, config.RESULTS_DIR / f"ablation_{args.cache_tag}.png")
    print(f"\nAblation summary → {out}", flush=True)

    # Brief winner printout
    by_target = {}
    for r in results:
        if r["branch_mode"] == "single" and r["trunk_set"] == "full":
            by_target[r["target"]] = r["test"]
    if len(by_target) == 2:
        print(
            "\nPrimary comparison (single + full trunk) — residual metrics:", flush=True
        )
        for t, m in by_target.items():
            print(
                f"  {t}: R²_R={m['r2_R']:.3f}  pearson_R_freq={m['pearson_R_freq']:.3f}  "
                f"Δr2_TF={m['delta_r2_TF']:.4f}  (TF₁D-only R²={m['r2_TF_1d_only']:.3f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
