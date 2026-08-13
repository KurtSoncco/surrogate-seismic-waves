"""Diagnostic plots for residual screening."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config


def plot_r_central_3x3(
    cache_dir: Path,
    *,
    results_dir: Path | None = None,
    seed: int = 42,
    n_combos: int = 9,
) -> Path:
    """3x3 grid of |R|(x_central, f) for random subsample combinations.

    Overlays R_col and R_nom at the central recorder.
    """
    results_dir = results_dir or config.RESULTS_DIR
    r_col = np.load(cache_dir / "r_col.npy")
    r_nom = np.load(cache_dir / "r_nom.npy")
    meta = dict(np.load(cache_dir / "meta.npz", allow_pickle=True))
    freq = np.load(config.TF_FREQ_PATH)
    n_s = r_col.shape[0]
    rng = np.random.default_rng(seed)
    n_combos = min(n_combos, n_s)
    picks = rng.choice(n_s, size=n_combos, replace=False)
    c = config.CENTRAL_RECORDER_IDX

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=False)
    axes = axes.ravel()
    for ax, i in zip(axes, picks):
        ax.semilogx(freq, r_col[i, c], label=r"$|R_{\mathrm{col}}|$", lw=1.2)
        ax.semilogx(
            freq, r_nom[i, c], label=r"$|R_{\mathrm{nom}}|$", lw=1.2, alpha=0.85
        )
        ax.set_title(
            f"idx={int(meta['sample_idx'][i])}  "
            f"CoV={float(meta['CoV'][i]):.2f}  "
            f"H={float(meta['H'][i]):.0f}  "
            f"rH={float(meta['rH'][i]):.1f}",
            fontsize=9,
        )
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes[n_combos:]:
        ax.axis("off")
    axes[0].legend(loc="upper right", fontsize=8)
    for ax in axes[6:]:
        ax.set_xlabel("f [Hz]")
    for ax in axes[::3]:
        ax.set_ylabel(r"$|R|(x_{\mathrm{central}}, f)$")
    fig.suptitle(
        r"Central-recorder residuals: $|R_{\mathrm{col}}|=|TF_{2D}-TF_{1D,col}|$, "
        r"$|R_{\mathrm{nom}}|=|TF_{2D}-TF_{1D}(V_{s1},H,V_{s2})|$",
        fontsize=11,
    )
    fig.tight_layout()
    out = results_dir / "r_central_3x3.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_importance_bars(
    mi_csv: Path,
    rf_csv: Path,
    target: str,
    *,
    results_dir: Path | None = None,
    top_n: int = 20,
) -> Path:
    """Side-by-side MI and permutation-importance bars."""
    import pandas as pd

    results_dir = results_dir or config.RESULTS_DIR
    mi = pd.read_csv(mi_csv)
    if "band" in mi.columns:
        mi = mi[mi["band"] == "all"]
    rf = pd.read_csv(rf_csv)
    mi = mi.head(top_n)
    rf = rf.head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(mi["feature"][::-1], mi["mi"][::-1], color="#2c6e49")
    axes[0].set_title(f"Mutual information → {target}")
    axes[0].set_xlabel("MI")
    axes[1].barh(rf["feature"][::-1], rf["perm_importance_mean"][::-1], color="#bc4749")
    axes[1].set_title(f"RF permutation importance → {target}")
    axes[1].set_xlabel(r"$\Delta R^2$")
    fig.tight_layout()
    out = results_dir / f"importance_{target}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
