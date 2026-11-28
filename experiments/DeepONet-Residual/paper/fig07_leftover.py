"""Fig. 7 — leftover calibration. The line is the cached trough-windowed OLS."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import calibration_cloud, load_pack, load_slopes  # noqa: E402
from style import (  # noqa: E402
    DOMAIN_LABEL,
    DOMAIN_ORDER,
    GINO,
    METHOD_STYLE,
    figsize,
    panel_letter,
    save_figure,
    use_style,
)


def main() -> None:
    use_style()
    slopes = load_slopes()
    fig, axes = plt.subplots(1, 2, figsize=figsize("double", height_mm=72))
    for ax, domain, letter in zip(axes, DOMAIN_ORDER, "abc"):
        pack = load_pack(domain)
        x, y = calibration_cloud(pack, seed=0)
        ax.scatter(
            x,
            y,
            s=3,
            c=METHOD_STYLE[GINO]["color"],
            alpha=0.55,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        fit = slopes[domain]
        both = np.concatenate([x, y])
        lo = float(np.quantile(both, 0.01))
        hi = float(np.quantile(both, 0.99))
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        xx = np.linspace(lo, hi, 40)
        ax.plot(xx, xx, color="0.45", ls="--", lw=0.7, zorder=3, label="1:1")
        ax.plot(
            xx,
            fit["a"] + fit["b"] * xx,
            color=METHOD_STYLE[GINO]["color"],
            lw=1.2,
            zorder=4,
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)
        ax.set_xlabel(r"$R$")
        ax.set_title(
            f"{DOMAIN_LABEL[domain]}\n"
            rf"$b$={fit['b']:.2f} [{fit['b_lo']:.2f}, {fit['b_hi']:.2f}]",
            fontsize=7,
        )
        panel_letter(ax, letter)
    axes[0].set_ylabel(r"$\hat{R}$")
    axes[0].legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    save_figure(fig, "fig07_leftover")


if __name__ == "__main__":
    main()
