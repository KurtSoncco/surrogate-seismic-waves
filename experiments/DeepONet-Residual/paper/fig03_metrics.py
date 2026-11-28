"""Fig. 3 — frozen-test scores, and validation versus test Pearson."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_test_metrics, val_test_pearson  # noqa: E402
from style import (  # noqa: E402
    DOMAIN_COLOR,
    DOMAIN_LABEL,
    DOMAIN_ORDER,
    figsize,
    panel_letter,
    save_figure,
    use_style,
)

_METRICS = (
    ("rel_l2_TF", r"relative $L_2$ of $|\mathrm{TF}|$", False),
    ("pearson_TF_freq", r"Pearson of $|\mathrm{TF}|$", False),
    ("r2_R", r"leftover $R^2$", False),
    ("harm_rate", "harm rate (%)", True),
)


def main() -> None:
    use_style()
    metrics = load_test_metrics()
    fig = plt.figure(figsize=figsize("double", height_mm=118))
    grid = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.15), hspace=0.55, wspace=0.55)

    x = np.arange(len(DOMAIN_ORDER), dtype=float)
    for col, (key, ylabel, as_percent) in enumerate(_METRICS):
        ax = fig.add_subplot(grid[0, col])
        vals = []
        for i, domain in enumerate(DOMAIN_ORDER):
            v = metrics[domain][key]
            if as_percent:
                v *= 100.0
            vals.append(v)
            ax.scatter(
                [i],
                [v],
                s=28,
                color=DOMAIN_COLOR[domain],
                zorder=3,
            )
            label = f"{v:.1f}" if as_percent else f"{v:.2f}"
            ax.annotate(
                label,
                (i, v),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                va="bottom",
                fontsize=6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([DOMAIN_LABEL[d] for d in DOMAIN_ORDER], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        lo, hi = min(vals), max(vals)
        pad = 0.18 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(lo - pad, hi + pad * 1.8)
        ax.set_xlim(-0.45, 2.45)
        panel_letter(ax, "abcd"[col], x=-0.28, y=1.06)

    ax = fig.add_subplot(grid[1, :])
    positions = []
    labels = []
    pos = 0
    for domain in ("iid", "dipping"):
        scores = val_test_pearson(domain)
        for split, filled in (("val", False), ("test", True)):
            y = scores[split]
            y = y[np.isfinite(y)]
            box = ax.boxplot(
                y,
                positions=[pos],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "0.1", "lw": 0.8},
                whiskerprops={"color": DOMAIN_COLOR[domain], "lw": 0.6},
                capprops={"color": DOMAIN_COLOR[domain], "lw": 0.6},
                boxprops={"linewidth": 0.6},
            )
            face = DOMAIN_COLOR[domain] if filled else "white"
            for patch in box["boxes"]:
                patch.set_facecolor(face)
                patch.set_edgecolor(DOMAIN_COLOR[domain])
            positions.append(pos)
            labels.append(f"{DOMAIN_LABEL[domain]}\n{split}")
            pos += 1
        pos += 0.6
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Pearson of $|\mathrm{TF}|$")
    ax.set_ylim(0.70, 1.03)
    panel_letter(ax, "e", x=-0.06, y=1.04)
    fig.text(
        0.99,
        0.02,
        "open: validation    filled: test",
        ha="right",
        va="bottom",
        fontsize=6,
        color="0.25",
    )
    save_figure(fig, "fig03_metrics")


if __name__ == "__main__":
    main()
