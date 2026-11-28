"""Fig. 2 — direct LOGLO versus residual GINO on the same nested tests."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_compare  # noqa: E402
from style import (  # noqa: E402
    DOMAIN_LABEL,
    DOMAIN_ORDER,
    GINO,
    HASKELL,
    LOGLO,
    METHOD_STYLE,
    figsize,
    panel_letter,
    save_figure,
    use_style,
)

_SERIES = (
    ("haskell", HASKELL),
    ("loglo", LOGLO),
    ("gino", GINO),
)


def main() -> None:
    use_style()
    compare = load_compare()
    fig, ax = plt.subplots(figsize=figsize("single", height_mm=78))
    x = np.arange(len(DOMAIN_ORDER), dtype=float)
    width = 0.24
    offsets = (-width, 0.0, width)
    for (key, name), dx in zip(_SERIES, offsets):
        style = METHOD_STYLE[name]
        heights = [compare[key][d] for d in DOMAIN_ORDER]
        bars = ax.bar(
            x + dx,
            heights,
            width=width * 0.92,
            color=style["color"],
            edgecolor="0.15",
            linewidth=0.4,
            hatch=style["hatch"],
            label=name,
            zorder=2,
        )
        for bar, h in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="0.15",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABEL[d] for d in DOMAIN_ORDER])
    ax.set_ylabel(r"relative $L_2$ of $|\mathrm{TF}|$")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", fontsize=6)
    panel_letter(ax, "a", x=-0.16, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig02_residual")


if __name__ == "__main__":
    main()
