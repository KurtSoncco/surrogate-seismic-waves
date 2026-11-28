#!/usr/bin/env python3
"""Fig. 1. The metric canvas, then the path from that canvas to g.

    python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "scripts" / "_entry.py").is_file():
        sys.path.insert(0, str(_parent / "scripts"))
        break
from _entry import enter

enter(_HERE)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from encoding import CONTEXT_M, NX_METRIC, NZ_METRIC, encode
from plotting import apply_publication_style, panel_letter, save_figure

OUT = _HERE.parent
WIDTH = 1000
STATION = CONTEXT_M + WIDTH // 2


def _canvas() -> np.ndarray:
    vs = np.full((80, WIDTH), 220.0)
    vs[:, WIDTH // 3 :] = 360.0
    vs[40:] = 1400.0
    zeta = np.full_like(vs, 0.03)
    zeta[40:] = 0.004
    return encode("metric", vs, zeta, 80)


def _box(ax, x, y, w, h, text):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="#F7F7F7", edgecolor="#000000", linewidth=0.6)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5)


def main() -> None:
    field = _canvas()
    apply_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.48, 4.6), gridspec_kw={"height_ratios": [1.15, 1.0]})
    ax = axes[0]
    image = ax.imshow(field[0], cmap="cividis", aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02, label="Encoded column")
    left, right = CONTEXT_M, CONTEXT_M + WIDTH
    ax.add_patch(Rectangle((left, 0), WIDTH, NZ_METRIC - 1, fill=False, edgecolor="#D55E00", lw=0.9))
    ax.axvline(STATION, color="#0072B2", lw=0.9)
    ax.annotate(
        "",
        xy=(right, NZ_METRIC * 0.08),
        xytext=(left, NZ_METRIC * 0.08),
        arrowprops={"arrowstyle": "<->", "color": "#D55E00", "lw": 0.8},
    )
    ax.text((left + right) / 2, NZ_METRIC * 0.16, f"width {WIDTH} m", color="#D55E00", ha="center", fontsize=7)
    ax.text(STATION + 40, NZ_METRIC * 0.45, "station", color="#0072B2", fontsize=7)
    ax.set_title(f"{NZ_METRIC} × {NX_METRIC} canvas, interior from column {CONTEXT_M}")
    ax.set_xlabel("Canvas column")
    ax.set_ylabel("Depth row")
    panel_letter(ax, "a")

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    steps = (
        (0.2, "Depth collapse\nto surface tokens"),
        (2.6, "Global and local\nlateral paths"),
        (5.0, "Log-basis\noutput g"),
        (7.4, r"$|\mathrm{TF}|=|\mathrm{TF}_{\mathrm{col}}|e^{g}$"),
    )
    for x, text in steps:
        _box(ax, x, 1.1, 2.1, 1.1, text)
    for x in (2.3, 4.7, 7.1):
        ax.annotate("", xy=(x + 0.25, 1.65), xytext=(x, 1.65), arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#000000"})
    ax.text(5.0, 0.35, "g = 0 is the column", ha="center", fontsize=8, fontweight="bold")
    panel_letter(ax, "b", x=-0.02, y=1.02)
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    print(f"canvas {field.shape} station column {STATION}")


if __name__ == "__main__":
    main()
