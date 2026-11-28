#!/usr/bin/env python3
"""Fig. 1. What the network is asked to learn, in site-response terms."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from plotting import apply_publication_style, panel_letter, save_figure  # noqa: E402

OUT = HERE / "01_method_overview"


def _field() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1000.0, 201)
    z = np.arange(80.0)
    contact = 28.0 + 10.0 * np.tanh((x - 650.0) / 80.0)
    vs = np.empty((z.size, x.size))
    for i, depth in enumerate(z):
        vs[i] = np.where(depth < contact, 160.0 + 2.2 * depth, 1400.0)
    return x, z, vs


def _box(ax, x, y, w, h, text, face):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=face, edgecolor="#222222", linewidth=0.6,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.4)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x, z, vs = _field()
    apply_publication_style()
    fig = plt.figure(figsize=(7.48, 5.4))
    ax = fig.add_subplot(2, 1, 1)
    image = ax.pcolormesh(x, z, vs, cmap="cividis", shading="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=r"$V_s$ (m/s)")
    station = 700.0
    ax.plot([station, station], [0, 70], color="#0072B2", lw=1.2)
    ax.annotate(
        "1D column\nat the station", xy=(station, 36), xytext=(station - 280, 18),
        fontsize=7, color="#0072B2", arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 0.7},
    )
    ax.annotate(
        "", xy=(950, 6), xytext=(50, 6),
        arrowprops={"arrowstyle": "<->", "color": "#D55E00", "lw": 0.8},
    )
    ax.text(500, 12, "2D window the operator sees", color="#D55E00", ha="center", fontsize=7)
    ax.set_ylim(z[-1], 0)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("A lateral change in stiffness moves the surface motion off the column")
    panel_letter(ax, "a")

    ax = fig.add_subplot(2, 1, 2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    _box(ax, 0.2, 2.3, 2.8, 1.5, "Known profile\n" + r"$V_s(x,z)$, $\zeta(x,z)$", "#F4F1DE")
    _box(ax, 3.6, 2.3, 2.9, 1.5, "1D outcrop column\ncarries resonance\nand impedance", "#D6EAF8")
    _box(ax, 7.0, 2.3, 2.7, 1.5, "Operator learns\n" + r"$g=\ln(|TF_{2D}|/|TF_{1D}|)$", "#FDEBD0")
    _box(ax, 2.4, 0.25, 5.2, 1.15, r"$|TF|=|TF_{1D}|\,e^{g}$" + "\n" + r"$g=0$ returns the column exactly", "#E8F6EF")
    for start, end in ((3.05, 3.55), (6.55, 6.95)):
        ax.add_patch(FancyArrowPatch((start, 3.05), (end, 3.05), arrowstyle="-|>", mutation_scale=8, lw=0.7, color="#222222"))
    ax.annotate("", xy=(5.0, 1.45), xytext=(5.0, 2.25), arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": "#222222"})
    ax.text(5.0, 3.95, "Two paths. The column is kept. The network only learns the lateral correction.", ha="center", fontsize=7)
    panel_letter(ax, "b", x=-0.02, y=1.02)
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")

    pd.DataFrame([
        {"checkpoint": "P128n128", "epoch": 46, "parameters": 5560320, "narrow": 128, "latent": 128, "layers": 5, "pod_modes": 128, "role": "reported model"},
    ]).to_csv(OUT / "model_summary.csv", index=False)
    print(f"wrote {OUT / 'central.pdf'}")


if __name__ == "__main__":
    main()
