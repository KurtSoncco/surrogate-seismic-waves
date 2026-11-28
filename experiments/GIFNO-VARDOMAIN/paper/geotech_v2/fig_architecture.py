#!/usr/bin/env python3
"""Appendix diagram of P128n128. The numbers are the checkpoint config."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from plotting import apply_publication_style, save_figure  # noqa: E402

OUT = HERE / "appendix" / "architecture"


def _box(ax, x, y, w, h, text, face):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=face, edgecolor="#222222", linewidth=0.6,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.48, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("P128n128, the reported checkpoint", fontsize=9, loc="left")
    steps = (
        (0.3, 6.2, "Window\n$V_s$, damping, $\\log f_1$, $\\log f_2$\ninterior up to 2000 m\nedge columns repeated", "#F4F1DE"),
        (0.3, 4.15, "Depth code\nfrozen PCA, 128 coefficients\nper column, then a 1×1 map\nto 128 channels", "#D6EAF8"),
        (0.3, 2.1, "Five lateral layers\nlong path: 96 Fourier modes in $x$\nshort path: 20 m patches,\nnothing shorter than 10 m", "#FDEBD0"),
        (0.3, 0.15, "Station query\n128 POD modes rebuild $g(f)$\ncaller adds $\\ln|TF_{column}|$", "#E8F6EF"),
    )
    for x, y, text, face in steps:
        _box(ax, x, y, 5.4, 1.75, text, face)
    for y0, y1 in ((6.15, 5.95), (4.1, 3.9), (2.05, 1.95)):
        ax.add_patch(FancyArrowPatch((3.0, y0), (3.0, y1), arrowstyle="-|>", mutation_scale=8, lw=0.7, color="#222222"))
    _box(
        ax, 6.3, 2.3, 5.3, 4.6,
        "What is fixed\n\n"
        "5,560,320 parameters\n"
        "latent width 128\n"
        "loss: log-L1 of $g$\n"
        "seed 0, batch 16\n"
        "learning rate 0.001\n"
        "saved at epoch 46\n"
        "training ran to epoch 61\n"
        "float32 at scoring time",
        "#F7F7F7",
    )
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    pd.DataFrame([
        {"item": "checkpoint", "value": "P128n128, epoch 46"},
        {"item": "parameters", "value": "5560320"},
        {"item": "target", "value": "g, residual on the column"},
        {"item": "depth", "value": "frozen PCA, rank 128, mapped to 128 channels"},
        {"item": "lateral layers", "value": "5"},
        {"item": "Fourier modes in x", "value": "96"},
        {"item": "local patch", "value": "20 m, wavelengths shorter than 10 m removed"},
        {"item": "high-frequency kernel", "value": "4 m stride 4 m"},
        {"item": "readout", "value": "128 POD modes of the log spectrum"},
        {"item": "loss", "value": "log-L1"},
        {"item": "selection", "value": "validation score on horizontal and realistic sections"},
        {"item": "seed", "value": "0"},
        {"item": "batch", "value": "16"},
        {"item": "learning rate", "value": "0.001, cosine schedule"},
        {"item": "weight decay", "value": "0.0001"},
        {"item": "patience", "value": "15 epochs; weights kept from epoch 46"},
    ]).to_csv(OUT / "spec.csv", index=False)
    print(f"wrote {OUT / 'central.pdf'}")


if __name__ == "__main__":
    main()
