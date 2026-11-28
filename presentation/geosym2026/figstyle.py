"""Shared look for the GeoSymposium 2026 figures (slides 7-16).

Figures are drawn at slide scale (13.33 x 7.5 in template), so a 16 pt label
here is 16 pt on the slide. Palette: navia for Vs and |TF| maps, dark green
for the AI, dashed brick red for 1D, near-black for the 2D truth.
"""

from __future__ import annotations

import os
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"
DATA = HERE / "data"
FONT_DIR = Path.home() / "geosymposium" / "figure_scripts" / "fonts"

# The model every figure reports. Stage 2 is fine-tuned from stage 1 on the
# horizontal runs plus the variable-width sections.
STAGE = "stage2"
STAGE_LABEL = "Stage 2"
RESULTS = REPO / "experiments" / "GIFNO-VARDOMAIN" / "figures" / STAGE
CHECKPOINT = REPO / "checkpoints" / "vardomain" / "early_width" / "stage2_rise50"

# Training-set Vs examples: slide 8 (horizontal), slide 9 icon (dipping), and
# the slide 10 variable-width card (randomized section with a bedrock step).
HORIZONTAL_RUN = 810
DIPPING_RUN = 300
EXAMPLE_SECTION = REPO / "data" / "vardomain" / "cache" / "sections" / "manifold__man_0266.npz"

AI = "#2E7D5B"
AI_LIGHT = "#9CC9B2"
ONED = "#A23B3B"
TRUTH = "#1A1A1A"
REAL = "#C9CDD2"
PRETELL = "#6A1B9A"
TORO = "#C2185B"
DMULT = "#6D4C41"
NAVY = "#003262"
AMBER = "#C4820E"
MUTED = "#6B7480"
PANEL = "#EEF1F5"
CMAP = cmc.navia

LABEL = {
    "ai": "This work (1D × 10$^{g}$)",
    "1d": "1D column",
    "truth": "2D simulation (OpenSees)",
    "toro": "Toro (1995)",
    "dmult": "Tao & Rathje (2019)",
}

# Content area under the slide title.
FULL_W, FULL_H = 12.3, 5.0


def apply() -> None:
    if FONT_DIR.is_dir():
        for name in os.listdir(FONT_DIR):
            if name.endswith(".ttf"):
                font_manager.fontManager.addfont(str(FONT_DIR / name))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Open Sans", "Liberation Sans", "DejaVu Sans"],
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.titleweight": "semibold",
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "legend.frameon": False,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": TRUTH,
        "axes.labelcolor": TRUTH,
        "text.color": TRUTH,
        "xtick.color": TRUTH,
        "ytick.color": TRUTH,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "lines.linewidth": 2.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 220,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Open Sans",
        "mathtext.it": "Open Sans:italic",
        "mathtext.bf": "Open Sans:bold",
    })


def save(fig: plt.Figure, stem: str, *, tight: bool = True) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight", "pad_inches": 0.08} if tight else {}
    fig.savefig(OUT / f"{stem}.png", **kw)
    plt.close(fig)
    print(f"wrote {OUT / stem}.png")
    return OUT / f"{stem}.png"


def log_freq_axis(ax, lo: float = 0.1, hi: float = 10.0) -> None:
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10) if lo <= t <= hi]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Frequency (Hz)")
