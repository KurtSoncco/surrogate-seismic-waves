"""Nature journal figure style (single 89 mm / double 183 mm columns)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1.0 / 25.4
SINGLE_COL_IN = 89.0 * MM
DOUBLE_COL_IN = 183.0 * MM
DPI = 300


def apply_nature_style() -> None:
    """Sans-serif, small type, thin spines — suitable for Nature print."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.borderpad": 0.2,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.formatter.use_mathtext": True,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "mathtext.fontset": "dejavusans",
        }
    )


def figsize(width: str = "double", *, height_mm: float | None = None) -> tuple[float, float]:
    w = DOUBLE_COL_IN if width == "double" else SINGLE_COL_IN
    if height_mm is None:
        h = w * (0.72 if width == "double" else 1.05)
    else:
        h = height_mm * MM
    return (w, h)


def panel_letter(ax, letter: str, *, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def savefig(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
