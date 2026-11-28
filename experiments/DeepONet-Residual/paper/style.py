"""SDEE figure style for the results set.

Uses :mod:`wave_surrogate.plot_utils.publication` (90 / 140 / 190 mm, 7 pt,
open spines). Color is never the only cue: every series also has a linestyle,
and ranking markers differ. Domain colors and method colors are not mixed
on one axes.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from wave_surrogate.plot_utils.publication import (
    DPI,
    apply_publication_style,
    canvas_bbox,
    figsize as figsize,
    panel_letter as panel_letter,
)

PAPER_DIR = Path(__file__).resolve().parent
FIG_DIR = PAPER_DIR / "figures"

# Wong (2011), with Toro moved to green so it does not sit next to Pretell orange.
OPENSEES = "OpenSees 2-D"
GINO = "GINO"
HASKELL = "1D base case"
LOGLO = "LOGLO"
PRETELL = "Pretell median"
PRETELL_P84 = "Pretell percentile"
TORO = "Toro"
PASSERI = "Passeri"
DMULT = "Dmult"

METHOD_STYLE: dict[str, dict] = {
    OPENSEES: {
        "color": "#000000",
        "ls": "-",
        "lw": 1.5,
        "marker": None,
        "ms": 3.0,
        "zorder": 4,
        "hatch": "",
    },
    GINO: {
        "color": "#0057B8",
        "ls": "-",
        "lw": 1.45,
        "marker": "o",
        "ms": 3.2,
        "zorder": 5,
        "hatch": "",
    },
    HASKELL: {
        "color": "#3A3A3A",
        "ls": "-.",
        "lw": 1.25,
        "marker": "x",
        "ms": 3.2,
        "zorder": 2,
        "hatch": "...",
    },
    LOGLO: {
        "color": "#C2410C",
        "ls": "--",
        "lw": 1.3,
        "marker": None,
        "ms": 3.2,
        "zorder": 3,
        "hatch": "//",
    },
    PRETELL: {
        "color": "#B86E00",
        "ls": "--",
        "lw": 1.3,
        "marker": "o",
        "ms": 3.2,
        "zorder": 3,
        "hatch": "",
    },
    PRETELL_P84: {
        "color": "#B86E00",
        "ls": ":",
        "lw": 1.15,
        "marker": "o",
        "ms": 3.0,
        "zorder": 2,
        "hatch": "",
        "mfc": "white",
    },
    TORO: {
        "color": "#0B7A45",
        "ls": ":",
        "lw": 1.2,
        "marker": "^",
        "ms": 3.3,
        "zorder": 2,
        "hatch": "",
    },
    PASSERI: {
        "color": "#8E2F68",
        "ls": "-.",
        "lw": 1.2,
        "marker": "s",
        "ms": 2.8,
        "zorder": 2,
        "hatch": "",
    },
    DMULT: {
        "color": "#0C5F8A",
        "ls": (0, (4, 1.6)),
        "lw": 1.2,
        "marker": "D",
        "ms": 2.8,
        "zorder": 2,
        "hatch": "",
    },
}

# One method across domains. Do not use these on an axes that already encodes method.
DOMAIN_COLOR = {
    "iid": "#0057B8",
    "dipping": "#0B7A45",
}
DOMAIN_LABEL = {
    "iid": "IID",
    "dipping": "Dipping",
}
DOMAIN_ORDER = ("iid", "dipping")

RANK_METHODS = (GINO, PRETELL, PRETELL_P84, HASKELL, TORO, PASSERI, DMULT)


def use_style() -> None:
    apply_publication_style()
    plt.rcParams.update(
        {
            "legend.handlelength": 2.6,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 1.0,
            "lines.solid_capstyle": "round",
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def save_figure(
    fig: Figure, stem: str, *, fixed_canvas: bool = True, tiff: bool = False
) -> tuple[Path, Path]:
    """Write ``figures/<stem>.pdf`` and ``.png`` at the designed canvas size.

    ``fixed_canvas`` remains for existing callers. The media box is always the
    figure canvas, so a later LaTeX width does not rescale type. ``tiff`` adds
    an LZW ``.tiff`` at the same resolution.
    """
    if not fixed_canvas:
        warnings.warn(
            "savefig no longer crops with bbox_inches='tight'; "
            "the designed canvas size is kept.",
            stacklevel=2,
        )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    kw = {"dpi": DPI, "facecolor": "white", "bbox_inches": canvas_bbox(fig)}
    fig.savefig(pdf, **kw)
    fig.savefig(png, **kw)
    if tiff:
        fig.savefig(FIG_DIR / f"{stem}.tiff", pil_kwargs={"compression": "tiff_lzw"}, **kw)
    plt.close(fig)
    return pdf, png


def log_marker_index(freq, n_marks: int = 7, shift: float = 0.0) -> list[int]:
    """Indices spaced evenly in log-frequency, optionally shifted by a fraction of a step."""
    import numpy as np

    f = np.asarray(freq, dtype=float)
    positive = np.isfinite(f) & (f > 0)
    if int(positive.sum()) < 2:
        return []
    logf = np.log10(f)
    lo = float(np.nanmin(logf[positive]))
    hi = float(np.nanmax(logf[positive]))
    step = (hi - lo) / max(n_marks, 1)
    targets = np.linspace(lo, hi, n_marks) + float(shift) * step
    idx: list[int] = []
    for t in targets:
        i = int(np.nanargmin(np.abs(logf - t)))
        if i not in idx and positive[i]:
            idx.append(i)
    return idx


def plot_curve(
    ax: Axes,
    x,
    y,
    style: dict,
    *,
    markers: bool = False,
    n_marks: int = 7,
    mark_shift: float = 0.0,
    label: str | None = None,
) -> None:
    import numpy as np

    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ax.plot(
        xx,
        yy,
        color=style["color"],
        ls=style["ls"],
        lw=style["lw"],
        zorder=style.get("zorder", 2),
        label=label,
        solid_capstyle="round",
    )
    marker = style.get("marker")
    if not markers or marker is None:
        return
    idx = log_marker_index(xx, n_marks=n_marks, shift=mark_shift)
    if not idx:
        return
    mfc = style.get("mfc", style["color"])
    ax.plot(
        xx[idx],
        yy[idx],
        ls="none",
        marker=marker,
        ms=style.get("ms", 3.2),
        color=style["color"],
        markerfacecolor=mfc,
        markeredgecolor=style["color"],
        markeredgewidth=0.4,
        zorder=style.get("zorder", 2) + 1,
    )


def errorbar_method(ax: Axes, x, y, lo, hi, style: dict, *, label: str | None = None) -> None:
    mfc = style.get("mfc", style["color"])
    ax.errorbar(
        x,
        y,
        xerr=[[x - lo], [hi - x]],
        fmt=style.get("marker") or "o",
        color=style["color"],
        ms=style.get("ms", 3.2),
        lw=0,
        elinewidth=0.6,
        capsize=1.3,
        capthick=0.5,
        markerfacecolor=mfc,
        markeredgecolor=style["color"],
        markeredgewidth=0.45,
        zorder=style.get("zorder", 3),
        label=label,
    )
