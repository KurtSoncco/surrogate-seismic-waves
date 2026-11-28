"""Elsevier / SDEE figure style.

Production widths are the artwork-sizing table, not an author-template
``\\columnwidth``: single 90 mm, 1.5-column 140 mm, full width 190 mm.
Export at that canvas. In LaTeX, include the PDF at the same absolute width
(``width=90mm``, ``140mm``, or ``190mm``) so type is not rescaled.

Finished type is about 7 pt, and not below 6 pt. Elsevier line art allows
strokes from 0.1 pt to 1.5 pt; this style keeps them in
[:data:`STROKE_MIN_PT`, :data:`STROKE_MAX_PT`] (0.5–1.5 pt).
``pdf.fonttype`` 42 embeds TrueType instead of Type 3.

Vector PDF is the submission file. Raster siblings are RGB: 300 dpi
halftone, 500 dpi combination, 1000 dpi pure line art, with TIFF LZW.
The default line cycle is Okabe–Ito without yellow, each series also
changing linestyle. ``image.cmap`` is cividis.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

MM_PER_IN = 25.4
MM = 1.0 / MM_PER_IN

SINGLE_MM = 90.0
ONEHALF_MM = 140.0
DOUBLE_MM = 190.0
SINGLE_COL_IN = SINGLE_MM * MM
ONEHALF_COL_IN = ONEHALF_MM * MM
DOUBLE_COL_IN = DOUBLE_MM * MM

HALFTONE_DPI = 300
DPI = HALFTONE_DPI
COMBINATION_DPI = 500
LINE_ART_DPI = 1000
PANEL_FONTSIZE = 8

# Elsevier allows 0.1–1.5 pt. The floor is above that legal minimum so
# hairlines survive print.
STROKE_MIN_PT = 0.5
STROKE_MAX_PT = 1.5

CMAP_SEQUENTIAL = "cividis"
CMAP_SEQUENTIAL_ALT = "viridis"
CMAP_DIVERGING = "managua"

# Wong (2011). For new figures; the paper method palette stays in paper/style.py.
# Yellow is for fills only: it is too light on white for a line.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

LINE_COLORS = (
    OKABE_ITO["black"],
    OKABE_ITO["orange"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["reddish_purple"],
)
LINE_STYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)))
LINE_MARKERS = ("o", "s", "^", "D", "v", "P", "X")

_DPI_FOR = {
    "halftone": HALFTONE_DPI,
    "combination": COMBINATION_DPI,
    "line": LINE_ART_DPI,
}
_RASTER_SUFFIXES = {".png", ".tif", ".tiff"}

_WIDTHS = {
    "single": SINGLE_COL_IN,
    "onehalf": ONEHALF_COL_IN,
    "1.5": ONEHALF_COL_IN,
    "double": DOUBLE_COL_IN,
}
_DEFAULT_ASPECT = {
    "single": 1.05,
    "onehalf": 0.85,
    "1.5": 0.85,
    "double": 0.72,
}

_WINDOWS_FONTS = Path("/mnt/c/Windows/Fonts")
_ARIAL_FILES = ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")
_arial_warned = False


def _register_arial() -> bool:
    """Register Windows Arial when Matplotlib does not already have it."""
    global _arial_warned
    known = {f.name for f in font_manager.fontManager.ttflist}
    if "Arial" not in known:
        for name in _ARIAL_FILES:
            path = _WINDOWS_FONTS / name
            if path.exists():
                font_manager.fontManager.addfont(str(path))
        known = {f.name for f in font_manager.fontManager.ttflist}
    if "Arial" not in known:
        if not _arial_warned:
            warnings.warn(
                "Arial not found; figures fall back to another sans-serif.",
                stacklevel=3,
            )
            _arial_warned = True
        return False
    return True


def dpi_for(kind: str) -> int:
    """Raster DPI for an Elsevier artwork kind.

    ``halftone`` is 300, ``combination`` (image plus lines or labels) is 500,
    and pure ``line`` art is 1000. Vector PDF does not use this.
    """
    try:
        return _DPI_FOR[kind]
    except KeyError:
        raise ValueError(
            f"Unknown artwork kind {kind!r}; use 'halftone', 'combination', or 'line'."
        ) from None


def series_style(i: int) -> dict:
    """Color, linestyle, and marker for series ``i``.

    Color is never the only cue. Markers are not on the default cycle, so
    dense transfer functions stay unmarked until a caller asks for them.
    """
    n = len(LINE_COLORS)
    slot = int(i) % n
    return {
        "color": LINE_COLORS[slot],
        "linestyle": LINE_STYLES[slot],
        "marker": LINE_MARKERS[slot],
    }


def _line_cycle():
    return cycler(color=LINE_COLORS) + cycler(linestyle=LINE_STYLES)


def apply_publication_style() -> None:
    """Sans-serif, 7 pt type, open spines, embedded TrueType fonts."""
    has_arial = _register_arial()
    plt.rcParams.update(
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
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.formatter.use_mathtext": True,
            "axes.prop_cycle": _line_cycle(),
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.2,
            "lines.markersize": 3.0,
            "image.cmap": CMAP_SEQUENTIAL,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "mathtext.fontset": "dejavusans",
        }
    )
    if has_arial:
        plt.rcParams.update(
            {
                "mathtext.fontset": "custom",
                "mathtext.rm": "Arial",
                "mathtext.it": "Arial:italic",
                "mathtext.bf": "Arial:bold",
                "mathtext.sf": "Arial",
                "mathtext.cal": "Arial:italic",
                "mathtext.tt": "Arial",
            }
        )


def figsize(
    width: str = "double", *, height_mm: float | None = None
) -> tuple[float, float]:
    """Return ``(width, height)`` in inches for an Elsevier column width.

    ``width`` is ``"single"`` (90 mm), ``"onehalf"`` (140 mm), or ``"double"``
    (190 mm). ``"1.5"`` is accepted as an alias of ``"onehalf"``. When
    ``height_mm`` is omitted, the height stays proportional to the width.
    """
    if width not in _WIDTHS:
        raise ValueError(
            f"Unknown figure width {width!r}; use 'single', 'onehalf', or 'double'."
        )
    w = _WIDTHS[width]
    if height_mm is None:
        h = w * _DEFAULT_ASPECT[width]
    else:
        h = height_mm * MM
    return (w, h)


def panel_letter(ax: Axes, letter: str, *, x: float = -0.12, y: float = 1.08) -> None:
    """Draw a bold panel letter. A bare ``"a"`` is rendered as ``"(a)"``."""
    label = letter.strip()
    if not (label.startswith("(") and label.endswith(")")):
        label = f"({label})"
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def canvas_bbox(fig: Figure) -> Bbox:
    """Media box of the designed figure, in inches."""
    return Bbox.from_bounds(0, 0, fig.get_figwidth(), fig.get_figheight())


def _write_figure(fig: Figure, dest: Path, **kw) -> None:
    extra = {}
    if dest.suffix.lower() in {".tif", ".tiff"}:
        extra["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(dest, **extra, **kw)


def _flatten_rgb(path: Path) -> None:
    """Rewrite a raster file as RGB on white. TIFF stays LZW."""
    from PIL import Image

    with Image.open(path) as im:
        if im.mode == "RGB":
            rgb = im.copy()
        else:
            rgba = im.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, "white")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
    save_kw = {}
    if path.suffix.lower() in {".tif", ".tiff"}:
        save_kw["compression"] = "tiff_lzw"
    rgb.save(path, **save_kw)


def savefig(
    fig: Figure,
    path: str | Path,
    *,
    png: bool = False,
    tiff: bool = False,
    eps: bool = False,
    dpi: float | None = None,
    kind: str | None = None,
) -> Path:
    """Write ``path`` at the figure's designed size and close ``fig``.

    The media box stays at ``figsize``. ``png``, ``tiff``, and ``eps`` add
    those siblings beside ``path`` when that is not already their suffix.
    Vector PDF is the submission file. Raster DPI stays 300 unless ``dpi``
    is set, or ``kind`` is set and ``dpi`` is omitted (``halftone`` 300,
    ``combination`` 500, ``line`` 1000). PNG and TIFF are flattened to RGB;
    TIFF uses LZW.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if dpi is None and kind is not None:
        dpi = dpi_for(kind)
    elif dpi is None:
        dpi = DPI
    kw = {
        "dpi": dpi,
        "facecolor": "white",
        "bbox_inches": canvas_bbox(fig),
    }
    _write_figure(fig, path, **kw)
    written = [path]
    stem = path.with_suffix("")
    if png and path.suffix.lower() != ".png":
        sibling = stem.with_suffix(".png")
        _write_figure(fig, sibling, **kw)
        written.append(sibling)
    if tiff and path.suffix.lower() not in {".tif", ".tiff"}:
        sibling = stem.with_suffix(".tiff")
        _write_figure(fig, sibling, **kw)
        written.append(sibling)
    if eps and path.suffix.lower() != ".eps":
        _write_figure(fig, stem.with_suffix(".eps"), **kw)
    for saved in written:
        if saved.suffix.lower() in _RASTER_SUFFIXES:
            _flatten_rgb(saved)
    plt.close(fig)
    return path
