import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.image import imread
from PIL import Image

from wave_surrogate.plot_utils.publication import (
    CMAP_DIVERGING,
    CMAP_SEQUENTIAL,
    COMBINATION_DPI,
    DPI,
    DOUBLE_MM,
    HALFTONE_DPI,
    LINE_ART_DPI,
    MM,
    OKABE_ITO,
    ONEHALF_MM,
    SINGLE_MM,
    STROKE_MAX_PT,
    STROKE_MIN_PT,
    apply_publication_style,
    dpi_for,
    figsize,
    panel_letter,
    savefig,
)

def test_column_widths_are_elsevier_mm():
    assert SINGLE_MM == 90.0
    assert ONEHALF_MM == 140.0
    assert DOUBLE_MM == 190.0

    single_w, single_h = figsize("single", height_mm=50)
    onehalf_w, _ = figsize("onehalf")
    alias_w, _ = figsize("1.5", height_mm=40)
    double_w, double_h = figsize("double", height_mm=100)

    assert single_w == pytest.approx(90.0 * MM)
    assert single_h == pytest.approx(50.0 * MM)
    assert onehalf_w == pytest.approx(140.0 * MM)
    assert alias_w == pytest.approx(140.0 * MM)
    assert double_w == pytest.approx(190.0 * MM)
    assert double_h == pytest.approx(100.0 * MM)


def test_publication_style_type_and_strokes():
    apply_publication_style()
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42
    assert plt.rcParams["axes.labelsize"] == 8
    assert plt.rcParams["xtick.labelsize"] == 7
    assert plt.rcParams["ytick.labelsize"] == 7
    for key in (
        "axes.linewidth",
        "xtick.major.width",
        "ytick.major.width",
        "xtick.minor.width",
        "ytick.minor.width",
        "lines.linewidth",
    ):
        width = plt.rcParams[key]
        assert STROKE_MIN_PT <= width <= STROKE_MAX_PT
    assert plt.rcParams["image.cmap"] == CMAP_SEQUENTIAL


def test_line_cycle_is_colorblind_and_redundant():
    apply_publication_style()
    cycle = plt.rcParams["axes.prop_cycle"].by_key()
    colors = cycle["color"]
    styles = [str(style) for style in cycle["linestyle"]]
    assert OKABE_ITO["yellow"] not in colors
    assert len(colors) == len(set(colors))
    assert len(styles) == len(set(styles))
    assert colors[0] != colors[1]
    assert styles[0] != styles[1]
    assert "marker" not in cycle


def test_dpi_for_artwork_kind():
    assert dpi_for("halftone") == HALFTONE_DPI == 300
    assert dpi_for("combination") == COMBINATION_DPI == 500
    assert dpi_for("line") == LINE_ART_DPI == 1000
    with pytest.raises(ValueError):
        dpi_for("photo")


def test_named_colormaps_resolve():
    assert CMAP_SEQUENTIAL in matplotlib.colormaps
    assert CMAP_DIVERGING in matplotlib.colormaps
    matplotlib.colormaps[CMAP_SEQUENTIAL]
    matplotlib.colormaps[CMAP_DIVERGING]


def test_savefig_keeps_designed_canvas(tmp_path):
    apply_publication_style()
    width_in, height_in = figsize("single", height_mm=40)
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("frequency " * 12)
    ax.set_ylabel("amplitude " * 8)
    panel_letter(ax, "a")
    dest = savefig(fig, tmp_path / "column.png")

    image = imread(dest)
    assert image.shape[1] == pytest.approx(width_in * DPI, abs=2)
    assert image.shape[0] == pytest.approx(height_in * DPI, abs=2)
    with Image.open(dest) as saved:
        assert saved.mode == "RGB"


def test_savefig_combination_canvas(tmp_path):
    apply_publication_style()
    width_in, height_in = figsize("single", height_mm=40)
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.plot([0, 1], [0, 1])
    dest = savefig(fig, tmp_path / "combo.png", kind="combination")

    image = imread(dest)
    assert image.shape[1] == pytest.approx(width_in * COMBINATION_DPI, abs=2)
    assert image.shape[0] == pytest.approx(height_in * COMBINATION_DPI, abs=2)
    with Image.open(dest) as saved:
        assert saved.mode == "RGB"
