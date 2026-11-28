"""Graphical abstract: site, leftover, held-out Pearson. Read left to right."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (  # noqa: E402
    central_curves,
    load_pack,
    load_ranking,
    pick_median_case,
    site_section,
)
from style import (  # noqa: E402
    GINO,
    HASKELL,
    METHOD_STYLE,
    OPENSEES,
    PRETELL,
    plot_curve,
    save_figure,
    use_style,
)

_CM = 1.0 / 2.54
# Elsevier graphical abstract: readable at 13 cm wide (about 1535 px at 300 dpi),
# which clears the 1328 px minimum. 1328 x 531 px is about 11.2 cm, not 13 cm.
_WIDTH_CM = 13.0
_HEIGHT_CM = 5.2
_INK = "#111111"
_FONT = 8

# Abstract-only colors: blue and orange stay apart under color-vision deficiency,
# the 1-D baseline is neutral, and labels are ink beside a colored mark.
_COLOR = {OPENSEES: "#000000", HASKELL: "#6B6A66", GINO: "#2A78D6", PRETELL: "#EB6834"}
_MARKER = {HASKELL: "x", GINO: "o", PRETELL: "s"}
_LEFTOVER_FILL = "#CFE1F7"
# Sand to dark rock, one hue, light to dark.
_SITE_CMAP = LinearSegmentedColormap.from_list(
    "site", ["#FBF3E1", "#E9D3A4", "#C9A36A", "#7A5A3C", "#3B2A1E"]
)


def _style(name: str) -> dict:
    return {**METHOD_STYLE[name], "color": _COLOR[name]}


def _panel_site(ax: Axes, section: dict) -> None:
    vs = section["vs"]
    width = float(section["width"])
    depth = float(section["depth"])
    ax.imshow(
        vs,
        extent=(0.0, width, depth, 0.0),
        aspect="auto",
        cmap=_SITE_CMAP,
        norm=PowerNorm(0.5, vmin=0.0, vmax=float(np.nanmax(vs)) * 1.25),
        interpolation="nearest",
    )
    rec = section["recorder_x"]
    centre = float(rec[rec.size // 2])
    lift = -0.035 * depth
    ax.plot(
        rec,
        np.full(rec.size, lift),
        ls="none",
        marker="v",
        ms=2.6,
        color=_INK,
        clip_on=False,
    )
    column = _style(HASKELL)
    ax.plot([centre, centre], [0.0, depth], color="#FFFFFF", lw=2.6)
    ax.plot(
        [centre, centre], [0.0, depth], color=column["color"], ls=column["ls"], lw=1.3
    )
    ax.text(
        centre + 0.04 * width,
        0.30 * depth,
        "1-D\nColumn",
        fontsize=_FONT,
        color=_INK,
        va="center",
    )
    ax.text(0.05 * width, 0.30 * depth, "Soil", fontsize=_FONT, color=_INK, va="center")
    ax.text(
        0.05 * width, 0.88 * depth, "Rock", fontsize=_FONT, color="#FFFFFF", va="center"
    )
    ax.set_xlim(0.0, width)
    ax.set_ylim(depth, 1.6 * lift)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    ax.set_xlabel(f"{width:.0f} m × {depth:.0f} m", fontsize=_FONT, labelpad=3)
    ax.set_title("Dipping Bedrock", fontsize=_FONT + 1, pad=7)


def _panel_residual(ax: Axes, curves: dict[str, np.ndarray]) -> None:
    freq = curves["freq"]
    ops = np.maximum(curves["opensees"], 1e-3)
    one = np.maximum(curves["haskell"], 1e-3)
    gino = np.maximum(curves["gino"], 1e-3)
    ax.fill_between(freq, one, ops, color=_LEFTOVER_FILL, linewidth=0, zorder=1)
    for name, y, label in (
        (OPENSEES, ops, "2-D"),
        (HASKELL, one, "1-D"),
        (GINO, gino, "GINO"),
    ):
        plot_curve(ax, freq, y, _style(name), label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(freq.min()), float(freq.max()))
    ax.set_ylim(0.4, float(max(ops.max(), one.max(), gino.max())) * 5.5)
    ax.set_xlabel("Frequency (Hz)", fontsize=_FONT, labelpad=1)
    ax.set_ylabel("Amplification", fontsize=_FONT, labelpad=1)
    ax.legend(
        loc="upper left",
        fontsize=_FONT,
        handlelength=1.7,
        handletextpad=0.5,
        labelspacing=0.15,
        borderaxespad=0.2,
        ncol=3,
        columnspacing=0.9,
    )
    # Tag the leftover where the gap between 1-D and 2-D is widest above 2 Hz.
    high = freq >= 2.0
    k = int(np.argmax(np.where(high, np.abs(np.log(ops / one)), -np.inf)))
    ax.annotate(
        r"$R$",
        xy=(freq[k], float(np.sqrt(ops[k] * one[k]))),
        xytext=(freq[k] * 0.62, 0.52),
        fontsize=_FONT + 1,
        color=_INK,
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "lw": 0.6,
            "color": _INK,
            "shrinkA": 1.0,
            "shrinkB": 0.0,
        },
    )
    ax.set_title(r"GINO Learns $R$ = 2-D $-$ 1-D", fontsize=_FONT + 1, pad=7)


def _panel_slope(ax: Axes, ranking: list[dict]) -> None:
    median = {(row["method"], row["domain"]): float(row["pearson"]) for row in ranking}
    for name, label in ((HASKELL, "1-D"), (PRETELL, "Pretell"), (GINO, "GINO")):
        style = _style(name)
        ys = [median[name, "iid"], median[name, "dipping"]]
        heavy = name == GINO
        ax.plot(
            [0.0, 1.0],
            ys,
            color=style["color"],
            ls=style["ls"],
            lw=2.4 if heavy else 1.5,
            marker=_MARKER[name],
            ms=5.0,
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            zorder=style["zorder"],
            clip_on=False,
        )
        ax.text(
            1.10,
            ys[1],
            f"{label} {ys[1]:.2f}",
            fontsize=_FONT,
            color=_INK,
            fontweight="bold" if heavy else "normal",
            va="center",
            ha="left",
        )
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0.6, 1.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Flat", "Dipping"], fontsize=_FONT)
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.set_ylabel("Pearson Correlation", fontsize=_FONT, labelpad=1)
    ax.set_title("Held-Out Tests", fontsize=_FONT + 1, pad=7)


def main() -> None:
    use_style()
    plt.rcParams.update(
        {
            "font.size": _FONT,
            "xtick.labelsize": _FONT,
            "ytick.labelsize": _FONT,
            "axes.labelcolor": _INK,
            "xtick.color": _INK,
            "ytick.color": _INK,
            "text.color": _INK,
        }
    )
    pack = load_pack("dipping")
    case = pick_median_case(pack)

    fig = plt.figure(figsize=(_WIDTH_CM * _CM, _HEIGHT_CM * _CM))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(0.95, 1.5, 0.72),
        wspace=0.46,
        left=0.016,
        right=0.862,
        top=0.86,
        bottom=0.19,
    )
    _panel_site(fig.add_subplot(grid[0, 0]), site_section(pack, case))
    _panel_residual(fig.add_subplot(grid[0, 1]), central_curves(pack, case))
    _panel_slope(fig.add_subplot(grid[0, 2]), load_ranking())
    save_figure(fig, "graphical_abstract", fixed_canvas=True, tiff=True)


if __name__ == "__main__":
    main()
