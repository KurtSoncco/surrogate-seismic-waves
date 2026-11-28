#!/usr/bin/env python3
"""Presentation figure of the Treasure Island Case 7 Vs section.

Plots the geologic interval only. West of -650 m and east of 1050 m the
instruction-sheet model repeats one column; those extensions are left off.
Depth is positive downward, with a thin negative band above z = 0 for the
ground surface. Color is Crameri *navia* on a logarithmic Vs scale.

    python plot_case7_section.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_REPO = _HERE.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from case7a import (  # noqa: E402
    DEPTH_M,
    ELEM_M,
    GEOLOGIC_X_MAX,
    GEOLOGIC_X_MIN,
    N_ROW,
    X_LEFT_M,
    ensure_cache,
)
from wave_surrogate.plot_utils.publication import apply_publication_style  # noqa: E402

OUT_DIR = _HERE / "figures"
# Metres of axis above z = 0, so the ground surface sits inside the frame.
SURFACE_ABOVE_M = 10.0
VERTICAL_EXAGGERATION = 4.0
DPI = 600

# Inches. Margins hold only the labels; the axes take the rest.
_LEFT_IN = 0.70
_RIGHT_IN = 1.12
_TOP_IN = 0.66
_BOTTOM_IN = 0.10
_AX_W_IN = 12.8
_CBAR_GAP_IN = 0.08
_CBAR_W_IN = 0.16
_TITLE_TOP_IN = 0.04


def _crop(vs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Element Vs on [-650, 1050] m and the cell edges, in metres."""
    field = np.asarray(vs, dtype=np.float64)
    x_edge = X_LEFT_M + ELEM_M * np.arange(field.shape[1] + 1)
    j0 = int(np.searchsorted(x_edge, GEOLOGIC_X_MIN, side="left"))
    j1 = int(np.searchsorted(x_edge, GEOLOGIC_X_MAX, side="left"))
    z_edge = np.arange(N_ROW + 1, dtype=np.float64) * ELEM_M
    return field[:, j0:j1], x_edge[j0 : j1 + 1], z_edge


def _figure_size() -> tuple[float, float, float]:
    y_span = float(DEPTH_M - (-SURFACE_ABOVE_M))
    x_span = float(GEOLOGIC_X_MAX - GEOLOGIC_X_MIN)
    ax_h = VERTICAL_EXAGGERATION * y_span * _AX_W_IN / x_span
    fig_w = _LEFT_IN + _AX_W_IN + _RIGHT_IN
    fig_h = _BOTTOM_IN + ax_h + _TOP_IN
    return fig_w, fig_h, ax_h


def plot_section(out_path: Path) -> Path:
    """Draw the section and write ``out_path`` plus a 600 dpi PNG sibling."""
    apply_publication_style()
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "savefig.dpi": DPI,
        }
    )

    vs, x_edge, z_edge = _crop(ensure_cache()["vs"])
    fig_w, fig_h, ax_h = _figure_size()
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    ax = fig.add_axes(
        [
            _LEFT_IN / fig_w,
            _BOTTOM_IN / fig_h,
            _AX_W_IN / fig_w,
            ax_h / fig_h,
        ]
    )
    # Colorbar spans the soil column only, so its top sits on the ground surface.
    mesh_frac = float(DEPTH_M) / (float(DEPTH_M) + SURFACE_ABOVE_M)
    cax = fig.add_axes(
        [
            (_LEFT_IN + _AX_W_IN + _CBAR_GAP_IN) / fig_w,
            _BOTTOM_IN / fig_h,
            _CBAR_W_IN / fig_w,
            ax_h * mesh_frac / fig_h,
        ]
    )

    ax.axhspan(-SURFACE_ABOVE_M, 0.0, facecolor="#ececec", edgecolor="none", zorder=0)
    norm = LogNorm(vmin=float(vs.min()), vmax=float(vs.max()))
    mesh = ax.pcolormesh(
        x_edge,
        z_edge,
        vs,
        shading="flat",
        cmap=cmc.navia,
        norm=norm,
        edgecolors="face",
        linewidth=0.0,
        antialiased=False,
        zorder=2,
    )
    ax.axhline(0.0, color="black", linewidth=1.15, zorder=4, solid_capstyle="butt")
    ax.set_xlim(float(x_edge[0]), float(x_edge[-1]))
    ax.set_ylim(float(DEPTH_M), -SURFACE_ABOVE_M)

    ax.set_xlabel("Horizontal distance (m)", labelpad=3)
    ax.set_ylabel("Depth (m)", labelpad=3)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-600.0, 1001.0, 200.0))
    ax.set_yticks([0.0, 50.0, 100.0, 150.0])
    ax.tick_params(axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False, pad=2)
    ax.tick_params(axis="y", which="both", left=True, right=False, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)

    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.outline.set_linewidth(0.9)
    ticks = [150, 300, 600, 1200, 2400]
    colorbar.locator = FixedLocator(ticks)
    colorbar.formatter = FuncFormatter(lambda value, _pos: f"{int(round(value))}")
    colorbar.update_ticks()
    colorbar.ax.yaxis.set_minor_locator(NullLocator())
    colorbar.ax.tick_params(labelsize=12, width=0.9, length=3.5, direction="out", pad=2)
    colorbar.set_label(r"$V_{\mathrm{S}}$ (m/s)", fontsize=13, labelpad=6)

    fig.text(
        (_LEFT_IN + 0.5 * _AX_W_IN) / fig_w,
        1.0 - _TITLE_TOP_IN / fig_h,
        "Treasure Island Vs section",
        ha="center",
        va="top",
        fontsize=14,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    png = out_path.with_suffix(".png")
    if png != out_path:
        fig.savefig(png, dpi=DPI, facecolor="white")
    plt.close(fig)
    return png


def main() -> None:
    png = plot_section(OUT_DIR / "case7_treasure_island_vs.pdf")
    print(png)
    print(png.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
