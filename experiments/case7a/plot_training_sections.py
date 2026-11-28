#!/usr/bin/env python3
"""Two training windows, one figure: two-layer horizontal and dipping bedrock.

Both are the 500 m interior the network sees (the repeated edge pads are
omitted). Inverted triangles mark the 21 surface recorders. Color is Crameri
*navia* on one logarithmic Vs scale.

    python plot_training_sections.py
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
_REPO = _HERE.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from wave_surrogate.plot_utils.publication import apply_publication_style  # noqa: E402

OUT_DIR = _HERE / "figures"
PACK_DIR = _REPO / "experiments" / "DeepONet-Residual" / "results" / "presentation"

DX_M = 1.0
WIDTH_M = 500.0
# Metres of axis above z = 0, so the ground surface and the stations sit in frame.
SURFACE_ABOVE_M = 6.0
VERTICAL_EXAGGERATION = 5.0
DPI = 600

_LEFT_IN = 0.68
_RIGHT_IN = 1.12
_HEADER_IN = 0.64
_BOTTOM_IN = 0.12
_AX_W_IN = 6.2
_CBAR_GAP_IN = 0.08
_CBAR_W_IN = 0.16
_TITLE_PAD_IN = 0.03


def _finite_block(vs: np.ndarray) -> np.ndarray:
    """Drop the NaN rows under the stored mesh. ``vs`` is (depth, x)."""
    rows = np.flatnonzero(np.isfinite(vs).any(axis=1))
    if rows.size == 0:
        raise ValueError("Vs block is empty")
    return np.asarray(vs[: int(rows[-1]) + 1], dtype=np.float64)


def _extend_bedrock(vs: np.ndarray, n_rows: int) -> np.ndarray:
    """Repeat the base row so a shallower mesh reaches ``n_rows``."""
    if vs.shape[0] > n_rows:
        raise ValueError(f"Vs has {vs.shape[0]} rows, above the target {n_rows}")
    if vs.shape[0] == n_rows:
        return vs
    extra = np.repeat(vs[-1:, :], n_rows - vs.shape[0], axis=0)
    return np.vstack((vs, extra))


def _bedrock_row(vs: np.ndarray) -> np.ndarray:
    """Row of the largest downward Vs jump in each column."""
    jump = np.diff(vs, axis=0)
    jump = np.where(np.isfinite(jump), jump, -np.inf)
    return np.argmax(jump, axis=0) + 1


def _pick_horizontal(pack: np.lib.npyio.NpzFile) -> int:
    """Flat two-layer case near 50 m of soil and a typical coefficient of variation."""
    height = np.asarray(pack["H"], dtype=np.float64)
    cov = np.asarray(pack["cov"], dtype=np.float64)
    score = np.abs(height - 50.0) + 80.0 * np.abs(cov - np.median(cov))
    return int(np.argmin(score))


def _pick_dipping(pack: np.lib.npyio.NpzFile) -> int:
    """Steepest bedrock contact in the dipping pack."""
    vs = np.asarray(pack["vs_2d"], dtype=np.float64)
    best_i, best_slope = 0, -1.0
    x = np.arange(vs.shape[2], dtype=np.float64)
    for i in range(vs.shape[0]):
        row = _bedrock_row(_finite_block(vs[i]))
        slope = abs(float(np.polyfit(x, row, 1)[0]))
        if slope > best_slope:
            best_i, best_slope = i, slope
    return best_i


def _load_pair() -> list[tuple[str, np.ndarray, np.ndarray]]:
    horizontal = np.load(PACK_DIR / "iid_pack.npz")
    dipping = np.load(PACK_DIR / "dipping_pack.npz")
    recorders = np.asarray(horizontal["recorder_x"], dtype=np.float64) * DX_M
    if not np.array_equal(recorders, np.asarray(dipping["recorder_x"], dtype=np.float64) * DX_M):
        raise ValueError("IID and dipping recorder positions differ")
    dipping_vs = _finite_block(dipping["vs_2d"][_pick_dipping(dipping)])
    horizontal_vs = _extend_bedrock(
        _finite_block(horizontal["vs_2d"][_pick_horizontal(horizontal)]),
        dipping_vs.shape[0],
    )
    panels = [
        ("Two-layer horizontal", horizontal_vs),
        ("Dipping", dipping_vs),
    ]
    return [(title, vs, recorders) for title, vs in panels]


def _style() -> None:
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


def _panel_height(depth_m: float) -> float:
    y_span = depth_m + SURFACE_ABOVE_M
    return VERTICAL_EXAGGERATION * y_span * _AX_W_IN / WIDTH_M


def plot_sections(out_path: Path) -> Path:
    """Draw both windows in one column and write ``out_path`` plus a 600 dpi PNG."""
    _style()
    panels = _load_pair()
    heights = [_panel_height(vs.shape[0] * DX_M) for _, vs, _ in panels]
    fig_w = _LEFT_IN + _AX_W_IN + _RIGHT_IN
    fig_h = _HEADER_IN + heights[0] + _HEADER_IN + heights[1] + _BOTTOM_IN
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    y_in = [
        _BOTTOM_IN + heights[1] + _HEADER_IN,
        _BOTTOM_IN,
    ]
    axes = [
        fig.add_axes([_LEFT_IN / fig_w, y / fig_h, _AX_W_IN / fig_w, height / fig_h])
        for y, height in zip(y_in, heights)
    ]

    lo = min(float(vs.min()) for _, vs, _ in panels)
    hi = max(float(vs.max()) for _, vs, _ in panels)
    norm = LogNorm(vmin=lo, vmax=hi)
    mesh = None
    for ax, (title, vs, recorders), y, height in zip(axes, panels, y_in, heights):
        depth = vs.shape[0] * DX_M
        x_edge = np.arange(vs.shape[1] + 1, dtype=np.float64) * DX_M
        z_edge = np.arange(vs.shape[0] + 1, dtype=np.float64) * DX_M
        ax.axhspan(-SURFACE_ABOVE_M, 0.0, facecolor="#ececec", edgecolor="none", zorder=0)
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
        ax.axhline(0.0, color="black", linewidth=1.05, zorder=4, solid_capstyle="butt")
        ax.plot(
            recorders,
            np.full(recorders.shape, -0.55 * SURFACE_ABOVE_M),
            linestyle="none",
            marker="v",
            markersize=5,
            color="black",
            markeredgewidth=0.0,
            zorder=5,
            clip_on=False,
        )
        ax.set_xlim(0.0, WIDTH_M)
        ax.set_ylim(depth, -SURFACE_ABOVE_M)
        ax.set_xlabel("Horizontal distance (m)", labelpad=2)
        ax.set_ylabel("Depth (m)", labelpad=3)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(0.0, WIDTH_M + 1.0, 100.0))
        ax.set_yticks(np.arange(0.0, depth + 1.0, 20.0))
        ax.tick_params(
            axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False, pad=2
        )
        ax.tick_params(axis="y", which="both", left=True, right=False, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
        fig.text(
            (_LEFT_IN + 0.5 * _AX_W_IN) / fig_w,
            (y + height + _HEADER_IN - _TITLE_PAD_IN) / fig_h,
            title,
            ha="center",
            va="top",
            fontsize=14,
        )

    cax = fig.add_axes(
        [
            (_LEFT_IN + _AX_W_IN + _CBAR_GAP_IN) / fig_w,
            _BOTTOM_IN / fig_h,
            _CBAR_W_IN / fig_w,
            (heights[0] + _HEADER_IN + heights[1]) / fig_h,
        ]
    )
    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.outline.set_linewidth(0.9)
    ticks = [150, 300, 600, 1000]
    colorbar.locator = FixedLocator(ticks)
    colorbar.formatter = FuncFormatter(lambda value, _pos: f"{int(round(value))}")
    colorbar.update_ticks()
    colorbar.ax.yaxis.set_minor_locator(NullLocator())
    colorbar.ax.tick_params(labelsize=12, width=0.9, length=3.5, direction="out", pad=2)
    colorbar.set_label(r"$V_{\mathrm{S}}$ (m/s)", fontsize=13, labelpad=6)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    png = out_path.with_suffix(".png")
    if png != out_path:
        fig.savefig(png, dpi=DPI, facecolor="white")
    plt.close(fig)
    return png


def main() -> None:
    png = plot_sections(OUT_DIR / "training_sections_vs.pdf")
    print(png)
    print(png.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
