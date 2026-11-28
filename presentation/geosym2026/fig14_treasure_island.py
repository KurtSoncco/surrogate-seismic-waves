"""Slide 14: the Treasure Island (Case 7) Vs section with annotations.

Same data and color scale as case7a/plot_case7_section.py, at slide scale.

    python fig14_treasure_island.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import figstyle as fs

sys.path.insert(0, str(fs.REPO / "experiments" / "case7a"))
from case7a import ELEM_M, GEOLOGIC_X_MAX, GEOLOGIC_X_MIN, X_LEFT_M, ensure_cache, window_center_x  # noqa: E402

ROCK_VS = 760.0
TICKS = [150, 300, 600, 1200, 2400]


def main() -> None:
    fs.apply()
    vs = np.asarray(ensure_cache()["vs"], dtype=np.float64)
    x_edge = X_LEFT_M + ELEM_M * np.arange(vs.shape[1] + 1)
    j0 = int(np.searchsorted(x_edge, GEOLOGIC_X_MIN))
    j1 = int(np.searchsorted(x_edge, GEOLOGIC_X_MAX))
    vs, x_edge = vs[:, j0:j1], x_edge[j0 : j1 + 1]
    z_edge = ELEM_M * np.arange(vs.shape[0] + 1)
    xc = 0.5 * (x_edge[:-1] + x_edge[1:])
    zc = 0.5 * (z_edge[:-1] + z_edge[1:])
    rock = np.array([zc[np.argmax(vs[:, j] >= ROCK_VS)] - ELEM_M / 2 for j in range(vs.shape[1])])
    mud = vs[:2]
    print(f"bedrock (Vs >= {ROCK_VS:.0f}) top: {rock[0]:.0f} m at x = {xc[0]:.0f}, {rock[-1]:.0f} m at x = {xc[-1]:.0f}; "
          f"x=600: {rock[np.argmin(abs(xc - 600))]:.0f} m")
    print(f"top 10 m Vs: {np.median(mud):.0f} m/s")
    stations = window_center_x()
    print(f"SVLAB stations: {stations.size}, {stations.min():.0f} to {stations.max():.0f} m")

    fig, ax = plt.subplots(figsize=(fs.FULL_W, 4.3))
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.17, top=0.83)
    mesh = ax.pcolormesh(x_edge, z_edge, vs, cmap=fs.CMAP, norm=LogNorm(150, 2400), shading="flat",
                         rasterized=True)
    ax.set_xlim(x_edge[0], x_edge[-1])
    ax.set_ylim(z_edge[-1], -16)
    ax.axhline(0, color=fs.TRUTH, lw=1.2)
    inside = stations[(stations >= x_edge[0]) & (stations <= x_edge[-1])]
    ax.scatter(inside, np.full(inside.size, -6.5), marker="v", s=22, color=fs.TRUTH, zorder=5, clip_on=False)
    ax.set_xticks(np.arange(-600, 1001, 200))
    ax.set_yticks([0, 50, 100, 150])
    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.text(x_edge[0], -22, "SVLAB 2D simulation: 233 surface stations, 15 m apart",
            ha="left", va="bottom", fontsize=15, fontweight="semibold", color=fs.TRUTH)

    ax.annotate(f"Soft Bay Mud, $V_s$ ≈ {np.median(mud):.0f} m/s", xy=(-380, 6), xytext=(-560, 42),
                fontsize=15, color="white", fontweight="semibold",
                arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 1.8})
    xs = np.array([600, 1000])
    zs = np.interp(xs, xc, rock)
    ax.annotate("", xy=(xs[1], zs[1] - 3), xytext=(xs[0], zs[0] - 3),
                arrowprops={"arrowstyle": "-|>", "color": fs.AMBER, "lw": 3.2, "mutation_scale": 22})
    ax.text(780, zs.mean() - 22, f"Bedrock Incline\n{zs[0]:.0f} $\\rightarrow$ {zs[1]:.0f} m", ha="center", va="bottom",
            fontsize=15, fontweight="semibold", color="white", linespacing=1.15)
    ax.text(-560, rock[np.argmin(abs(xc + 560))] + 14, f"Bedrock ≈ {rock[np.argmin(abs(xc + 560))]:.0f} m",
            ha="left", va="top", fontsize=14, color=fs.TRUTH)

    cax = fig.add_axes([0.915, 0.17, 0.013, 0.66 * 150 / 166])
    cb = fig.colorbar(mesh, cax=cax, ticks=TICKS)
    cb.ax.set_yticklabels([str(t) for t in TICKS])
    cb.ax.yaxis.set_minor_locator(plt.NullLocator())
    cb.set_label("$V_s$ (m/s)", fontsize=16)
    cb.ax.tick_params(labelsize=14)
    fs.save(fig, "fig14_treasure_island", tight=True)


if __name__ == "__main__":
    main()
