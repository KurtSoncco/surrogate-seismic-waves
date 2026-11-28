"""Slide 7: three random-field seeds of one site, and correlation length 10 vs 100 m.

All fields are real training-campaign realizations (horizontal set).

    python fig07_random_fields.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch

import figstyle as fs

VS = fs.REPO / "data" / "vardomain" / "cache" / "iid" / "vs.npy"
# Rock (760-1500 m/s) saturates to cream so the soil spans the colormap.
NORM = LogNorm(vmin=100, vmax=700)
CB_TICKS = [100, 200, 400, 700]
CHIPS = ("Soil $V_s$  100–360 m/s", "CoV  0.10–0.30", "Correlation length  10–100 m", "Bedrock $V_s$  760–1500 m/s")


def _pick_site(params: pd.DataFrame) -> int:
    """A test site near the median depth, with CoV high enough to see."""
    test = params[(params.family == "iid") & (params.split == "test")]
    site = test.groupby("group")[["CoV", "H", "rH", "Vs1"]].first()
    score = (site.CoV - 0.25).abs() / 0.05 + (site.H - 55).abs() / 15 + (site.rH - 50).abs() / 30
    return int(score.idxmin())


def _pick_lengths(params: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Two runs alike in CoV, depth, and Vs1, at about 10 m and 100 m correlation length."""
    iid = params[params.family == "iid"]
    pool = iid[(iid.CoV.between(0.22, 0.30)) & (iid.H.between(45, 70)) & (iid.Vs1.between(150, 230))]
    short = pool.iloc[(pool.rH - 10).abs().argsort()].iloc[0]
    long = pool.iloc[(pool.rH - 100).abs().argsort()].iloc[0]
    return short, long


def _section(ax, vs: np.ndarray, depth: float):
    nz = int(min(vs.shape[0], depth))
    im = ax.imshow(vs[:nz].astype(np.float32), cmap=fs.CMAP, norm=NORM, aspect="auto",
                   extent=(0, vs.shape[1], nz, 0), interpolation="nearest")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    return im


def main() -> None:
    fs.apply()
    params = pd.read_csv(fs.DATA / "run_params.csv")
    vs_all = np.load(VS, mmap_mode="r")
    group = _pick_site(params)
    runs = params[(params.family == "iid") & (params.group == group)].sort_values("run").head(3)
    short, long = _pick_lengths(params)
    depth = float(runs.H.iloc[0]) + 15.0

    fig = plt.figure(figsize=(fs.FULL_W, 5.6))
    left, width, top, h, gap = 0.075, 0.53, 0.92, 0.19, 0.06
    axes = [fig.add_axes([left, top - (k + 1) * h - k * gap, width, h]) for k in range(3)]
    for k, (ax, (_, row)) in enumerate(zip(axes, runs.iterrows())):
        im = _section(ax, vs_all[int(row.run)], depth)
        ax.set_yticks([0, 50])
        ax.set_xticks([0, 250, 500] if k == 2 else [])
        ax.text(1.0, 1.03, f"Seed {k + 1}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=15, fontweight="semibold", color=fs.NAVY)
    axes[2].set_xlabel("Distance (m)", labelpad=2)
    fig.text(0.012, top - 1.5 * h - gap, "Depth (m)", rotation=90, va="center", fontsize=18)
    axes[0].text(0.0, 1.03, "One site, three random-field seeds", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=15, color=fs.MUTED)

    ins = [fig.add_axes([0.65, 0.60, 0.25, 0.23]), fig.add_axes([0.65, 0.23, 0.25, 0.23])]
    for ax, row, label in ((ins[0], short, "10 m"), (ins[1], long, "100 m")):
        _section(ax, vs_all[int(row.run)], float(row.H) + 10.0)
        ax.set_xlim(0, 250)
        ax.set_xticks([0, 250] if ax is ins[1] else [])
        ax.set_yticks([])
        ax.text(0.0, 1.03, f"Correlation length ≈ {label}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=15, fontweight="semibold", color=fs.NAVY)
        print(f"{label}: run {int(row.run)} rH {row.rH:.1f} m, CoV {row.CoV:.2f}, H {row.H:.0f} m")
    ins[1].set_xlabel("Distance (m)", labelpad=2)

    cax = fig.add_axes([0.925, 0.23, 0.013, 0.60])
    cb = fig.colorbar(im, cax=cax, ticks=CB_TICKS, extend="max", extendfrac=0.06)
    cb.ax.set_yticklabels([str(t) for t in CB_TICKS])
    cb.ax.yaxis.set_minor_locator(plt.NullLocator())
    cb.set_label("$V_s$ (m/s)", fontsize=16)
    cb.ax.tick_params(labelsize=14)

    chip_w, chip_h, y0, chip_gap = 0.207, 0.075, 0.01, 0.012
    x = 0.075
    for text in CHIPS:
        fig.patches.append(FancyBboxPatch((x, y0), chip_w, chip_h, boxstyle="round,pad=0,rounding_size=0.012",
                                          transform=fig.transFigure, facecolor=fs.PANEL, edgecolor="none"))
        fig.text(x + chip_w / 2, y0 + chip_h / 2, text, ha="center", va="center", fontsize=14.5, color=fs.NAVY)
        x += chip_w + chip_gap

    print(f"site group {group}: runs {runs.run.tolist()}, Vs1 {runs.Vs1.iloc[0]:.0f}, CoV {runs.CoV.iloc[0]:.2f}, "
          f"rH {runs.rH.iloc[0]:.0f} m, H {runs.H.iloc[0]:.0f} m, Vs2 {runs.Vs2.iloc[0]:.0f}")
    iid = params[params.family == "iid"]
    print("ranges: Vs1", iid.Vs1.min().round(), iid.Vs1.max().round(), "CoV", iid.CoV.min().round(3),
          iid.CoV.max().round(3), "rH", iid.rH.min().round(1), iid.rH.max().round(1), "Vs2",
          iid.Vs2.min().round(), iid.Vs2.max().round(), "H", iid.H.min(), iid.H.max())
    fs.save(fig, "fig07_random_fields", tight=False)


if __name__ == "__main__":
    main()
