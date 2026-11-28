"""Slide 13 companion: the 4000 m Case 7 section versus the column.

full_qvs.h5 is the 15 s Q–Vs solve of the whole interior. The shaded interval is
the 500 m flat-west crop scored separately. full_svlab.h5 repeats this surface
record, and there is no 30 s or 60 s full-section file.

    python fig13d_full_section.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import figstyle as fs

SRC = fs.REPO / "experiments" / "case7a" / "results" / "full_section_vs_column.json"
BANDS = (("0.1-2.5", "0.1–2.5 Hz", fs.NAVY), ("0.1-10", "0.1–10 Hz", fs.AMBER))
CROP = (-1750.0, -1250.0)
# case7a.ZONES, named for the slide.
ZONE_LABELS = ((-1325.0, "Flat west"), (200.0, "Basin"), (1175.0, "Ramp"), (1650.0, "Flat east"))
ZONE_EDGES = (-650.0, 1050.0, 1300.0)


def main() -> None:
    fs.apply()
    row = json.loads(SRC.read_text())["row"]
    x = np.asarray(row["x_m"])
    order = np.argsort(x)
    x = x[order]
    fig, axes = plt.subplots(2, 1, figsize=(fs.FULL_W, 6.6), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.84, hspace=0.18)
    for key, label, color in BANDS:
        pear = np.asarray(row["bands"][key]["station_pearson"])[order]
        gap = np.asarray(row["bands"][key]["station_abs_log"])[order]
        axes[0].plot(x, pear, color=color, lw=2.6, label=label, zorder=3)
        axes[1].plot(x, gap, color=color, lw=2.6, zorder=3)
        print(f"{key}: Pearson {pear.min():.3f}–{pear.max():.3f}  |log| median {np.median(gap):.3f}")
    for ax in axes:
        ax.axvspan(*CROP, color=fs.PANEL, zorder=0)
        for edge in ZONE_EDGES:
            ax.axvline(edge, color=fs.MUTED, lw=0.8, zorder=1)
    axes[0].set_xlim(-2050, 2050)
    axes[0].set_ylim(0.84, 1.08)
    axes[0].set_yticks([0.85, 0.90, 0.95, 1.00], ["0.85", "0.90", "0.95", "1.00"])
    axes[0].set_ylabel("Pearson of log$|TF|$")
    axes[0].set_title("15 s full section, Q–Vs damping", fontsize=18, color=fs.NAVY, pad=16)
    for pos, name in ZONE_LABELS:
        axes[0].text(pos, 1.055, name, ha="center", va="center", fontsize=13, color=fs.MUTED)
    axes[1].set_ylim(0, 0.18)
    axes[1].set_yticks([0, 0.05, 0.10, 0.15])
    axes[1].set_ylabel("Mean |log| versus\nthe column")
    axes[1].set_xlabel("Position (m)")
    axes[1].set_xticks([-2000, -1000, 0, 1000, 2000])
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor=fs.PANEL, edgecolor="none", label="500 m flat-west crop"))
    labels.append("500 m flat-west crop")
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.53, 0.91),
               fontsize=15, bbox_transform=fig.transFigure)
    fs.save(fig, "fig13d_full_section", tight=True)


if __name__ == "__main__":
    main()
