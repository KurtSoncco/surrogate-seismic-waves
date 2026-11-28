"""Slide 13: Case 7 crops, Pearson of log |TF| (0.4-2.5 Hz, all stations pooled) vs crop width, AI vs 1D, per zone.

Source: GIFNO-VARDOMAIN/figures/<figstyle.STAGE>/case7_width.json. Stage 2
trained on variable-width sections up to 2000 m; Case 7 geometry was excluded
from those sections.

    python fig13_width.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

SRC = fs.RESULTS / "case7_width.json"
ZONES = (("flat_west", "Flat west"), ("center", "Center"), ("thinning", "Thinning"), ("ramp", "Ramp"),
         ("flat_east", "Flat east"))


def main() -> None:
    fs.apply()
    data = json.loads(SRC.read_text())
    fig, axes = plt.subplots(1, len(ZONES), figsize=(fs.FULL_W, 3.9), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.2, top=0.8, wspace=0.12)
    skills = []
    for ax, (key, name) in zip(axes, ZONES):
        widths = sorted(int(w) for w in data[key])
        ai = np.array([data[key][str(w)]["pearson"] for w in widths])
        oned = np.array([data[key][str(w)]["pearson_1d"] for w in widths])
        skills += [data[key][str(w)]["skill"] for w in widths]
        ax.plot(widths, oned, color=fs.ONED, lw=2.6, ls=(0, (5, 3)), marker="o", ms=7, zorder=2)
        ax.plot(widths, ai, color=fs.AI, lw=3.0, marker="o", ms=8, zorder=3)
        ax.text(widths[-1] + 60, ai[-1], f"{ai[-1]:.2f}", color=fs.AI, fontsize=14, va="center",
                fontweight="bold")
        ax.set_title(name, fontsize=17, color=fs.NAVY)
        ax.set_xlim(300, 2300)
        ax.set_xticks([500, 1000, 1500, 2000], ["0.5", "1", "1.5", "2"])
        print(f"{name}: AI {np.round(ai, 3)}  1D {np.round(oned, 3)}")
    axes[0].set_ylim(0.95, 1.0)
    axes[0].set_yticks([0.95, 0.975, 1.0], ["0.95", "0.975", "1.00"])
    axes[0].set_ylabel("Pearson of log$|TF|$\n0.4–2.5 Hz")
    fig.text(0.535, 0.03, "Crop width (km)", ha="center", fontsize=18)
    axes[0].plot([], [], color=fs.AI, lw=3, marker="o", label="This work")
    axes[0].plot([], [], color=fs.ONED, lw=2.6, ls=(0, (5, 3)), marker="o", label="1D column")
    fig.legend(loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.0), fontsize=15)
    print(f"skill vs 1D: {min(skills):.2f} to {max(skills):.2f}")
    fs.save(fig, "fig13_width", tight=True)


if __name__ == "__main__":
    main()
