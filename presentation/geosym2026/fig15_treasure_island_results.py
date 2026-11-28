"""Slide 15: Treasure Island, Pearson of |TF| vs x, AI (figstyle.STAGE) vs 1D, 0.1-2.5 Hz.

The SVLAB truth is valid only to 2.5 Hz (case7a/labels.TRUSTED_BAND_HZ): above
about 2.5 Hz its |TF| jumps to 30-500 at every station.

Curves from prep_svlab.py. Pearson and zone medians use the same functions as
case7a/score_svlab.py. Zones are labeled by their x ranges.

    python fig15_treasure_island_results.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

sys.path.insert(0, str(fs.REPO / "experiments" / "case7a"))
from labels import TRUSTED_BAND_HZ  # noqa: E402
from score_svlab import _pearson  # noqa: E402

BAND = TRUSTED_BAND_HZ
BAND_TEXT = f"{BAND[0]:g}–{BAND[1]:g} Hz"
SECTION = (-650.0, 1050.0)
RAMP = (600.0, 1000.0)
ZONES = (("Section, −650 to 1050 m", -650.0, 1050.0), ("East edge, 1050–1300 m", 1050.0, 1300.0),
         ("Flat east, > 1300 m", 1300.0, np.inf), ("All 233 stations", -np.inf, np.inf))


def main() -> None:
    fs.apply()
    d = np.load(fs.DATA / f"svlab_{fs.STAGE}_tf.npz")
    x, freq, truth = d["x"], d["freq"], d["tf_gt"]
    band = (freq >= BAND[0]) & (freq <= BAND[1])
    p_ai = np.array([_pearson(d["pred"][i, band], truth[i, band]) for i in range(x.size)])
    p_1d = np.array([_pearson(d["prior"][i, band], truth[i, band]) for i in range(x.size)])
    gain = p_ai - p_1d

    rows = []
    for name, lo, hi in ZONES:
        m = (x >= lo) & (x < hi)
        rows.append((name, float(np.median(p_ai[m])), float(np.median(p_1d[m]))))
        print(f"{name:26s} n={m.sum():3d}  AI {rows[-1][1]:+.3f}  1D {rows[-1][2]:+.3f}")
    ramp = (x >= RAMP[0]) & (x <= RAMP[1])
    k = np.flatnonzero(ramp)[np.argmax(gain[ramp])]
    sec = (x >= SECTION[0]) & (x <= SECTION[1])
    worst = np.flatnonzero(sec)[np.argmin(gain[sec])]
    print(f"ramp 600-1000 m: best AI gain {gain[k]:+.3f} at x = {x[k]:.0f} m (AI {p_ai[k]:.2f}, 1D {p_1d[k]:.2f}); "
          f"median gain {np.median(gain[ramp]):+.3f}; AI ahead at {np.mean(gain[ramp] > 0):.0%} of ramp stations")
    print(f"section: worst AI gain {gain[worst]:+.3f} at x = {x[worst]:.0f} m; "
          f"AI ahead at {np.mean(gain[sec] > 0):.0%} of section stations")

    fig = plt.figure(figsize=(fs.FULL_W, 4.9))
    ax = fig.add_axes([0.075, 0.14, 0.60, 0.74])
    ax.axvspan(RAMP[0], RAMP[1], color=fs.AMBER, alpha=0.14, lw=0, zorder=0)
    ax.plot(x, p_1d, color=fs.ONED, lw=2.6, ls=(0, (5, 3)), label="1D column", zorder=2)
    ax.plot(x, p_ai, color=fs.AI, lw=3.0, label=f"This work ({fs.STAGE_LABEL.lower()})", zorder=3)
    ax.axvspan(x.min() - 20, SECTION[0], color="white", alpha=0.75, lw=0, zorder=4)
    ax.axvspan(x.min() - 20, SECTION[0], color=fs.REAL, alpha=0.35, lw=0, zorder=4)
    ax.text(-1200, 0.60, "Outside the\nmodeled section", ha="center", va="top", fontsize=14,
            color=fs.MUTED, zorder=5)
    ax.set_xlim(x.min() - 20, x.max() + 20)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.5, 0.75, 1.0], ["0.5", "0.75", "1"])
    ax.set_xticks([-1500, -650, 0, 1050, 1500], ["−1500", "−650", "0", "1050", "1500"])
    ax.set_xlabel("Station position x (m)")
    ax.set_ylabel(f"Pearson of $|TF|$, {BAND_TEXT}")
    ax.legend(loc="lower right", ncol=1, fontsize=15)
    if gain[k] >= 0.02:
        ax.annotate(f"AI {gain[k]:+.2f} over 1D\nat x ≈ {x[k]:.0f} m", xy=(x[k], p_ai[k]),
                    xytext=(x[k] - 650, 0.62), fontsize=15, fontweight="semibold", color=fs.AI, ha="center",
                    va="top", arrowprops={"arrowstyle": "-|>", "color": fs.AI, "lw": 1.8}, zorder=6)
    else:
        spread = float(np.max(np.abs(gain[sec])))
        ax.text(200, 0.62, f"AI within {spread:.2f} of 1D\nacross the section", fontsize=15,
                fontweight="semibold", color=fs.AI, ha="center", va="top", zorder=6)
    ax.text(np.mean(RAMP), 0.995, "Bedrock Incline", ha="center", va="top", fontsize=13, color=fs.AMBER,
            fontweight="semibold")

    tx, ty, rh = 0.715, 0.80, 0.085
    fig.text(tx, ty + 0.06, f"Median Pearson, {BAND_TEXT}", fontsize=15, fontweight="semibold", color=fs.NAVY)
    fig.text(tx + 0.205, ty, "This work", fontsize=15, fontweight="bold", color=fs.AI, ha="right")
    fig.text(tx + 0.275, ty, "1D", fontsize=15, fontweight="bold", color=fs.ONED, ha="right")
    for r, (name, a, b) in enumerate(rows):
        y = ty - (r + 1) * rh
        if r == len(rows) - 1:
            fig.lines.append(plt.Line2D([tx, tx + 0.28], [y + rh * 0.55] * 2, transform=fig.transFigure,
                                        color=fs.MUTED, lw=0.8))
        fig.text(tx, y, name.replace(", ", "\n"), fontsize=13.5, va="center", linespacing=1.05)
        fig.text(tx + 0.205, y, f"{a:.2f}".replace("-", "−"), fontsize=15, ha="right", va="center",
                 fontweight="bold" if a > b + 0.005 else "normal")
        fig.text(tx + 0.275, y, f"{b:.2f}".replace("-", "−"), fontsize=15, ha="right", va="center",
                 fontweight="bold" if b > a + 0.005 else "normal")
    fs.save(fig, "fig15_treasure_island_results", tight=True)


if __name__ == "__main__":
    main()
