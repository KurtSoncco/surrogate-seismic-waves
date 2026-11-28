"""Slide 13 companion: flat-west Pearson at every station, not only the center.

Stations are every 20 m across the 500 m crop, plus the center column.
Each point is the Pearson of log|TF| between that station's OpenSees record
and the 1D column at the same place, on one 1310.72 s Fourier window.

    python fig13c_stations.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

SRC = fs.REPO / "experiments" / "case7a" / "results" / "flat_west_record_length.json"
PANELS = (("qvs", "Q–Vs damping"), ("svlab", "SVLAB damping"))
BANDS = ("0.1-2.5", "0.1-10")
LENGTHS = ((15.0, "15 s", fs.MUTED), (30.0, "30 s", fs.NAVY), (60.0, "60 s", fs.AI))


def _match(rows: list[dict], damping: str, duration: float) -> dict:
    hit = [r for r in rows if r["damping"] == damping and abs(r["duration_s"] - duration) < 1.0]
    if len(hit) != 1:
        raise KeyError(f"{damping} {duration} s: {len(hit)} rows")
    return hit[0]


def main() -> None:
    fs.apply()
    rows = json.loads(SRC.read_text())["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(fs.FULL_W, 6.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.84, hspace=0.28, wspace=0.12)
    for col, (damping, title) in enumerate(PANELS):
        center = _match(rows, damping, 15.0)["center_x_m"]
        for row_i, key in enumerate(BANDS):
            ax = axes[row_i, col]
            if row_i == 0:
                ax.set_title(title, fontsize=18, color=fs.NAVY, pad=8)
            for duration, label, color in LENGTHS:
                rec = _match(rows, damping, duration)
                x = np.asarray(rec["x_m"])
                pear = np.asarray(rec["bands"][key]["station_pearson"])
                order = np.argsort(x)
                ax.plot(x[order], pear[order], color=color, lw=2.6, marker="o", ms=5,
                        label=label if col == 0 and row_i == 0 else None, zorder=3)
                print(
                    f"{title} {duration:.0f} s {key}: "
                    f"{pear.min():.3f}–{pear.max():.3f}  pooled {rec['bands'][key]['pooled_pearson']:.3f}"
                )
            ax.axvline(center, color=fs.TRUTH, lw=1.1, ls="--", zorder=1,
                       label="Center station" if col == 0 and row_i == 0 else None)
    for ax in axes[1]:
        ax.set_xlabel("Position (m)")
        ax.set_xticks([-1750, -1500, -1250])
    axes[0, 0].set_xlim(-1780, -1240)
    axes[0, 0].set_ylim(0.85, 1.015)
    axes[0, 0].set_yticks([0.85, 0.90, 0.95, 1.00], ["0.85", "0.90", "0.95", "1.00"])
    axes[0, 0].set_ylabel("0.1–2.5 Hz\nPearson of log$|TF|$")
    axes[1, 0].set_ylabel("0.1–10 Hz\nPearson of log$|TF|$")
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.53, 0.93), fontsize=15,
               bbox_transform=fig.transFigure)
    fs.save(fig, "fig13c_stations", tight=True)


if __name__ == "__main__":
    main()
