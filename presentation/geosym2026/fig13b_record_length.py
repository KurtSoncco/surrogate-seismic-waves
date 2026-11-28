"""Slide 13 companion: flat-west OpenSees versus the column, by record length.

The 15 s crop is data/case7a/opensees_crops/h5/crop_west.h5. The 30 s and 60 s
pilots are the Stampede job 3669168 records. Both rows are the center station
on the same 1310.72 s Fourier window as the training labels. The lower row is
the Pearson of log|TF|.

    python fig13b_record_length.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt

import figstyle as fs

SRC = fs.REPO / "experiments" / "case7a" / "results" / "flat_west_record_length.json"
PANELS = (("qvs", "Q–Vs damping"), ("svlab", "SVLAB damping"))
BANDS = (("0.1-2.5", "0.1–2.5 Hz", fs.NAVY), ("0.1-10", "0.1–10 Hz", fs.AMBER))


def _group(rows: list[dict], damping: str) -> list[dict]:
    return sorted((r for r in rows if r["damping"] == damping), key=lambda r: r["duration_s"])


def main() -> None:
    fs.apply()
    rows = json.loads(SRC.read_text())["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(fs.FULL_W, 6.8), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.09, top=0.86, hspace=0.28, wspace=0.16)
    for col, (damping, title) in enumerate(PANELS):
        group = _group(rows, damping)
        duration = [r["duration_s"] for r in group]
        top, bot = axes[0, col], axes[1, col]
        top.set_title(title, fontsize=18, color=fs.NAVY, pad=8)
        for key, label, color in BANDS:
            gap = [r["bands"][key]["sensor_abs_log"] for r in group]
            pear = [r["bands"][key]["sensor_pearson"] for r in group]
            top.plot(duration, gap, color=color, lw=3.0, marker="o", ms=8,
                     label=label if col == 0 else None, zorder=3)
            bot.plot(duration, pear, color=color, lw=3.0, marker="o", ms=8, zorder=3)
            print(f"{title} {key}: |log| {[round(v, 3) for v in gap]}  Pearson {[round(v, 3) for v in pear]}")
        low60 = group[-1]["bands"]["0.1-2.5"]["sensor_pearson"]
        mid60 = group[-1]["bands"]["0.1-10"]["sensor_pearson"]
        if damping == "svlab":
            bot.annotate(f"{low60:.3f}", xy=(60, low60), xytext=(6, 8), textcoords="offset points",
                         color=fs.NAVY, fontsize=14, fontweight="bold", ha="left", va="bottom")
            bot.annotate(f"{mid60:.3f}", xy=(60, mid60), xytext=(-6, -14), textcoords="offset points",
                         color=fs.AMBER, fontsize=14, fontweight="bold", ha="right", va="top")
        else:
            bot.annotate(f"{low60:.3f}", xy=(60, low60), xytext=(-10, 8), textcoords="offset points",
                         color=fs.NAVY, fontsize=14, fontweight="bold", ha="right", va="bottom")
            bot.annotate(f"{mid60:.3f}", xy=(60, mid60), xytext=(8, -12), textcoords="offset points",
                         color=fs.AMBER, fontsize=14, fontweight="bold", ha="left", va="top")
        top.set_xlim(8, 74)
        top.set_xticks([15, 30, 60])
    for ax in axes[0]:
        ax.set_ylim(0, 0.18)
        ax.set_yticks([0, 0.05, 0.10, 0.15])
    axes[0, 0].set_ylabel("Mean |log| versus\nthe column")
    axes[1, 0].set_ylim(0.85, 1.015)
    axes[1, 0].set_yticks([0.85, 0.90, 0.95, 1.00], ["0.85", "0.90", "0.95", "1.00"])
    axes[1, 1].set_ylim(0.85, 1.015)
    axes[1, 1].set_yticks([0.85, 0.90, 0.95, 1.00], ["0.85", "0.90", "0.95", "1.00"])
    axes[1, 0].set_ylabel("Pearson of log$|TF|$")
    for ax in axes[1]:
        ax.set_xlabel("Record length (s)")
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.53, 0.93), fontsize=15,
               bbox_transform=fig.transFigure)
    fs.save(fig, "fig13b_record_length", tight=True)


if __name__ == "__main__":
    main()
