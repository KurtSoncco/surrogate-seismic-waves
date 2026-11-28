#!/usr/bin/env python3
"""Elastic half-space comparison for the in-family tests.

The bars are pooled g log-L1 of P128n128, the stored column (g = 0), the
center-column elastic outcrop, and the 9-column Pretell geometric mean.
Numbers are read from analysis/frozen/ladder.csv and baselines.csv. The
outcrop uses Q(Vs) matched to Rayleigh damping at the record (f1, f2).
An earlier docstring described this figure as a rigid-base box. That
description is withdrawn.

Practice Pearson is of log|TF| only. The previous box plotted Pearson of
linear |TF| on the full spectrum (analysis/G0.md). This script does not
recompute Pearson, Anderson, or a per-record median.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parents[1] / "analysis" / "frozen"
sys.path.insert(0, str(HERE.parent / "scripts"))
from plotting import apply_publication_style, panel_letter, save_figure  # noqa: E402

OUT = HERE / "03_state_of_practice"
RUNGS = (("iid_test", "Horizontal"), ("dipping_test", "Dipping"))
BANDS = (("mid", "0.4–2.5 Hz"), ("high", "2.5–10 Hz"))
SERIES = (
    ("P128n128", "#D55E00"),
    ("g = 0", "#0072B2"),
    ("Center column", "#009E73"),
    ("Pretell, 9 columns", "#CC79A7"),
)


def _one(frame: pd.DataFrame, **query) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for key, value in query.items():
        mask &= frame[key] == value
    hit = frame[mask]
    if len(hit) != 1:
        raise SystemExit(f"expected one row for {query}, found {len(hit)}")
    return hit.iloc[0]


def main() -> None:
    ladder = pd.read_csv(FROZEN / "ladder.csv")
    baselines = pd.read_csv(FROZEN / "baselines.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.5), sharey=True)
    x = np.arange(len(BANDS))
    width = 0.18
    for ax, (rung, title), letter in zip(axes, RUNGS, "ab"):
        for i, ((name, color), key) in enumerate(zip(SERIES, ("model", "g0", "center_column", "pretell_geomean"))):
            means, low, high = [], [], []
            for band, _label in BANDS:
                if key == "model":
                    row = _one(ladder, rung=rung, band=band)
                else:
                    row = _one(baselines, rung=rung, band=band, baseline=key)
                mean = float(row.g_log_l1)
                means.append(mean)
                low.append(mean - float(row.ci_low))
                high.append(float(row.ci_high) - mean)
                rows.append({
                    "rung": rung,
                    "band": band,
                    "series": name,
                    "g_log_l1": mean,
                    "ci_low": float(row.ci_low),
                    "ci_high": float(row.ci_high),
                    "ci_statistic": row.ci_statistic,
                    "source": "ladder.csv" if key == "model" else "baselines.csv",
                })
            positions = x + (i - 1.5) * width
            ax.bar(positions, means, width=width * 0.92, color=color, label=name)
            ax.errorbar(positions, means, yerr=np.vstack([low, high]), fmt="none", ecolor="#222222", elinewidth=0.6, capsize=1.5)
        ax.set_xticks(x, [label for _band, label in BANDS])
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("Pooled g log-L1" if letter == "a" else "")
        panel_letter(ax, letter)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    pd.DataFrame(rows).to_csv(OUT / "plotted_from_frozen.csv", index=False)
    print(f"wrote {OUT / 'central.pdf'}")


if __name__ == "__main__":
    main()
