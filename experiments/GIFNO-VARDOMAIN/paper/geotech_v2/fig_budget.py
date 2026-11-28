#!/usr/bin/env python3
"""Error budget drawn only from analysis/frozen/error_budget.csv.

The FE floor is unmeasured and is not drawn. Treasure Island intervals in
that file are moving-block medians, so they are not drawn on the pooled mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUDGET = HERE.parents[1] / "analysis" / "frozen" / "error_budget.csv"
sys.path.insert(0, str(HERE.parent / "scripts"))
from plotting import apply_publication_style, panel_letter, save_figure  # noqa: E402

OUT = HERE / "06_reliability_and_broadband"
FAMILIES = (
    ("iid", "IID"),
    ("ood_dipping", "Dipping"),
    ("ood_three_layer", "Three-layer"),
    ("manifold_blind", "Manifold blind"),
    ("treasure_island", "Treasure Island"),
)
SERIES = (
    ("prior_g_log_l1", "1D prior", "#0072B2"),
    ("model_g_log_l1", "P128n128", "#D55E00"),
    ("realization_floor", "Realization floor", "#009E73"),
    ("pod_floor", "POD floor", "#CC79A7"),
)
PANELS = (
    ("low", "0.1–0.4 Hz"),
    ("mid", "0.4–2.5 Hz"),
    ("high", "2.5–10 Hz"),
    ("overall", "0.1–10 Hz"),
)


def main() -> None:
    frame = pd.read_csv(BUDGET)
    OUT.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 6.2), sharex=True)
    x = np.arange(len(FAMILIES))
    width = 0.18
    for ax, (band, title), letter in zip(axes.ravel(), PANELS, "abcd"):
        for i, (column, label, color) in enumerate(SERIES):
            heights = []
            for family, _name in FAMILIES:
                row = frame[(frame["family"] == family) & (frame["band"] == band)]
                if len(row) != 1:
                    raise SystemExit(f"expected one error-budget row for {family} {band}")
                value = row.iloc[0][column]
                heights.append(np.nan if pd.isna(value) else float(value))
            positions = x + (i - 1.5) * width
            ax.bar(positions, heights, width=width * 0.92, color=color, label=label)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("g log-L1")
        panel_letter(ax, letter)
    for ax in axes[1, :]:
        ax.set_xticks(x, [name for _family, name in FAMILIES], rotation=20, ha="right", fontsize=6.5)
    axes[0, 1].legend(frameon=False, fontsize=6)
    fig.suptitle("Error budget. The FE floor is unmeasured, so ratios are provisional.", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, OUT / "error_budget.pdf")
    print(f"wrote {OUT / 'error_budget.pdf'}")


if __name__ == "__main__":
    main()
