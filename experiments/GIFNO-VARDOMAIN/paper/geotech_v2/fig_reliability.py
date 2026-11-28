#!/usr/bin/env python3
"""One aggregation: pooled g log-L1, and validation against blind.

Bars are read from analysis/frozen/ladder.csv. Stage skill bars are not drawn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LADDER = HERE.parents[1] / "analysis" / "frozen" / "ladder.csv"
sys.path.insert(0, str(HERE.parent / "scripts"))
from plotting import apply_publication_style, panel_letter, save_figure  # noqa: E402

OUT = HERE / "06_reliability_and_broadband"
IN_FAMILY = (
    ("iid_test", "IID"),
    ("dipping_test", "Dipping"),
    ("three_layer_test", "Three-layer"),
)
HOLDOUT = (
    ("manifold_val", "Validation"),
    ("manifold_blind", "Blind"),
)


def _row(frame: pd.DataFrame, rung: str, band: str) -> pd.Series:
    hit = frame[(frame["rung"] == rung) & (frame["band"] == band)]
    if len(hit) != 1:
        raise SystemExit(f"expected one ladder row for {rung} {band}, found {len(hit)}")
    return hit.iloc[0]


def _grouped(ax, frame: pd.DataFrame, rungs: tuple[tuple[str, str], ...], bands: tuple[str, ...], band_labels: tuple[str, ...]) -> None:
    x = np.arange(len(rungs))
    width = 0.8 / len(bands)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for i, (band, label, color) in enumerate(zip(bands, band_labels, colors)):
        means, low, high = [], [], []
        for rung, _name in rungs:
            row = _row(frame, rung, band)
            mean = float(row.g_log_l1)
            means.append(mean)
            low.append(mean - float(row.ci_low))
            high.append(float(row.ci_high) - mean)
        positions = x + (i - (len(bands) - 1) / 2) * width
        ax.bar(positions, means, width=width * 0.92, color=color, label=label)
        ax.errorbar(positions, means, yerr=np.vstack([low, high]), fmt="none", ecolor="#222222", elinewidth=0.6, capsize=2)
    ax.set_xticks(x, [name for _rung, name in rungs], fontsize=7)
    ax.set_ylabel("Pooled g log-L1")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=6)


def main() -> None:
    frame = pd.read_csv(LADDER)
    OUT.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.5))
    _grouped(axes[0], frame, IN_FAMILY, ("mid", "high"), ("0.4–2.5 Hz", "2.5–10 Hz"))
    axes[0].set_title("In-family tests", fontsize=8)
    panel_letter(axes[0], "a")
    _grouped(
        axes[1],
        frame,
        HOLDOUT,
        ("low", "mid", "high", "overall"),
        ("0.1–0.4 Hz", "0.4–2.5 Hz", "2.5–10 Hz", "0.1–10 Hz"),
    )
    axes[1].set_title("Manifold validation and blind", fontsize=8)
    panel_letter(axes[1], "b")
    fig.suptitle("Pooled absolute g log-L1. Blind was not in the selection score.", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, OUT / "central.pdf")
    print(f"wrote {OUT / 'central.pdf'}")


if __name__ == "__main__":
    main()
