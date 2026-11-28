#!/usr/bin/env python3
"""Context invariance, plus the Case 7 column diagnostic.

The synthetic curves are the context_invariance rows of
analysis/frozen/consistency_tests.csv. A wider window repeats the edge of the
500 m training strip. The old mid-band skill curve is not drawn.

case7_column.pdf is OpenSees versus the corrected column on existing H5s.
The network is not scored. Open markers meet the domain edge.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[1] / "analysis"
FROZEN = ANALYSIS / "frozen" / "consistency_tests.csv"
COVERAGE = ANALYSIS / "audit" / "coverage.png"
sys.path.insert(0, str(HERE.parent / "scripts"))
from plotting import apply_publication_style, panel_letter, save_figure  # noqa: E402

OUT = HERE / "04_width_and_manifold"
FAMILIES = ("iid", "ood_dipping", "ood_three_layer")
LABEL = {"iid": "Horizontal", "ood_dipping": "Dipping", "ood_three_layer": "Three-layer"}
COLOR = {"iid": "#0072B2", "ood_dipping": "#D55E00", "ood_three_layer": "#009E73"}
CASE7 = ANALYSIS / "frozen" / "case7_column.csv"
CASE_ORDER = ("flat_west", "center", "thinning", "ramp", "flat_east")
CASE_LABEL = {
    "flat_west": "Flat west",
    "center": "Center",
    "thinning": "Thinning",
    "ramp": "Ramp",
    "flat_east": "Flat east",
}
CASE_COLOR = {
    "flat_west": "#0072B2",
    "center": "#009E73",
    "thinning": "#E69F00",
    "ramp": "#D55E00",
    "flat_east": "#CC79A7",
}


def main() -> None:
    frame = pd.read_csv(FROZEN)
    part = frame[frame["test"] == "context_invariance"].copy()
    if part.empty:
        raise SystemExit(f"no context_invariance rows in {FROZEN}")
    OUT.mkdir(parents=True, exist_ok=True)
    part.to_csv(OUT / "context_invariance.csv", index=False)
    if COVERAGE.is_file():
        shutil.copyfile(COVERAGE, OUT / "coverage.png")
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.4), sharex=True)
    for ax, band, letter in zip(axes, ("mid", "high"), "ab"):
        for family in FAMILIES:
            rows = part[(part["family"] == family) & (part["band"] == band)].sort_values("width_m")
            ax.plot(
                rows["width_m"].astype(float),
                rows["value"].astype(float),
                color=COLOR[family],
                marker="o",
                lw=1.1,
                ms=4,
                label=LABEL[family],
            )
        ax.set_xticks([500, 1000, 1500])
        ax.set_xlabel("Window width (m)")
        ax.set_ylabel("Pooled g log-L1")
        name = "0.4–2.5 Hz" if band == "mid" else "2.5–10 Hz"
        ax.set_title(name, fontsize=8)
        panel_letter(ax, letter)
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.suptitle(
        "Context invariance. Training strip 500 m; wider windows repeat the edge.",
        fontsize=8,
    )
    save_figure(fig, OUT / "central.pdf")
    print(f"wrote {OUT / 'central.pdf'}")
    _case7()


def _case7() -> None:
    """OpenSees versus the corrected column. The network is not drawn."""

    if not CASE7.is_file():
        raise SystemExit(f"missing {CASE7}; run case7_column_rescore.py")
    frame = pd.read_csv(CASE7)
    frame.to_csv(OUT / "case7_column.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.5), sharex=True)
    for ax, band, letter, title in zip(
        axes, ("low", "mid"), "ab", ("0.1–0.4 Hz", "0.4–2.5 Hz"),
    ):
        for name in CASE_ORDER:
            rows = frame[(frame["name"] == name) & (frame["band"] == band)].sort_values("width_m")
            if rows.empty:
                raise SystemExit(f"no Case 7 rows for {name} {band}")
            width = rows["width_m"].to_numpy(dtype=float)
            value = rows["mean_abs_log"].to_numpy(dtype=float)
            edge = rows["domain_flag"].astype(str).str.contains("domain_edge").to_numpy()
            ax.plot(width, value, color=CASE_COLOR[name], lw=1.1, label=CASE_LABEL[name])
            ax.plot(width[~edge], value[~edge], color=CASE_COLOR[name], marker="o", ms=4, ls="none")
            if edge.any():
                ax.plot(
                    width[edge], value[edge], color=CASE_COLOR[name],
                    marker="o", ms=5, ls="none", markerfacecolor="none",
                )
        ax.set_xticks([500, 1000, 1500, 2000])
        ax.set_xlabel("Interior width (m)")
        ax.set_ylabel("Mean |log| gap")
        ax.set_title(title, fontsize=8)
        panel_letter(ax, letter)
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.suptitle(
        "Case 7. OpenSees versus the corrected column. Open markers meet the domain edge. Network not scored.",
        fontsize=8,
    )
    save_figure(fig, OUT / "case7_column.pdf")
    print(f"wrote {OUT / 'case7_column.pdf'}")


if __name__ == "__main__":
    main()
