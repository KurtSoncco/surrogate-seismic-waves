"""Fig. 5 — two IID cases: a typical Pearson and a rougher CoV."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import case_blurb, central_curves, load_pack, pick_iid_cases, vs_profiles  # noqa: E402
from responses import plot_leftover_curves, plot_tf, plot_vs  # noqa: E402
from style import figsize, panel_letter, save_figure, use_style  # noqa: E402


def _headline(pack, i: int, title: str) -> str:
    return (
        f"{title}\n"
        f"Pearson {float(pack['pearson_gino'][i]):.2f}, "
        f"CoV {float(pack['cov'][i]):.2f}, "
        f"$H$={float(pack['H'][i]):.0f} m"
    )


def main() -> dict[str, str]:
    use_style()
    pack = load_pack("iid")
    chosen = pick_iid_cases(pack)
    rows = (
        ("typical", "IID, typical"),
        ("rough", "IID, high CoV"),
    )
    fig, axes = plt.subplots(2, 3, figsize=figsize("double", height_mm=132))
    notes: dict[str, str] = {}
    letters = "abcdef"
    for r, (key, title) in enumerate(rows):
        i = chosen[key]
        notes[key] = case_blurb(pack, i)
        profiles = vs_profiles(pack, i)
        curves = central_curves(pack, i)
        plot_vs(axes[r, 0], profiles, domain="iid")
        plot_tf(axes[r, 1], curves)
        plot_leftover_curves(axes[r, 2], curves)
        axes[r, 1].set_title(_headline(pack, i, title), fontsize=7, pad=3)
        for c in range(3):
            panel_letter(axes[r, c], letters[r * 3 + c])
    handles, labels = axes[0, 1].get_legend_handles_labels()
    r_handles, r_labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(
        handles + r_handles,
        labels + r_labels,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "fig05_iid")
    return notes


if __name__ == "__main__":
    print(main())
