"""Fig. 4 — classical ranking with bootstrap intervals, plus IID frequency bands."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_iid_bands, load_ranking  # noqa: E402
from style import (  # noqa: E402
    DOMAIN_LABEL,
    DOMAIN_ORDER,
    METHOD_STYLE,
    RANK_METHODS,
    errorbar_method,
    figsize,
    panel_letter,
    save_figure,
    use_style,
)


def _rows(ranking: list[dict], domain: str) -> dict[str, dict]:
    return {row["method"]: row for row in ranking if row["domain"] == domain}


def _rank_panel(ax, ranking: list[dict], metric: str, xlabel: str) -> None:
    n_m = len(RANK_METHODS)
    y_base = []
    y = 0.0
    for _domain in DOMAIN_ORDER:
        y_base.append(y)
        y += n_m + 1.4
    for d_i, domain in enumerate(DOMAIN_ORDER):
        block = _rows(ranking, domain)
        for m_i, name in enumerate(RANK_METHODS):
            row = block[name]
            yy = y_base[d_i] + (n_m - 1 - m_i)
            errorbar_method(
                ax,
                row[metric],
                yy,
                row[f"{metric}_lo"],
                row[f"{metric}_hi"],
                METHOD_STYLE[name],
                label=name if d_i == 0 else None,
            )
    centers = [base + (n_m - 1) / 2 for base in y_base]
    ax.set_yticks(centers)
    ax.set_yticklabels([DOMAIN_LABEL[d] for d in DOMAIN_ORDER])
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.8, y_base[-1] + n_m - 0.2)


def _band_panel(ax, bands: dict[str, dict[str, float]], prefix: str, ylabel: str) -> None:
    names = list(RANK_METHODS)
    x = np.arange(3, dtype=float)
    dodge = np.linspace(-0.28, 0.28, len(names))
    keys = [f"{prefix}_{b}" for b in ("low", "mid", "high")]
    for name, dx in zip(names, dodge):
        style = METHOD_STYLE[name]
        ys = [bands[name][k] for k in keys]
        mfc = style.get("mfc", style["color"])
        ax.plot(
            x + dx,
            ys,
            color=style["color"],
            ls=style["ls"],
            lw=0.8,
            marker=style["marker"] or "o",
            ms=style.get("ms", 3.0),
            markerfacecolor=mfc,
            markeredgecolor=style["color"],
            markeredgewidth=0.4,
            label=name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["0.1–0.5 Hz", "0.5–2 Hz", "2–10 Hz"])
    ax.set_ylabel(ylabel)


def main() -> None:
    use_style()
    ranking = load_ranking()
    bands = load_iid_bands()
    fig, axes = plt.subplots(2, 2, figsize=figsize("double", height_mm=128))
    _rank_panel(axes[0, 0], ranking, "anderson", "median Anderson misfit")
    _rank_panel(axes[0, 1], ranking, "pearson", r"median Pearson of $|\mathrm{TF}|$")
    axes[0, 0].set_xlim(0.0, 1.15)
    axes[0, 1].set_xlim(-0.15, 1.05)
    _band_panel(axes[1, 0], bands, "pearson", r"mean Pearson of $|\mathrm{TF}|$")
    _band_panel(axes[1, 1], bands, "anderson", "mean Anderson misfit")
    axes[1, 0].set_ylim(0.35, 1.05)
    axes[1, 1].set_ylim(0.0, 1.35)
    for ax, letter in zip(axes.ravel(), "abcd"):
        panel_letter(ax, letter)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, "fig04_sota")


if __name__ == "__main__":
    main()
