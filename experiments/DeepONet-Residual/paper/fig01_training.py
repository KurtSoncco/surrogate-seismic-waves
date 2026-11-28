"""Fig. 1 — fine-tune learning curves stored in the shipped checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_history  # noqa: E402
from style import (  # noqa: E402
    GINO,
    HASKELL,
    METHOD_STYLE,
    figsize,
    panel_letter,
    save_figure,
    use_style,
)


def main() -> None:
    use_style()
    history, best = load_history()
    epoch = np.array([row["epoch"] for row in history], dtype=float)
    train = np.array([row["train_smooth_l1"] for row in history])
    val_loss = np.array([row["val_smooth_l1"] for row in history])
    val_l2 = np.array([row["val_rel_l2_TF"] for row in history])

    fig, axes = plt.subplots(1, 2, figsize=figsize("double", height_mm=62))

    ax = axes[0]
    ax.plot(epoch, train, color=METHOD_STYLE[GINO]["color"], lw=1.45, label="Train")
    ax.plot(
        epoch,
        val_loss,
        color=METHOD_STYLE[HASKELL]["color"],
        ls="-.",
        lw=1.25,
        label="Validation",
    )
    ax.axvline(best, color="0.2", lw=0.5, ls=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Smooth L1 on $R$")
    ax.set_xlim(1, float(epoch.max()))
    ax.legend(loc="center", bbox_to_anchor=(0.58, 0.62), frameon=False)
    panel_letter(ax, "a")

    ax = axes[1]
    ax.plot(
        epoch,
        val_l2,
        color=METHOD_STYLE[GINO]["color"],
        lw=1.45,
        label="Validation",
    )
    shipped = float(val_l2[np.where(epoch == best)[0][0]])
    ax.axvline(best, color="0.15", lw=0.6, ls=":")
    ax.scatter([best], [shipped], s=18, color="#000000", zorder=5)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"relative $L_2$ of $|\mathrm{TF}|$")
    ax.set_xlim(1, float(epoch.max()))
    ax.set_ylim(0.415, 0.47)
    ax.legend(loc="upper left", frameon=False)
    ax.annotate(
        "shipped epoch",
        (best, shipped),
        textcoords="offset points",
        xytext=(10, 8),
        fontsize=6,
        color="#111111",
        va="bottom",
    )
    panel_letter(ax, "b")

    fig.tight_layout()
    save_figure(fig, "fig01_training")


if __name__ == "__main__":
    main()
