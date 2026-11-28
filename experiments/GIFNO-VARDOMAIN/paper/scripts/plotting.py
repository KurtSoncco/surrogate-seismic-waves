"""Save one central figure at the publication canvas size."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wave_surrogate.plot_utils.publication import apply_publication_style, panel_letter, savefig

__all__ = ["apply_publication_style", "panel_letter", "save_figure"]


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    apply_publication_style()
    return savefig(fig, path, png=True, kind="line")
