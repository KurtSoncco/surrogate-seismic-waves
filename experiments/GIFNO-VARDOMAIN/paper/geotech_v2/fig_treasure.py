#!/usr/bin/env python3
"""Treasure Island from stored fields. No new forward.

g_pred is 07_one_model/treasure_g.npz, the array named in
ladder_treasure.provenance.json. g_true is fields.npz. The 0.4 Hz boundary
is marked, and panel (c) is g_pred - g_true. Panel (d) reads pooled g log-L1
from ladder_treasure.csv and baselines.csv. The low band is not an accuracy
claim. Per-station scores are not recomputed, and full_svlab.h5 is not read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parents[1] / "analysis" / "frozen"
OUT = HERE / "05_treasure_island_full"
FIELDS = OUT / "fields.npz"
G_PATH = HERE / "07_one_model" / "treasure_g.npz"
FULL = HERE.parents[3] / "data" / "case7a" / "opensees_crops" / "context" / "h5" / "full_qvs.h5"
X_LEFT = -2000.0
sys.path.insert(0, str(HERE.parent / "scripts"))
from plotting import apply_publication_style, panel_letter, save_figure  # noqa: E402


def _one(frame: pd.DataFrame, **query) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for key, value in query.items():
        mask &= frame[key] == value
    hit = frame[mask]
    if len(hit) != 1:
        raise SystemExit(f"expected one row for {query}, found {len(hit)}")
    return hit.iloc[0]


def _vs_panel() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if FULL.is_file():
        import h5py
        import hdf5plugin  # noqa: F401

        with h5py.File(FULL, "r") as handle:
            vs = np.asarray(handle["Vs_realization_2D"][:], dtype=np.float64)
        x_edge = X_LEFT + np.arange(vs.shape[1] + 1)
        z_edge = np.arange(vs.shape[0] + 1)
        return vs, x_edge, z_edge
    blob = np.load(FIELDS)
    vs = np.asarray(blob["vs"])
    x_edge = np.linspace(X_LEFT, X_LEFT + 4000.0, vs.shape[1] + 1)
    z_edge = np.arange(vs.shape[0] + 1)
    return vs, x_edge, z_edge


def main() -> None:
    if not FIELDS.is_file() or not G_PATH.is_file():
        raise SystemExit("stored Treasure Island fields are missing; a forward would be required")
    blob = np.load(FIELDS)
    stored = np.load(G_PATH)
    g_true = np.asarray(blob["g_true"], dtype=np.float64)
    g_pred = np.asarray(stored["g"], dtype=np.float64)
    if g_pred.shape != g_true.shape:
        raise SystemExit(f"g shapes differ: pred {g_pred.shape}, true {g_true.shape}")
    x = np.asarray(blob["x"], dtype=np.float64)
    freq = np.asarray(blob["freq"], dtype=np.float64)
    ladder = pd.read_csv(FROZEN / "ladder_treasure.csv")
    baselines = pd.read_csv(FROZEN / "baselines.csv")
    OUT.mkdir(parents=True, exist_ok=True)

    apply_publication_style()
    fig = plt.figure(figsize=(7.48, 8.2))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.05, 1.15, 1.15, 1.15], width_ratios=[1.0, 0.045], hspace=0.55, wspace=0.08)
    vs, x_edge, z_edge = _vs_panel()
    ax = fig.add_subplot(grid[0, 0])
    image = ax.pcolormesh(x_edge, z_edge, vs, cmap="cividis", shading="flat")
    fig.colorbar(image, cax=fig.add_subplot(grid[0, 1]), label=r"$V_s$ (m/s)")
    ax.set_ylim(z_edge[-1], z_edge[0])
    ax.set_xlim(x_edge[0], x_edge[-1])
    ax.set_ylabel("Depth (m)")
    ax.set_title("Treasure Island, 4000 m section")
    panel_letter(ax, "a")

    residual = g_pred - g_true
    limit = max(float(np.quantile(np.abs(g_true), 0.98)), 0.05)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    panels = (
        (1, g_true, r"OpenSees $g$", "b"),
        (2, residual, r"$g_{\mathrm{pred}} - g_{\mathrm{true}}$", "c"),
    )
    drawn = None
    for row, field, title, letter in panels:
        ax = fig.add_subplot(grid[row, 0])
        drawn = ax.pcolormesh(x, freq, field.T, cmap="RdBu_r", norm=norm, shading="auto")
        ax.axhline(0.4, color="#222222", lw=0.7, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(0.1, 10)
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title, fontsize=8)
        ax.text(float(x.min()) + 40, 0.42, "0.4 Hz", fontsize=6, va="bottom")
        panel_letter(ax, letter)
    fig.colorbar(drawn, cax=fig.add_subplot(grid[1:3, 1]), label=r"$g$")

    ax = fig.add_subplot(grid[3, 0])
    bands = ("low", "mid", "high", "overall")
    labels = ("0.1–0.4", "0.4–2.5", "2.5–10", "0.1–10")
    positions = np.arange(len(bands))
    model = [float(_one(ladder, rung="treasure_island", band=band).g_log_l1) for band in bands]
    prior = [float(_one(baselines, rung="treasure_island", band=band, baseline="g0").g_log_l1) for band in bands]
    ax.bar(positions - 0.16, prior, width=0.32, color="#0072B2", label="g = 0")
    ax.bar(positions + 0.16, model, width=0.32, color="#D55E00", label="P128n128")
    for index in (0, 3):
        ax.patches[index].set_hatch("///")
        ax.patches[index + 4].set_hatch("///")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Pooled g log-L1")
    ax.set_xlabel("Band (Hz). Hatched bars include 0.1–0.4 Hz and are not an accuracy result.")
    fig.suptitle(
        "The sub-0.4 Hz mismatch is visible. The 15 s record does not validate that band.",
        fontsize=8,
    )
    ax.legend(frameon=False, fontsize=6)
    panel_letter(ax, "d")
    fig.subplots_adjust(top=0.94)
    save_figure(fig, OUT / "central.pdf")
    print(f"wrote {OUT / 'central.pdf'}")


if __name__ == "__main__":
    main()
