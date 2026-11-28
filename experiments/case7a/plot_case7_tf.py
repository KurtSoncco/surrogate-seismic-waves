#!/usr/bin/env python3
"""Transfer functions on the real Case 7 section.

SVLAB ground truth, the 1D prior, the residual head as scored before, and the
same head with the training soil/rock zeta and Rayleigh prior.

    python plot_case7_tf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from case7a import (
    REPO,
    ROCK_VS,
    STRIP_HALF_M,
    ZONES,
    element_center_x,
    ensure_cache,
    paint_section_1m,
)
from diagnose_svlab import HEADS, _score_head, _training_tf1d, soil_rock_zeta
from labels import TRUSTED_BAND_HZ, band_mask
from score_svlab import _pearson, _ylim_in_band
OUT = REPO / "checkpoints" / "case7_generalization" / "svlab_diagnosis" / "plots"
STATIONS_M = (-1500.0, 0.0, 1175.0, 1605.0)
STATION_ZONE = ("flat west", "geologic", "ramp", "flat east")
LABEL_GT = "SVLAB ground truth"
LABEL_1D = "1D theoretical"
LABEL_RES = "LOGLO residual"
LABEL_ALN = "LOGLO residual, training zeta"


def _nearest(x: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(x - value)))


def plot_stations(freq, x, tf_gt, tf_1d, series: list, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True, constrained_layout=True)
    m = band_mask(freq, TRUSTED_BAND_HZ)
    for ax, x_m, zone in zip(axes.ravel(), STATIONS_M, STATION_ZONE):
        i = _nearest(x, x_m)
        curves = [tf_gt[i], tf_1d[i]]
        ax.semilogx(freq[m], tf_gt[i, m], color="k", lw=1.5, label=LABEL_GT)
        ax.semilogx(freq[m], tf_1d[i, m], color="C3", lw=1.25, label=LABEL_1D)
        for arr, color, ls, label in series:
            ax.semilogx(freq[m], arr[i, m], color=color, lw=1.15, ls=ls, label=label)
            curves.append(arr[i])
        ax.set_xlim(*TRUSTED_BAND_HZ)
        ax.set_ylim(*_ylim_in_band(freq, curves))
        ax.set_title(f"x = {x[i]:.0f} m, {zone}")
        ax.set_ylabel("|TF| (surface / input)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].set_xlabel("frequency (Hz)")
    axes[1, 1].set_xlabel("frequency (Hz)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pearson(freq, x, tf_gt, tf_1d, series: list, out_path: Path) -> None:
    m = band_mask(freq, TRUSTED_BAND_HZ)
    fig, ax = plt.subplots(figsize=(9.0, 3.8), constrained_layout=True)
    ax.plot(x, [_pearson(tf_1d[i, m], tf_gt[i, m]) for i in range(len(x))], color="C3", lw=1.4, label=LABEL_1D)
    for arr, color, ls, label in series:
        ax.plot(
            x,
            [_pearson(arr[i, m], tf_gt[i, m]) for i in range(len(x))],
            color=color,
            lw=1.2,
            ls=ls,
            label=label,
        )
    for name, (lo, hi) in ZONES.items():
        ax.axvspan(max(lo, float(x[0])), min(hi, float(x[-1])), alpha=0.06 if name != "geologic" else 0.14, color="0.3")
    ax.set_ylim(-0.2, 1.02)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Case 7 vs SVLAB ground truth, 0.1–2.5 Hz")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vs(out_path: Path) -> None:
    """Case 7 Vs, the score-bin names, and the 500 m crop each plotted station feeds LOGLO."""
    pack = ensure_cache()
    vs = np.asarray(pack["vs"], dtype=np.float64)
    x = element_center_x(np.arange(vs.shape[1]))
    soil = np.array(
        [(np.flatnonzero(vs[:, j] > ROCK_VS)[0] * 5.0 if np.any(vs[:, j] > ROCK_VS) else 150.0) for j in range(vs.shape[1])]
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(9.4, 6.2), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [1, 2.4]}
    )
    axh, ax = axes
    axh.plot(x, soil, color="k", lw=1.2)
    axh.set_ylabel("soil thickness (m)")
    axh.set_ylim(110, 60)
    axh.set_title("Case 7 Vs, and the 500 m crop LOGLO sees at each plotted station")
    image = ax.imshow(
        vs,
        origin="upper",
        aspect="auto",
        extent=(-2000, 2000, 150, 0),
        cmap="viridis",
        vmin=100,
        vmax=2600,
    )
    ax.plot(x, soil, color="w", lw=0.8)
    bounds = ((-650, "flat west | geologic"), (1050, "geologic | ramp"), (1300, "ramp | flat east"))
    for xv, _label in bounds:
        axh.axvline(xv, color="0.4", lw=0.6, ls=":")
        ax.axvline(xv, color="w", lw=0.6, ls=":")
    axh.text(-1320, 104, "flat west", ha="center", fontsize=8, color="0.25")
    axh.text(200, 104, "geologic", ha="center", fontsize=8, color="0.25")
    axh.text(1175, 104, "ramp", ha="center", fontsize=8, color="0.25")
    axh.text(1650, 104, "flat east", ha="center", fontsize=8, color="0.25")
    colors = ("C0", "C1", "C3", "C2")
    for (xc, zone), color in zip(zip(STATIONS_M, STATION_ZONE), colors):
        ax.plot([xc - STRIP_HALF_M, xc + STRIP_HALF_M], [8, 8], color=color, lw=2.0, solid_capstyle="butt")
        ax.plot(xc, 8, marker="|", color=color, ms=8)
        axh.annotate(
            "",
            xy=(xc + STRIP_HALF_M, 66),
            xytext=(xc - STRIP_HALF_M, 66),
            arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.0},
        )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth (m)")
    fig.colorbar(image, ax=ax, label="Vs (m/s)", pad=0.01)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    pack = ensure_cache()
    vs_1m = paint_section_1m(pack["vs"])
    cache = REPO / "checkpoints" / "case7_generalization" / "svlab_diagnosis" / "centers.npz"
    saved = np.load(cache)
    x = saved["x"]
    freq = saved["freq"]
    tf_gt = saved["tf_gt"]
    tf_1d = saved["tf1d"]
    residual_dir = dict(HEADS)["residual"]
    print("residual, uniform zeta", flush=True)
    raw = _score_head("residual", residual_dir, pack, vs_1m, x, freq, tf_gt, tf_1d)
    tf_raw = raw["_arrays"]["tf_center"]
    zeta = soil_rock_zeta(vs_1m)
    tf_train = _training_tf1d(vs_1m, zeta, x, freq)
    print("residual, training zeta", flush=True)
    aligned = _score_head(
        "residual",
        residual_dir,
        pack,
        vs_1m,
        x,
        freq,
        tf_gt,
        tf_1d,
        zeta=zeta,
        prior_mode="rayleigh",
        tf1d_mult=tf_train,
    )
    tf_aln = aligned["_arrays"]["tf_center"]
    series = [
        (tf_raw, "C0", "-", LABEL_RES),
        (tf_aln, "C2", "--", LABEL_ALN),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    plot_stations(freq, x, tf_gt, tf_1d, series, OUT / "tf_stations.png")
    plot_pearson(freq, x, tf_gt, tf_1d, series, OUT / "pearson_vs_x.png")
    np.savez(OUT / "tf_compare.npz", x=x, freq=freq, tf_gt=tf_gt, tf_1d=tf_1d, tf_residual=tf_raw, tf_aligned=tf_aln)
    print(OUT / "tf_stations.png")
    print(OUT / "pearson_vs_x.png")


if __name__ == "__main__":
    if "--vs-only" in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        plot_vs(OUT / "case7_vs.png")
        print(OUT / "case7_vs.png")
    else:
        sys.exit(main())
