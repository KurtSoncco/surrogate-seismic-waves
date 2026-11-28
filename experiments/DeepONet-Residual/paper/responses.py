"""Shared transfer-function panels: four curves, log frequency, sparse markers."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from style import DOMAIN_COLOR, GINO, HASKELL, OPENSEES, PRETELL, METHOD_STYLE, plot_curve

_TF_ORDER = (
    (OPENSEES, "opensees", 0.0),
    (HASKELL, "haskell", 0.35),
    (PRETELL, "pretell", 0.65),
    (GINO, "gino", 0.15),
)


def plot_vs(ax: Axes, profiles: dict[str, tuple[np.ndarray, np.ndarray]], *, domain: str) -> None:
    if "site" in profiles:
        vs, z = profiles["site"]
        ax.plot(vs, z, color="#000000", ls="-", lw=1.2, zorder=3, label="Site")
        vs_n, z_n = profiles["nominal"]
        ax.plot(
            vs_n,
            z_n,
            color=METHOD_STYLE[HASKELL]["color"],
            ls=METHOD_STYLE[HASKELL]["ls"],
            lw=1.05,
            zorder=2,
            label="1-layer nom.",
        )
        ax.legend(loc="upper right", fontsize=6, handlelength=1.6, borderpad=0.2)
    else:
        vs, z = profiles["nominal"]
        ax.plot(vs, z, color=DOMAIN_COLOR[domain], ls="-", lw=1.25, zorder=3)
    ax.invert_yaxis()
    ax.set_xlabel(r"$V_s$ (m s$^{-1}$)")
    ax.set_ylabel("depth (m)")


def plot_tf(ax: Axes, curves: dict[str, np.ndarray]) -> None:
    freq = curves["freq"]
    drawn = []
    for name, key, shift in _TF_ORDER:
        y = np.maximum(np.asarray(curves[key], dtype=float), 1e-3)
        plot_curve(
            ax,
            freq,
            y,
            METHOD_STYLE[name],
            markers=True,
            n_marks=6,
            mark_shift=shift,
            label=name,
        )
        drawn.append(y)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(freq.min()), float(freq.max()))
    stack = np.concatenate(drawn)
    hi = float(np.nanmax(stack)) * 1.35
    ax.set_ylim(0.45, hi)
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$|\mathrm{TF}|$")


def plot_leftover_curves(ax: Axes, curves: dict[str, np.ndarray]) -> None:
    freq = curves["freq"]
    ax.axhline(0.0, color="0.65", lw=0.4, zorder=0)
    plot_curve(
        ax,
        freq,
        curves["R"],
        METHOD_STYLE[OPENSEES],
        markers=False,
        label=r"$R$",
    )
    plot_curve(
        ax,
        freq,
        curves["Rhat"],
        METHOD_STYLE[GINO],
        markers=True,
        n_marks=6,
        label=r"$\hat{R}$",
    )
    ax.set_xscale("log")
    ax.set_xlim(float(freq.min()), float(freq.max()))
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$R$")


def style_legend(fig, handles_labels_ax: Axes) -> None:
    handles, labels = handles_labels_ax.get_legend_handles_labels()
    # Drop the Vs-panel legend entries if they leaked; TF panel is the source.
    keep = []
    seen = set()
    for h, lab in zip(handles, labels):
        if lab in seen or lab in {"Site", "1-layer nom."}:
            continue
        seen.add(lab)
        keep.append((h, lab))
    fig.legend(
        [h for h, _ in keep],
        [lab for _, lab in keep],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
