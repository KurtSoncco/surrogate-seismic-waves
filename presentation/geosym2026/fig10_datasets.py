"""Slide 10: the three OpenSees campaigns as thumbnails, counts, and how the
stage-2 model used them. Stage 2 starts from stage 1 (horizontal + dipping
train splits) and then trains on the horizontal runs and the variable-width
sections; the 16 blind sections stay out of training and early stopping.

Also writes each card's Vs model on its own: fig10_vs_horizontal,
fig10_vs_dipping, fig10_vs_variable_width.

    python fig10_datasets.py
"""

from __future__ import annotations

import glob

import h5py
import hdf5plugin  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch

import collections

import figstyle as fs

CACHE = fs.REPO / "data" / "vardomain" / "cache"
H5 = {"iid": fs.REPO / "data" / "gifno_screen" / "h5",
      "ood_dipping": fs.REPO / "data" / "gifno_screen" / "ood_dipping" / "h5"}
STRIP = (500, 1000)
NORM = LogNorm(vmin=100, vmax=700)


def _mesh(family: str, run: int) -> np.ndarray:
    with h5py.File(H5[family] / f"run_{run}.h5", "r") as handle:
        return np.asarray(handle["Vs_realization_2D"][:], dtype=np.float32)


def _depth(vs: np.ndarray) -> int:
    return min(vs.shape[0], int((vs < 700).sum(axis=0).max()) + 25)


def _vs_alone(vs: np.ndarray, width: int, stem: str) -> None:
    if width == 1500:
        vs, width = vs[:, STRIP[0]:STRIP[1]], STRIP[1] - STRIP[0]
    depth = _depth(vs)
    fig = plt.figure(figsize=(6.4, 2.9))
    ax = fig.add_axes([0.11, 0.24, 0.74, 0.70])
    cax = fig.add_axes([0.875, 0.24, 0.02, 0.70])
    im = ax.imshow(vs[:depth], cmap=fs.CMAP, norm=NORM, aspect="auto", extent=(0, width, depth, 0),
                   interpolation="nearest")
    ax.set_xticks(np.linspace(0, width, 3 if width == 500 else 5))
    ax.set_yticks([0, 50, 100] if depth > 100 else [0, 50])
    ax.set_xlabel("Distance (m)", labelpad=1)
    ax.set_ylabel("Depth (m)")
    ax.tick_params(labelsize=14)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    cb = fig.colorbar(im, cax=cax, extend="max", ticks=[100, 200, 400, 700])
    cb.ax.set_yticklabels(["100", "200", "400", "≥700"])
    cb.ax.yaxis.set_minor_locator(plt.NullLocator())
    cb.ax.tick_params(labelsize=13)
    cb.set_label("$V_s$ (m/s)", fontsize=14)
    fs.save(fig, stem, tight=True)


def _chip(fig, x, y, w, h, text, fc, tc="white"):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01",
                                      transform=fig.transFigure, fc=fc, ec="none"))
    fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=13.5, color=tc, fontweight="semibold")


def main() -> None:
    fs.apply()
    params = pd.read_csv(fs.DATA / "run_params.csv")
    counts = params.groupby(["family", "split"]).size()
    sites = params.groupby(["family", "split"]).group.nunique()
    iid = params[(params.family == "iid") & (params.split == "test") & params.H.between(50, 65)
                 & params.CoV.between(0.2, 0.3)].iloc[0]
    dip_runs = params[params.family == "ood_dipping"].run.astype(int).to_numpy()
    meshes = {r: _mesh("ood_dipping", r) for r in dip_runs}
    relief = {r: int(np.ptp((meshes[r] < 700).sum(axis=0))) for r in dip_runs}
    dip_run = max(relief, key=relief.get)
    test_relief = max(relief[r] for r in params[(params.family == "ood_dipping") & (params.split == "test")].run)
    print(f"thumbnail dipping run {dip_run}: relief {relief[dip_run]} m; held-out test runs: at most {test_relief} m")
    sec, sec_path = np.load(fs.EXAMPLE_SECTION, allow_pickle=True)["vs"], fs.EXAMPLE_SECTION
    sec_split = collections.Counter(str(np.load(p, allow_pickle=True)["split"])
                                    for p in glob.glob(str(CACHE / "sections" / "*.npz")))
    print("thumbs: iid run", int(iid.run), "section", sec_path)

    cards = (
        ("7,680", "Horizontal bedrock", "256 sites × 30 seeds", _mesh("iid", int(iid.run)), 1500,
         (("Train", f"{counts['iid', 'train']:,}", fs.AI), ("Val", f"{counts['iid', 'val']:,}", fs.MUTED),
          ("Test", f"{counts['iid', 'test']:,} · {sites['iid', 'test']} sites", fs.NAVY))),
        ("960", "Dipping bedrock", "32 sites × 30 seeds, dip ±3°", meshes[dip_run], 1500,
         (("Train (stage 1)", f"{counts['ood_dipping', 'train']:,}", fs.AI),
          ("Test", f"{counts['ood_dipping', 'test']} · {sites['ood_dipping', 'test']} sites, dip < 0.4°", fs.NAVY))),
        ("301", "Variable width", "500–2,000 m wide sections", sec, 2000,
         (("Train", f"{sec_split['train']}", fs.AI), ("Val", f"{sec_split['val']}", fs.MUTED),
          ("Blind test", f"{sec_split['blind']}", fs.NAVY))),
    )

    fig = plt.figure(figsize=(fs.FULL_W, 5.0))
    cw, gap, x0 = 0.31, 0.035, 0.0
    for k, (big, name, sub, vs, width, tags) in enumerate(cards):
        cx = x0 + k * (cw + gap)
        fig.patches.append(FancyBboxPatch((cx, 0.155), cw, 0.825, boxstyle="round,pad=0,rounding_size=0.015",
                                          transform=fig.transFigure, fc=fs.PANEL, ec="none"))
        fig.text(cx + 0.02, 0.955, big, ha="left", va="top", fontsize=40, fontweight="bold", color=fs.NAVY)
        fig.text(cx + 0.02, 0.80, name, ha="left", va="top", fontsize=17, fontweight="semibold", color=fs.TRUTH)
        fig.text(cx + 0.02, 0.735, sub, ha="left", va="top", fontsize=14, color=fs.MUTED)
        depth = _depth(vs)
        ax = fig.add_axes([cx + 0.02, 0.40, cw - 0.04, 0.28], zorder=3)
        ax.imshow(vs[:depth], cmap=fs.CMAP, norm=NORM, aspect="auto", extent=(0, width, depth, 0),
                  interpolation="nearest")
        ax.set_xticks([0, width], ["0", f"{width:,} m"])
        if width == 1500:
            for xs in STRIP:
                ax.axvline(xs, color="white", lw=1.6, ls=(0, (4, 3)))
            ax.text(sum(STRIP) / 2, depth * 0.04, "500 m stations", ha="center", va="top", fontsize=12.5,
                    color="white", fontweight="semibold")
        ax.set_yticks([])
        ax.tick_params(labelsize=13)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.8)
        tx, ty = cx + 0.02, 0.265
        for label, value, color in tags:
            text = f"{label} {value}".strip()
            w = 0.0085 * len(text) + 0.02
            if tx + w > cx + cw - 0.01:
                tx, ty = cx + 0.02, ty - 0.08
            _chip(fig, tx, ty, w, 0.065, text, color)
            tx += w + 0.008
    fig.text(0.0, 0.07, "Linear-elastic, soft sites, median $f_0$ ≈ 0.85 Hz (horizontal set)", ha="left",
             va="center", fontsize=15, color=fs.MUTED)
    print(counts.to_string())
    print("sections:", dict(sec_split))
    fs.save(fig, "fig10_datasets", tight=True)
    for (_big, _name, _sub, vs, width, _tags), stem in zip(
            cards, ("fig10_vs_horizontal", "fig10_vs_dipping", "fig10_vs_variable_width")):
        _vs_alone(vs, width, stem)


if __name__ == "__main__":
    main()
