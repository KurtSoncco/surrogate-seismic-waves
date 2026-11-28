"""Slide 9: the six-block LOGLO pipeline, and one real held-out prediction.

fig09a_pipeline: static redraw of loglo_architecture.png as a horizontal flow.
fig09b_prediction: stage-2 AI vs 2D truth vs 1D at the center station of one
horizontal-bedrock test record whose AI Pearson is at the test-set median.

    python fig09_loglo.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge

import figstyle as fs

CACHE = fs.REPO / "data" / "vardomain" / "cache"

BLOCKS = (
    ("1", "The site as\na picture", "measured\nphysics in"),
    ("2", "Read each\ncell", "every cell\njudged alone"),
    ("3", "Collapse\ndepth", "soil shakes as\none column"),
    ("4", "Site dials\n($f_0$, contrast)", "a site = a few\nglobal dials"),
    ("5", "Global FNO $\\Vert$\nlocal conv", "echoes global,\ndetail local"),
    ("6", "64 POD numbers\n$\\rightarrow g(f)$", "corrections\nsmooth in $f$"),
)


def _icon_section(ax, x0, y0, w, h, field):
    ax.imshow(field, cmap=fs.CMAP, norm=LogNorm(80, 1100), aspect="auto",
              extent=(x0, x0 + w, y0, y0 + h), zorder=3)


def _icon_cells(ax, x0, y0, w, h, rng):
    nx, nz = 14, 6
    vals = rng.random((nz, nx))
    ax.imshow(vals, cmap=fs.CMAP, aspect="auto", extent=(x0, x0 + w, y0, y0 + h), zorder=3)
    ax.add_patch(Rectangle((x0 + 6 * w / nx, y0), w / nx, h, fill=False, ec=fs.ONED, lw=2.2, zorder=4))


def _icon_collapse(ax, x0, y0, w, h, rng):
    nx = 14
    ax.imshow(rng.random((5, nx)), cmap=fs.CMAP, aspect="auto",
              extent=(x0, x0 + w, y0 + 0.45 * h, y0 + h), zorder=3)
    ax.imshow(rng.random((1, nx)), cmap=fs.CMAP, aspect="auto",
              extent=(x0, x0 + w, y0, y0 + 0.16 * h), zorder=3)
    ax.add_patch(FancyArrowPatch((x0 + w / 2, y0 + 0.42 * h), (x0 + w / 2, y0 + 0.19 * h),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.8, color=fs.AI, zorder=4))


def _icon_dials(ax, x0, y0, w, h):
    r = 0.3 * h
    for cx, frac in ((x0 + 0.25 * w, 0.35), (x0 + 0.75 * w, 0.7)):
        cy = y0 + 0.45 * h
        ax.add_patch(Circle((cx, cy), r, fc="white", ec=fs.TRUTH, lw=1.4, zorder=3))
        ang = np.deg2rad(210 - 240 * frac)
        ax.plot([cx, cx + 0.8 * r * np.cos(ang)], [cy, cy + 0.8 * r * np.sin(ang)],
                color=fs.ONED, lw=2.4, zorder=4, solid_capstyle="round")


def _icon_rounds(ax, x0, y0, w, h):
    bw, bh = 0.40 * w, 0.42 * h
    for bx, label in ((x0, "Global\n(FNO)"), (x0 + 0.60 * w, "Local\n(conv)")):
        ax.add_patch(FancyBboxPatch((bx, y0 + 0.25 * h), bw, bh, boxstyle="round,pad=0.02",
                                    fc="white", ec=fs.AI, lw=1.6, zorder=3))
        ax.text(bx + bw / 2, y0 + 0.25 * h + bh / 2, label, ha="center", va="center", fontsize=10.5,
                zorder=4, linespacing=1.0)
    ax.text(x0 + w / 2, y0 + 0.86 * h, "×5 rounds", ha="center", va="center", fontsize=12,
            color=fs.AI, fontweight="bold", zorder=4)


def _icon_pod(ax, x0, y0, w, h, rng):
    n = 20
    heights = np.abs(rng.normal(0, 1, n)) * np.exp(-np.arange(n) / 6)
    heights = heights / heights.max() * 0.45 * h
    bx = x0 + np.linspace(0, w, n, endpoint=False)
    for xb, hb in zip(bx, heights):
        ax.add_patch(Rectangle((xb, y0 + 0.5 * h), 0.6 * w / n, hb, fc=fs.AI_LIGHT, ec="none", zorder=3))
    xs = np.linspace(0, 1, 100)
    ax.plot(x0 + xs * w, y0 + 0.25 * h + 0.12 * h * np.sin(5 * xs) + 0.1 * h * xs, color=fs.AI, lw=2.2, zorder=4)


def pipeline() -> None:
    dip = CACHE / "ood_dipping"
    k = int(np.flatnonzero(np.load(dip / "run_index.npy") == fs.DIPPING_RUN)[0])
    assert str(np.load(dip / "split.npy")[k]) == "train"
    field = np.asarray(np.load(dip / "vs.npy", mmap_mode="r")[k, : int(np.load(dip / "n_active.npy")[k])])
    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(fs.FULL_W, 3.92))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fs.FULL_W)
    ax.set_ylim(-0.2, 3.72)
    ax.axis("off")
    n = len(BLOCKS)
    gap = 0.34
    bw = (fs.FULL_W - 0.3 - gap * (n - 1)) / n
    y0, bh = 0.62, 2.78
    for k, (num, title, prior) in enumerate(BLOCKS):
        x0 = 0.24 + k * (bw + gap)
        ax.add_patch(FancyBboxPatch((x0, y0), bw, bh, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="#E8F2EC", ec=fs.AI, lw=1.8))
        ax.add_patch(Circle((x0 + 0.02, y0 + bh - 0.02), 0.2, fc=fs.AI, ec="white", lw=1.5, zorder=5))
        ax.text(x0 + 0.02, y0 + bh - 0.03, num, ha="center", va="center", fontsize=15, fontweight="bold",
                color="white", zorder=6)
        ax.text(x0 + bw / 2, y0 + bh - 0.2, title, ha="center", va="top", fontsize=14.5,
                fontweight="semibold", color=fs.TRUTH, linespacing=1.15)
        ix, iy, iw, ih = x0 + 0.15, y0 + 0.22, bw - 0.30, 1.15
        if k == 0:
            _icon_section(ax, ix, iy, iw, ih, field)
        elif k == 1:
            _icon_cells(ax, ix, iy, iw, ih, rng)
        elif k == 2:
            _icon_collapse(ax, ix, iy, iw, ih, rng)
        elif k == 3:
            _icon_dials(ax, ix, iy, iw, ih)
        elif k == 4:
            _icon_rounds(ax, ix, iy, iw, ih)
        else:
            _icon_pod(ax, ix, iy, iw, ih, rng)
        ax.text(x0 + bw / 2, y0 - 0.1, prior, ha="center", va="top", fontsize=13.5, style="italic",
                color=fs.AI, linespacing=1.1)
        if k < n - 1:
            ax.add_patch(FancyArrowPatch((x0 + bw + 0.04, y0 + bh / 2), (x0 + bw + gap - 0.04, y0 + bh / 2),
                                         arrowstyle="-|>", mutation_scale=18, lw=2.2, color=fs.AI))
    fs.save(fig, "fig09a_pipeline", tight=True)


def prediction() -> None:
    freq = np.load(CACHE / "freq.npy").astype(np.float64)
    split = np.load(CACHE / "iid" / "split.npy")
    runs = np.flatnonzero(split == "test")
    truth = np.exp(np.load(CACHE / "iid" / "log_tf.npy", mmap_mode="r")[runs, 10].astype(np.float64))
    oned = np.exp(np.load(CACHE / "iid" / "log_tf1d.npy", mmap_mode="r")[runs, 10].astype(np.float64))
    ai = np.load(fs.RESULTS / "discarded" / "test_model_iid_stage2_epoch53_float32.npy").astype(np.float64)
    p_ai = np.array([np.corrcoef(ai[i], truth[i])[0, 1] for i in range(runs.size)])
    p_1d = np.array([np.corrcoef(oned[i], truth[i])[0, 1] for i in range(runs.size)])
    hi = freq >= 0.5
    err_1d = np.median(np.abs(np.log(oned[:, hi] / truth[:, hi])), axis=1)
    lo, up = np.percentile(p_ai, [40, 60])
    typical = np.flatnonzero((p_ai >= lo) & (p_ai <= up))
    i = int(typical[np.argmax(err_1d[typical])])
    ratio = oned[i, hi] / truth[i, hi]
    print(f"record run {runs[i]}: AI Pearson {p_ai[i]:.3f} (test median {np.median(p_ai):.3f}), "
          f"1D Pearson {p_1d[i]:.3f}; above 0.5 Hz 1D/truth from {ratio.min():.2f} to {ratio.max():.2f}, "
          f"median |factor| {np.exp(err_1d[i]):.2f}")

    fs.apply()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.loglog(freq, truth[i], color=fs.TRUTH, lw=3.2, label="2D simulation (OpenSees)", zorder=3)
    ax.loglog(freq, ai[i], color=fs.AI, lw=3.0, label="This work (1D × 10$^{g}$)", zorder=4)
    ax.loglog(freq, oned[i], color=fs.ONED, lw=2.6, ls=(0, (5, 3)), label="1D column", zorder=2)
    fs.log_freq_axis(ax)
    ax.set_ylabel("$|TF|$")
    ax.set_yticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0.15, 20)
    ax.legend(loc="upper right")
    ax.text(0.02, 0.03, f"Held-out site, center station\nPearson: AI {p_ai[i]:.2f}, 1D {p_1d[i]:.2f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=14, color=fs.MUTED)
    fs.save(fig, "fig09b_prediction", tight=True)


if __name__ == "__main__":
    fs.apply()
    pipeline()
    prediction()
