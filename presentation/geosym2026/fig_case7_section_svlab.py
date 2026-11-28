"""Case 7 against SVLAB: same stations as fig_case7_section.py.

The AI is the stage-2 model on a 500 m Q–Vs window. SVLAB is the recorded
motion, so the damping is not the damping in the window. SVLAB |TF| above
about 2.5 Hz runs to tens or hundreds at every station, so the curves and the
Pearson both stop at 2.5 Hz. Stations are the 15 m SVLAB nodes nearest
−1660 m, the one-third and two-thirds points, and x = 0.

    python fig_case7_section_svlab.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
import fig_case7_section as sec

NPZ = fs.DATA / "svlab_stage2_tf.npz"
# Nearest 15 m nodes to −1660, −667, 0, 667.
SHOW = (
    (-1665.0, "−1665 m", ""),
    (-660.0, "−660 m", "one-third"),
    (0.0, "0", ""),
    (660.0, "660 m", "two-thirds"),
)
SVLAB = fs.NAVY


def _band(freq: np.ndarray) -> np.ndarray:
    return (freq >= sec.BAND[0]) & (freq <= sec.BAND[1])


def _pearson_along(freq: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    band = _band(freq)
    return np.array([
        sec._pearson(np.log(np.maximum(pred[i, band], 1e-6)), np.log(np.maximum(truth[i, band], 1e-6)))
        for i in range(pred.shape[0])
    ])


def _freq_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlim(0.1, 2.5)
    ax.set_xticks([0.1, 0.5, 1, 2], ["0.1", "0.5", "1", "2"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Frequency (Hz)")


def main() -> None:
    d = np.load(NPZ)
    x, freq, truth, pred = d["x"], d["freq"], d["tf_gt"], d["pred"]
    pear = _pearson_along(freq, pred, truth)
    curves = {}
    for xq, _title, _note in SHOW:
        i = int(np.argmin(np.abs(x - xq)))
        if abs(float(x[i]) - xq) > 0.5:
            raise RuntimeError(f"x = {xq:.0f} m is not an SVLAB station")
        band = _band(freq)
        curves[xq] = (truth[i, band], pred[i, band], float(pear[i]), freq[band])
        print(f"x = {xq:.0f} m  Pearson {pear[i]:.3f}")
    for name, a, b, _color in sec.ZONES:
        m = (x >= a) & (x < b)
        print(f"{name:12} n={m.sum():3d}  median {np.median(pear[m]):.3f}  min {pear[m].min():.3f}")

    fs.apply()
    fig = plt.figure(figsize=(fs.FULL_W, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1.0], left=0.08, right=0.985, bottom=0.08,
                          top=0.93, hspace=0.55, wspace=0.14)
    ax = fig.add_subplot(gs[0, :])
    sec._shade(ax, x)
    ax.plot(x, pear, color=fs.AI, lw=2.8, zorder=3, label="This work")
    mark_x = [xq for xq, _t, _n in SHOW]
    ax.scatter(mark_x, [curves[xq][2] for xq in mark_x], s=46, color=fs.AI, edgecolor="white",
               linewidths=0.8, zorder=4)
    west = SHOW[0][0]
    ax.annotate("−1665 m", xy=(west, curves[west][2]), xytext=(-1610, 0.46),
                fontsize=15, color=fs.AI, fontweight="semibold", ha="center", va="bottom",
                arrowprops={"arrowstyle": "-|>", "color": fs.AI, "lw": 1.5}, zorder=5)
    sec._zone_labels(ax, x)
    ax.set_xlim(float(x.min()) - 30, float(x.max()) + 30)
    ax.set_ylim(-0.08, 1.05)
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Pearson of log$|TF|$\n0.1–2.5 Hz")
    ax.set_title("(A)", loc="left", fontweight="semibold", color=fs.NAVY, pad=16)
    ax.legend(loc="center right", fontsize=15)

    shown = [np.concatenate([t, p]) for t, p, _r, _f in curves.values()]
    ymax = max(float(np.max(v)) for v in shown)
    ymin = min(float(np.min(v)) for v in shown)
    b_axes = []
    for i, (xq, title, note) in enumerate(SHOW):
        bx = fig.add_subplot(gs[1, i], sharey=b_axes[0] if b_axes else None)
        b_axes.append(bx)
        truth_i, pred_i, p_i, freq_i = curves[xq]
        bx.loglog(freq_i, np.maximum(pred_i, 1e-3), color=fs.AI, lw=2.6, label="This work", zorder=3)
        bx.loglog(freq_i, np.maximum(truth_i, 1e-3), color=SVLAB, lw=1.8, label="SVLAB", zorder=4)
        _freq_axis(bx)
        head = "(B)  " + title if i == 0 else title
        bx.set_title(f"{head}\n{note or ' '}", fontsize=14, color=fs.TRUTH, pad=4)
        bx.text(0.50, 0.04, f"Pearson {p_i:.2f}", transform=bx.transAxes, ha="center", va="bottom",
                fontsize=12, color=fs.MUTED)
        if i == 0:
            bx.set_ylabel("$|TF|$")
        else:
            bx.tick_params(labelleft=False)
    b_axes[0].set_ylim(max(ymin / 1.5, 0.08), ymax * 1.6)
    b_axes[0].set_yticks([0.3, 1, 10], ["0.3", "1", "10"])
    b_axes[0].yaxis.set_minor_locator(plt.NullLocator())
    b_axes[2].legend(loc="lower left", bbox_to_anchor=(0.0, 1.18), ncol=2, fontsize=12,
                     frameon=False, borderaxespad=0)
    fs.save(fig, "fig_case7_section_svlab")


if __name__ == "__main__":
    main()
