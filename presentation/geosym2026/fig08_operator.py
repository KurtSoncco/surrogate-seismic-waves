"""Slide 8: the operator maps the Vs section to the |TF(x, f)| map.

Uses horizontal-bedrock training record fs.HORIZONTAL_RUN from the vardomain
cache: the 500 m interior the model sees, and the OpenSees surface/outcrop
|TF| at its 21 stations (x = 100-400 m).

    python fig08_operator.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch

import figstyle as fs

CACHE = fs.REPO / "data" / "vardomain" / "cache"
VS_NORM = LogNorm(vmin=80, vmax=1100)


def main() -> None:
    fs.apply()
    run = fs.HORIZONTAL_RUN
    iid = CACHE / "iid"
    assert str(np.load(iid / "split.npy")[run]) == "train"
    depth = int(np.load(iid / "n_active.npy")[run])
    vs = np.asarray(np.load(iid / "vs.npy", mmap_mode="r")[run, :depth], dtype=np.float64)
    nz, nx = vs.shape
    x_st = np.load(iid / "rec_x.npy").astype(np.float64)
    freqs = np.load(CACHE / "freq.npy").astype(np.float64)
    tf = np.exp(np.load(iid / "log_tf.npy", mmap_mode="r")[run].astype(np.float64))
    print(f"iid run {run} (train): {nx} x {nz} m, {x_st.size} stations {x_st[0]:.0f}-{x_st[-1]:.0f} m, "
          f"|TF| max {tf.max():.1f}")

    fig = plt.figure(figsize=(fs.FULL_W, 4.6))
    ax_in = fig.add_axes([0.075, 0.44, 0.33, 0.38])
    ax_out = fig.add_axes([0.625, 0.17, 0.29, 0.65])
    cax_out = fig.add_axes([0.928, 0.17, 0.012, 0.65])
    cax_in = fig.add_axes([0.075, 0.19, 0.33, 0.035])

    im = ax_in.imshow(vs, cmap=fs.CMAP, norm=VS_NORM, aspect="auto", extent=(0, nx, nz, 0),
                      interpolation="nearest")
    ax_in.plot(x_st, np.full_like(x_st, -1.5), "v", color=fs.TRUTH, ms=6, clip_on=False, zorder=5)
    ax_in.set_ylim(nz, 0)
    ax_in.set_xticks([0, 250, 500])
    ax_in.set_yticks([0, nz])
    ax_in.set_ylabel("Depth (m)")
    ax_in.set_xlabel("Distance (m)", labelpad=1)
    ax_in.set_title("Input: the site, $V_s(x, z)$", loc="left", color=fs.NAVY, pad=14)
    cb = fig.colorbar(im, cax=cax_in, orientation="horizontal", ticks=[100, 200, 400, 800])
    cb.ax.set_xticklabels(["100", "200", "400", "800"])
    cb.ax.xaxis.set_minor_locator(plt.NullLocator())
    cb.set_label("$V_s$ (m/s)", fontsize=15, labelpad=2)
    cb.ax.tick_params(labelsize=14)

    vmax = float(np.percentile(tf, 99.8))
    vmin = vmax / 30.0
    mesh = ax_out.pcolormesh(x_st, freqs, tf.T, cmap=fs.CMAP, norm=LogNorm(vmin=vmin, vmax=vmax),
                             shading="gouraud", rasterized=True)
    ax_out.set_yscale("log")
    ax_out.set_ylim(freqs[0], freqs[-1])
    ax_out.set_yticks([0.1, 0.3, 1, 3, 10], ["0.1", "0.3", "1", "3", "10"])
    ax_out.yaxis.set_minor_locator(plt.NullLocator())
    ax_out.set_xlim(x_st[0], x_st[-1])
    ax_out.set_xticks([x_st[0], np.mean(x_st[[0, -1]]), x_st[-1]])
    ax_out.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:.0f}"))
    ax_out.set_xlabel("Station position (m)", labelpad=1)
    ax_out.set_ylabel("Frequency (Hz)")
    ax_out.set_title("Output: the response, $|TF(x, f)|$", loc="left", color=fs.NAVY, pad=10)
    ticks = [t for t in (0.3, 1, 3, 10) if vmin <= t <= vmax]
    cb2 = fig.colorbar(mesh, cax=cax_out, ticks=ticks)
    cb2.ax.set_yticklabels([f"{t:g}" for t in ticks])
    cb2.ax.yaxis.set_minor_locator(plt.NullLocator())
    cb2.set_label("$|TF|$", fontsize=15)
    cb2.ax.tick_params(labelsize=14)

    arrow = FancyArrowPatch((0.43, 0.63), (0.555, 0.63), transform=fig.transFigure, arrowstyle="-|>",
                            mutation_scale=34, lw=4, color=fs.AI)
    fig.patches.append(arrow)
    fig.text(0.49, 0.70, "operator", ha="center", va="bottom", fontsize=18, fontweight="bold", color=fs.AI)
    fig.text(0.49, 0.56, "field $\\rightarrow$ field\nresolution-free", ha="center", va="top", fontsize=15,
             color=fs.TRUTH, linespacing=1.4)
    fs.save(fig, "fig08_operator", tight=False)


if __name__ == "__main__":
    main()
