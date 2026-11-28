"""Slide 11: AI vs the state of practice on the held-out test records.

fig11_comparison: median Pearson of |TF| per method, horizontal and dipping
test sets. Methods are identified by a shared color legend.
fig11b_dipping_site: the 30 AI curves of one held-out dipping site, with the
Pretell median from GIFNO-VARDOMAIN/figures/test_arms_ood_dipping.npz.

Source: GIFNO-VARDOMAIN/figures/<fs.STAGE>/test_vs_classical.csv.

    python fig11_comparison.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import figstyle as fs

CACHE = fs.REPO / "data" / "vardomain" / "cache"
ARMS = fs.REPO / "experiments" / "GIFNO-VARDOMAIN" / "figures" / "test_arms_ood_dipping.npz"
METHODS = (("Model", "This work", fs.AI), ("Pretell", "Pretell et al. (2022)", fs.PRETELL), ("Toro", "Toro (1995)", fs.TORO),
           ("Dmult", "Tao & Rathje (2019)", fs.DMULT))
DOMAINS = (("Horizontal", "Horizontal bedrock", "720 records"),
           ("Dipping", "Dipping bedrock", "90 records"))
X_MIN = 0.5


def _bars(ax, rows: pd.DataFrame, title: str, sub: str) -> None:
    rows = rows.set_index("method")
    order = sorted(METHODS, key=lambda m: -rows.loc[m[0], "pearson_median"])
    y = np.arange(len(order))[::-1]
    for yi, (key, label, color) in zip(y, order):
        v = float(rows.loc[key, "pearson_median"])
        ax.barh(yi, v - X_MIN, left=X_MIN, height=0.68, color=color)
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", ha="left", fontsize=17, fontweight="bold",
                color=fs.AI if key == "Model" else fs.TRUTH)
    ax.set_yticks(y, [])
    ax.set_xlim(X_MIN, 1.12)
    ax.set_xticks([0.5, 0.75, 1.0], ["0.5", "0.75", "1"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontsize=18, color=fs.NAVY, pad=30)
    ax.text(0.0, 1.03, sub, transform=ax.transAxes, fontsize=14, color=fs.MUTED, va="bottom")


def comparison() -> None:
    table = pd.read_csv(fs.RESULTS / "test_vs_classical.csv")
    table = table[(table.level == "record") & table.method.isin([m[0] for m in METHODS])]
    fig = plt.figure(figsize=(fs.FULL_W, 4.6))
    axes = [fig.add_axes([0.045, 0.15, 0.44, 0.58]), fig.add_axes([0.535, 0.15, 0.44, 0.58])]
    for ax, (dom, title, sub) in zip(axes, DOMAINS):
        _bars(ax, table[table.domain == dom], title, sub)
    handles = [Patch(facecolor=color, edgecolor="none") for _key, _label, color in METHODS]
    fig.legend(handles, [label for _key, label, _color in METHODS], loc="lower center",
               bbox_to_anchor=(0.5, 0.90), ncol=4, frameon=False, fontsize=15,
               handlelength=1.15, handleheight=0.7, columnspacing=1.5, borderaxespad=0)
    fig.text(0.5, 0.02, "Median Pearson of $|TF|$ vs 2D simulation  (higher is better; axis starts at 0.5)",
             ha="center", fontsize=15, color=fs.MUTED)
    fs.save(fig, "fig11_comparison", tight=True)
    print(table[["domain", "method", "pearson_median", "gof_median", "beat_pearson", "beat_anderson"]].round(3)
          .to_string(index=False))


def dipping_site() -> None:
    freq = np.load(CACHE / "freq.npy").astype(np.float64)
    split = np.load(CACHE / "ood_dipping" / "split.npy")
    group = np.load(CACHE / "ood_dipping" / "group.npy")
    runs = np.flatnonzero(split == "test")
    ai = np.load(fs.RESULTS / "discarded" / "test_model_ood_dipping_stage2_epoch53_float32.npy").astype(np.float64)
    truth = np.exp(np.load(CACHE / "ood_dipping" / "log_tf.npy", mmap_mode="r")[runs, 10].astype(np.float64))
    arms = np.load(ARMS)
    row = {int(r): k for k, r in enumerate(arms["run"])}
    pretell = arms["pretell"][[row[int(r)] for r in runs]].astype(np.float64)
    p_ai = np.array([np.corrcoef(ai[i], truth[i])[0, 1] for i in range(runs.size)])
    g = group[runs]
    med = {k: np.median(p_ai[g == k]) for k in np.unique(g)}
    site = sorted(med, key=med.get)[len(med) // 2]
    m = g == site
    params = pd.read_csv(fs.DATA / "run_params.csv")
    print(f"dipping test site group {site}: {m.sum()} records, median AI Pearson {med[site]:.3f}")

    fs.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for curve in ai[m]:
        ax.loglog(freq, curve, color=fs.AI_LIGHT, lw=1.0, alpha=0.7, zorder=1)
    gm = lambda a: np.exp(np.median(np.log(a), axis=0))  # noqa: E731
    ax.loglog(freq, gm(truth[m]), color=fs.TRUTH, lw=3.0, label="2D simulation, median", zorder=3)
    ax.loglog(freq, gm(ai[m]), color=fs.AI, lw=3.0, label="This work, median", zorder=4)
    ax.loglog(freq, gm(pretell[m]), color=fs.PRETELL, lw=2.6, ls=(0, (5, 3)), label="Pretell et al. (2022), median", zorder=2)
    ax.plot([], [], color=fs.AI_LIGHT, lw=2, label=f"This work, each of {m.sum()} seeds")
    fs.log_freq_axis(ax)
    ax.set_yticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0.15, 45)
    ax.set_ylabel("$|TF|$")
    ax.legend(loc="upper right", fontsize=14, ncol=2)
    fs.save(fig, "fig11b_dipping_site", tight=True)


if __name__ == "__main__":
    fs.apply()
    comparison()
    dipping_site()
