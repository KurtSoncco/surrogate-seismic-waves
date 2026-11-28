"""Slide 12: between-seed variability at one held-out horizontal site.

The 30 random-field seeds of one test site, 2D truth vs AI (figstyle.STAGE), with
median and +-1 sigma bands, and sigma_ln(TF) vs frequency for truth, AI, the
1D center columns, and Toro (1995). Toro's sigma is the spread of its own 40
randomized profiles for this site's parameters; the saved Toro curve is the
same for all 30 seeds, so it has no between-seed spread.

    python fig12_seed_spread.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

sys.path.insert(0, str(fs.REPO / "experiments" / "DeepONet-Residual"))

CACHE = fs.REPO / "data" / "vardomain" / "cache"
BANDS = ((0.1, 1.0), (1.0, 3.0), (3.0, 10.0))


def _toro_sigma(freq: np.ndarray, row: pd.Series) -> np.ndarray:
    dest = fs.DATA / f"toro_sigma_run{int(row.run)}.npy"
    if dest.is_file():
        return np.load(dest)
    from response_variability.seiskit_arms import hallal_geomean_tf

    _gm, sig = hallal_geomean_tf(freq=freq, vs1=float(row.Vs1), H=float(row.H), cov=float(row.CoV),
                                 vs2=float(row.Vs2), xi=0.05, n_seeds=40, kind="toro")
    np.save(dest, sig)
    return sig


def main() -> None:
    fs.apply()
    freq = np.load(CACHE / "freq.npy").astype(np.float64)
    split = np.load(CACHE / "iid" / "split.npy")
    group = np.load(CACHE / "iid" / "group.npy")
    runs = np.flatnonzero(split == "test")
    g = group[runs]
    ln_truth = np.load(CACHE / "iid" / "log_tf.npy", mmap_mode="r")[runs, 10].astype(np.float64)
    ln_1d = np.load(CACHE / "iid" / "log_tf1d.npy", mmap_mode="r")[runs, 10].astype(np.float64)
    ln_ai = np.log(np.load(fs.RESULTS / "discarded" / "test_model_iid_stage2_epoch53_float32.npy").astype(np.float64))

    p_ai = np.array([np.corrcoef(np.exp(ln_ai[i]), np.exp(ln_truth[i]))[0, 1] for i in range(runs.size)])
    med = {k: float(np.median(p_ai[g == k])) for k in np.unique(g)}
    site = sorted(med, key=med.get)[len(med) // 2]
    m = g == site

    ratios = {b: [] for b in BANDS}
    for k in np.unique(g):
        mk = g == k
        s_t, s_a = ln_truth[mk].std(axis=0, ddof=1), ln_ai[mk].std(axis=0, ddof=1)
        for lo, hi in BANDS:
            band = (freq >= lo) & (freq < hi)
            ratios[(lo, hi)].append(float(np.median(s_a[band] / s_t[band])))
    print(f"site group {site}: {m.sum()} seeds, median AI Pearson {med[site]:.3f}")
    for (lo, hi), r in ratios.items():
        r = np.array(r)
        print(f"  sigma_AI / sigma_truth, {lo}-{hi} Hz, over 24 sites: median {np.median(r):.2f} "
              f"(IQR {np.percentile(r, 25):.2f}-{np.percentile(r, 75):.2f})")

    params = pd.read_csv(fs.DATA / "run_params.csv")
    row = params[(params.family == "iid") & (params.run == int(runs[m][0]))].iloc[0]
    print(f"  site params Vs1 {row.Vs1:.0f}, H {row.H:.0f}, CoV {row.CoV:.2f}, rH {row.rH:.0f}, Vs2 {row.Vs2:.0f}")
    sig = {
        "truth": ln_truth[m].std(axis=0, ddof=1),
        "ai": ln_ai[m].std(axis=0, ddof=1),
        "1d": ln_1d[m].std(axis=0, ddof=1),
        "toro": _toro_sigma(freq, row),
    }
    for key, s in sig.items():
        print(f"  sigma_ln {key}: " + ", ".join(
            f"{lo}-{hi} Hz {np.median(s[(freq >= lo) & (freq < hi)]):.2f}" for lo, hi in BANDS))

    fig = plt.figure(figsize=(fs.FULL_W, 4.7))
    ax_t = fig.add_axes([0.065, 0.15, 0.27, 0.70])
    ax_a = fig.add_axes([0.375, 0.15, 0.27, 0.70], sharey=ax_t)
    ax_s = fig.add_axes([0.735, 0.15, 0.255, 0.70])
    for ax, ln, color, title in ((ax_t, ln_truth[m], fs.TRUTH, "2D simulation (OpenSees)"),
                                 (ax_a, ln_ai[m], fs.AI, "This work")):
        for curve in ln:
            ax.plot(freq, np.exp(curve), color=fs.REAL, lw=0.9, zorder=1)
        mu, sd = ln.mean(axis=0), ln.std(axis=0, ddof=1)
        ax.fill_between(freq, np.exp(mu - sd), np.exp(mu + sd), color=color, alpha=0.22, lw=0, zorder=2)
        ax.plot(freq, np.exp(mu), color=color, lw=3.0, zorder=3)
        ax.set_yscale("log")
        fs.log_freq_axis(ax)
        ax.set_title(title, loc="left", color=color if color == fs.AI else fs.NAVY)
    ax_t.set_yticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"])
    ax_t.yaxis.set_minor_locator(plt.NullLocator())
    ax_t.set_ylim(0.2, 15)
    ax_t.set_ylabel("$|TF|$")
    plt.setp(ax_a.get_yticklabels(), visible=False)
    ax_a.plot([], [], color=fs.REAL, lw=2, label=f"{m.sum()} seeds")
    ax_a.plot([], [], color=fs.TRUTH, lw=3, label="median ± 1σ")
    ax_a.legend(loc="upper right", fontsize=14, handlelength=1.4)

    ax_s.plot(freq, sig["truth"], color=fs.TRUTH, lw=3.0, label="2D simulation")
    ax_s.plot(freq, sig["ai"], color=fs.AI, lw=2.8, label="This work")
    ax_s.plot(freq, sig["1d"], color=fs.ONED, lw=2.4, ls=(0, (5, 3)), label="1D")
    ax_s.plot(freq, sig["toro"], color=fs.TORO, lw=2.4, label="Toro, 40 profiles")
    fs.log_freq_axis(ax_s)
    ax_s.set_ylim(0, max(0.85, float(np.nanmax([s.max() for s in sig.values()])) * 1.05))
    ax_s.set_ylabel("$\\sigma_{\\ln}$ of $|TF|$")
    ax_s.set_title("Spread across seeds", loc="left", color=fs.NAVY)
    ax_s.legend(loc="upper left", fontsize=13.5, ncol=2, columnspacing=0.8, handlelength=1.4)
    fs.save(fig, "fig12_seed_spread", tight=True)


if __name__ == "__main__":
    main()
