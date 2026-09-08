"""Nature-style figures for Sobol covering and frequency probes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from response_variability.names import (  # noqa: E402
    GINO,
    HASKELL_NOMINAL,
    METHOD_COLORS,
    PRETELL,
    PRETELL_P84,
)
from response_variability.plot_presentation import DOMAIN_SPECS  # noqa: E402
from response_variability.sobol_design import AHV_FIXED, RH_FIXED  # noqa: E402
from response_variability.style import (  # noqa: E402
    apply_nature_style,
    figsize,
    panel_letter,
    savefig,
)

DOMAIN_COLORS = {
    "iid": "#0072B2",
    "dipping": "#D55E00",
    "three_layer": "#009E73",
}
SEISKIT_AGG = config.RESULTS_DIR / "response_variability" / "seiskit" / "aggregate.json"


def _title(domain: str) -> str:
    return DOMAIN_SPECS[domain]["title"]


def plot_covering(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    rows = blob["summary"]["covering"]
    fig, ax = plt_subplots("double", 72)
    x = np.arange(len(rows))
    w = 0.35
    files = [r["n_files"] for r in rows]
    uniq = [r["n_unique_6d_iid"] for r in rows]
    ax.bar(x - w / 2, files, w, color="#CCCCCC", label="mix train files")
    ax.bar(x + w / 2, uniq, w, color=DOMAIN_COLORS["iid"], label="unique IID 6D Sobol")
    ax.set_xticks(x)
    ax.set_xticklabels([r["mix"] for r in rows])
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    ax.set_title("Extra mix files are mostly RF replicates, not new Sobol IDs")
    fig.tight_layout()
    return savefig(fig, out_dir / "covering_unique_vs_n.png")


def plot_n_ladder(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    pub = blob["summary"]["published_rel_l2"]
    methods = list(pub.keys())
    domains = ("iid", "dipping", "three_layer")
    fig, ax = plt_subplots("double", 78)
    x = np.arange(len(methods))
    w = 0.22
    for k, d in enumerate(domains):
        vals = [pub[m][d] for m in methods]
        ax.bar(
            x + (k - 1) * w,
            vals,
            w,
            color=DOMAIN_COLORS[d],
            label=_title(d),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=12, ha="right")
    ax.set_ylabel(r"Held-out relative $L_2$")
    ax.legend(loc="upper right")
    ax.set_title("More IID files without rebalancing hurts three-layer leftover")
    fig.tight_layout()
    return savefig(fig, out_dir / "n_ladder_scores.png")


def plot_n_ladder_pearson(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    pub = blob["summary"]["published_pearson"]
    methods = list(pub.keys())
    domains = ("iid", "dipping", "three_layer")
    fig, ax = plt_subplots("double", 78)
    x = np.arange(len(methods))
    w = 0.22
    for k, d in enumerate(domains):
        vals = [pub[m][d] for m in methods]
        ax.bar(
            x + (k - 1) * w,
            vals,
            w,
            color=DOMAIN_COLORS[d],
            label=_title(d),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=12, ha="right")
    ax.set_ylabel(r"Held-out Pearson of $|\mathrm{TF}|$")
    ax.set_ylim(0.80, 0.95)
    ax.legend(loc="lower right")
    ax.set_title("Pearson: rebalance keeps three-layer shape while extra IID files do not")
    fig.tight_layout()
    return savefig(fig, out_dir / "n_ladder_pearson.png")


def plot_fill(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    fig, ax = plt_subplots("single", 72)
    fill = blob["fill"]
    styles = {
        "iid_6d": ("-", DOMAIN_COLORS["iid"], "IID test vs IID unique 6D"),
        "iid_4d": ("--", DOMAIN_COLORS["iid"], "IID test vs IID unique 4D"),
        "dipping_vs_iid": ("-", DOMAIN_COLORS["dipping"], "dipping vs IID unique 4D"),
        "three_layer_vs_iid": (
            "-",
            DOMAIN_COLORS["three_layer"],
            "three-layer vs IID unique 4D",
        ),
    }
    for key, (ls, color, label) in styles.items():
        if key not in fill:
            continue
        rec = fill[key]
        ax.plot(rec["n_unique"], rec["fill_distance"], ls=ls, color=color, label=label)
    ax.set_xlabel("Nested unique Sobol IDs in train subset")
    ax.set_ylabel("Mean 1-NN distance (z-scored)")
    ax.legend(loc="upper right")
    ax.set_title("Covering gap vs nested unique-ID prefixes")
    fig.tight_layout()
    return savefig(fig, out_dir / "fill_distance.png")


def plot_rh_ahv(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    fig, ax = plt_subplots("double", 90)
    train = blob["train6_iid"]
    ax.scatter(
        train[:, 3],
        train[:, 4],
        s=8,
        c="#BBBBBB",
        alpha=0.7,
        linewidths=0,
        label="IID train",
        zorder=1,
    )
    pack = blob["packs"]["iid"]
    if "rH" in pack:
        ax.scatter(
            pack["rH"],
            pack["aHV"],
            s=14,
            c=DOMAIN_COLORS["iid"],
            linewidths=0,
            label="IID test",
            zorder=2,
        )
    ax.scatter(
        [RH_FIXED],
        [AHV_FIXED],
        s=48,
        marker="*",
        c="#000000",
        label=r"RV 64 ($r_H{=}10$, $a_{HV}{=}50$)",
        zorder=3,
    )
    ax.axvline(RH_FIXED, color="#666666", lw=0.6, ls=":")
    ax.axhline(AHV_FIXED, color="#666666", lw=0.6, ls=":")
    ax.set_xlabel(r"$r_H$ (m)")
    ax.set_ylabel(r"$a_{HV}$")
    ax.legend(loc="lower right")
    ax.set_title("RV campaign sits at the min-$r_H$ / max-$a_{HV}$ corner")
    fig.tight_layout()
    return savefig(fig, out_dir / "cloud_rh_ahv.png")


def plot_vs1_h(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    fig, axes = plt_subplots("double", 82, ncols=2)
    train = blob["train4_iid"]
    rv4 = blob["rv"]["rv4"]
    for ax, domain, letter in zip(axes, ("iid", "three_layer"), "ab"):
        pack = blob["packs"][domain]
        ax.scatter(
            train[:, 0],
            train[:, 1],
            s=7,
            c="#BBBBBB",
            alpha=0.65,
            linewidths=0,
            label="IID train",
            zorder=1,
        )
        ax.scatter(
            pack["vs1"],
            pack["H"],
            s=12,
            c=DOMAIN_COLORS[domain],
            linewidths=0,
            label=_title(domain) + " test",
            zorder=2,
        )
        ax.scatter(
            rv4[:, 0],
            rv4[:, 1],
            s=10,
            marker="x",
            c="#000000",
            label="RV 64 (4D)",
            zorder=3,
        )
        ax.set_xlabel(r"$V_{s1}$ (m s$^{-1}$)")
        ax.set_ylabel(r"$H$ (m)")
        ax.legend(loc="upper right")
        panel_letter(ax, letter)
    axes[0].set_title("4D Sobol slice is in-range; three-layer $H$ is not")
    fig.tight_layout()
    return savefig(fig, out_dir / "cloud_vs1_H.png")


def plot_error_vs_knn(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    fig, axes = plt_subplots("double", 72, ncols=3)
    for ax, domain, letter in zip(axes, DOMAIN_SPECS, "abc"):
        rec = blob["dist"][domain]
        ax.scatter(
            rec["knn_4d"],
            rec["rel_l2_gino"],
            s=10,
            c=DOMAIN_COLORS[domain],
            linewidths=0,
            label=GINO,
        )
        ax.scatter(
            rec["knn_4d"],
            rec["rel_l2_1d"],
            s=8,
            c="#999999",
            linewidths=0,
            alpha=0.7,
            label=HASKELL_NOMINAL,
        )
        rho, pval = rec["spearman_knn4d_l2"]
        ax.set_xlabel(r"1-NN to IID train (4D $z$)")
        ax.set_title(_title(domain))
        ax.text(
            0.05,
            0.95,
            rf"$\rho$={rho:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
        panel_letter(ax, letter)
    axes[0].set_ylabel(r"Relative $L_2$ vs OpenSees")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    return savefig(fig, out_dir / "error_vs_knn.png")


def plot_error_vs_knn_pearson(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    fig, axes = plt_subplots("double", 72, ncols=3)
    for ax, domain, letter in zip(axes, DOMAIN_SPECS, "abc"):
        rec = blob["dist"][domain]
        ax.scatter(
            rec["knn_4d"],
            rec["pearson_gino"],
            s=10,
            c=DOMAIN_COLORS[domain],
            linewidths=0,
            label=GINO,
        )
        ax.scatter(
            rec["knn_4d"],
            rec["pearson_1d"],
            s=8,
            c="#999999",
            linewidths=0,
            alpha=0.7,
            label=HASKELL_NOMINAL,
        )
        rho, pval = rec["spearman_knn4d_pearson"]
        ax.set_xlabel(r"1-NN to IID train (4D $z$)")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(_title(domain))
        ax.text(
            0.05,
            0.05,
            rf"$\rho$={rho:.2f}",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
        )
        panel_letter(ax, letter)
    axes[0].set_ylabel(r"Pearson of $|\mathrm{TF}|$ vs OpenSees")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    return savefig(fig, out_dir / "pearson_vs_knn.png")


def plot_freq_train_heldout(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    fig, ax = plt_subplots("double", 72)
    domains = list(DOMAIN_SPECS)
    x = np.arange(len(domains))
    w = 0.22
    train_m = [blob["summary"]["freq"][d]["gino_train_bins_mean"] for d in domains]
    hold_m = [blob["summary"]["freq"][d]["gino_heldout_bins_mean"] for d in domains]
    high_m = [blob["summary"]["freq"][d]["gino_high_mean"] for d in domains]
    ax.bar(x - w, train_m, w, color="#56B4E9", label="200 train-query bins")
    ax.bar(x, hold_m, w, color="#0072B2", label="800 held-out bins (0.1–10 Hz)")
    ax.bar(x + w, high_m, w, color="#D55E00", label="2–10 Hz band")
    ax.set_xticks(x)
    ax.set_xticklabels([_title(d) for d in domains])
    ax.set_ylabel(r"GINO relative $L_2$")
    ax.legend(loc="upper left")
    ax.set_title("Held-out frequencies are in-band interpolation; high $f$ is the leftover")
    fig.tight_layout()
    return savefig(fig, out_dir / "freq_train_vs_heldout.png")


def plot_freq_train_heldout_pearson(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    fig, ax = plt_subplots("double", 72)
    domains = list(DOMAIN_SPECS)
    x = np.arange(len(domains))
    w = 0.22
    train_m = [blob["summary"]["freq"][d]["gino_train_bins_pearson"] for d in domains]
    hold_m = [blob["summary"]["freq"][d]["gino_heldout_bins_pearson"] for d in domains]
    high_m = [blob["summary"]["freq"][d]["gino_high_pearson"] for d in domains]
    ax.bar(x - w, train_m, w, color="#56B4E9", label="200 train-query bins")
    ax.bar(x, hold_m, w, color="#0072B2", label="800 held-out bins (0.1–10 Hz)")
    ax.bar(x + w, high_m, w, color="#D55E00", label="2–10 Hz band")
    ax.set_xticks(x)
    ax.set_xticklabels([_title(d) for d in domains])
    ax.set_ylabel(r"GINO Pearson of $|\mathrm{TF}|$")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower left")
    ax.set_title("Pearson: held-out bins match train queries; high $f$ loses shape")
    fig.tight_layout()
    return savefig(fig, out_dir / "freq_train_vs_heldout_pearson.png")


def plot_error_vs_freq(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    fig, ax = plt_subplots("double", 78)
    for domain in DOMAIN_SPECS:
        rec = blob["freq"][domain]
        freq = rec["freq"]
        ax.plot(
            freq,
            rec["abs_log_gino"],
            color=DOMAIN_COLORS[domain],
            label=_title(domain),
        )
    rec0 = blob["freq"]["iid"]
    ax.set_xscale("log")
    ymin = ax.get_ylim()[0]
    ticks = rec0["freq"][np.asarray(rec0["mask_train"], dtype=bool)]
    ax.plot(
        ticks,
        np.full(ticks.shape, ymin),
        "|",
        color="#333333",
        markersize=4,
        label="train queries",
    )
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"mean $|\ln(\widehat{\mathrm{TF}}/\mathrm{TF})|$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return savefig(fig, out_dir / "error_vs_freq.png")


def plot_error_vs_f0(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    fig, ax = plt_subplots("double", 78)
    edges = np.logspace(-1, 1.2, 18)
    centers = np.sqrt(edges[:-1] * edges[1:])
    for domain in DOMAIN_SPECS:
        pack = blob["packs"][domain]
        rec = blob["freq"][domain]
        freq = rec["freq"]
        err_sf = rec["abs_log_gino_sf"]
        f0 = pack["f0"]
        acc = np.zeros(len(centers))
        wts = np.zeros(len(centers))
        for i, f0i in enumerate(f0):
            if not np.isfinite(f0i) or f0i <= 0:
                continue
            ratio = freq / f0i
            bins = np.digitize(ratio, edges) - 1
            for j in range(len(centers)):
                m = bins == j
                if np.any(m):
                    acc[j] += float(np.mean(err_sf[i, m]))
                    wts[j] += 1.0
        y = np.divide(acc, np.clip(wts, 1e-12, None))
        y[wts == 0] = np.nan
        ax.plot(centers, y, color=DOMAIN_COLORS[domain], label=_title(domain))
    ax.axvline(1.0, color="#666666", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$f / f_0$")
    ax.set_ylabel(r"mean $|\ln(\widehat{\mathrm{TF}}/\mathrm{TF})|$")
    ax.legend(loc="upper left")
    ax.set_title("Leftover vs site harmonics, not vs unqueried bins")
    fig.tight_layout()
    return savefig(fig, out_dir / "error_vs_f_over_f0.png")


def plot_seiskit_arms(out_dir: Path) -> Path | None:
    path = SEISKIT_AGG
    if not path.is_file():
        return None
    apply_nature_style()
    agg = json.loads(path.read_text())
    order = [GINO, PRETELL, PRETELL_P84, HASKELL_NOMINAL, "Toro Vs", "Passeri tts"]
    methods = [m for m in order if m in agg]
    fig, ax = plt_subplots("double", 72)
    x = np.arange(len(methods))
    w = 0.35
    central = [agg[m]["rel_l2_central_mean"] for m in methods]
    high = [agg[m]["rel_l2_high_mean"] for m in methods]
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
    ax.bar(x - w / 2, central, w, color=colors, label="full band (central rec.)")
    ax.bar(
        x + w / 2,
        high,
        w,
        color=colors,
        alpha=0.45,
        label="2–10 Hz",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(r"Relative $L_2$ vs OpenSees 2-D")
    ax.legend(loc="upper left")
    ax.set_title("Nested IID test: residual GINO vs seiskit Response_Variability arms")
    fig.tight_layout()
    return savefig(fig, out_dir / "seiskit_arms_iid.png")


def plot_seiskit_arms_pearson(out_dir: Path) -> Path | None:
    path = SEISKIT_AGG
    if not path.is_file():
        return None
    apply_nature_style()
    agg = json.loads(path.read_text())
    order = [GINO, PRETELL, PRETELL_P84, HASKELL_NOMINAL, "Toro Vs", "Passeri tts"]
    methods = [m for m in order if m in agg and "pearson_central_mean" in agg[m]]
    if not methods:
        return None
    fig, ax = plt_subplots("double", 72)
    x = np.arange(len(methods))
    w = 0.35
    central = [agg[m]["pearson_central_mean"] for m in methods]
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
    ax.bar(x - w / 2, central, w, color=colors, label="full band (central rec.)")
    if all("pearson_high_mean" in agg[m] for m in methods):
        high = [agg[m]["pearson_high_mean"] for m in methods]
        ax.bar(x + w / 2, high, w, color=colors, alpha=0.45, label="2–10 Hz")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(r"Pearson of $|\mathrm{TF}|$ vs OpenSees 2-D")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower left")
    ax.set_title("Nested IID test: Pearson vs seiskit Response_Variability arms")
    fig.tight_layout()
    return savefig(fig, out_dir / "seiskit_arms_iid_pearson.png")


def plot_rv_proxy(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    fig, ax = plt_subplots("single", 72)
    proxy = blob["proxy"]
    ax.scatter(
        proxy["nn_dist_4d"],
        proxy["nn_rel_l2_gino"],
        s=14,
        c=DOMAIN_COLORS["iid"],
        label=f"{GINO} at nearest IID test",
    )
    ax.scatter(
        proxy["nn_dist_4d"],
        proxy["nn_rel_l2_1d"],
        s=12,
        c="#999999",
        label=HASKELL_NOMINAL,
    )
    ax.set_xlabel(r"4D $z$-distance of RV case to nearest IID test")
    ax.set_ylabel(r"Relative $L_2$ (proxy sample)")
    ax.legend(loc="upper left")
    ax.set_title("RV 64 without OpenSees H5s: 4D nearest-neighbour proxy")
    fig.tight_layout()
    return savefig(fig, out_dir / "rv64_nearest_proxy.png")


def plot_rv_proxy_pearson(blob: dict[str, Any], out_dir: Path) -> Path:
    apply_nature_style()
    fig, ax = plt_subplots("single", 72)
    proxy = blob["proxy"]
    ax.scatter(
        proxy["nn_dist_4d"],
        proxy["nn_pearson_gino"],
        s=14,
        c=DOMAIN_COLORS["iid"],
        label=f"{GINO} at nearest IID test",
    )
    ax.scatter(
        proxy["nn_dist_4d"],
        proxy["nn_pearson_1d"],
        s=12,
        c="#999999",
        label=HASKELL_NOMINAL,
    )
    ax.set_xlabel(r"4D $z$-distance of RV case to nearest IID test")
    ax.set_ylabel(r"Pearson of $|\mathrm{TF}|$ (proxy sample)")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left")
    ax.set_title("RV 64 4D proxy: Pearson at nearest nested-IID test")
    fig.tight_layout()
    return savefig(fig, out_dir / "rv64_nearest_proxy_pearson.png")


def plt_subplots(width: str, height_mm: float, *, ncols: int = 1):
    import matplotlib.pyplot as plt

    if ncols == 1:
        fig, ax = plt.subplots(figsize=figsize(width, height_mm=height_mm))
        return fig, ax
    fig, axes = plt.subplots(
        1, ncols, figsize=figsize(width, height_mm=height_mm), sharey=False
    )
    return fig, axes


def plot_all(blob: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_covering(blob, out_dir),
        plot_n_ladder(blob, out_dir),
        plot_n_ladder_pearson(blob, out_dir),
        plot_fill(blob, out_dir),
        plot_rh_ahv(blob, out_dir),
        plot_vs1_h(blob, out_dir),
        plot_error_vs_knn(blob, out_dir),
        plot_error_vs_knn_pearson(blob, out_dir),
        plot_freq_train_heldout(blob, out_dir),
        plot_freq_train_heldout_pearson(blob, out_dir),
        plot_error_vs_freq(blob, out_dir),
        plot_error_vs_f0(blob, out_dir),
        plot_rv_proxy(blob, out_dir),
        plot_rv_proxy_pearson(blob, out_dir),
    ]
    arms = plot_seiskit_arms(out_dir)
    if arms is not None:
        paths.append(arms)
    arms_p = plot_seiskit_arms_pearson(out_dir)
    if arms_p is not None:
        paths.append(arms_p)
    return paths
