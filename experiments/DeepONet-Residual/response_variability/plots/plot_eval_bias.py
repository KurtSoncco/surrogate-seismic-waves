#!/usr/bin/env python3
"""Nature figures: leftover calibration, covariate Pearson/Anderson, SOTA ranking.

uv run python experiments/DeepONet-Residual/response_variability/plots/plot_eval_bias.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from response_variability.covariates import (  # noqa: E402
    COVARIATE_LABELS,
    attach_extracted_f0,
    attach_h5_covariates,
    pearson_anderson_per_sample,
    present_covariates,
)
from response_variability.evals.eval_classical import merge_classical_into_pack  # noqa: E402
from response_variability.evals.eval_iid import (  # noqa: E402
    aggregate_json,
    band_misfit_table,
    summarize_methods,
)
from response_variability.gino_bias import (  # noqa: E402
    _boot_mean_ci,
    _spearman,
    band_mask,
    leftover,
    rhat_on_r_slope,
    trough_keep_mask,
)
from response_variability.metrics import FREQ_BANDS  # noqa: E402
from response_variability.names import (  # noqa: E402
    DMULT,
    GINO,
    HASKELL_NOMINAL,
    METHOD_COLORS,
    PASSERI,
    PASSERI_DIP,
    PASSERI_FIXED,
    PRETELL,
    PRETELL_P84,
    TORO,
    TORO_DIP,
    TORO_FIXED,
)
from response_variability.plots.plot_iid import _boxplot_with_points  # noqa: E402
from response_variability.plots.plot_presentation import (  # noqa: E402
    DOMAIN_SPECS,
    leftover_central,
    load_pack,
    pack_path,
)
from response_variability.style import (  # noqa: E402
    apply_nature_style,
    figsize,
    panel_letter,
    savefig,
)

OUT_DIR = config.RESULTS_DIR / "response_variability" / "eval_bias"
PACK_DIR = config.RESULTS_DIR / "presentation"
ATLAS_DIR = config.RESULTS_DIR / "response_variability" / "tf_atlas"
RANK_METHODS = (
    GINO,
    HASKELL_NOMINAL,
    PRETELL,
    PRETELL_P84,
    TORO,
    TORO_FIXED,
    TORO_DIP,
    PASSERI,
    PASSERI_FIXED,
    PASSERI_DIP,
    DMULT,
)
PEARSON_HELDOUT_PANELS = (
    ("iid", "IID (2-layer) val+test"),
    ("dipping", "dipping OOD val+test"),
)
DOMAIN_COLORS = {"iid": "#0072B2", "dipping": "#D55E00", "three_layer": "#009E73"}
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _merge_classical(pack: dict[str, np.ndarray], domain: str, out_dir: Path) -> dict:
    return merge_classical_into_pack(pack, domain, out_dir=out_dir)


def _median_ci(
    x: np.ndarray, rng: np.random.Generator, n_boot: int = 400
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(x))
    if x.size < 2:
        return med, float("nan"), float("nan")
    boots = np.empty(n_boot)
    for t in range(n_boot):
        boots[t] = float(np.median(x[rng.integers(0, x.size, size=x.size)]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return med, float(lo), float(hi)


def plot_leftover_calibration(
    packs: dict[str, dict[str, np.ndarray]], out_path: Path
) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize("double", height_mm=68), sharey=True)
    for ax, (domain, letter) in zip(axes, zip(DOMAIN_SPECS, "abc")):
        pack = packs[domain]
        slope = rhat_on_r_slope(
            pack["tf_gino"],
            pack["tf_opensees"],
            pack["tf_haskell_nominal"],
            pack["freq"],
            n_boot=80,
            seed=0,
        )
        r_hat = leftover(pack["tf_gino"], pack["tf_haskell_nominal"])
        r = leftover(pack["tf_opensees"], pack["tf_haskell_nominal"])
        rec = r_hat.shape[1] // 2
        ops_c = np.asarray(pack["tf_opensees"], dtype=np.float64)
        ops_c = ops_c[:, rec, :] if ops_c.ndim == 3 else ops_c
        keep = trough_keep_mask(ops_c) & band_mask(pack["freq"])[None, :]
        x = r[:, rec, :][keep].ravel()
        y = r_hat[:, rec, :][keep].ravel()
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size == 0:
            ax.set_title(DOMAIN_SPECS[domain]["title"])
            panel_letter(ax, letter, x=0.02, y=0.98)
            continue
        if x.size > 8000:
            rng = np.random.default_rng(0)
            idx = rng.choice(x.size, 8000, replace=False)
            x, y = x[idx], y[idx]
        ax.scatter(x, y, s=4, c=METHOD_COLORS[GINO], alpha=0.25, edgecolors="none")
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 50)
        ax.plot(xx, xx, color="0.4", lw=0.6, ls="--")
        ax.plot(xx, slope["a"] + slope["b"] * xx, color=METHOD_COLORS[GINO], lw=1.2)
        ax.set_xlabel(r"$R$")
        ax.set_title(
            rf"{DOMAIN_SPECS[domain]['title']}: $b$={slope['b']:.2f} "
            rf"[{slope['b_lo']:.2f},{slope['b_hi']:.2f}]"
        )
        panel_letter(ax, letter, x=0.02, y=0.98)
    axes[0].set_ylabel(r"$\hat R$")
    fig.tight_layout()
    return savefig(fig, out_path)


def plot_leftover_vs_freq(
    packs: dict[str, dict[str, np.ndarray]], out_path: Path
) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    fig, axes = plt.subplots(2, 3, figsize=figsize("double", height_mm=120))
    for col, domain in enumerate(DOMAIN_SPECS):
        pack = (
            attach_extracted_f0(packs[domain])
            if "f0_calc" not in packs[domain]
            else packs[domain]
        )
        freq = np.asarray(pack["freq"], dtype=float)
        n = int(pack["tf_opensees"].shape[0])
        r_stack = []
        rh_stack = []
        r_fn = []
        rh_fn = []
        for i in range(n):
            r_true, r_hat = leftover_central(pack, i)
            r_stack.append(r_true)
            rh_stack.append(r_hat)
            f0 = float(pack["f0"][i])
            if np.isfinite(f0) and f0 > 0:
                fn = freq / f0
                order = np.argsort(fn)
                grid = np.logspace(-1, 1, 80)
                r_fn.append(
                    np.interp(grid, fn[order], r_true[order], left=np.nan, right=np.nan)
                )
                rh_fn.append(
                    np.interp(grid, fn[order], r_hat[order], left=np.nan, right=np.nan)
                )
        r_m = np.nanmean(np.vstack(r_stack), axis=0)
        rh_m = np.nanmean(np.vstack(rh_stack), axis=0)
        ax = axes[0, col]
        ax.axhline(0.0, color="0.6", lw=0.4)
        ax.plot(freq, r_m, color=METHOD_COLORS[GINO], ls="--", lw=1.0, label=r"$R$")
        ax.plot(freq, rh_m, color=METHOD_COLORS[GINO], lw=1.2, label=r"$\hat R$")
        ax.set_xscale("log")
        ax.set_title(DOMAIN_SPECS[domain]["title"])
        ax.set_xlabel(r"$f$ (Hz)")
        panel_letter(ax, "abc"[col], x=0.02, y=0.98)
        ax2 = axes[1, col]
        fn_grid = np.logspace(-1, 1, 80)
        ax2.axhline(0.0, color="0.6", lw=0.4)
        if r_fn:
            stacked_r = np.vstack(r_fn)
            stacked_h = np.vstack(rh_fn)
            if np.isfinite(stacked_r).any():
                with np.errstate(all="ignore"):
                    ax2.plot(
                        fn_grid,
                        np.nanmean(stacked_r, axis=0),
                        color=METHOD_COLORS[GINO],
                        ls="--",
                        lw=1.0,
                    )
            if np.isfinite(stacked_h).any():
                with np.errstate(all="ignore"):
                    ax2.plot(
                        fn_grid,
                        np.nanmean(stacked_h, axis=0),
                        color=METHOD_COLORS[GINO],
                        lw=1.2,
                    )
        ax2.set_xscale("log")
        ax2.set_xlabel(r"$f/f_0$")
        panel_letter(ax2, "def"[col], x=0.02, y=0.98)
    axes[0, 0].set_ylabel(r"mean leftover")
    axes[1, 0].set_ylabel(r"mean leftover")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    return savefig(fig, out_path)


def _gino_metrics(pack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pack = attach_extracted_f0(pack) if "f0_calc" not in pack else pack
    return pearson_anderson_per_sample(
        pack["tf_gino"],
        pack["tf_opensees"],
        pack["freq"],
        f0_extracted=pack["f0"],
    )


def plot_bias_vs_covariates(
    packs: dict[str, dict[str, np.ndarray]],
    out_path: Path,
    *,
    metric: str = "pearson",
) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    keys: list[str] = []
    for domain, pack in packs.items():
        for k in present_covariates(pack, domain):
            if k not in keys:
                keys.append(k)
    if not keys:
        keys = ["vs1", "H", "cov", "vs2", "f0"]
    n = len(keys)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=figsize("double", height_mm=42 * nrow + 18), squeeze=False
    )
    ylab = r"Pearson of $|\mathrm{TF}|$" if metric == "pearson" else "Anderson misfit"
    ykey = "pearson" if metric == "pearson" else "gof_af"
    for ax, key, letter in zip(axes.ravel(), keys, _LETTERS):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for domain, pack in packs.items():
            if key not in pack:
                continue
            mets = _gino_metrics(pack)
            y = mets[ykey]
            ax.scatter(
                pack[key],
                y,
                s=8,
                c=DOMAIN_COLORS[domain],
                alpha=0.55,
                edgecolors="none",
                label=DOMAIN_SPECS[domain]["title"],
            )
            xs.append(np.asarray(pack[key], dtype=float))
            ys.append(np.asarray(y, dtype=float))
        rho = _spearman(np.concatenate(xs), np.concatenate(ys)) if xs else float("nan")
        rho_s = rf"$\rho$={rho:.2f}" if np.isfinite(rho) else r"$\rho$=n/a"
        ax.set_xlabel(COVARIATE_LABELS.get(key, key))
        ax.set_title(rho_s, fontsize=7)
        panel_letter(ax, letter, x=0.02, y=0.98)
    axes[0, 0].set_ylabel(ylab)
    if nrow > 1:
        axes[1, 0].set_ylabel(ylab)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02)
        )
    fig.tight_layout()
    return savefig(fig, out_path)


def plot_bias_vs_covariates_bands(
    packs: dict[str, dict[str, np.ndarray]],
    out_path: Path,
    *,
    metric: str = "pearson",
) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    bands = ("low", "mid", "high", "all")
    cov_keys = ("vs1", "H", "cov", "f0")
    present = [k for k in cov_keys if any(k in p for p in packs.values())]
    if not present:
        present = ["cov"]
    fig, axes = plt.subplots(
        len(bands),
        len(present),
        figsize=figsize("double", height_mm=38 * len(bands) + 16),
        squeeze=False,
    )
    ylab = r"Pearson" if metric == "pearson" else "Anderson"
    prefix = "pearson" if metric == "pearson" else "gof"
    letter_i = 0
    for row, band in enumerate(bands):
        lo, hi = FREQ_BANDS[band]
        for col, key in enumerate(present):
            ax = axes[row, col]
            xs: list[np.ndarray] = []
            ys: list[np.ndarray] = []
            for domain, pack in packs.items():
                if key not in pack:
                    continue
                mets = _gino_metrics(pack)
                y = mets[f"{prefix}_{band}"]
                ax.scatter(
                    pack[key],
                    y,
                    s=6,
                    c=DOMAIN_COLORS[domain],
                    alpha=0.55,
                    edgecolors="none",
                    label=DOMAIN_SPECS[domain]["title"],
                )
                xs.append(np.asarray(pack[key], dtype=float))
                ys.append(np.asarray(y, dtype=float))
            rho = (
                _spearman(np.concatenate(xs), np.concatenate(ys))
                if xs
                else float("nan")
            )
            rho_s = f", $\\rho$={rho:.2f}" if np.isfinite(rho) else ""
            ax.set_title(f"{band} {lo:g}–{hi:g} Hz{rho_s}", fontsize=6.5)
            if row == len(bands) - 1:
                ax.set_xlabel(COVARIATE_LABELS.get(key, key))
            if col == 0:
                ax.set_ylabel(ylab)
            panel_letter(ax, _LETTERS[letter_i], x=0.02, y=0.98)
            letter_i += 1
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02)
        )
    fig.tight_layout()
    return savefig(fig, out_path)


def plot_quartile_forest(
    packs: dict[str, dict[str, np.ndarray]], out_path: Path
) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize("double", height_mm=90))
    rng = np.random.default_rng(0)
    for ax, metric, letter, ylab in zip(
        axes,
        ("pearson", "gof_af"),
        "ab",
        (r"Pearson of $|\mathrm{TF}|$", "Anderson misfit"),
    ):
        y = 0
        yticks = []
        ylabels = []
        for domain, pack in packs.items():
            mets = _gino_metrics(pack)
            vals = mets[metric]
            for key in present_covariates(pack, domain)[:6]:
                v = np.asarray(pack[key], dtype=float)
                finite = np.isfinite(v) & np.isfinite(vals)
                if finite.sum() < 8:
                    continue
                edges = np.quantile(v[finite], [0.0, 0.25, 0.5, 0.75, 1.0])
                for q in range(4):
                    sel = finite & (v > edges[q]) & (v <= edges[q + 1])
                    mm = vals[sel]
                    if mm.size == 0:
                        continue
                    mean = float(np.mean(mm))
                    lo, hi = _boot_mean_ci(mm, rng, 200)
                    ax.plot([lo, hi], [y, y], color=DOMAIN_COLORS[domain], lw=0.8)
                    ax.plot(mean, y, "o", color=DOMAIN_COLORS[domain], ms=3)
                    yticks.append(y)
                    ylabels.append(f"{domain[:3]} {key} Q{q + 1}")
                    y -= 1
        ax.set_yticks(yticks[::2] if len(yticks) > 16 else yticks)
        ax.set_yticklabels(ylabels[::2] if len(yticks) > 16 else ylabels, fontsize=5)
        ax.set_xlabel(ylab)
        panel_letter(ax, letter, x=0.02, y=1.02)
    fig.tight_layout()
    return savefig(fig, out_path)


def _rank_methods_in(summary: pd.DataFrame) -> list[str]:
    present = set(summary["method"].unique())
    return [m for m in RANK_METHODS if m in present]


def plot_pearson_boxes_heldout(
    summaries: dict[str, pd.DataFrame],
    out_path: Path,
    *,
    panels: tuple[tuple[str, str], ...] | None = None,
    label_rotation: float = 0.0,
) -> Path:
    """Pearson boxplots for two (or more) summary slices, side by side."""
    import matplotlib.pyplot as plt

    apply_nature_style()
    panels = panels or PEARSON_HELDOUT_PANELS
    height = 118 if label_rotation else 95
    fig, axes = plt.subplots(
        1, len(panels), figsize=figsize("double", height_mm=height), sharey=True
    )
    if len(panels) == 1:
        axes = [axes]
    for ax, (domain, title), letter in zip(axes, panels, _LETTERS):
        summary = summaries[domain]
        labels = _rank_methods_in(summary)
        data = [
            summary.loc[summary["method"] == m, "pearson"].to_numpy() for m in labels
        ]
        n = int(max((len(v) for v in data), default=0))
        _boxplot_with_points(ax, data, labels, ref_line=1.0)
        if label_rotation:
            ax.tick_params(axis="x", labelrotation=label_rotation)
            for lab in ax.get_xticklabels():
                lab.set_ha("right")
        ax.set_title(rf"{title} ($n$={n})")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("")
        panel_letter(ax, letter, x=0.02, y=0.98)
    axes[0].set_ylabel(r"Pearson of $|\mathrm{TF}|$")
    fig.tight_layout()
    return savefig(fig, out_path)


CORNER_PEARSON_PANELS = (
    ("corner_train", "IS corner, train-eligible"),
    ("corner_held", "IS corner, held-out"),
)


def plot_pearson_boxes_corner(
    summaries: dict[str, pd.DataFrame], out_path: Path | None = None
) -> Path:
    out_path = Path(out_path or (OUT_DIR / "method_ranking_pearson_corner.png"))
    return plot_pearson_boxes_heldout(summaries, out_path, panels=CORNER_PEARSON_PANELS)


def load_heldout_summaries(atlas_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    atlas_dir = Path(atlas_dir or ATLAS_DIR)
    out: dict[str, pd.DataFrame] = {}
    for domain, _title in PEARSON_HELDOUT_PANELS:
        path = atlas_dir / f"{domain}_heldout_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing held-out summary {path}")
        out[domain] = pd.read_csv(path)
    return out


def plot_pearson_boxes_from_atlas(
    out_path: Path | None = None, *, atlas_dir: Path | None = None
) -> Path:
    out_path = Path(out_path or (OUT_DIR / "method_ranking_pearson_heldout.png"))
    return plot_pearson_boxes_heldout(load_heldout_summaries(atlas_dir), out_path)


def plot_method_ranking_iid(
    summary: pd.DataFrame, misfit: pd.DataFrame, out_path: Path
) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    labels = _rank_methods_in(summary)
    fig, axes = plt.subplots(1, 2, figsize=figsize("double", height_mm=95))
    data_g = [summary.loc[summary["method"] == m, "gof_af"].to_numpy() for m in labels]
    _boxplot_with_points(axes[0], data_g, labels, ref_line=None)
    axes[0].set_ylabel("Anderson misfit")
    panel_letter(axes[0], "a", x=0.02, y=0.98)
    data_p = [summary.loc[summary["method"] == m, "pearson"].to_numpy() for m in labels]
    _boxplot_with_points(axes[1], data_p, labels, ref_line=1.0)
    axes[1].set_ylabel(r"Pearson of $|\mathrm{TF}|$")
    axes[1].set_ylim(0.0, 1.02)
    panel_letter(axes[1], "b", x=0.02, y=0.98)
    fig.tight_layout()
    p = savefig(fig, out_path)

    apply_nature_style()
    bands = ["low", "mid", "high"]
    band_labels = ["0.1–0.5 Hz", "0.5–2 Hz", "2–10 Hz"]
    fig, axes = plt.subplots(1, 2, figsize=figsize("double", height_mm=78))
    x = np.arange(len(bands), dtype=float)
    n = max(len(labels), 1)
    width = min(0.14, 0.8 / n)
    for ax, prefix, ylab in zip(
        axes,
        ("gof", "pearson"),
        ("Anderson misfit", r"Pearson of $|\mathrm{TF}|$"),
    ):
        for k, method in enumerate(labels):
            sub = misfit[misfit["method"] == method]
            means = [float(sub[f"{prefix}_{b}"].mean()) for b in bands]
            offset = (k - (n - 1) / 2.0) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor="none",
                label=method,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels)
        ax.set_ylabel(ylab)
        if prefix == "pearson":
            ax.set_ylim(0.0, 1.0)
        ax.legend(loc="best", fontsize=5.5, ncol=2)
    panel_letter(axes[0], "c", x=0.02, y=0.98)
    panel_letter(axes[1], "d", x=0.02, y=0.98)
    fig.tight_layout()
    band_path = out_path.with_name(out_path.stem + "_bands.png")
    savefig(fig, band_path)
    return p


def plot_method_ranking_ood_compact(
    packs: dict[str, dict[str, np.ndarray]], out_path: Path
) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for domain, pack in packs.items():
        summary, _peaks = summarize_methods(pack)
        for method in _rank_methods_in(summary):
            sub = summary.loc[summary["method"] == method]
            g_med, g_lo, g_hi = _median_ci(sub["gof_af"].to_numpy(), rng)
            p_med, p_lo, p_hi = _median_ci(sub["pearson"].to_numpy(), rng)
            rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "gof_median": g_med,
                    "gof_lo": g_lo,
                    "gof_hi": g_hi,
                    "pearson_median": p_med,
                    "pearson_lo": p_lo,
                    "pearson_hi": p_hi,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_path.with_name("ood_method_ranking.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=figsize("double", height_mm=88))
    domains = list(DOMAIN_SPECS)
    methods = [m for m in RANK_METHODS if m in set(df["method"])]
    y = np.arange(len(domains))
    h = 0.12
    for ax, col, lo_c, hi_c, xlab in zip(
        axes,
        ("gof_median", "pearson_median"),
        ("gof_lo", "pearson_lo"),
        ("gof_hi", "pearson_hi"),
        ("median Anderson misfit", r"median Pearson of $|\mathrm{TF}|$"),
    ):
        for k, method in enumerate(methods):
            sub = df[df["method"] == method].set_index("domain")
            yy = y + (k - (len(methods) - 1) / 2.0) * h
            med = [
                float(sub.loc[d, col]) if d in sub.index else np.nan for d in domains
            ]
            lo = [
                float(sub.loc[d, lo_c]) if d in sub.index else np.nan for d in domains
            ]
            hi = [
                float(sub.loc[d, hi_c]) if d in sub.index else np.nan for d in domains
            ]
            ax.errorbar(
                med,
                yy,
                xerr=[
                    np.clip(np.asarray(med) - np.asarray(lo), 0, None),
                    np.clip(np.asarray(hi) - np.asarray(med), 0, None),
                ],
                fmt="o",
                color=METHOD_COLORS[method],
                ms=3.5,
                lw=0.8,
                label=method,
            )
        ax.set_yticks(y)
        ax.set_yticklabels([DOMAIN_SPECS[d]["title"] for d in domains])
        ax.set_xlabel(xlab)
        ax.invert_yaxis()
    axes[1].legend(loc="lower left", fontsize=6)
    panel_letter(axes[0], "a", x=0.02, y=1.02)
    panel_letter(axes[1], "b", x=0.02, y=1.02)
    fig.tight_layout()
    savefig(fig, out_path)
    return out_path


def plot_all(
    *,
    pack_dir: Path,
    out_dir: Path,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    packs: dict[str, dict[str, np.ndarray]] = {}
    for domain in DOMAIN_SPECS:
        path = pack_path(pack_dir, domain)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        pack = load_pack(path)
        pack = attach_h5_covariates(pack, domain=domain)
        pack = _merge_classical(pack, domain, out_dir)
        packs[domain] = pack
    paths = [
        plot_leftover_calibration(packs, out_dir / "leftover_calibration.png"),
        plot_leftover_vs_freq(packs, out_dir / "leftover_vs_freq.png"),
        plot_bias_vs_covariates(
            packs, out_dir / "bias_vs_covariates.png", metric="pearson"
        ),
        plot_bias_vs_covariates(
            packs, out_dir / "bias_vs_covariates_anderson.png", metric="gof_af"
        ),
        plot_bias_vs_covariates_bands(
            packs, out_dir / "bias_vs_covariates_bands.png", metric="pearson"
        ),
        plot_bias_vs_covariates_bands(
            packs, out_dir / "bias_vs_covariates_bands_anderson.png", metric="gof_af"
        ),
        plot_quartile_forest(packs, out_dir / "bias_quartile_forest.png"),
    ]
    iid = packs["iid"]
    summary, _peaks = summarize_methods(iid)
    misfit = band_misfit_table(iid)
    summary.to_csv(out_dir / "iid_method_summary.csv", index=False)
    misfit.to_csv(out_dir / "iid_band_misfit.csv", index=False)
    (out_dir / "iid_aggregate.json").write_text(
        json.dumps(aggregate_json(summary, misfit), indent=2)
    )
    paths.append(
        plot_method_ranking_iid(summary, misfit, out_dir / "method_ranking_iid.png")
    )
    paths.append(
        plot_method_ranking_ood_compact(
            packs, out_dir / "method_ranking_ood_compact.png"
        )
    )
    try:
        paths.append(
            plot_pearson_boxes_from_atlas(
                out_dir / "method_ranking_pearson_heldout.png"
            )
        )
    except FileNotFoundError:
        pass
    for p in paths:
        print(f"Wrote {p}", flush=True)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    plot_all(pack_dir=args.pack_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
