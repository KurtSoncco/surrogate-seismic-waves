"""Nature-style Response_Variability figures for the nested IID test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from response_variability.metrics import spatial_percentiles  # noqa: E402
from response_variability.names import (  # noqa: E402
    ALL_METHODS,
    GINO,
    HASKELL_COLUMN,
    HASKELL_NOMINAL,
    METHOD_COLORS,
    METHOD_LINESTYLES,
    METHOD_ZORDER,
    OPENSEES,
    TF_KEYS,
)
from response_variability.style import (  # noqa: E402
    apply_nature_style,
    figsize,
    panel_letter,
    savefig,
)

OUT_DIR = config.RESULTS_DIR / "response_variability"
IID_CACHE = config.CACHE_DIR / "n1000_seed42"
CENTRAL_REC = config.N_LATERAL // 2
N_PANEL = 5
_LETTERS = "abcde"


def methods_in_pack(pack: dict[str, np.ndarray]) -> list[str]:
    return [m for m, key in TF_KEYS.items() if key in pack]


def compare_methods_in(obj) -> list[str]:
    """Candidate methods (excludes OpenSees 2-D), in canonical order."""
    if isinstance(obj, dict):
        present = {m for m, key in TF_KEYS.items() if key in obj and m != OPENSEES}
    else:
        present = {m for m in obj["method"].unique() if m != OPENSEES}
    return [m for m in ALL_METHODS if m in present]


def _curve_at_sample(tf: np.ndarray, i: int) -> np.ndarray:
    a = np.asarray(tf[i], dtype=np.float64)
    if a.ndim == 1:
        return a
    return a[CENTRAL_REC]


def _load_pack(out_dir: Path) -> dict[str, np.ndarray]:
    path = out_dir / "predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Run eval_iid.py first; missing {path}")
    blob = np.load(path, allow_pickle=True)
    return {k: blob[k] for k in blob.files}


def attach_geometry_from_cache(
    pack: dict[str, np.ndarray], cache_dir: Path | None = None
) -> dict[str, np.ndarray]:
    """Add rH / aHV from the nested IID meta if the pack does not already store them."""
    if "rH" in pack and "aHV" in pack:
        return pack
    cache_dir = Path(cache_dir or IID_CACHE)
    meta_path = cache_dir / "meta.npz"
    if not meta_path.is_file() or "local_idx" not in pack:
        return pack
    meta = np.load(meta_path, allow_pickle=True)
    loc = np.asarray(pack["local_idx"], dtype=int)
    out = dict(pack)
    if "rH" in meta.files:
        out["rH"] = np.asarray(meta["rH"][loc], dtype=float)
    if "aHV" in meta.files:
        out["aHV"] = np.asarray(meta["aHV"][loc], dtype=float)
    return out


def _zscore_matrix(pack: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    cols = []
    for key in keys:
        if key not in pack:
            continue
        cols.append(np.asarray(pack[key], dtype=np.float64))
    if not cols:
        raise KeyError(f"none of {keys} present in pack")
    x = np.column_stack(cols)
    mu = x.mean(axis=0)
    sd = np.clip(x.std(axis=0), 1e-8, None)
    return (x - mu) / sd


def select_maximin_indices(
    pack: dict[str, np.ndarray],
    keys: tuple[str, ...],
    n: int = N_PANEL,
    *,
    sort_key: str = "f0",
) -> np.ndarray:
    """Greedy maximin in standardized ``keys``; left-to-right by ``sort_key``."""
    z = _zscore_matrix(pack, keys)
    n_pick = min(int(n), len(z))
    start = int(np.argmin(np.sum(z**2, axis=1)))
    chosen = [start]
    remaining = set(range(len(z))) - {start}
    while len(chosen) < n_pick:
        best_i, best_d = None, -1.0
        for i in remaining:
            d = min(float(np.linalg.norm(z[i] - z[j])) for j in chosen)
            if d > best_d:
                best_d, best_i = d, i
        chosen.append(int(best_i))
        remaining.remove(int(best_i))
    idx = np.asarray(chosen, dtype=int)
    if sort_key in pack:
        idx = idx[np.argsort(np.asarray(pack[sort_key])[idx])]
    return idx


def select_diverse_indices(
    pack: dict[str, np.ndarray], n: int = N_PANEL, *, seed: int = 42
) -> np.ndarray:
    """Spread n samples in (Vs1, H, CoV[, rH, aHV]) via greedy maximin."""
    del seed
    keys = ("vs1", "H", "cov", "rH", "aHV")
    return select_maximin_indices(pack, keys, n=n)


def select_f0_quantile_indices(
    pack: dict[str, np.ndarray], n: int = N_PANEL
) -> np.ndarray:
    """n unique samples nearest the f0 quantiles, sorted by f0."""
    f0 = np.asarray(pack["f0"], dtype=np.float64)
    finite = np.isfinite(f0)
    qs = np.linspace(0.0, 1.0, n)
    targets = np.quantile(f0[finite], qs)
    chosen: list[int] = []
    used: set[int] = set()
    for t in targets:
        order = np.argsort(np.abs(f0 - t))
        for i in order:
            i = int(i)
            if i not in used and finite[i]:
                chosen.append(i)
                used.add(i)
                break
    idx = np.asarray(chosen, dtype=int)
    return idx[np.argsort(f0[idx])]


def select_impedance_indices(
    pack: dict[str, np.ndarray], n: int = N_PANEL
) -> np.ndarray:
    """Spread n samples in (r_H, a_HV, CoV) to check stochastic covering."""
    keys = tuple(k for k in ("rH", "aHV", "cov") if k in pack)
    if len(keys) < 2:
        keys = ("vs1", "H", "cov")
    return select_maximin_indices(pack, keys, n=n)


def _panel_title(pack: dict[str, np.ndarray], i: int) -> str:
    line1 = (
        rf"$V_{{s1}}$={pack['vs1'][i]:.0f} m s$^{{-1}}$, $H$={pack['H'][i]:.0f} m"
    )
    bits = [rf"CoV={pack['cov'][i]:.2f}"]
    if "rH" in pack:
        bits.insert(0, rf"$r_H$={pack['rH'][i]:.0f}")
    if "aHV" in pack:
        bits.insert(1, rf"$a_{{HV}}$={pack['aHV'][i]:.0f}")
    return line1 + "\n" + ", ".join(bits)


def _tf_legend_handles(methods: list[str], *, has_spatial: bool) -> list:
    handles: list = []
    if has_spatial:
        handles.append(
            Patch(
                facecolor=METHOD_COLORS[OPENSEES],
                edgecolor="none",
                alpha=0.18,
                label="OpenSees 2-D 16–84%",
            )
        )
        if GINO in methods:
            handles.append(
                Patch(
                    facecolor=METHOD_COLORS[GINO],
                    edgecolor="none",
                    alpha=0.18,
                    label="GINO 16–84%",
                )
            )
    for method in methods:
        handles.append(
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                ls=METHOD_LINESTYLES[method],
                lw=1.6 if method in (OPENSEES, GINO) else 1.15,
                label=method,
            )
        )
    return handles


def _plot_tf_curve(ax, freq, af, method: str) -> None:
    lw = 1.45 if method in (OPENSEES, GINO) else 1.05
    ax.plot(
        freq,
        af,
        color=METHOD_COLORS[method],
        ls=METHOD_LINESTYLES[method],
        lw=lw,
        zorder=METHOD_ZORDER[method],
    )


def plot_tf_panels(
    pack: dict[str, np.ndarray],
    out_dir: Path,
    *,
    idx: np.ndarray | None = None,
    filename: str = "tf_panels_iid.png",
    note: str | None = None,
) -> Path:
    apply_nature_style()
    pack = attach_geometry_from_cache(pack)
    if idx is None:
        idx = select_diverse_indices(pack)
    idx = np.asarray(idx, dtype=int)
    freq = pack["freq"]
    fig, axes = plt.subplots(
        1, len(idx), figsize=figsize("double", height_mm=115), sharey=True
    )
    if len(idx) == 1:
        axes = [axes]
    tfs = {m: pack[TF_KEYS[m]] for m in methods_in_pack(pack)}
    has_spatial = False
    for ax, i, letter in zip(axes, idx, _LETTERS):
        ops = np.asarray(pack["tf_opensees"][i])
        if ops.ndim == 2:
            has_spatial = True
            p16, _, p84 = spatial_percentiles(ops)
            ax.fill_between(
                freq,
                p16,
                p84,
                color=METHOD_COLORS[OPENSEES],
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
        if GINO in tfs:
            gino = np.asarray(tfs[GINO][i])
            if gino.ndim == 2:
                gp16, _, gp84 = spatial_percentiles(gino)
                ax.fill_between(
                    freq,
                    gp16,
                    gp84,
                    color=METHOD_COLORS[GINO],
                    alpha=0.14,
                    linewidth=0,
                    zorder=1,
                )
        for method in tfs:
            _plot_tf_curve(ax, freq, _curve_at_sample(tfs[method], i), method)
        f0 = float(pack["f0"][i])
        if np.isfinite(f0):
            ax.axvline(f0, color="0.35", ls=":", lw=0.7, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 10.0)
        ax.set_xlabel(r"$f$ (Hz)")
        ax.set_title(_panel_title(pack, i), fontsize=6.5, pad=4, linespacing=1.25)
        panel_letter(ax, letter, x=0.02, y=0.98)
    axes[0].set_ylabel(r"$|\mathrm{TF}|$")
    handles = _tf_legend_handles(list(tfs), has_spatial=has_spatial)
    n_h = len(handles)
    ncol = 4 if n_h >= 6 else min(3, n_h)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.15,
        handletextpad=0.45,
        fontsize=6.5,
    )
    if note:
        fig.text(0.5, -0.10, note, ha="center", va="top", fontsize=6.5, color="0.35")
    fig.tight_layout(w_pad=0.45, rect=(0, 0.18, 1, 1.0))
    return savefig(fig, out_dir / filename)


def plot_tf_panel_variants(
    pack: dict[str, np.ndarray], out_dir: Path
) -> list[Path]:
    """Three sample-covering layouts for the same method overlay."""
    pack = attach_geometry_from_cache(pack)
    layouts = (
        (
            select_diverse_indices(pack),
            "tf_panels_param_span.png",
            r"Maximin covering in $(V_{s1},\,H,\,\mathrm{CoV},\,r_H,\,a_{HV})$",
        ),
        (
            select_f0_quantile_indices(pack),
            "tf_panels_f0.png",
            r"Quantiles of $f_0=V_{s1}/(4H)$ (low $\rightarrow$ high site frequency)",
        ),
        (
            select_impedance_indices(pack),
            "tf_panels_impedance.png",
            r"Maximin covering in $(r_H,\,a_{HV},\,\mathrm{CoV})$",
        ),
    )
    paths = []
    for idx, name, note in layouts:
        paths.append(plot_tf_panels(pack, out_dir, idx=idx, filename=name, note=note))
    paths.append(
        plot_tf_panels(
            pack,
            out_dir,
            idx=layouts[0][0],
            filename="tf_panels_iid.png",
            note=layouts[0][2],
        )
    )
    return paths


def _tick_labels(labels: list[str]) -> list[str]:
    wrap = {
        HASKELL_NOMINAL: "1D Base\nCase",
        HASKELL_COLUMN: "Pretell's\napproach",
    }
    return [wrap.get(lab, lab.replace(" (", "\n(")) for lab in labels]


def _boxplot_with_points(ax, data: list[np.ndarray], labels: list[str]) -> None:
    rng = np.random.default_rng(0)
    ticks = _tick_labels(labels)
    box_kw = dict(
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "0.3", "linewidth": 0.6},
        capprops={"color": "0.3", "linewidth": 0.6},
        boxprops={"linewidth": 0.6},
    )
    try:
        bp = ax.boxplot(data, tick_labels=ticks, **box_kw)
    except TypeError:
        bp = ax.boxplot(data, labels=ticks, **box_kw)
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(METHOD_COLORS[lab])
        patch.set_alpha(0.35)
        patch.set_edgecolor(METHOD_COLORS[lab])
    for i, (vals, lab) in enumerate(zip(data, labels), start=1):
        x = rng.normal(i, 0.06, size=len(vals))
        ax.scatter(
            x,
            vals,
            s=7,
            alpha=0.4,
            c=METHOD_COLORS[lab],
            edgecolors="none",
            zorder=3,
        )
    ax.axhline(0.0, color="0.5", lw=0.5, ls="--", zorder=0)


def _compare_figsize(n_methods: int):
    if n_methods >= 5:
        return figsize("double", height_mm=105)
    return figsize("single", height_mm=100)


def plot_peak_bias(peaks: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    apply_nature_style()
    labels = compare_methods_in(peaks)
    cmp_df = peaks[peaks["method"].isin(labels)]

    fig, ax = plt.subplots(figsize=_compare_figsize(len(labels)))
    data = [cmp_df.loc[cmp_df["method"] == m, "delta_f_peak"].to_numpy() for m in labels]
    _boxplot_with_points(ax, data, labels)
    ax.set_ylabel(r"$\Delta f_{\mathrm{peak}}$ (Hz)")
    ax.set_xlabel("")
    fig.tight_layout()
    p1 = savefig(fig, out_dir / "tf_peak_freq_bias.png")

    fig, ax = plt.subplots(figsize=_compare_figsize(len(labels)))
    data = [
        cmp_df.loc[cmp_df["method"] == m, "delta_ln_A_peak"].to_numpy() for m in labels
    ]
    _boxplot_with_points(ax, data, labels)
    ax.set_ylabel(r"$\Delta \ln A_{\mathrm{peak}}$")
    ax.set_xlabel("")
    fig.tight_layout()
    p2 = savefig(fig, out_dir / "tf_peak_amp_bias.png")
    return p1, p2


def plot_band_misfit(misfit: pd.DataFrame, out_dir: Path) -> Path:
    apply_nature_style()
    labels = compare_methods_in(misfit)
    bands = ["low", "mid", "high"]
    band_labels = ["0.1–0.5 Hz", "0.5–2 Hz", "2–10 Hz"]
    x = np.arange(len(bands), dtype=float)
    n = max(len(labels), 1)
    width = min(0.24, 0.8 / n)
    fig, ax = plt.subplots(figsize=_compare_figsize(n))
    for k, method in enumerate(labels):
        sub = misfit[misfit["method"] == method]
        means = [float(sub[f"rel_l2_{b}"].mean()) for b in bands]
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
    ax.set_ylabel(r"Relative $L_2$ vs OpenSees 2-D")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return savefig(fig, out_dir / "tf_band_misfit.png")


def plot_error_vs_params(summary: pd.DataFrame, out_dir: Path) -> Path:
    apply_nature_style()
    gino = summary[summary["method"] == GINO]
    params = [
        ("Vs1", r"$V_{s1}$ (m s$^{-1}$)"),
        ("H", r"$H$ (m)"),
        ("CoV", r"CoV"),
        ("Vs2", r"$V_{s2}$ (m s$^{-1}$)"),
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=figsize("double", height_mm=145), sharey=True
    )
    for ax, (col, xlab), letter in zip(axes.ravel(), params, "abcd"):
        ax.scatter(
            gino[col],
            gino["gof_af"],
            s=10,
            c=METHOD_COLORS[GINO],
            alpha=0.65,
            edgecolors="none",
        )
        ax.set_xlabel(xlab)
        panel_letter(ax, letter, x=0.02, y=0.98)
    axes[0, 0].set_ylabel("Anderson GOF (ln |TF|)")
    axes[1, 0].set_ylabel("Anderson GOF (ln |TF|)")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    return savefig(fig, out_dir / "tf_error_vs_params.png")


def plot_gof_boxplot(summary: pd.DataFrame, out_dir: Path) -> Path:
    apply_nature_style()
    labels = compare_methods_in(summary)
    fig, ax = plt.subplots(figsize=_compare_figsize(len(labels)))
    data = [summary.loc[summary["method"] == m, "gof_af"].to_numpy() for m in labels]
    _boxplot_with_points(ax, data, labels)
    ax.set_ylabel("Anderson GOF (ln |TF|)")
    ax.set_xlabel("")
    fig.tight_layout()
    return savefig(fig, out_dir / "tf_gof.png")


def _refresh_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map stored CSV labels onto the current display names."""
    df = df.copy()
    df["method"] = df["method"].replace(
        {
            "Haskell (nominal)": HASKELL_NOMINAL,
            "Haskell (column)": HASKELL_COLUMN,
        }
    )
    return df


def plot_all(out_dir: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir or OUT_DIR)
    pack = attach_geometry_from_cache(_load_pack(out_dir))
    summary = _refresh_method_labels(pd.read_csv(out_dir / "method_comparison_summary.csv"))
    peaks = _refresh_method_labels(pd.read_csv(out_dir / "per_sample_peaks.csv"))
    misfit = _refresh_method_labels(pd.read_csv(out_dir / "tf_band_misfit.csv"))
    summary.to_csv(out_dir / "method_comparison_summary.csv", index=False)
    peaks.to_csv(out_dir / "per_sample_peaks.csv", index=False)
    misfit.to_csv(out_dir / "tf_band_misfit.csv", index=False)
    agg_path = out_dir / "aggregate.json"
    if agg_path.is_file():
        import json

        agg = json.loads(agg_path.read_text())
        for old, new in (
            ("Haskell (nominal)", HASKELL_NOMINAL),
            ("Haskell (column)", HASKELL_COLUMN),
        ):
            if old in agg:
                agg[new] = agg.pop(old)
        agg_path.write_text(json.dumps(agg, indent=2))
    paths = plot_tf_panel_variants(pack, out_dir)
    paths.extend(plot_peak_bias(peaks, out_dir))
    paths.append(plot_band_misfit(misfit, out_dir))
    paths.append(plot_error_vs_params(summary, out_dir))
    paths.append(plot_gof_boxplot(summary, out_dir))
    for p in paths:
        print(f"Wrote {p}", flush=True)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    plot_all(args.out_dir)


if __name__ == "__main__":
    main()
