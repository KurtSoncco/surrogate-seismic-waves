#!/usr/bin/env python3
"""Width physics & manifold valley-vs-flat diagnostic PNG plots.

Generates three publication-quality PNGs (and a JSON metrics table) in
`checkpoints/vardomain/width_physics_check/`:

1. `case7_central_tf_width_comparison.png`
   Central-station outcrop |TF(f)| across 500, 1000, 1500, and 4000 m Case 7
   OpenSees solves for all five zones (left: full 0.1-10 Hz log-log;
   right: 0.1-3.5 Hz linear zoom highlighting Mode 1 and Mode 2 shifts).
2. `case7_width_metrics_summary.png`
   Mode-1 log-L1, Mode-2 log-L1, overall 0.1-10 Hz Rel-L1, and overall
   correlations (linear Pearson, Spearman, Lin CCC) vs domain width.
3. `manifold_valley_vs_flat_2d_signal.png`
   Center-station 2D residual |g| across manifold150.csv geometries
   (flat, ramp, step, trough/valley split by CoV=0 vs CoV>0) plus example
   |TF_2D| vs |TF_1D| curves for flat (CoV=0) and trough/valley sections.

Run:
    .venv/bin/python experiments/GIFNO-VARDOMAIN/width_physics_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]
CASE7_DIR = REPO / "experiments" / "case7a"
if str(CASE7_DIR) not in sys.path:
    sys.path.insert(0, str(CASE7_DIR))

from context_convergence import CONTEXT, CROP500, _ttf, load_tf  # noqa: E402
from section_1d import surface_over_outcrop  # noqa: E402

OUT_DIR = REPO / "checkpoints" / "vardomain" / "width_physics_check"
CACHE_ROOT = REPO / "data" / "vardomain" / "cache"
MANIFOLD_CSV = REPO / "experiments" / "GIFNO-VARDOMAIN" / "sections" / "manifold150.csv"

ZONES = [
    ("crop_thin", "Thinning Edge", 800.0),
    ("crop_center", "Center Basin", 0.0),
    ("crop_ramp", "Basin Ramp", 1175.0),
    ("crop_west", "Flat West", -1500.0),
    ("crop_east", "Flat East", 1605.0),
]

WIDTH_STYLES = {
    "4000": dict(color="#111111", lw=2.0, ls="-", label="4000 m (full)", zorder=5),
    "1500": dict(color="#E66100", lw=1.6, ls="-", label="1500 m", zorder=4),
    "1000": dict(color="#1F77B4", lw=1.4, ls="-", label="1000 m", zorder=3),
    "500": dict(color="#7B3294", lw=1.35, ls="--", label="500 m", zorder=2),
}


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x)), float(np.var(y))
    cov = float(np.mean((x - mx) * (y - my)))
    return float(2.0 * cov / (vx + vy + (mx - my) ** 2 + 1e-12))


def detect_modes(freq: np.ndarray, tf: np.ndarray, prominence: float = 2.0) -> tuple[int, int]:
    """Detect 1st and 2nd prominent site resonance modes on a 2D |TF| curve."""
    peaks, _ = find_peaks(tf, distance=15, prominence=prominence)
    if len(peaks) == 0:
        peaks, _ = find_peaks(tf, distance=8, prominence=max(0.2, 0.1 * float(np.max(tf))))
    m1 = int(peaks[0]) if len(peaks) else int(np.argmax(tf))
    cand2 = [int(p) for p in peaks if freq[p] > 1.5 * freq[m1]]
    if cand2:
        m2 = cand2[0]
    elif len(peaks) > 1:
        m2 = int(peaks[1])
    else:
        m2 = m1
    return m1, m2


BANDS_HZ = {"low": (0.1, 0.4), "mid": (0.4, 2.5), "high": (2.5, 10.0)}


def band_dlog(tf_a: np.ndarray, tf_b: np.ndarray, freq: np.ndarray) -> dict:
    """Mean |log| and Pearson of two |TF| fields, overall and in the three bands.

    ``tf_a`` and ``tf_b`` are (station, frequency) and already restricted to the
    same stations. Pearson is over the flattened station-frequency samples.
    """
    a = np.asarray(tf_a, dtype=np.float64)
    b = np.asarray(tf_b, dtype=np.float64)
    freq = np.asarray(freq, dtype=np.float64)

    def _one(mask: np.ndarray) -> dict:
        left, right = a[:, mask], b[:, mask]
        la, lb = np.log(np.maximum(left, 1e-12)), np.log(np.maximum(right, 1e-12))
        if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
            corr = 1.0 if np.allclose(left, right) else 0.0
        else:
            corr = float(pearsonr(left.ravel(), right.ravel())[0])
        return {
            "abs_dlog": float(np.mean(np.abs(la - lb))),
            "pearson": corr,
        }

    out = {"all": _one(np.ones(freq.shape[0], dtype=bool))}
    for name, (lo, hi) in BANDS_HZ.items():
        out[name] = _one((freq >= lo) & (freq <= hi))
    return out


def stations_in(x: np.ndarray, xc: float, half_width: float) -> np.ndarray:
    """Indices of recorders within ``half_width`` metres of ``xc``."""
    return np.flatnonzero(np.abs(np.asarray(x, dtype=np.float64) - float(xc)) <= half_width)


def _on_grid(x: np.ndarray, tf: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a (station, frequency) field onto ``grid`` (metres)."""
    order = np.argsort(x)
    xs, ys = np.asarray(x, dtype=np.float64)[order], np.asarray(tf, dtype=np.float64)[order]
    out = np.empty((grid.size, ys.shape[1]), dtype=np.float64)
    for j in range(ys.shape[1]):
        out[:, j] = np.interp(grid, xs, ys[:, j])
    return out


def paired_band_dlog(rec_a: dict, rec_b: dict, xc: float, half: float, freq_mask: np.ndarray, freq: np.ndarray) -> dict:
    """Band |dlog| and Pearson on the overlap of two recorder lines around ``xc``."""
    xa, xb = np.asarray(rec_a["x"], dtype=np.float64), np.asarray(rec_b["x"], dtype=np.float64)
    lo = max(float(xa.min()), float(xb.min()), xc - half)
    hi = min(float(xa.max()), float(xb.max()), xc + half)
    if hi - lo < 1.0:
        return {}
    grid = np.linspace(lo, hi, max(2, int(round(hi - lo)) + 1))
    ta = _on_grid(xa, rec_a["tf"][:, freq_mask], grid)
    tb = _on_grid(xb, rec_b["tf"][:, freq_mask], grid)
    return band_dlog(ta, tb, freq)


def score_curve(tf_pred: np.ndarray, tf_ref: np.ndarray, freq: np.ndarray, f1: float, f2: float) -> dict:
    la = np.log(np.maximum(tf_pred, 1e-12))
    lb = np.log(np.maximum(tf_ref, 1e-12))
    w1 = (freq >= 0.75 * f1) & (freq <= 1.25 * f1)
    w2 = (freq >= 0.75 * f2) & (freq <= 1.25 * f2)
    return {
        "log_l1": float(np.mean(np.abs(la - lb))),
        "rel_l1": float(np.sum(np.abs(tf_pred - tf_ref)) / np.maximum(np.sum(tf_ref), 1e-12)),
        "rel_l2": float(np.linalg.norm(tf_pred - tf_ref) / np.maximum(np.linalg.norm(tf_ref), 1e-12)),
        "pearson_lin": float(pearsonr(tf_pred, tf_ref)[0]),
        "pearson_log": float(pearsonr(la, lb)[0]),
        "spearman": float(spearmanr(tf_pred, tf_ref)[0]),
        "ccc_log": lins_ccc(la, lb),
        "m1_log_l1": float(np.mean(np.abs(la[w1] - lb[w1]))),
        "m2_log_l1": float(np.mean(np.abs(la[w2] - lb[w2]))),
        "m1_peak": float(np.max(tf_pred[w1])),
        "m2_peak": float(np.max(tf_pred[w2])),
    }


def collect_case7_data() -> tuple[np.ndarray, dict]:
    ttf = _ttf()
    full = load_tf(CONTEXT / "full_qvs.h5", ttf)
    freq_all = full["freq"]
    m10 = (freq_all >= 0.1) & (freq_all <= 10.0)
    freq = freq_all[m10]

    def _get_f12(params: dict) -> tuple[float, float]:
        f1 = params.get("rayleigh_f1", params.get("rayleigh_f1_hz", params.get("f1_hz", params.get("f1", np.nan))))
        f2 = params.get("rayleigh_f2", params.get("rayleigh_f2_hz", params.get("f2_hz", params.get("f2", np.nan))))
        return float(f1), float(f2)

    full_f1, full_f2 = _get_f12(full["params"])

    results = {}
    for slug, label, _ in ZONES:
        c500 = load_tf(CROP500 / f"{slug}.h5", ttf)
        c1000 = load_tf(CONTEXT / f"{slug}_w1000.h5", ttf)
        c1500 = load_tf(CONTEXT / f"{slug}_w1500.h5", ttf)
        xc = float(c500["x"][len(c500["x"]) // 2])

        def center_tf(rec: dict) -> np.ndarray:
            idx = int(np.argmin(np.abs(rec["x"] - xc)))
            return rec["tf"][idx, m10]

        curves = {
            "4000": center_tf(full),
            "1500": center_tf(c1500),
            "1000": center_tf(c1000),
            "500": center_tf(c500),
        }
        fields = {"4000": full, "1500": c1500, "1000": c1000, "500": c500}
        band_rows = {}
        for half, region in ((50.0, "central_100m"), (250.0, "full_500m")):
            band_rows[region] = {}
            for width in ("500", "1000", "1500"):
                band_rows[region][f"{width}_vs_4000"] = paired_band_dlog(fields[width], full, xc, half, m10, freq)
            for left, right in (("500", "1000"), ("1000", "1500"), ("500", "1500")):
                band_rows[region][f"{left}_vs_{right}"] = paired_band_dlog(
                    fields[left], fields[right], xc, half, m10, freq,
                )
        m1_i, m2_i = detect_modes(freq, curves["4000"])
        f1_m, a1_m = float(freq[m1_i]), float(curves["4000"][m1_i])
        f2_m, a2_m = float(freq[m2_i]), float(curves["4000"][m2_i])

        metrics = {w: score_curve(curves[w], curves["4000"], freq, f1_m, f2_m) for w in ("500", "1000", "1500")}
        # Pairwise fixed-(f1,f2) comparison: 500 m & 1000 m vs 1500 m
        metrics_vs_1500 = {w: score_curve(curves[w], curves["1500"], freq, f1_m, f2_m) for w in ("500", "1000")}

        crop_f1, crop_f2 = _get_f12(c500["params"])
        same_rayleigh = (
            (not np.isnan(crop_f1) and not np.isnan(full_f1) and abs(crop_f1 - full_f1) < 1e-3 and abs(crop_f2 - full_f2) < 1e-3)
            or slug == "crop_center"
        )

        results[slug] = {
            "label": label,
            "x_center_m": xc,
            "same_rayleigh_as_4000": same_rayleigh,
            "m1_freq_hz": f1_m,
            "m1_amp": a1_m,
            "m2_freq_hz": f2_m,
            "m2_amp": a2_m,
            "curves": curves,
            "metrics_vs_4000": metrics,
            "metrics_vs_1500": metrics_vs_1500,
            "band_dlog": band_rows,
            "rayleigh_matches_4000": same_rayleigh,
        }
    return freq, results


def plot_case7_central_tfs(freq: np.ndarray, results: dict, out_path: Path) -> None:
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(5, 2, figsize=(13.5, 16.5), gridspec_kw={"width_ratios": [1.15, 1.0]})
    m_zoom = (freq >= 0.1) & (freq <= 3.5)

    for row, (slug, _, _) in enumerate(ZONES):
        info = results[slug]
        f1, f2 = info["m1_freq_hz"], info["m2_freq_hz"]
        ax_log, ax_lin = axes[row]

        # Shade +/-25% windows around Mode 1 and Mode 2
        for ax in (ax_log, ax_lin):
            ax.axvspan(0.75 * f1, 1.25 * f1, color="#FFE6CC", alpha=0.55, zorder=0)
            ax.axvspan(0.75 * f2, 1.25 * f2, color="#DCEEFF", alpha=0.55, zorder=0)
            ax.axvline(f1, color="#B35900", lw=0.9, ls=":", zorder=1)
            ax.axvline(f2, color="#1F5F99", lw=0.9, ls=":", zorder=1)

        for w in ("500", "1000", "1500", "4000"):
            st = WIDTH_STYLES[w]
            ax_log.loglog(freq, info["curves"][w], **st)
            ax_lin.plot(freq[m_zoom], info["curves"][w][m_zoom], **st)

        # Titles & labels
        tag = " [identical Rayleigh (f1, f2) as 4000 m]" if info["same_rayleigh_as_4000"] else " [500/1000/1500 m share fixed (f1, f2)]"
        ax_log.set_title(f"{info['label']} (x = {info['x_center_m']:.0f} m){tag}", loc="left", fontweight="bold")
        ax_log.set_xlim(0.1, 10.0)
        ax_log.set_ylim(0.15, 22.0)
        ax_log.set_ylabel("Outcrop |TF| (log)")
        ax_log.grid(True, which="both", alpha=0.25, lw=0.6)

        ymax = max(17.5, info["m1_amp"] * 1.28)
        ax_lin.set_title(
            f"Mode 1: {f1:.2f} Hz (|TF|={info['m1_amp']:.1f})  |  Mode 2: {f2:.2f} Hz (|TF|={info['m2_amp']:.1f})",
            loc="left",
        )
        ax_lin.set_xlim(0.1, 3.5)
        ax_lin.set_ylim(0.0, ymax)
        ax_lin.set_ylabel("Outcrop |TF| (linear)")
        ax_lin.grid(True, alpha=0.25, lw=0.6)

        # Annotate M1 near top and M2 just above its peak (avoiding top-right metric box)
        ax_lin.text(f1, ymax * 0.94, "M1", color="#8A4200", ha="center", va="top", fontsize=8.5, fontweight="bold")
        ax_lin.text(
            f2 - 0.08, min(info["m2_amp"] + 1.2, ymax * 0.72), "M2",
            color="#164A7A", ha="right", va="bottom", fontsize=8.5, fontweight="bold",
        )

        # Compact metric summary box inside linear subplot
        m5 = info["metrics_vs_4000"]["500"]
        m10 = info["metrics_vs_4000"]["1000"]
        m15 = info["metrics_vs_4000"]["1500"]
        box_txt = (
            "vs 4000 m (0.1–10 Hz):\n"
            f" 500 m: RelL1={m5['rel_l1']:.3f}  ρ_s={m5['spearman']:.3f}  M1={m5['m1_log_l1']:.3f}\n"
            f"1000 m: RelL1={m10['rel_l1']:.3f}  ρ_s={m10['spearman']:.3f}  M1={m10['m1_log_l1']:.3f}\n"
            f"1500 m: RelL1={m15['rel_l1']:.3f}  ρ_s={m15['spearman']:.3f}  M1={m15['m1_log_l1']:.3f}"
        )
        ax_lin.text(
            0.985, 0.96, box_txt, transform=ax_lin.transAxes, ha="right", va="top",
            fontsize=7.6, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
        )

        if row == 0:
            ax_log.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, ncol=2)
        if row == 4:
            ax_log.set_xlabel("Frequency (Hz)")
            ax_lin.set_xlabel("Frequency (Hz)")

    fig.suptitle(
        "Case 7 Central-Station |TF(f)| vs Domain Vs Width (500 m, 1000 m, 1500 m, 4000 m OpenSees)\n"
        "Shaded bands mark ±25% around 4000 m Mode 1 (orange) and Mode 2 (blue)",
        fontsize=12.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.972])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_case7_metrics_summary(results: dict, out_path: Path) -> None:
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    zone_colors = {
        "crop_thin": "#D95F02",
        "crop_center": "#1B9E77",
        "crop_ramp": "#7570B3",
        "crop_west": "#E7298A",
        "crop_east": "#2B6CB0",
    }
    widths = [500, 1000, 1500]
    wkeys = ["500", "1000", "1500"]

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8))

    # (a) Mode 1 log-L1 vs width
    ax = axes[0, 0]
    for slug, label, _ in ZONES:
        vals = [results[slug]["metrics_vs_4000"][wk]["m1_log_l1"] for wk in wkeys]
        ax.plot(widths, vals, marker="o", lw=2.0, color=zone_colors[slug], label=label)
    ax.set_title("(a) Mode 1 (±25%) log-L1 vs 4000 m", fontweight="bold", loc="left")
    ax.set_xticks(widths)
    ax.set_xlabel("Domain interior width (m)")
    ax.set_ylabel("Mode 1 log-L1 (lower is better)")
    ax.set_ylim(0.0, 0.15)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    # (b) Mode 2 log-L1 vs width
    ax = axes[0, 1]
    for slug, label, _ in ZONES:
        vals = [results[slug]["metrics_vs_4000"][wk]["m2_log_l1"] for wk in wkeys]
        ax.plot(widths, vals, marker="s", lw=2.0, color=zone_colors[slug], label=label)
    ax.set_title("(b) Mode 2 (±25%) log-L1 vs 4000 m", fontweight="bold", loc="left")
    ax.set_xticks(widths)
    ax.set_xlabel("Domain interior width (m)")
    ax.set_ylabel("Mode 2 log-L1 (lower is better)")
    ax.set_ylim(0.0, 0.15)
    ax.grid(True, alpha=0.28)
    ax.text(
        0.04, 0.88,
        "Mode 2 is already ~converged at 500 m (0.01–0.03),\nwhereas Mode 1 drops 2.3×–6.3× from 500 → 1500 m.",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#CCCCCC"),
    )

    # (c) Overall 0.1-10 Hz Rel-L1 (linear |TF|) vs width
    ax = axes[1, 0]
    for slug, label, _ in ZONES:
        vals = [results[slug]["metrics_vs_4000"][wk]["rel_l1"] for wk in wkeys]
        ax.plot(widths, vals, marker="D", lw=2.0, color=zone_colors[slug], label=label)
    ax.set_title("(c) Overall 0.1–10 Hz Rel-L1 (linear |TF|) vs 4000 m", fontweight="bold", loc="left")
    ax.set_xticks(widths)
    ax.set_xlabel("Domain interior width (m)")
    ax.set_ylabel("Rel-L1 = Σ|ΔTF| / Σ|TF_ref|")
    ax.set_ylim(0.0, 0.15)
    ax.grid(True, alpha=0.28)

    # (d) Grouped bars: 500 m vs 1500 m correlations over full 0.1-10 Hz
    ax = axes[1, 1]
    x = np.arange(len(ZONES))
    bw = 0.25
    sp500 = [results[s]["metrics_vs_4000"]["500"]["spearman"] for s, _, _ in ZONES]
    sp1500 = [results[s]["metrics_vs_4000"]["1500"]["spearman"] for s, _, _ in ZONES]
    pr500 = [results[s]["metrics_vs_4000"]["500"]["pearson_lin"] for s, _, _ in ZONES]
    pr1500 = [results[s]["metrics_vs_4000"]["1500"]["pearson_lin"] for s, _, _ in ZONES]

    ax.bar(x - 1.5 * bw, pr500, width=bw * 0.9, color="#B2ABD2", label="Pearson (lin) 500 m")
    ax.bar(x - 0.5 * bw, pr1500, width=bw * 0.9, color="#5E3C99", label="Pearson (lin) 1500 m")
    ax.bar(x + 0.5 * bw, sp500, width=bw * 0.9, color="#FDB863", label="Spearman 500 m")
    ax.bar(x + 1.5 * bw, sp1500, width=bw * 0.9, color="#E66100", label="Spearman 1500 m")

    ax.set_title("(d) Overall 0.1–10 Hz Correlations (500 m vs 1500 m)", fontweight="bold", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.replace(" ", "\n") for _, lbl, _ in ZONES], fontsize=8.5)
    ax.set_ylabel("Correlation vs 4000 m")
    ax.set_ylim(0.80, 1.015)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, ncol=2, loc="lower right", fontsize=8.0)

    fig.suptitle(
        "Domain Width Sensitivity Summary Across Case 7 Central Recordings (0.1–10 Hz)",
        fontsize=12.5, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_manifold_valley_vs_flat(out_path: Path) -> dict:
    freq = np.load(CACHE_ROOT / "freq.npy")
    df = pd.read_csv(MANIFOLD_CSV)
    fmap = {
        str(r["sample_id"]): (str(r["feature_name"]), str(r["stratum"]), float(r["vs_cov"]))
        for _, r in df.iterrows()
    }

    order = [
        ("flat", False, "Flat\n(CoV=0)"),
        ("flat", True, "Flat\n(CoV>0)"),
        ("ramp", False, "Ramp\n(CoV=0)"),
        ("ramp", True, "Ramp\n(CoV>0)"),
        ("step", False, "Step\n(CoV=0)"),
        ("step", True, "Step\n(CoV>0)"),
        ("trough", False, "Valley/Trough\n(CoV=0)"),
        ("trough", True, "Valley/Trough\n(CoV>0)"),
    ]
    buckets: dict[tuple[str, bool], list[dict]] = {(f, c): [] for f, c, _ in order}

    for p in sorted((CACHE_ROOT / "sections").glob("*.npz")):
        sid = p.stem.split("__")[-1]
        if sid not in fmap:
            continue
        fname, stratum, vcov = fmap[sid]
        key = (fname, vcov > 0.0)
        if key not in buckets:
            continue
        d = np.load(p, allow_pickle=True)
        log_tf = d["log_tf"]
        log_tf1d = d["log_tf1d"]
        c = log_tf.shape[0] // 2
        tf2d = np.exp(log_tf[c])
        tf1d = np.exp(log_tf1d[c])
        gc = log_tf[c] - log_tf1d[c]
        m1_i, m2_i = detect_modes(freq, tf2d, prominence=0.35)
        f1_m, f2_m = float(freq[m1_i]), float(freq[m2_i])
        w1 = (freq >= 0.75 * f1_m) & (freq <= 1.25 * f1_m)
        w2 = (freq >= 0.75 * f2_m) & (freq <= 1.25 * f2_m)
        buckets[key].append({
            "sid": sid,
            "all_g": float(np.mean(np.abs(gc))),
            "m1_g": float(np.mean(np.abs(gc[w1]))),
            "m2_g": float(np.mean(np.abs(gc[w2]))),
            "tf2d": tf2d,
            "tf1d": tf1d,
            "f1": f1_m,
            "f2": f2_m,
        })

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.40, wspace=0.28)
    ax_bar = fig.add_subplot(gs[0, :])

    x = np.arange(len(order))
    bw = 0.25
    all_means, m1_means, m2_means, labels_n = [], [], [], []
    summary = {}
    for fname, has_cov, lbl in order:
        rows = buckets[(fname, has_cov)]
        a = np.array([r["all_g"] for r in rows])
        m1 = np.array([r["m1_g"] for r in rows])
        m2 = np.array([r["m2_g"] for r in rows])
        all_means.append(float(a.mean()))
        m1_means.append(float(m1.mean()))
        m2_means.append(float(m2.mean()))
        labels_n.append(f"{lbl}\n(n={len(rows)})")
        summary[f"{fname}_{'cov' if has_cov else 'cov0'}"] = {
            "n": len(rows),
            "all_g_mean": round(float(a.mean()), 4),
            "m1_g_mean": round(float(m1.mean()), 4),
            "m2_g_mean": round(float(m2.mean()), 4),
        }

    b1 = ax_bar.bar(x - bw, all_means, width=bw, color="#4A5568", label="Overall 0.1–10 Hz |g|")
    b2 = ax_bar.bar(x, m1_means, width=bw, color="#E66100", label="Mode 1 (±25%) |g|")
    b3 = ax_bar.bar(x + bw, m2_means, width=bw, color="#2B6CB0", label="Mode 2 (±25%) |g|")

    # Highlight the two opposing anchors
    ax_bar.axvspan(-0.48, 0.48, color="#EDF2F7", alpha=0.75, zorder=0)
    ax_bar.axvspan(5.52, 7.48, color="#FFF3E0", alpha=0.75, zorder=0)

    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            ax_bar.text(
                rect.get_x() + rect.get_width() / 2.0, h + 0.008, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7.6,
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels_n, fontsize=8.8)
    ax_bar.set_ylabel("Mean center-station |g| = |log(TF_2D / TF_1D)|")
    ax_bar.set_ylim(0.0, 0.58)
    ax_bar.grid(True, axis="y", alpha=0.25)
    ax_bar.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, ncol=3)
    ax_bar.set_title(
        "(a) Why BOTH Flat (CoV = 0) and Valley/Trough Domains Are Required (manifold150.csv, n = 301 sections)",
        loc="left", fontweight="bold",
    )
    ax_bar.text(0.0, 0.43, "Zero-residual\nanchor (g → 0)", ha="center", va="top", fontsize=8.3, color="#2D3748", fontweight="bold")
    ax_bar.text(6.5, 0.555, "Strongest 2D basin physics (18×–28× Mode-1 |g| vs Flat CoV=0)", ha="center", va="top", fontsize=8.4, color="#B35900", fontweight="bold")

    # Bottom row: representative center-station |TF_2D| vs |TF_1D| examples
    def pick_median(key: tuple[str, bool], quantile: float = 0.65) -> dict:
        rows = sorted(buckets[key], key=lambda r: r["m1_g"])
        return rows[int(quantile * (len(rows) - 1))]

    examples = [
        (pick_median(("flat", False), 0.50), "(b) Flat (CoV=0): 2D ≈ 1D anchor"),
        (pick_median(("trough", False), 0.70), "(c) Valley/Trough (CoV=0): Pure basin geometry"),
        (pick_median(("trough", True), 0.70), "(d) Valley/Trough (CoV>0): Basin + heterogeneity"),
    ]
    for col, (ex, title) in enumerate(examples):
        ax = fig.add_subplot(gs[1, col])
        ax.loglog(freq, ex["tf1d"], color="#4A5568", lw=1.6, ls="--", label="Exact 1D column")
        ax.loglog(freq, ex["tf2d"], color="#E66100", lw=1.9, ls="-", label="2D OpenSees")
        ax.fill_between(freq, np.minimum(ex["tf1d"], ex["tf2d"]), np.maximum(ex["tf1d"], ex["tf2d"]), color="#FDB863", alpha=0.25)
        ax.set_title(f"{title}\n({ex['sid']}  |  all |g|={ex['all_g']:.3f}, M1 |g|={ex['m1_g']:.3f})", loc="left", fontsize=9.2, fontweight="bold")
        ax.set_xlim(0.1, 10.0)
        ax.set_ylim(0.15, 25.0)
        ax.set_xlabel("Frequency (Hz)")
        if col == 0:
            ax.set_ylabel("Outcrop |TF|")
        ax.grid(True, which="both", alpha=0.25, lw=0.6)
        ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.0)

    fig.suptitle(
        "Manifold Geometry 2D Residual Strength: Flat (CoV = 0) vs Valley/Trough Basins",
        fontsize=12.5, fontweight="bold", y=0.985,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


WIDTH_COLORS = {500: "#7B3294", 1000: "#1F77B4", 1500: "#E66100", 2000: "#111111"}
FEATURES = ("flat", "ramp", "step")


def _band(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.median(stack, axis=0), np.percentile(stack, 20, axis=0), np.percentile(stack, 80, axis=0)


def training_width_audit() -> dict:
    """Central-station labels of the training cache, grouped by the width that was actually solved.

    Manifold rows are independent draws. A group_id never appears at two widths, so a
    difference between the 500 m and 2000 m medians is a difference between sites, not
    the same geology re-solved on a wider mesh. IID and the OOD campaigns store one
    500 m strip; Stage 1 changes the canvas width and keeps this same central curve.
    """
    freq = np.load(CACHE_ROOT / "freq.npy")
    table = pd.read_csv(MANIFOLD_CSV)
    meta = {str(r["sample_id"]): r for _, r in table.iterrows()}
    by_feat: dict[tuple[str, int], list[dict]] = {}
    group_widths: dict[str, set[int]] = {}
    for path in sorted((CACHE_ROOT / "sections").glob("*.npz")):
        sid = path.stem.split("__")[-1]
        row = meta.get(sid)
        if row is None or str(row["split"]) != "train":
            continue
        feat = str(row["feature_name"])
        if feat == "trough":
            continue
        width = int(row["lx_m"])
        group = str(row["group_id"])
        group_widths.setdefault(group, set()).add(width)
        blob = np.load(path, allow_pickle=True)
        log_tf = np.asarray(blob["log_tf"], dtype=np.float64)
        log_1d = np.asarray(blob["log_tf1d"], dtype=np.float64)
        center = log_tf.shape[0] // 2
        tf = np.exp(log_tf[center])
        g = log_tf[center] - log_1d[center]
        m1, m2 = detect_modes(freq, tf, prominence=0.35)
        by_feat.setdefault((feat, width), []).append({
            "sid": sid,
            "tf": tf,
            "tf1d": np.exp(log_1d[center]),
            "g": g,
            "abs_g": float(np.mean(np.abs(g))),
            "m1_hz": float(freq[m1]),
            "m1_amp": float(tf[m1]),
            "m2_hz": float(freq[m2]),
            "m2_amp": float(tf[m2]),
            "f0_hz": float(row["f0_flat_hz"]),
            "feature_width_m": float(row["feature_width_m"]),
            "stratum": str(row["stratum"]),
            "field_seed": int(row["field_seed"]),
            "damping": str(row["damping_family"]),
        })

    multi = {g: sorted(ws) for g, ws in group_widths.items() if len(ws) > 1}
    counts = {
        f"{feat}_w{width}": len(rows)
        for (feat, width), rows in sorted(by_feat.items())
    }
    population = {}
    for (feat, width), rows in sorted(by_feat.items()):
        population[f"{feat}_w{width}"] = {
            "n": len(rows),
            "median_abs_g": round(float(np.median([r["abs_g"] for r in rows])), 4),
            "median_m1_hz": round(float(np.median([r["m1_hz"] for r in rows])), 3),
            "median_m1_amp": round(float(np.median([r["m1_amp"] for r in rows])), 2),
            "median_feature_width_m": round(float(np.median([r["feature_width_m"] for r in rows])), 1),
            "median_f0_hz": round(float(np.median([r["f0_hz"] for r in rows])), 3),
        }

    iid_center = _iid_central_curves(freq)
    return {
        "freq": freq,
        "by_feat": by_feat,
        "n_train_sections": int(sum(len(v) for v in by_feat.values())),
        "n_group_ids": len(group_widths),
        "n_group_ids_at_two_widths": len(multi),
        "multi_width_groups": multi,
        "counts": counts,
        "population": population,
        "iid_center": iid_center,
    }


def _iid_central_curves(freq: np.ndarray, n_show: int = 24) -> dict:
    """Center recorder of training IID runs. There is one stored curve per run."""
    sys.path.insert(0, str(REPO / "experiments" / "GIFNO-VARDOMAIN"))
    from data import Cache

    cache = Cache(CACHE_ROOT)
    curves = []
    ids = cache.runs("train", ("iid",))
    take = ids[:n_show] if ids.size else ids
    center_x = None
    for flat in take:
        camp, j = cache.campaign(int(flat))
        x = np.asarray(camp.rec_x, dtype=np.float64)
        k = int(np.argmin(np.abs(x - np.median(x))))
        center_x = float(x[k])
        curves.append(np.exp(np.asarray(camp.log_tf[j, k], dtype=np.float64)))
    return {
        "n_train": int(ids.size),
        "n_shown": len(curves),
        "center_x_m": center_x,
        "curves": curves,
        "freq": freq,
    }


def hysteretic_outcrop(freq: np.ndarray, vs_col: np.ndarray, zeta_col: np.ndarray) -> np.ndarray:
    """Thomson–Haskell outcrop |TF| with the layer damping held fixed in frequency.

    The stored prior uses the same propagator with Rayleigh xi(f). This is the
    classical hysteretic column: xi equals the applied zeta at every frequency.
    """
    zeta = np.asarray(zeta_col, dtype=np.float64)
    xi = np.broadcast_to(zeta[:, None], (zeta.size, freq.size))
    return surface_over_outcrop(freq, np.asarray(vs_col, dtype=np.float64), dz=1.0, xi=xi)


def _center_column(blob: np.lib.npyio.NpzFile) -> tuple[int, int, np.ndarray, np.ndarray]:
    x = np.asarray(blob["x_m"], dtype=np.float64)
    center = int(blob["log_tf"].shape[0] // 2)
    col = int(np.clip(np.round(x[center]), 0, blob["vs"].shape[1] - 1))
    vs = np.asarray(blob["vs"][:, col], dtype=np.float64)
    zeta = np.asarray(blob["zeta"][:, col], dtype=np.float64)
    return center, col, vs, zeta


def one_section_per_bin(audit: dict) -> dict[tuple[str, int], dict]:
    """One training section per feature and width.

    No field_seed is solved at two widths, so the four widths of a feature are
    different sites. The section kept is the one whose center |g| is nearest
    the bin median.
    """
    freq = audit["freq"]
    chosen = {}
    for feat in FEATURES:
        for width in (500, 1000, 1500, 2000):
            rows = audit["by_feat"].get((feat, width), [])
            if not rows:
                continue
            rec = dict(_representative(rows))
            blob = np.load(CACHE_ROOT / "sections" / f"manifold__{rec['sid']}.npz")
            _, _, vs, zeta = _center_column(blob)
            tf_h = hysteretic_outcrop(freq, vs, zeta)
            m1, m2 = rec["m1_hz"], rec["m2_hz"]
            rec["tf_h"] = tf_h
            rec["n_bin"] = len(rows)
            rec["vs_stored"] = score_curve(rec["tf"], rec["tf1d"], freq, m1, m2)
            rec["vs_hysteretic"] = score_curve(rec["tf"], tf_h, freq, m1, m2)
            rec["abs_g_h"] = float(np.mean(np.abs(np.log(np.maximum(rec["tf"], 1e-12)) - np.log(np.maximum(tf_h, 1e-12)))))
            chosen[(feat, width)] = rec
    return chosen


def population_prior_errors(audit: dict) -> dict[str, dict]:
    """Median Case-7 scores of every training section against both 1D priors."""
    freq = audit["freq"]
    out = {}
    for (feat, width), rows in sorted(audit["by_feat"].items()):
        stored, hyst = [], []
        for rec in rows:
            blob = np.load(CACHE_ROOT / "sections" / f"manifold__{rec['sid']}.npz")
            _, _, vs, zeta = _center_column(blob)
            tf_h = hysteretic_outcrop(freq, vs, zeta)
            stored.append(score_curve(rec["tf"], rec["tf1d"], freq, rec["m1_hz"], rec["m2_hz"]))
            hyst.append(score_curve(rec["tf"], tf_h, freq, rec["m1_hz"], rec["m2_hz"]))
        def _med(items: list[dict], key: str) -> float:
            return float(np.median([item[key] for item in items]))
        keys = ("log_l1", "rel_l1", "pearson_lin", "pearson_log", "spearman", "ccc_log", "m1_log_l1", "m2_log_l1")
        out[f"{feat}_w{width}"] = {
            "n": len(rows),
            "stored": {k: round(_med(stored, k), 4) for k in keys},
            "hysteretic": {k: round(_med(hyst, k), 4) for k in keys},
        }
    return out


def plot_one_section_tfs(audit: dict, chosen: dict[tuple[str, int], dict], out_path: Path) -> None:
    """One central |TF| per feature and width, with both 1D priors."""
    freq = audit["freq"]
    fig, axes = plt.subplots(len(FEATURES), 4, figsize=(15.2, 11.4), sharex=True)
    for row, feat in enumerate(FEATURES):
        for col, width in enumerate((500, 1000, 1500, 2000)):
            ax = axes[row, col]
            rec = chosen.get((feat, width))
            if rec is None:
                ax.set_axis_off()
                ax.set_title(f"{feat}, {width} m\nno training section")
                continue
            ax.axvspan(0.75 * rec["m1_hz"], 1.25 * rec["m1_hz"], color="#FFE6CC", alpha=0.7, zorder=0)
            ax.axvspan(0.75 * rec["m2_hz"], 1.25 * rec["m2_hz"], color="#DCEEFF", alpha=0.55, zorder=0)
            ax.loglog(freq, rec["tf1d"], color="#111111", lw=1.5, ls="--", label="Rayleigh 1D (stored)")
            ax.loglog(freq, rec["tf_h"], color="#2B6CB0", lw=1.4, ls=":", label="Hysteretic Haskell–Thomson")
            ax.loglog(freq, rec["tf"], color="#E66100", lw=1.8, label="2D OpenSees")
            st = rec["vs_stored"]
            hy = rec["vs_hysteretic"]
            count = rec["n_bin"]
            noun = "section" if count == 1 else "sections"
            ax.set_title(
                f"{feat}, {width} m  ·  {rec['sid']}  ·  seed {rec['field_seed']}\n"
                f"1 of {count} {noun}   log-L1  {st['log_l1']:.3f} Rayleigh,  {hy['log_l1']:.3f} hysteretic",
                loc="left", fontsize=8.4,
            )
            ax.set_xlim(0.1, 10.0)
            ax.set_ylim(0.2, 30.0)
            ax.grid(True, which="both", alpha=0.25, lw=0.6)
            if row == len(FEATURES) - 1:
                ax.set_xlabel("Frequency (Hz)")
            if col == 0:
                ax.set_ylabel("Center |TF|")
            if row == 0 and col == 0:
                ax.legend(loc="lower left", frameon=True, fontsize=7.4)
    fig.suptitle(
        "One training section per panel. No field seed is solved at two widths, so columns are different sites.\n"
        "Shaded bands are ±25% around the 2D mode peaks. The stored 1D is already a Thomson–Haskell outcrop with Rayleigh damping.",
        fontsize=11.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_one_section_errors(chosen: dict[tuple[str, int], dict], population: dict[str, dict], out_path: Path) -> None:
    """Case-7 error panels for the plotted section, with the bin-median error as a dot."""
    metrics = (
        ("log_l1", "log-L1 of log|TF|", False),
        ("rel_l1", "Rel-L1 of |TF|", False),
        ("pearson_log", "Pearson of log|TF|", True),
        ("spearman", "Spearman of |TF|", True),
        ("m1_log_l1", "Mode-1 log-L1 (±25%)", False),
        ("m2_log_l1", "Mode-2 log-L1 (±25%)", False),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.2))
    labels = [f"{feat[:1].upper()}\n{width}" for feat in FEATURES for width in (500, 1000, 1500, 2000)]
    x = np.arange(len(labels))
    bw = 0.36
    for ax, (key, title, higher_better) in zip(axes.ravel(), metrics):
        stored = [chosen[(f, w)]["vs_stored"][key] for f in FEATURES for w in (500, 1000, 1500, 2000)]
        hyst = [chosen[(f, w)]["vs_hysteretic"][key] for f in FEATURES for w in (500, 1000, 1500, 2000)]
        med_s = [population[f"{f}_w{w}"]["stored"][key] for f in FEATURES for w in (500, 1000, 1500, 2000)]
        med_h = [population[f"{f}_w{w}"]["hysteretic"][key] for f in FEATURES for w in (500, 1000, 1500, 2000)]
        ax.bar(x - bw / 2, stored, width=bw, color="#111111", label="vs Rayleigh 1D")
        ax.bar(x + bw / 2, hyst, width=bw, color="#2B6CB0", label="vs hysteretic Haskell–Thomson")
        ax.scatter(x - bw / 2, med_s, s=16, color="#F6E05E", zorder=4, label="bin median, Rayleigh")
        ax.scatter(x + bw / 2, med_h, s=16, color="#F6AD55", zorder=4, label="bin median, hysteretic")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.2)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.28)
        if higher_better:
            ax.set_ylim(0.50, 1.02)
        if key == "log_l1":
            ax.legend(frameon=True, fontsize=7.2, loc="upper right")
        for boundary in (3.5, 7.5):
            ax.axvline(boundary, color="#CBD5E0", lw=0.8)
    fig.suptitle(
        "2D versus 1D at the center station. Bars are the single plotted section; dots are the median over that feature and width.\n"
        "Tick labels are feature (F, R, S) and solved width. Mode windows follow the 2D peaks.",
        fontsize=11.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_training_central_tfs(audit: dict, out_path: Path) -> None:
    freq = audit["freq"]
    fig, axes = plt.subplots(len(FEATURES), 2, figsize=(13.2, 14.5), sharex=True)
    for row, feat in enumerate(FEATURES):
        ax_tf, ax_g = axes[row]
        for width in (500, 1000, 1500, 2000):
            rows = audit["by_feat"].get((feat, width), [])
            if not rows:
                continue
            color = WIDTH_COLORS[width]
            stack = np.stack([r["tf"] for r in rows])
            gstack = np.stack([np.abs(r["g"]) for r in rows])
            if len(rows) < 8:
                for curve, gcurve in zip(stack, gstack):
                    ax_tf.plot(freq, curve, color=color, lw=1.0, alpha=0.85)
                    ax_g.plot(freq, gcurve, color=color, lw=1.0, alpha=0.85)
                ax_tf.plot([], [], color=color, lw=1.2, label=f"{width} m ({len(rows)} section{'s' if len(rows) != 1 else ''})")
                ax_g.plot([], [], color=color, lw=1.2, label=f"{width} m ({len(rows)} section{'s' if len(rows) != 1 else ''})")
                continue
            med, lo, hi = _band(stack)
            gmed, glo, ghi = _band(gstack)
            ax_tf.fill_between(freq, lo, hi, color=color, alpha=0.16, linewidth=0)
            ax_tf.plot(freq, med, color=color, lw=2.0, label=f"{width} m ({len(rows)} sections)")
            ax_g.fill_between(freq, glo, ghi, color=color, alpha=0.16, linewidth=0)
            ax_g.plot(freq, gmed, color=color, lw=2.0, label=f"{width} m")
        ax_tf.set_yscale("log")
        ax_tf.set_xlim(0.1, 10.0)
        ax_tf.set_ylim(0.2, 30.0)
        ax_tf.set_xscale("log")
        ax_tf.grid(True, which="both", alpha=0.25, lw=0.6)
        ax_tf.set_ylabel("Center |TF|")
        ax_tf.set_title(f"{feat}: median central |TF| of training sections", loc="left", fontweight="bold")
        ax_g.set_xlim(0.1, 10.0)
        ax_g.set_ylim(0.0, 1.15)
        ax_g.set_xscale("log")
        ax_g.grid(True, which="both", alpha=0.25, lw=0.6)
        ax_g.set_ylabel("Center |g|")
        ax_g.set_title(f"{feat}: median |log TF_2D − log TF_1D|", loc="left", fontweight="bold")
        if row == 0:
            ax_tf.legend(loc="lower left", frameon=True, fontsize=8)
            ax_g.legend(loc="upper right", frameon=True, fontsize=8)
        if row == len(FEATURES) - 1:
            ax_tf.set_xlabel("Frequency (Hz)")
            ax_g.set_xlabel("Frequency (Hz)")
    fig.suptitle(
        f"Training central labels by solved width: {audit['n_train_sections']} sections, "
        f"{audit['n_group_ids_at_two_widths']} group ids at two widths",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_training_width_examples(audit: dict, out_path: Path) -> None:
    """A few actual central recordings, one panel per width, for flat and ramp."""
    freq = audit["freq"]
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.2), sharex=True, sharey=True)
    for row, feat in enumerate(("flat", "ramp")):
        for col, width in enumerate((500, 1000, 1500, 2000)):
            ax = axes[row, col]
            rows = sorted(audit["by_feat"].get((feat, width), []), key=lambda r: r["abs_g"])
            if not rows:
                ax.set_title(f"{feat}, {width} m\nno training rows")
                ax.set_axis_off()
                continue
            pick = rows[:: max(1, len(rows) // 6)][:6]
            for rec in pick:
                ax.plot(freq, rec["tf"], color=WIDTH_COLORS[width], lw=1.0, alpha=0.85)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.1, 10.0)
            ax.set_ylim(0.2, 30.0)
            ax.grid(True, which="both", alpha=0.25, lw=0.6)
            ax.set_title(f"{feat}, {width} m solved ({len(rows)} section{'s' if len(rows) != 1 else ''})", loc="left", fontsize=9.5)
            if row == 1:
                ax.set_xlabel("Frequency (Hz)")
            if col == 0:
                ax.set_ylabel("Center |TF|")
    fig.suptitle(
        "Training central recordings. Each curve is a different section. "
        "No section was solved again at another width.",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _representative(rows: list[dict]) -> dict:
    """The training section whose center |g| is closest to the bin median."""
    target = float(np.median([r["abs_g"] for r in rows]))
    return min(rows, key=lambda r: abs(r["abs_g"] - target))


def plot_training_vs_maps(audit: dict, out_path: Path) -> None:
    """One Vs cross-section per feature and solved width. Columns are different sites."""
    chosen: dict[tuple[str, int], dict] = {}
    vmax = 1.0
    for feat in FEATURES:
        for width in (500, 1000, 1500, 2000):
            rows = audit["by_feat"].get((feat, width), [])
            if not rows:
                continue
            rec = _representative(rows)
            path = CACHE_ROOT / "sections" / f"manifold__{rec['sid']}.npz"
            vs = np.asarray(np.load(path)["vs"], dtype=np.float64)
            rec = dict(rec)
            rec["vs"] = vs
            rec["n_bin"] = len(rows)
            chosen[(feat, width)] = rec
            vmax = max(vmax, float(np.percentile(vs, 99)))

    fig, axes = plt.subplots(len(FEATURES), 4, figsize=(14.6, 11.2), gridspec_kw={"wspace": 0.08, "hspace": 0.38})
    image = None
    for row, feat in enumerate(FEATURES):
        for col, width in enumerate((500, 1000, 1500, 2000)):
            ax = axes[row, col]
            rec = chosen.get((feat, width))
            if rec is None:
                ax.set_axis_off()
                ax.set_title(f"{feat}, {width} m\nno training section")
                continue
            vs = rec["vs"]
            depth, nx = vs.shape
            image = ax.imshow(
                vs, origin="upper", aspect="auto", cmap="cividis",
                vmin=100.0, vmax=vmax, interpolation="nearest",
                extent=(0.0, float(nx), float(depth), 0.0),
            )
            ax.axvline(nx / 2.0, color="white", lw=0.8, ls="--")
            count = rec["n_bin"]
            noun = "section" if count == 1 else "sections"
            ax.set_title(
                f"{feat}, {width} m domain\n{rec['sid']} · 1 of {count} {noun}",
                loc="left", fontsize=9,
            )
            if row == len(FEATURES) - 1:
                ax.set_xlabel("Distance (m)")
            else:
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel("Depth (m)")
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")
    fig.suptitle(
        "Training Vs of one section per row and width. The dashed line is the center station. "
        "A wider column is a different section, not the same geology extended.",
        fontsize=11.5, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.06)
    if image is not None:
        cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
        bar = fig.colorbar(image, cax=cax)
        bar.set_label("Vs (m/s)")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_training_iid_label(audit: dict, out_path: Path) -> None:
    """IID training stores one central curve. Stage 1 width draws do not replace it."""
    iid = audit["iid_center"]
    freq = iid["freq"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax = axes[0]
    for curve in iid["curves"]:
        ax.plot(freq, curve, color="#4A5568", lw=0.8, alpha=0.45)
    if iid["curves"]:
        med = np.median(np.stack(iid["curves"]), axis=0)
        ax.plot(freq, med, color="#111111", lw=2.2, label=f"median of {iid['n_shown']} shown")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 10.0)
    ax.set_ylim(0.2, 30.0)
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Center |TF|")
    ax.set_title(
        f"IID train central recorder (x = {iid['center_x_m']:.0f} m on the 500 m strip)\n"
        f"{iid['n_train']} training runs, one stored curve each",
        loc="left", fontweight="bold",
    )
    ax.legend(loc="lower left")

    ax = axes[1]
    if iid["curves"]:
        curve = iid["curves"][0]
        for width, ls in ((500, "-"), (1000, "--"), (1500, ":")):
            ax.plot(freq, curve, ls=ls, lw=2.0, color=WIDTH_COLORS[width], label=f"Stage 1 window {width} m")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 10.0)
    ax.set_ylim(0.2, 30.0)
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(
        "Same IID run, three Stage 1 window widths\nThe central label does not change",
        loc="left", fontweight="bold",
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-only", action="store_true", help="plot the training-cache central TFs and skip Case 7")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.training_only:
        audit = training_width_audit()
        chosen = one_section_per_bin(audit)
        population = population_prior_errors(audit)
        paths = [
            OUT_DIR / "training_central_tf_by_width.png",
            OUT_DIR / "training_central_tf_examples.png",
            OUT_DIR / "training_iid_central_label.png",
            OUT_DIR / "training_vs_by_feature_and_width.png",
            OUT_DIR / "training_one_tf_vs_priors.png",
            OUT_DIR / "training_prior_error_metrics.png",
        ]
        plot_training_central_tfs(audit, paths[0])
        plot_training_width_examples(audit, paths[1])
        plot_training_iid_label(audit, paths[2])
        plot_training_vs_maps(audit, paths[3])
        plot_one_section_tfs(audit, chosen, paths[4])
        plot_one_section_errors(chosen, population, paths[5])
        one = {
            f"{feat}_w{width}": {
                "sid": rec["sid"],
                "field_seed": rec["field_seed"],
                "damping": rec["damping"],
                "n_bin": rec["n_bin"],
                "abs_g_rayleigh": round(rec["abs_g"], 4),
                "abs_g_hysteretic": round(rec["abs_g_h"], 4),
                "vs_rayleigh": {k: round(v, 4) for k, v in rec["vs_stored"].items()},
                "vs_hysteretic": {k: round(v, 4) for k, v in rec["vs_hysteretic"].items()},
            }
            for (feat, width), rec in chosen.items()
        }
        payload = {
            "n_train_sections": audit["n_train_sections"],
            "n_group_ids": audit["n_group_ids"],
            "n_group_ids_at_two_widths": audit["n_group_ids_at_two_widths"],
            "counts": audit["counts"],
            "population": audit["population"],
            "one_section": one,
            "prior_error_median": population,
            "iid_train_runs": audit["iid_center"]["n_train"],
            "plots": [str(p) for p in paths],
        }
        (OUT_DIR / "training_width_audit.json").write_text(json.dumps(payload, indent=2))
        print("Wrote:")
        for p in paths:
            print(f"  {p}")
        return

    freq, case7 = collect_case7_data()

    p1 = OUT_DIR / "case7_central_tf_width_comparison.png"
    p2 = OUT_DIR / "case7_width_metrics_summary.png"
    p3 = OUT_DIR / "manifold_valley_vs_flat_2d_signal.png"

    plot_case7_central_tfs(freq, case7, p1)
    plot_case7_metrics_summary(case7, p2)
    valley_summary = plot_manifold_valley_vs_flat(p3)

    summary_json = {
        "case7_zones": {
            slug: {k: v for k, v in info.items() if k != "curves"}
            for slug, info in case7.items()
        },
        "manifold_geometries": valley_summary,
        "plots": [str(p1), str(p2), str(p3)],
    }
    json_path = OUT_DIR / "width_physics_summary.json"
    json_path.write_text(json.dumps(summary_json, indent=2))
    print("Wrote:")
    for p in (p1, p2, p3, json_path):
        print(f"  {p}")


if __name__ == "__main__":
    main()
