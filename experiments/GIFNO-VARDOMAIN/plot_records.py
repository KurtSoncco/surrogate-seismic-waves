#!/usr/bin/env python3
"""Per-record transfer functions and box plots.

Horizontal is the IID test. Dipping is the dipping test. Case 7 is each crop
against that crop's own OpenSees record. The band is 0.4–2.5 Hz.

    python plot_records.py --model-dir ../../checkpoints/vardomain/early_width/stage2_rise50
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASE7A = HERE.parent / "case7a"
PAPER = REPO / "experiments" / "DeepONet-Residual" / "paper"
for path in (PAPER, CASE7A):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
sys.path.insert(0, str(HERE))

from data import KIND_RUN, Cache, make_window  # noqa: E402
from evaluate import load_model  # noqa: E402
from plot_results import (  # noqa: E402
    CROPS,
    FIG_DIR,
    SITE_LABEL,
    _save,
    _score_mesh,
    _tf_panel,
    case7,
)
from style import METHOD_STYLE, figsize, panel_letter, use_style  # noqa: E402

BAND = (0.4, 2.5)
MODEL_STYLE = {"color": "#C2410C", "ls": "--", "lw": 1.15, "marker": None, "ms": 2.4, "zorder": 3}


def _records(cache: Cache, family: str, n: int, limit: int | None) -> np.ndarray:
    runs = cache.runs("test", (family,))
    if limit is not None:
        runs = runs[:limit]
    if runs.size <= n:
        return runs
    return runs[np.linspace(0, runs.size - 1, n).round().astype(int)]


def _one(model, device, cache: Cache, flat: int) -> dict:
    x, query, log_tf, prior, valid, _info, _xa = make_window(cache, "metric", KIND_RUN, int(flat), 500, 500, 0, False)
    with torch.no_grad():
        g = model(x.unsqueeze(0).to(device), query.unsqueeze(0).to(device)).float().cpu().numpy()[0]
    v = np.flatnonzero(valid.numpy())
    freq = cache.freq
    band = (freq >= BAND[0]) & (freq <= BAND[1])
    log_p = prior.numpy()[v] + g[v]
    log_t = log_tf.numpy()[v]
    log_l1 = np.abs(log_p[:, band] - log_t[:, band]).mean(1)
    pear = np.empty(v.size)
    for i in range(v.size):
        a, b = log_p[i, band], log_t[i, band]
        a, b = a - a.mean(), b - b.mean()
        pear[i] = float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))
    mid = v[len(v) // 2]
    return {
        "freq": freq,
        "truth": np.exp(log_tf.numpy()[mid]),
        "prior": np.exp(prior.numpy()[mid]),
        "pred": np.exp(prior.numpy()[mid] + g[mid]),
        "log_l1": log_l1,
        "pearson": pear,
        "index": int(flat),
    }


def _gallery(model, device, cache: Cache, family: str, title: str, stem: str, n: int = 6) -> None:
    use_style()
    rows = [_one(model, device, cache, int(i)) for i in _records(cache, family, n, None)]
    fig, axes = plt.subplots(2, 3, figsize=figsize("double", height_mm=118), sharex=True, sharey=True)
    for ax, row, letter in zip(axes.ravel(), rows, "abcdef"):
        _tf_panel(ax, row["freq"], [
            (row["truth"], METHOD_STYLE["OpenSees 2-D"], "OpenSees"),
            (row["prior"], METHOD_STYLE["1D base case"], "1D"),
            (row["pred"], MODEL_STYLE, "Model"),
        ])
        ax.set_title(f"{title} {row['index']}", fontsize=7, pad=2)
        panel_letter(ax, letter)
    axes[0, 0].legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout()
    _save(fig, stem)


def _case7_pages(scored: dict) -> None:
    use_style()
    names = [c["name"] for c in CROPS]
    for width, stem in ((500, "fig_tf_case7_w500"), (1500, "fig_tf_case7_w1500"), (2000, "fig_tf_case7_w2000")):
        fig, axes = plt.subplots(1, 5, figsize=figsize("double", height_mm=72), sharey=True)
        for ax, name, letter in zip(axes, names, "abcde"):
            curves = scored[name][width]["curves"]
            _tf_panel(ax, curves["freq"], [
                (curves["truth"], METHOD_STYLE["OpenSees 2-D"], "OpenSees"),
                (curves["prior"], METHOD_STYLE["1D base case"], "1D"),
                (curves["pred"], MODEL_STYLE, "Model"),
            ])
            ax.set_title(f"{SITE_LABEL[name]}, {width} m", fontsize=6.5, pad=2)
            panel_letter(ax, letter)
        axes[0].legend(frameon=False, fontsize=5.5, loc="upper right")
        fig.tight_layout()
        _save(fig, stem)


def _boxes(groups: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize("double", height_mm=88))
    labels = [g[0] for g in groups]
    for ax, idx, ylabel, letter in (
        (axes[0], 1, r"log $|$TF$|$ L1, 0.4–2.5 Hz", "a"),
        (axes[1], 2, "Pearson of log |TF|", "b"),
    ):
        data = [g[idx] for g in groups]
        bp = ax.boxplot(
            data, widths=0.62, patch_artist=True, showfliers=False,
            medianprops={"color": "black", "lw": 0.8},
            whiskerprops={"lw": 0.6}, capprops={"lw": 0.6}, boxprops={"lw": 0.6},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#D6E4F0")
            patch.set_edgecolor("#333333")
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=30, ha="right", fontsize=6)
        ax.set_ylabel(ylabel)
        panel_letter(ax, letter, x=-0.16)
    fig.tight_layout()
    _save(fig, "fig_box_metrics")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True, help="directory containing best_model.pt")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "vardomain" / "cache")
    ap.add_argument("--box-records", type=int, default=80)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _cfg = load_model(args.model_dir, device)
    model.eval()
    cache = Cache(args.cache)
    _gallery(model, device, cache, "iid", "Horizontal", "fig_tf_horizontal")
    _gallery(model, device, cache, "ood_dipping", "Dipping", "fig_tf_dipping")
    print("scoring Case 7", flush=True)
    scored = case7(model, device)
    _case7_pages(scored)

    groups = []
    for family, label in (("iid", "Horizontal"), ("ood_dipping", "Dipping"), ("ood_three_layer", "Three-layer")):
        l1, pear = [], []
        ids = _records(cache, family, args.box_records, None)
        print(f"box {label} {ids.size}", flush=True)
        for flat in ids:
            row = _one(model, device, cache, int(flat))
            l1.append(row["log_l1"])
            pear.append(row["pearson"])
        groups.append((label, np.concatenate(l1), np.concatenate(pear)))
    for width in (500, 1000, 1500, 2000):
        l1, pear = [], []
        for crop in CROPS:
            row = scored[crop["name"]][width]
            l1.append(row["station_log_l1"])
            pear.append(row["station_pearson"])
        groups.append((f"C7 {width}", np.concatenate(l1), np.concatenate(pear)))
    _boxes(groups)
    print(f"wrote {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
