"""Held-out |TF| at the center station: two horizontal sites and two dipping sites.

Horizontal sites are the stage-2 test split, one thin and one thick. The dipping
test split is flatter than 0.4°, so the dipping panels are the two validation
sites near 1.4°. Each panel is the seed closest to that site's median Pearson
of this work. Pretell, Toro, and Tao & Rathje are the same curves as fig11.

    python fig16_tf_comparison.py
"""

from __future__ import annotations

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import figstyle as fs

sys.path.insert(0, str(fs.REPO / "experiments" / "GIFNO-VARDOMAIN"))

from data import KIND_RUN, Cache, make_window  # noqa: E402
from evaluate import load_model  # noqa: E402
from classical_worker import compute_arms  # noqa: E402

CACHE = fs.REPO / "data" / "vardomain" / "cache"
MANIFEST = fs.REPO / "data" / "gifno_screen" / "ood_dipping" / "manifest.csv"
ARMS_IID = fs.REPO / "experiments" / "GIFNO-VARDOMAIN" / "figures" / "test_arms_iid.npz"
# Thin site (f0 ≈ 1.3 Hz) and thick site (f0 ≈ 0.44 Hz), both near the test median.
HORIZONTAL_GROUPS = (168, 225)
# Validation Sobol sites, dips −1.36° and +1.44°.
DIPPING_SITES = (18, 27)
SERIES = (
    ("2D simulation", fs.TRUTH, "-", 2.8),
    ("This work", fs.AI, "-", 2.6),
    ("Pretell et al. (2022)", fs.PRETELL, (0, (5, 2.2)), 2.3),
    ("Toro (1995)", fs.TORO, (0, (1.4, 1.0)), 2.3),
    ("Tao & Rathje (2019)", fs.DMULT, (0, (0.6, 1.5)), 2.5),
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.clip(a, 1e-12, None), np.clip(b, 1e-12, None))[0, 1])


def _nearest_median(pear: np.ndarray) -> int:
    return int(np.argmin(np.abs(pear - np.median(pear))))


def _horizontal(freq: np.ndarray) -> list[dict]:
    split = np.load(CACHE / "iid" / "split.npy")
    group = np.load(CACHE / "iid" / "group.npy")
    runs = np.flatnonzero(split == "test")
    ai = np.load(fs.RESULTS / "discarded" / "test_model_iid_stage2_epoch53_float32.npy").astype(np.float64)
    truth = np.exp(np.load(CACHE / "iid" / "log_tf.npy", mmap_mode="r")[runs, 10].astype(np.float64))
    arms = np.load(ARMS_IID)
    if not np.array_equal(arms["run"], runs):
        raise RuntimeError("horizontal classical arms are not the test split")
    params = pd.read_csv(fs.DATA / "run_params.csv")
    params = params[params.family == "iid"].set_index("run")
    out = []
    for g in HORIZONTAL_GROUPS:
        m = np.flatnonzero(group[runs] == g)
        if m.size == 0:
            raise RuntimeError(f"horizontal group {g} is not in the test split")
        pear = np.array([_pearson(ai[i], truth[i]) for i in m])
        i = m[_nearest_median(pear)]
        row = params.loc[int(runs[i])]
        out.append({
            "title": "Horizontal bedrock",
            "note": f"$H$ = {row.H:.0f} m,  $f_0$ = {row.f0_effective:.2f} Hz",
            "curves": {
                "2D simulation": truth[i],
                "This work": ai[i],
                "Pretell et al. (2022)": arms["pretell"][i].astype(np.float64),
                "Toro (1995)": arms["toro"][i].astype(np.float64),
                "Tao & Rathje (2019)": arms["dmult"][i].astype(np.float64),
            },
            "pearson": _pearson(ai[i], truth[i]),
        })
        print(f"horizontal group {g} run {int(runs[i])}: H {row.H:.0f} f0 {row.f0_effective:.2f} "
              f"Pearson {out[-1]['pearson']:.3f}")
    return out


def _predict(local_runs: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _cfg = load_model(fs.CHECKPOINT, device)
    model.eval()
    cache = Cache(CACHE)
    camp = next(c for c in cache.campaigns if c.name == "ood_dipping")
    base = next(i for i, (ci, _j) in enumerate(cache.run_of) if cache.campaigns[ci] is camp)
    pred = np.empty((local_runs.size, cache.freq.size), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, local_runs.size, 8):
            chunk = local_runs[start:start + 8]
            xs, qs, priors, mids = [], [], [], []
            for run in chunk:
                x, query, _tf, prior, valid, _info, _xa = make_window(
                    cache, "metric", KIND_RUN, int(base + int(run)), 500, 500, 0, False,
                )
                v = np.flatnonzero(valid.numpy())
                mids.append(int(v[len(v) // 2]))
                xs.append(x)
                qs.append(query)
                priors.append(prior.numpy()[mids[-1]])
            g = model(torch.stack(xs).to(device), torch.stack(qs).to(device)).float().cpu().numpy()
            for j, prior in enumerate(priors):
                pred[start + j] = np.exp(prior + g[j, mids[j]])
    return pred


def _dipping(freq: np.ndarray) -> list[dict]:
    rows = list(csv.DictReader(MANIFEST.open()))
    split = np.load(CACHE / "ood_dipping" / "split.npy").astype(str)
    truth_all = np.exp(np.load(CACHE / "ood_dipping" / "log_tf.npy", mmap_mode="r")[:, 10].astype(np.float64))
    out = []
    for sid in DIPPING_SITES:
        local = np.array([
            i for i, row in enumerate(rows)
            if int(row["sobol_id"]) == sid and split[i] == "val"
        ], dtype=int)
        if local.size != 30:
            raise RuntimeError(f"dipping site {sid} has {local.size} validation seeds")
        pred = _predict(local)
        truth = truth_all[local]
        pear = np.array([_pearson(pred[i], truth[i]) for i in range(local.size)])
        k = _nearest_median(pear)
        run = int(local[k])
        row = rows[run]
        task = {
            "family": "ood_dipping",
            "run": run,
            "vs1": float(row["Vs1"]),
            "H": float(row["H_discretized"]),
            "cov": float(row["CoV"]),
            "vs2": float(row["Vs2"]),
            "soil_nz": int(row["soil_layer_count"]),
            "f0": float(row["f0_effective"]),
        }
        arms = compute_arms(task)
        dip = float(row["dip_angle_deg"])
        sign = "−" if dip < 0 else "+"
        out.append({
            "title": "Dipping bedrock",
            "note": f"dip {sign}{abs(dip):.1f}°,  $H$ = {float(row['H_discretized']):.0f} m",
            "curves": {
                "2D simulation": truth[k],
                "This work": pred[k],
                "Pretell et al. (2022)": arms["pretell"].astype(np.float64),
                "Toro (1995)": arms["toro"].astype(np.float64),
                "Tao & Rathje (2019)": arms["dmult"].astype(np.float64),
            },
            "pearson": pear[k],
        })
        print(f"dipping sobol {sid} run {run}: dip {dip:+.2f}° H {float(row['H_discretized']):.0f} "
              f"Pearson {pear[k]:.3f}")
    return out


def main() -> None:
    fs.apply()
    freq = np.load(CACHE / "freq.npy").astype(np.float64)
    panels = _horizontal(freq) + _dipping(freq)
    # Change order to be column wise, so that the horizontal sites are in the first column and the dipping sites are in the second column.
    panels = [panels[0], panels[2], panels[1], panels[3]]
    fig, axes = plt.subplots(2, 2, figsize=(fs.FULL_W, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.84, hspace=0.38, wspace=0.08)
    ymax = max(float(np.max(c)) for panel in panels for c in panel["curves"].values())
    for ax, panel, letter in zip(axes.ravel(), panels, "ABCD"):
        for name, color, ls, lw in SERIES:
            ax.loglog(freq, np.maximum(panel["curves"][name], 1e-3), color=color, ls=ls, lw=lw,
                      label=name, zorder=4 if name == "This work" else 3)
        fs.log_freq_axis(ax)
        ax.set_ylim(0.15, max(20.0, ymax * 1.4))
        ax.set_yticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_title(f"({letter})  {panel['title']}", loc="left", fontsize=16, color=fs.NAVY, pad=28)
        ax.text(0.0, 1.02, panel["note"], transform=ax.transAxes, ha="left", va="bottom",
                fontsize=13, color=fs.MUTED)
    for ax in axes[0]:
        ax.set_xlabel("")
    for ax in axes[:, 0]:
        ax.set_ylabel("$|TF|$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.955), ncol=5,
               fontsize=13, frameon=False, borderaxespad=0)
    fs.save(fig, "fig16_tf_comparison", tight=True)


if __name__ == "__main__":
    main()
