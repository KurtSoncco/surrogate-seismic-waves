#!/usr/bin/env python3
"""Center-station medians, the median-Pearson example curves, and the method box plot.

The example in each panel is the held-out record whose model Pearson is closest
to the median model Pearson. Intervals resample sites in group.npy. Toro and
Passeri are scored at xi = 0.05 and again at the record's median soil zeta.

    python run.py
    python run.py --skip-passeri --skip-zeta
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "scripts" / "_entry.py").is_file():
        sys.path.insert(0, str(_parent / "scripts"))
        break
from _entry import enter

enter(_HERE)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bootstrap_ci import bootstrap_cluster_statistic
from center_curves import FAMILIES, test_frame
from metrics import anderson, pearson
from passeri import ensure_passeri
from plotting import apply_publication_style, panel_letter, save_figure
from zeta_arms import ensure_zeta_arms

OUT = _HERE.parent
METHODS = ("Column", "Model", "Pretell", "Passeri", "Dmult", "Toro")
EXAMPLE = ("OpenSees", "Column", "Model", "Pretell", "Toro")
STYLE = {
    "OpenSees": {"color": "#000000", "ls": "-", "lw": 1.3},
    "Column": {"color": "#0072B2", "ls": "--", "lw": 1.1},
    "Model": {"color": "#D55E00", "ls": "-", "lw": 1.3},
    "Pretell": {"color": "#009E73", "ls": "-.", "lw": 1.0},
    "Passeri": {"color": "#56B4E9", "ls": "--", "lw": 1.0},
    "Dmult": {"color": "#E69F00", "ls": ":", "lw": 1.0},
    "Toro": {"color": "#CC79A7", "ls": ":", "lw": 1.1},
}
BOX_COLOR = {
    "Column": "#0072B2",
    "Model": "#D55E00",
    "Pretell": "#009E73",
    "Passeri": "#CC79A7",
    "Dmult": "#332288",
    "Toro": "#F0E442",
}
ARMS = {
    "Column": "The 1D column stored with the record.",
    "Model": "Mean of P128, P128s1, P128n128, and P128c (epoch 30), added to that column.",
    "Pretell": "Haskell on the record Vs strip and the record zeta strip.",
    "Passeri": "TTS random profile at a constant xi of 0.05, 40 seeds.",
    "Dmult": "Its own multiplier on a 2% base. It does not use the record zeta.",
    "Toro": "Toro random Vs at a constant xi of 0.05, 40 seeds.",
    "Passeri (record zeta)": "Same Passeri randomization at the record median soil zeta.",
    "Toro (record zeta)": "Same Toro randomization at the record median soil zeta.",
}


def _closest_to_median(score: np.ndarray) -> int:
    med = float(np.median(score))
    return int(np.argmin(np.abs(score - med)))


def _curves(frame: dict, passeri: np.ndarray | None, zeta: dict | None) -> dict[str, np.ndarray]:
    curves = {
        "OpenSees": frame["opensees"],
        "Column": frame["column"],
        "Model": frame["model"],
        "Pretell": frame["pretell"],
        "Dmult": frame["dmult"],
        "Toro": frame["toro"],
    }
    if passeri is not None:
        curves["Passeri"] = passeri
    if zeta is not None:
        curves["Toro (record zeta)"] = zeta["toro"]
        curves["Passeri (record zeta)"] = zeta["passeri"]
    return curves


def _site(values: np.ndarray, groups: np.ndarray) -> dict:
    return bootstrap_cluster_statistic(values, groups, np.median, rng=np.random.default_rng(0))


def _harm(model: np.ndarray, column: np.ndarray, groups: np.ndarray, *, lower_is_better: bool) -> dict:
    worse = model > column if lower_is_better else model < column
    return bootstrap_cluster_statistic(worse.astype(np.float64), groups, np.mean, rng=np.random.default_rng(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-passeri", action="store_true")
    ap.add_argument("--skip-zeta", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {}
    rows = []
    examples = {}
    harm_rows = []
    zeta_note = {}
    box = {}
    for name, family in FAMILIES.items():
        frame = test_frame(family)
        got = None if args.skip_passeri else ensure_passeri(family, frame["run"], workers=args.workers)
        zeta = None if args.skip_zeta else ensure_zeta_arms(family, frame["run"], workers=args.workers)
        curves = _curves(frame, got, zeta)
        if zeta is not None:
            zeta_note[name] = {
                "n": int(zeta["zeta"].size),
                "median": float(np.median(zeta["zeta"])),
                "min": float(np.min(zeta["zeta"])),
                "max": float(np.max(zeta["zeta"])),
            }
        pear = {method: np.array([pearson(curves[method][i], frame["opensees"][i]) for i in range(frame["run"].size)]) for method in curves if method != "OpenSees"}
        gof = {
            method: np.array(
                [anderson(frame["freq"], frame["opensees"][i], curves[method][i], frame["f0"][i]) for i in range(frame["run"].size)]
            )
            for method in pear
        }
        pick = _closest_to_median(pear["Model"])
        examples[name] = {
            "run": int(frame["run"][pick]),
            "model_pearson": float(pear["Model"][pick]),
            "median_model_pearson": float(np.median(pear["Model"])),
            "curves": {method: curves[method][pick] for method in EXAMPLE},
            "freq": frame["freq"],
        }
        for method in list(METHODS) + ["Toro (record zeta)", "Passeri (record zeta)"]:
            if method not in pear:
                continue
            for metric, values in (("pearson", pear[method]), ("anderson", gof[method])):
                ci = _site(values, frame["group"])
                rows.append(
                    {
                        "domain": name,
                        "method": method,
                        "metric": metric,
                        "level": "record",
                        "n_records": int(frame["run"].size),
                        "n_sites": ci["n_clusters"],
                        "median": ci["point"],
                        "low": ci["low"],
                        "high": ci["high"],
                    }
                )
                site = pd.DataFrame({"value": values, "group": frame["group"]}).groupby("group")["value"].median()
                rows.append(
                    {
                        "domain": name,
                        "method": method,
                        "metric": metric,
                        "level": "site",
                        "n_records": int(frame["run"].size),
                        "n_sites": int(site.size),
                        "median": float(site.median()),
                        "low": None,
                        "high": None,
                    }
                )
        harm_rows.append(
            {
                "domain": name,
                "n_records": int(frame["run"].size),
                "n_sites": int(np.unique(frame["group"]).size),
                "pearson_worse_than_column": _harm(pear["Model"], pear["Column"], frame["group"], lower_is_better=False),
                "anderson_worse_than_column": _harm(gof["Model"], gof["Column"], frame["group"], lower_is_better=True),
            }
        )
        box[name] = {"pearson": {method: pear[method] for method in METHODS if method in pear}, "anderson": {method: gof[method] for method in METHODS if method in gof}}
        frames[name] = frame
        print(f"{name} example run {examples[name]['run']} pearson {examples[name]['model_pearson']:.3f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table1_medians.csv", index=False)
    harm_out = []
    for row in harm_rows:
        harm_out.append(
            {
                "domain": row["domain"],
                "n_records": row["n_records"],
                "n_sites": row["n_sites"],
                "pearson_worse_than_column": row["pearson_worse_than_column"]["point"],
                "pearson_low": row["pearson_worse_than_column"]["low"],
                "pearson_high": row["pearson_worse_than_column"]["high"],
                "anderson_worse_than_column": row["anderson_worse_than_column"]["point"],
                "anderson_low": row["anderson_worse_than_column"]["low"],
                "anderson_high": row["anderson_worse_than_column"]["high"],
            }
        )
    pd.DataFrame(harm_out).to_csv(OUT / "harm_rate.csv", index=False)
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "example_rule": "held-out center station whose model Pearson is closest to the median model Pearson",
                "interval": "Sites in group.npy are resampled. Every record of a drawn site stays in the draw. Dipping is 3 sites of 30 seeds, so that panel is a three-geometry demonstration.",
                "arms": ARMS,
                "record_median_soil_zeta": zeta_note,
                "harm_rate": harm_out,
                "examples": {name: {k: v for k, v in spec.items() if k != "curves" and k != "freq"} for name, spec in examples.items()},
                "passeri": not args.skip_passeri,
                "zeta_arms": not args.skip_zeta,
            },
            indent=2,
        )
    )
    np.savez(
        OUT / "examples.npz",
        freq=examples["Horizontal"]["freq"],
        **{f"{name}_{method}": spec["curves"][method] for name, spec in examples.items() for method in EXAMPLE},
    )

    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.2), sharey=True)
    for ax, name, letter in zip(axes, ("Horizontal", "Dipping"), "ab"):
        spec = examples[name]
        freq = spec["freq"]
        for method in EXAMPLE:
            style = STYLE[method]
            ax.plot(freq, np.clip(spec["curves"][method], 1e-3, None), label=method, **style)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 10.0)
        ax.set_xlabel(r"$f$ (Hz)")
        ax.set_title(f"{name}, run {spec['run']}")
        panel_letter(ax, letter)
    axes[0].set_ylabel(r"$|\mathrm{TF}|$")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    _box(box)
    print(table.to_string(index=False))
    print(pd.DataFrame(harm_out).to_string(index=False))


def _box(box: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.6))
    panels = (
        (0, 0, "Horizontal", "pearson", r"Pearson of $|$TF$|$"),
        (0, 1, "Horizontal", "anderson", "Anderson misfit"),
        (1, 0, "Dipping", "pearson", r"Pearson of $|$TF$|$"),
        (1, 1, "Dipping", "anderson", "Anderson misfit"),
    )
    for letter, (row, col, name, metric, ylabel) in enumerate(panels):
        ax = axes[row, col]
        methods = [method for method in METHODS if method in box[name][metric]]
        drawn = ax.boxplot(
            [box[name][metric][method] for method in methods],
            widths=0.62,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "lw": 0.8},
            whiskerprops={"lw": 0.6},
            capprops={"lw": 0.6},
            boxprops={"lw": 0.6},
        )
        for patch, method in zip(drawn["boxes"], methods):
            patch.set_facecolor(BOX_COLOR[method])
            patch.set_alpha(0.55)
            patch.set_edgecolor("#333333")
        ax.set_xticks(range(1, len(methods) + 1), methods, fontsize=6, rotation=20, ha="right")
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        panel_letter(ax, "abcd"[letter])
    fig.tight_layout()
    save_figure(fig, OUT / "fig_box_test_methods.pdf")


if __name__ == "__main__":
    main()
