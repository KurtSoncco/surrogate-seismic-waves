#!/usr/bin/env python3
"""Column error against model error. The floor is the horizontal line.

Horizontal and dipping points are center stations of the saved stage-2 curves.
Treasure Island zones come from the zone table (mid band). Three-layer records
and manifold rows are added when the eval dump is present.

The crossover is where a Theil–Sen fit in log-log space crosses y = x.

    python run.py
"""

from __future__ import annotations

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
from scipy.stats import theilslopes

from center_curves import FAMILIES, test_frame
from metrics import band_mask
from paths import DUMP, FIG, FLOOR_JSON, current_pass
from plotting import apply_publication_style, panel_letter, save_figure

OUT = _HERE.parent
COLOR = {
    "Horizontal": "#0072B2",
    "Dipping": "#D55E00",
    "Three-layer": "#009E73",
    "Manifold": "#000000",
    "Treasure Island": "#CC79A7",
}


def _floor() -> float:
    if not FLOOR_JSON.is_file():
        raise SystemExit(f"run consistency_floor first: {FLOOR_JSON}")
    return float(json.loads(FLOOR_JSON.read_text())["floor_midband_median_abs_g"])


def _band_error(log_a: np.ndarray, log_b: np.ndarray, freq: np.ndarray, band: str) -> np.ndarray:
    mask = band_mask(freq, band)
    return np.mean(np.abs(log_a[:, mask] - log_b[:, mask]), axis=1)


def _saved_points(band: str) -> list[dict]:
    rows = []
    for name, family in FAMILIES.items():
        frame = test_frame(family)
        column = _band_error(frame["log_column"], frame["log_opensees"], frame["freq"], band)
        model = _band_error(frame["log_model"], frame["log_opensees"], frame["freq"], band)
        for x, y, run in zip(column, model, frame["run"]):
            rows.append({"band": band, "family": name, "x": float(x), "y": float(y), "id": f"{family}:{int(run)}"})
    if band == "mid":
        table = json.loads((FIG / "stage2" / "case7_width.json").read_text())
        for zone, widths in table.items():
            for width, cell in widths.items():
                skill = float(cell["skill"])
                y = float(cell["log_l1"])
                x = y / max(1.0 - skill, 1e-6)
                rows.append({"band": band, "family": "Treasure Island", "x": x, "y": y, "id": f"{zone}:{width}"})
    return rows


def _dump_points(band: str) -> list[dict]:
    if not (DUMP / "records.csv").is_file():
        return []
    frame = pd.read_csv(DUMP / "records.csv")
    frame = frame[frame["pass"] == current_pass(frame)]
    rows = []
    names = {"ood_three_layer": "Three-layer", "manifold": "Manifold"}
    for family, label in names.items():
        part = frame[frame["family"] == family]
        if "split" in part.columns and family == "manifold":
            part = part[part["split"].isin(["val", "blind"])]
        xcol = f"center_column_l1_{band}"
        ycol = f"center_log_l1_{band}"
        for rec in part.itertuples(index=False):
            rows.append(
                {
                    "band": band,
                    "family": label,
                    "x": float(getattr(rec, xcol)),
                    "y": float(getattr(rec, ycol)),
                    "id": f"{family}:{rec.record_id}:{int(rec.width)}",
                }
            )
    return rows


def _crossover(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 1e-6) & (y > 1e-6)
    lx, ly = np.log(x[mask]), np.log(y[mask])
    if lx.size < 5:
        return {"n": int(lx.size), "crossover": None}
    slope, intercept, low, high = theilslopes(ly, lx)
    if abs(1.0 - slope) < 1e-6:
        cross = None
    else:
        cross = float(np.exp(intercept / (1.0 - slope)))
    xmin = float(x[mask].min())
    xmax = float(x[mask].max())
    inside = cross is not None and xmin <= cross <= xmax and not (low <= 1.0 <= high)
    return {
        "n": int(lx.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_low": float(low),
        "slope_high": float(high),
        "crossover": cross,
        "x_min": xmin,
        "x_max": xmax,
        "inside_observations": bool(inside),
    }


def main() -> None:
    floor = _floor()
    rows = []
    for band in ("low", "mid"):
        rows.extend(_saved_points(band))
        rows.extend(_dump_points(band))
    frame = pd.DataFrame(rows)
    fits = {}
    for band, part in frame.groupby("band"):
        fits[band] = _crossover(part["x"].to_numpy(), part["y"].to_numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "points.csv", index=False)
    (OUT / "results.json").write_text(json.dumps({"floor": floor, "crossover": fits}, indent=2))

    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.5), sharex=True, sharey=True)
    for ax, band, letter in zip(axes, ("low", "mid"), "ab"):
        part = frame[frame["band"] == band]
        for family, color in COLOR.items():
            sub = part[part["family"] == family]
            if sub.empty:
                continue
            ax.scatter(sub["x"], sub["y"], s=10, c=color, alpha=0.65, label=family, linewidths=0)
        lo = max(float(part["x"].min()) * 0.7, 1e-4)
        hi = float(max(part["x"].max(), part["y"].max())) * 1.3
        grid = np.array([lo, hi])
        ax.plot(grid, grid, color="#000000", lw=0.7, label="y = x")
        ax.plot(grid, 0.5 * grid, color="#000000", lw=0.7, ls="--", label="y = 0.5x")
        ax.axhline(floor, color="#E69F00", lw=1.0, label="Floor")
        cross = fits[band]["crossover"]
        if fits[band].get("inside_observations") and lo < cross < hi:
            ax.axvline(cross, color="#56B4E9", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Column error")
        ax.set_title(band)
        panel_letter(ax, letter)
    axes[0].set_ylabel("Model error")
    axes[1].legend(frameon=False, fontsize=5.5, loc="upper left")
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
