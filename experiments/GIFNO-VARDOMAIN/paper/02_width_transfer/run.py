#!/usr/bin/env python3
"""Log-L1 and skill against width. Filled markers were trained; open markers are zero-shot.

Width is the interior window the network sees. The OpenSees mesh adds 500 m of
repeated edge on each side. The 21 recorders are fixed, so the station count
does not grow with width.

Point estimates come from eval_test.json. Record-bootstrap intervals, the
manifold series, and the edge check are filled when paper/dump/records.csv exists.
The floor-conditioned skill keeps center stations whose column log-L1 exceeds 0.014.

A width is trained if that family saw it in stage 1 or stage 2: horizontal at
500–1500 m, dipping and three-layer at 500 m, manifold at its native width.

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

from bootstrap_ci import bootstrap_median
from metrics import above_floor_skill
from paths import DUMP, STAGE2, current_pass
from plotting import apply_publication_style, panel_letter, save_figure

OUT = _HERE.parent
FAMILIES = ("iid", "ood_dipping", "ood_three_layer")
LABEL = {"iid": "Horizontal", "ood_dipping": "Dipping", "ood_three_layer": "Three-layer"}
COLOR = {"iid": "#0072B2", "ood_dipping": "#D55E00", "ood_three_layer": "#009E73"}
TRAINED = {("iid", 500), ("iid", 1000), ("iid", 1500), ("ood_dipping", 500), ("ood_three_layer", 500)}


def _from_json() -> list[dict]:
    path = DUMP / "ensemble_eval_test.json"
    if not path.is_file():
        path = STAGE2 / "eval_test.json"
    data = json.loads(path.read_text())["bins"]
    rows = []
    for family in FAMILIES:
        for width in (500, 1000, 1500):
            block = data[f"{family}_w{width}"]
            rows.append(
                {
                    "family": family,
                    "width": width,
                    "skill_mid": float(block["skill_mid"]),
                    "log_l1_mid": float(block["log_l1_mid"]),
                    "n_stations": int(block["n_stations"]),
                    "skill_low": float(block["skill_low"]),
                    "pearson": float(block["pearson_tr"]),
                    "trained": (family, width) in TRAINED,
                    "low": None,
                    "high": None,
                    "floor_skill": None,
                    "n_above_floor": None,
                }
            )
    return rows


def _from_dump(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    frame = pd.read_csv(DUMP / "records.csv")
    stage2 = frame[(frame["pass"] == current_pass(frame)) & (frame["family"].isin(FAMILIES))]
    updated = []
    for row in rows:
        part = stage2[(stage2["family"] == row["family"]) & (stage2["width"] == row["width"])]
        if part.empty:
            updated.append(row)
            continue
        ci = bootstrap_median(part["mean_skill_mid"].to_numpy(), part["record_id"].to_numpy(), rng=np.random.default_rng(0))
        low_ci = bootstrap_median(part["mean_skill_low"].to_numpy(), part["record_id"].to_numpy(), rng=np.random.default_rng(0))
        row = dict(row)
        row["skill_mid_station_mean"] = row["skill_mid"]
        row["skill_mid"] = ci["point"]
        row["low"] = ci["low"]
        row["high"] = ci["high"]
        row["skill_low_station_mean"] = row["skill_low"]
        row["skill_low"] = low_ci["point"]
        row["skill_low_low"] = low_ci["low"]
        row["skill_low_high"] = low_ci["high"]
        floor = above_floor_skill(part["center_column_l1_mid"], part["center_skill_mid"])
        row["floor_skill"] = floor["skill"]
        row["n_above_floor"] = floor["n_above_floor"]
        updated.append(row)
    manifold = frame[(frame["pass"] == current_pass(frame)) & (frame["family"] == "manifold") & (frame["split"].isin(["val", "blind"]))]
    series = []
    for width, part in manifold.groupby("width"):
        ci = bootstrap_median(part["mean_skill_mid"].to_numpy(), part["record_id"].to_numpy(), rng=np.random.default_rng(1))
        floor = above_floor_skill(part["center_column_l1_mid"], part["center_skill_mid"])
        series.append(
            {
                "width": int(width),
                "skill_mid": ci["point"],
                "low": ci["low"],
                "high": ci["high"],
                "log_l1_mid": float(part["center_log_l1_mid"].mean()),
                "floor_skill": floor["skill"],
                "n_above_floor": floor["n_above_floor"],
                "n": int(part["record_id"].nunique()),
                "n_stations": int(part.shape[0]),
            }
        )
    edge = {}
    station_path = DUMP / "ensemble_stations.npz"
    if not station_path.is_file():
        station_path = DUMP / "stations.npz"
    if station_path.is_file():
        stations = np.load(station_path)
        dist = stations["dist_edge"]
        skill = stations["skill_mid"]
        family_code = stations["family"]
        bins = (0.0, 50.0, 100.0, 250.0, 1.0e9)
        labels = ("0–50 m", "50–100 m", "100–250 m", ">250 m")
        names = {0: "iid", 1: "ood_dipping", 2: "ood_three_layer", 3: "manifold"}
        edge_rows = []
        for code, name in names.items():
            for lo, hi, label in zip(bins[:-1], bins[1:], labels):
                mask = (family_code == code) & (dist >= lo) & (dist < hi)
                if not np.any(mask):
                    continue
                edge_rows.append(
                    {
                        "family": name,
                        "bin": label,
                        "n": int(mask.sum()),
                        "skill_mid_median": float(np.median(skill[mask])),
                    }
                )
        edge = {
            "bins": edge_rows,
            "note": (
                "Stage-2 station skill against distance to the interior-mask edge, "
                "by family. Horizontal and dipping queries sit at least 100 m inside "
                "the mask. The outer bins are manifold stations."
            ),
        }
    return updated, series, edge


def _width_panel(ax, rows: list[dict], series: list[dict], key: str, ylabel: str, *, error: bool) -> None:
    for family in FAMILIES:
        part = [row for row in rows if row["family"] == family]
        widths = [row["width"] for row in part]
        values = [row[key] for row in part]
        color = COLOR[family]
        trained = [row["trained"] for row in part]
        ax.plot(widths, values, color=color, lw=1.1, label=LABEL[family])
        ax.scatter(
            [w for w, flag in zip(widths, trained) if flag],
            [s for s, flag in zip(values, trained) if flag],
            color=color, marker="o", s=22, zorder=3,
        )
        ax.scatter(
            [w for w, flag in zip(widths, trained) if not flag],
            [s for s, flag in zip(values, trained) if not flag],
            facecolors="none", edgecolors=color, marker="o", s=22, zorder=3,
        )
        if error and part and part[0]["low"] is not None:
            yerr = np.array([[row["skill_mid"] - row["low"], row["high"] - row["skill_mid"]] for row in part]).T
            ax.errorbar(widths, values, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7, capsize=2)
    if series:
        ax.plot(
            [row["width"] for row in series],
            [row[key] for row in series],
            color="#000000",
            ls="--",
            lw=1.0,
            label="Manifold",
        )
    ax.set_xlabel("Width (m)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([500, 1000, 1500, 2000])


def main() -> None:
    rows = _from_json()
    series: list[dict] = []
    edge: dict = {}
    if (DUMP / "records.csv").is_file():
        rows, series, edge = _from_dump(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "width.csv", index=False)
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "caption": (
                    "Width is the interior window the network sees. The OpenSees mesh adds 500 m "
                    "of repeated edge on each side. The 21 recorders are fixed, so n_stations does "
                    "not grow with width. Filled markers were trained. Open markers are zero-shot. "
                    "The left panel is mid-band log-L1: station means for the three families, and the "
                    "mean center-station error for the manifold. The right panel is the record-median "
                    "skill. Floor skill is the center-station skill where the column log-L1 exceeds 0.014."
                ),
                "trained_rule": "Filled if the family was trained at that width in stage 1 or stage 2.",
                "rows": rows,
                "manifold": series,
                "edge": edge,
            },
            indent=2,
        )
    )
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.4))
    _width_panel(axes[0], rows, series, "log_l1_mid", "Mid-band log-L1", error=False)
    _width_panel(axes[1], rows, series, "skill_mid", "Mid-band skill", error=True)
    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    print(json.dumps({"n": len(rows), "manifold": len(series), "edge": bool(edge)}, indent=2))


if __name__ == "__main__":
    main()
