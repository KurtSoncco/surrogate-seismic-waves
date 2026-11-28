#!/usr/bin/env python3
"""Stratum by width heatmap of mid-band skill on manifold rows.

Validation chose epoch 53. Blind did not enter that score. The figure shows
those splits apart. Stage 1 and stage 2 both report median skill and the mean
center-station mid-band log-L1.

Requires paper/dump/records.csv from dump_eval.py. Matched Q–Vs / explicit
pairs are written beside the heatmap.

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

from bootstrap_ci import bootstrap_cluster_statistic
from metrics import above_floor_skill
from paths import DUMP, current_pass
from plotting import apply_publication_style, save_figure

OUT = _HERE.parent


def _pairs(frame: pd.DataFrame) -> list[dict]:
    matched = frame[frame["stratum"] == "matched"]
    rows = []
    if matched.empty or "group_id" not in matched.columns:
        return rows
    for group, part in matched.groupby("group_id"):
        if part["damping"].nunique() < 2:
            continue
        skill = part.groupby("damping")["mean_skill_mid"].mean()
        rows.append({"group_id": str(group), **{f"skill_{k}": float(v) for k, v in skill.items()}})
    return rows


def _split_table(frame: pd.DataFrame) -> list[dict]:
    rows = []
    held = frame[(frame["family"] == "manifold") & (frame["split"].isin(["val", "blind"]))]
    for (stage, split), part in held.groupby(["pass", "split"]):
        floor = above_floor_skill(part["center_column_l1_mid"], part["center_skill_mid"])
        rows.append(
            {
                "stage": str(stage),
                "split": str(split),
                "n": int(part["record_id"].nunique()),
                "median_skill_mid": float(part["mean_skill_mid"].median()),
                "mean_center_log_l1_mid": float(part["center_log_l1_mid"].mean()),
                "floor_skill": floor["skill"],
                "n_above_floor": floor["n_above_floor"],
                "role": "epoch selection" if split == "val" else "not used to pick the epoch",
            }
        )
    return rows


def _blind_interval(frame: pd.DataFrame) -> dict:
    """Sixteen blind sections, one record each. The cluster is the section."""
    blind = frame[(frame["family"] == "manifold") & (frame["split"] == "blind")]
    wide = []
    for stage, part in blind.groupby("pass"):
        ci = bootstrap_cluster_statistic(
            part["mean_skill_mid"].to_numpy(),
            part["group_id"].to_numpy(),
            np.median,
            rng=np.random.default_rng(0),
        )
        wide.append(
            {
                "stage": str(stage),
                "n_sections": int(part["group_id"].nunique()),
                "n_records": int(part["record_id"].nunique()),
                "median_skill": ci["point"],
                "low": ci["low"],
                "high": ci["high"],
            }
        )
    first = blind[blind["pass"] == "stage1"][["record_id", "group_id", "mean_skill_mid"]]
    shown = current_pass(frame)
    second = blind[blind["pass"] == shown][["record_id", "mean_skill_mid"]]
    paired = first.merge(second, on="record_id", suffixes=("_stage1", "_stage2"))
    change = paired["mean_skill_mid_stage2"] - paired["mean_skill_mid_stage1"]
    delta = bootstrap_cluster_statistic(
        change.to_numpy(),
        paired["group_id"].to_numpy(),
        np.median,
        rng=np.random.default_rng(1),
    )
    return {
        "n_sections": int(paired["group_id"].nunique()),
        "n_records": int(len(paired)),
        "note": "One record per section, so the section cluster is a 16-record bootstrap.",
        "stages": wide,
        "paired_stage2_minus_stage1": {
            "median": delta["point"],
            "low": delta["low"],
            "high": delta["high"],
        },
    }


def _heatmap(ax, part: pd.DataFrame, title: str) -> None:
    if part.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes)
        return
    pivot = part.pivot_table(index="stratum", columns="width", values="mean_skill_mid", aggfunc="mean")
    error = part.pivot_table(index="stratum", columns="width", values="center_log_l1_mid", aggfunc="mean")
    image = ax.imshow(pivot.to_numpy(), cmap="cividis", aspect="auto", vmin=-0.2, vmax=0.8)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, label="Mid-band skill")
    ax.set_xticks(range(pivot.shape[1]), [str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), list(pivot.index))
    ax.set_xlabel("Width (m)")
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            skill = pivot.to_numpy()[i, j]
            log_l1 = error.to_numpy()[i, j]
            if not np.isfinite(skill):
                continue
            ax.text(j, i, f"{skill:.2f}\n{log_l1:.3f}", ha="center", va="center", fontsize=6, color="white")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = DUMP / "records.csv"
    if not path.is_file():
        (OUT / "results.json").write_text(json.dumps({"status": "waiting_for_dump", "path": str(path)}, indent=2))
        print(f"waiting for {path}")
        return
    frame = pd.read_csv(path)
    stage2 = frame[(frame["pass"] == current_pass(frame)) & (frame["family"] == "manifold")]
    val = stage2[stage2["split"] == "val"]
    blind = stage2[stage2["split"] == "blind"]
    summary = _split_table(frame)
    blind_interval = _blind_interval(frame)
    pd.DataFrame(summary).to_csv(OUT / "splits.csv", index=False)
    val.pivot_table(index="stratum", columns="width", values="mean_skill_mid", aggfunc="mean").to_csv(OUT / "skill.csv")
    pairs = _pairs(val)
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "caption": "Validation chose epoch 53. Blind did not enter that score. Each cell is mean mid-band skill over mean center-station log-L1.",
                "splits": summary,
                "blind_section_interval": blind_interval,
                "matched_pairs_val": pairs,
            },
            indent=2,
        )
    )

    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.6))
    _heatmap(axes[0], val, "Validation (epoch selection)")
    _heatmap(axes[1], blind, "Blind (not used to select)")
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
