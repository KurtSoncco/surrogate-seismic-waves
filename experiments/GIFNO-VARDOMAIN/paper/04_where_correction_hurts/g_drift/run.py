#!/usr/bin/env python3
"""Mean predicted g(f) on three-layer records, and error against scalar range.

Stage 2 with modes 64:80 zeroed is the third curve. Out-of-range distance is
the Euclidean excess, in training standard deviations, outside the axis-aligned
box of the manifold training scalars.

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

from paths import DUMP, current_pass
from plotting import apply_publication_style, panel_letter, save_figure

OUT = _HERE.parent
SCALAR = ("log_impedance", "log_thickness", "log_f0", "log_f2")


def _training_box() -> np.ndarray:
    import csv

    from paths import MANIFEST

    rows = []
    with MANIFEST.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("stratum") == "blind" or row.get("split") == "blind":
                continue
            rows.append(
                [
                    np.log(float(row["vs_contact_mps"]) / max(float(row["vs_surface_mps"]), 1.0)),
                    np.log(max(float(row["soil_thickness_m"]), 1.0)),
                    np.log(max(float(row["f0_flat_hz"]), 1e-6)),
                    np.log(max(float(row["rayleigh_f2_hz"]), 1e-6)),
                ]
            )
    return np.asarray(rows, dtype=np.float64)


def _range_distance(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    box = _training_box()
    sigma = np.maximum(np.nanstd(box, axis=0), 1e-6)
    lo = np.nanmin(box, axis=0)
    hi = np.nanmax(box, axis=0)
    three = frame[(frame["pass"] == current_pass(frame)) & (frame["family"] == "ood_three_layer") & (frame["width"] == 500)].copy()
    point = three[list(SCALAR)].to_numpy(dtype=np.float64)
    excess = np.maximum(lo - point, 0.0) + np.maximum(point - hi, 0.0)
    return three, np.linalg.norm(excess / sigma, axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = DUMP / "g_mean.npz"
    records = DUMP / "records.csv"
    if not spec.is_file() or not records.is_file():
        (OUT / "results.json").write_text(json.dumps({"status": "waiting_for_dump"}, indent=2))
        print("waiting for the eval dump")
        return
    blob = np.load(spec)
    frame = pd.read_csv(records)
    three, distance = _range_distance(frame)
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "n": int(len(three)),
                "median_out_of_range": float(np.median(distance)) if distance.size else None,
                "fraction_outside": float(np.mean(distance > 0)) if distance.size else None,
                "zeroed_recovers": _recovers(frame),
            },
            indent=2,
        )
    )
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.3))
    freq = blob["freq"]
    axes[0].plot(freq, blob["stage1"], color="#009E73", lw=1.1, label="Stage 1")
    axes[0].plot(freq, blob["stage2"], color="#E69F00", lw=1.0, label="Stage 2")
    if "ensemble" in blob.files:
        axes[0].plot(freq, blob["ensemble"], color="#D55E00", lw=1.2, label="Ensemble")
    axes[0].plot(freq, blob["stage2_zeroed"], color="#0072B2", ls="--", lw=1.0, label="Modes 64:80 zeroed")
    axes[0].axhline(0.0, color="#000000", lw=0.5)
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.1, 10.0)
    axes[0].set_xlabel(r"$f$ (Hz)")
    axes[0].set_ylabel(r"Mean predicted $g(f)$")
    axes[0].legend(frameon=False, fontsize=6)
    panel_letter(axes[0], "a")
    axes[1].scatter(distance, three["center_log_l1_mid"], s=12, c="#D55E00", linewidths=0)
    axes[1].set_xlabel("Out-of-range distance")
    axes[1].set_ylabel("Mid-band model error")
    panel_letter(axes[1], "b")
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")


def _recovers(frame: pd.DataFrame) -> dict:
    def skill(pass_name: str) -> float | None:
        part = frame[(frame["pass"] == pass_name) & (frame["family"] == "ood_three_layer") & (frame["width"] == 500)]
        if part.empty:
            return None
        return float(part["mean_skill_mid"].mean())

    return {"stage1": skill("stage1"), "stage2": skill("stage2"), "stage2_zeroed": skill("stage2_zeroed"), "ensemble": skill("ensemble")}


if __name__ == "__main__":
    main()
