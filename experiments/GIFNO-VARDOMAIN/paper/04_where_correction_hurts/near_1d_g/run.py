#!/usr/bin/env python3
"""CDFs of predicted and true |g| on held-out near-1D and blind CoV = 0 rows.

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

from paths import DUMP, FLOOR_JSON, current_pass
from plotting import apply_publication_style, save_figure

OUT = _HERE.parent


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.sort(values[np.isfinite(values)])
    if order.size == 0:
        return order, order
    return order, np.linspace(0.0, 1.0, order.size)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = DUMP / "records.csv"
    floor_path = FLOOR_JSON
    if not path.is_file() or not floor_path.is_file():
        (OUT / "results.json").write_text(json.dumps({"status": "waiting_for_dump"}, indent=2))
        print("waiting for the eval dump and the floor")
        return
    floor = float(json.loads(floor_path.read_text())["floor_midband_median_abs_g"])
    frame = pd.read_csv(path)
    part = frame[(frame["pass"] == current_pass(frame)) & (frame["family"] == "manifold") & (frame["near_1d"] == 1)]
    part = part[part["split"].isin(["val", "blind"])]
    pred = part["g_pred_abs_mid"].to_numpy()
    true = part["g_true_abs_mid"].to_numpy()
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "n": int(len(part)),
                "floor": floor,
                "median_predicted_abs_g": float(np.median(pred)) if pred.size else None,
                "median_true_abs_g": float(np.median(true)) if true.size else None,
            },
            indent=2,
        )
    )
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(3.54, 3.3))
    for values, color, label in ((true, "#000000", r"True $|g|$"), (pred, "#D55E00", r"Predicted $|g|$")):
        x, y = _cdf(values)
        ax.plot(x, y, color=color, lw=1.2, label=label)
    ax.axvline(floor, color="#E69F00", lw=1.0, label="Floor")
    ax.set_xlabel(r"Mid-band mean $|g|$")
    ax.set_ylabel("CDF")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    print(f"near-1D held-out n={len(part)}")


if __name__ == "__main__":
    main()
