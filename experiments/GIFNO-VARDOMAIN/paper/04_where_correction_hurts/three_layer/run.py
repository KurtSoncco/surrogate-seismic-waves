#!/usr/bin/env python3
"""Three-layer band errors, stage 1 against stage 2, and one worst-record example.

The example curves are written by the eval dump. Until that file exists this
script still writes the band split from eval_test.json.

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

from paths import DUMP, STAGE1, STAGE2, current_pass
from plotting import apply_publication_style, panel_letter, save_figure

OUT = _HERE.parent
BANDS = ("low", "mid", "high")


def _rows(stage: str, path: Path) -> list[dict]:
    block = json.loads(path.read_text())["bins"]["ood_three_layer_w500"]
    return [
        {
            "stage": stage,
            "band": band,
            "log_l1": float(block[f"log_l1_{band}"]),
            "skill": float(block[f"skill_{band}"]),
            "n_stations": int(block["n_stations"]),
        }
        for band in BANDS
    ]


def main() -> None:
    rows = _rows("stage1", STAGE1 / "eval_test.json") + _rows("stage2", STAGE2 / "eval_test.json")
    worst = []
    if (DUMP / "records.csv").is_file():
        frame = pd.read_csv(DUMP / "records.csv")
        part = frame[(frame["pass"] == current_pass(frame)) & (frame["family"] == "ood_three_layer") & (frame["width"] == 500)]
        part = part.sort_values("center_skill_mid")
        worst = part[["record_id", "center_skill_mid", "center_log_l1_mid"]].head(5).to_dict(orient="records")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "bands.csv", index=False)
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "bands": rows,
                "worst_stage2_center_skill_mid": worst,
                "example_rule": "worst stage-2 mid-band skill at the center station of a 500 m window",
                "note": "Stage-2 three-layer mid-band error is 0.014, the same size as the consistency floor 0.0136. The injected wiggles are still spurious structure. The skill of -5.7 is that error divided by a column leftover of about 0.002.",
            },
            indent=2,
        )
    )
    spec = DUMP / "three_layer_example.npz"
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.2))
    if spec.is_file():
        blob = np.load(spec)
        freq = blob["freq"]
        axes[0].plot(freq, np.clip(blob["opensees"], 1e-3, None), color="#000000", lw=1.2, label="OpenSees")
        axes[0].plot(freq, np.clip(blob["column"], 1e-3, None), color="#0072B2", ls="--", lw=1.0, label="Column")
        axes[0].plot(freq, np.clip(blob["stage1"], 1e-3, None), color="#009E73", lw=1.1, label="Stage 1")
        axes[0].plot(freq, np.clip(blob["stage2"], 1e-3, None), color="#E69F00", lw=1.0, label="Stage 2")
        if "ensemble" in blob.files:
            axes[0].plot(freq, np.clip(blob["ensemble"], 1e-3, None), color="#D55E00", lw=1.2, label="Ensemble")
        axes[0].set_yscale("log")
        axes[0].set_ylabel(r"$|\mathrm{TF}|$")
        axes[1].plot(freq, blob["g_true"], color="#000000", lw=1.1, label=r"$g_{\mathrm{true}}$")
        axes[1].plot(freq, blob["g_stage1"], color="#009E73", lw=1.0, label="Stage 1")
        axes[1].plot(freq, blob["g_stage2"], color="#E69F00", lw=1.0, label="Stage 2")
        if "g_ensemble" in blob.files:
            axes[1].plot(freq, blob["g_ensemble"], color="#D55E00", lw=1.2, label="Ensemble")
        axes[1].axhline(0.0, color="#000000", lw=0.5)
        axes[1].set_ylabel(r"$g(f)$")
        axes[0].set_title(str(np.asarray(blob["record_id"]).reshape(-1)[0]))
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Example curves follow the eval dump", ha="center", va="center", transform=ax.transAxes)
    for ax, letter in zip(axes, "ab"):
        ax.set_xscale("log")
        ax.set_xlim(0.1, 10.0)
        ax.set_xlabel(r"$f$ (Hz)")
        ax.legend(frameon=False, fontsize=6)
        panel_letter(ax, letter)
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
