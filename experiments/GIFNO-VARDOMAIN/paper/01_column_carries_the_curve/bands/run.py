#!/usr/bin/env python3
"""Skill and absolute log-L1 by family and band, plus the Vs-based trusted frequency.

Stage 1 and stage 2 numbers are the station means already stored in eval_test.json.
f_max = Vs_min / (10 Δz) with Δz = 1 m, on soil cells of each test record.

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

from metrics import ROCK_VS, f_max_hz
from paths import CACHE, DUMP, FLOOR_JSON, STAGE1, STAGE2
from plotting import apply_publication_style, panel_letter, save_figure

OUT = _HERE.parent
FAMILIES = ("iid", "ood_dipping", "ood_three_layer")
LABEL = {"iid": "Horizontal", "ood_dipping": "Dipping", "ood_three_layer": "Three-layer"}
BANDS = ("low", "mid", "high")
WIDTH = {"iid": 500, "ood_dipping": 500, "ood_three_layer": 500}


def _eval_rows(stage: str, path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = []
    for family in FAMILIES:
        key = f"{family}_w{WIDTH[family]}"
        block = data["bins"][key]
        for band in BANDS:
            rows.append(
                {
                    "stage": stage,
                    "family": LABEL[family],
                    "band": band,
                    "skill": float(block[f"skill_{band}"]),
                    "log_l1": float(block[f"log_l1_{band}"]),
                    "n_stations": int(block["n_stations"]),
                }
            )
    return rows


def _fmax(family: str) -> np.ndarray:
    vs = np.load(CACHE / family / "vs.npy", mmap_mode="r")
    n_active = np.load(CACHE / family / "n_active.npy")
    split = np.asarray(np.load(CACHE / family / "split.npy")).astype(str)
    out = []
    for i in np.flatnonzero(split == "test"):
        field = np.asarray(vs[int(i), : int(n_active[int(i)])], dtype=np.float64)
        soil = field[field < ROCK_VS]
        use = soil if soil.size else field
        out.append(f_max_hz(float(np.min(use))))
    return np.asarray(out, dtype=np.float64)


def _full_band(family: str, name: str) -> dict | None:
    """0.1–10 Hz is the whole stored grid. Nothing is saved above 10 Hz."""
    path = DUMP / f"ensemble_center_tf_{family}.npz"
    if not path.is_file():
        path = DUMP / f"stage2_center_tf_{family}.npz"
    if not path.is_file():
        return None
    blob = np.load(path)
    runs = np.asarray(blob["run"], dtype=np.int64)
    log_model = np.log(np.clip(np.asarray(blob["tf"], dtype=np.float64), 1e-12, None))
    freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
    log_true = np.asarray(np.load(CACHE / family / "log_tf.npy", mmap_mode="r")[runs, 10], dtype=np.float64)
    log_column = np.asarray(np.load(CACHE / family / "log_tf1d.npy", mmap_mode="r")[runs, 10], dtype=np.float64)
    mask = (freq >= 0.1) & (freq <= 10.0)
    model = np.mean(np.abs(log_model[:, mask] - log_true[:, mask]), axis=1)
    column = np.mean(np.abs(log_column[:, mask] - log_true[:, mask]), axis=1)
    skill = 1.0 - model / np.maximum(column, 1e-6)
    return {
        "family": name,
        "band_hz": "0.1–10",
        "source": "evaluate.predict center station; grid ends at 10 Hz",
        "n": int(model.size),
        "model_log_l1_mean": float(np.mean(model)),
        "model_log_l1_median": float(np.median(model)),
        "column_log_l1_mean": float(np.mean(column)),
        "column_log_l1_median": float(np.median(column)),
        "skill_median": float(np.median(skill)),
        "skill_of_means": float(1.0 - np.mean(model) / max(float(np.mean(column)), 1e-6)),
    }


def main() -> None:
    rows = _eval_rows("stage1", STAGE1 / "eval_test.json") + _eval_rows("stage2", STAGE2 / "eval_test.json")
    if (DUMP / "ensemble_eval_test.json").is_file():
        rows += _eval_rows("ensemble", DUMP / "ensemble_eval_test.json")
    table = pd.DataFrame(rows)
    trusted = []
    for family in FAMILIES:
        vals = _fmax(family)
        trusted.append(
            {
                "family": LABEL[family],
                "n": int(vals.size),
                "f_max_median_hz": float(np.median(vals)),
                "f_max_p05_hz": float(np.quantile(vals, 0.05)),
                "dz_m": 1.0,
            }
        )
    envelope = None
    if FLOOR_JSON.is_file():
        envelope = float(json.loads(FLOOR_JSON.read_text())["empirical_f_max_hz"])
    center_band = []
    centers_path = DUMP / "stage2_centers.csv"
    if centers_path.is_file():
        centers = pd.read_csv(centers_path)
        centers = centers[centers["split"] == "test"]
        for family, name in (("iid", "Horizontal"), ("ood_dipping", "Dipping")):
            part = centers[centers["family"] == family]
            for band, model_col, column_col in (
                ("0.4–2.5", "model_log_l1_mid", "column_log_l1_mid"),
                ("0.4–3.3", "model_log_l1_0p4_3p3", "column_log_l1_0p4_3p3"),
            ):
                model = part[model_col].to_numpy()
                column = part[column_col].to_numpy()
                skill = 1.0 - model / np.maximum(column, 1e-6)
                center_band.append(
                    {
                        "family": name,
                        "band_hz": band,
                        "source": "evaluate.predict center station",
                        "n": int(model.size),
                        "model_log_l1_mean": float(np.mean(model)),
                        "model_log_l1_median": float(np.median(model)),
                        "column_log_l1_mean": float(np.mean(column)),
                        "column_log_l1_median": float(np.median(column)),
                        "skill_median": float(np.median(skill)),
                        "skill_of_means": float(1.0 - np.mean(model) / max(float(np.mean(column)), 1e-6)),
                    }
                )
            full = _full_band(family, name)
            if full is not None:
                center_band.append(full)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "bands.csv", index=False)
    pd.DataFrame(trusted).to_csv(OUT / "trusted_band.csv", index=False)
    pd.DataFrame(center_band).to_csv(OUT / "center_0p4_3p3.csv", index=False)
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "bands": rows,
                "trusted_band": trusted,
                "center_0p4_3p3_hz": center_band,
                "high_band_note": (
                    "The 2.5–10 Hz skill stays in the band table. The flat-column envelope "
                    f"leaves the consistency floor at {envelope:.1f} Hz."
                    if envelope is not None
                    else "The 2.5–10 Hz skill stays in the band table."
                ),
            },
            indent=2,
        )
    )

    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.3))
    families = [LABEL[name] for name in FAMILIES]
    x = np.arange(len(families) * len(BANDS), dtype=float)
    width = 0.36
    for ax, column, ylabel, letter in (
        (axes[0], "skill", "Skill", "a"),
        (axes[1], "log_l1", r"Absolute log-L1", "b"),
    ):
        stages = [("stage1", "#0072B2", "Stage 1"), ("stage2", "#E69F00", "Stage 2")]
        if (table["stage"] == "ensemble").any():
            stages.append(("ensemble", "#D55E00", "Ensemble"))
        span = 0.8 / len(stages)
        for i, (stage, color, label) in enumerate(stages):
            part = table[table["stage"] == stage]
            vals = [float(part[(part["family"] == fam) & (part["band"] == band)][column].iloc[0]) for fam in families for band in BANDS]
            ax.bar(x + (i - (len(stages) - 1) / 2) * span, vals, width=span * 0.9, color=color, label=label)
        ax.set_xticks(x, [f"{fam[:4]}\n{band}" for fam in families for band in BANDS], fontsize=6)
        ax.set_ylabel(ylabel)
        panel_letter(ax, letter)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    print(pd.DataFrame(trusted).to_string(index=False))


if __name__ == "__main__":
    main()
