#!/usr/bin/env python3
"""Decision rule: the column when predicted |g| is below a threshold.

The threshold maximizes the unweighted mean of four validation-family median
hybrid skills: horizontal, dipping, three-layer, and manifold. A pooled median
would let the horizontal records hide the three-layer tail. Test rows, including
the manifold blind split, are reported at that threshold and do not choose it.
Lateral σ of log Vs stays the preset near-1D cut.

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

from data import NEAR_1D_LOG_STD, lateral_log_std
from metrics import soil_rows
from paths import CACHE, DUMP, FLOOR_JSON, current_pass
from plotting import apply_publication_style, save_figure

OUT = _HERE.parent
FAMILIES = ("iid", "ood_dipping", "ood_three_layer", "manifold")
LABEL = {"iid": "Horizontal", "ood_dipping": "Dipping", "ood_three_layer": "Three-layer", "manifold": "Manifold"}
COLOR = {"iid": "#0072B2", "ood_dipping": "#D55E00", "ood_three_layer": "#009E73", "manifold": "#000000"}


def _lateral() -> dict:
    root = CACHE / "ood_three_layer"
    vs = np.load(root / "vs.npy", mmap_mode="r")
    n_active = np.load(root / "n_active.npy")
    values = np.empty(n_active.shape[0], dtype=np.float64)
    for i, n in enumerate(n_active):
        values[i] = lateral_log_std(soil_rows(np.asarray(vs[i, : int(n)])))
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "fraction_below_near_1d_cut": float(np.mean(values < NEAR_1D_LOG_STD)),
        "near_1d_cut": NEAR_1D_LOG_STD,
        "rule": "Rejected. Three-layer interfaces are flat, but each soil layer is a random field. The switch is predicted |g|, not lateral σ.",
    }


def _slice(frame: pd.DataFrame, family: str, split: str | None) -> pd.DataFrame:
    part = frame[frame["family"] == family]
    if family == "manifold":
        part = part[part["split"] == split] if split is not None else part
    else:
        part = part[part["width"] == 500]
    return part


def _hybrid(part: pd.DataFrame, threshold: float) -> dict:
    pred = part["g_pred_abs_mid"].to_numpy()
    skill = part["center_skill_mid"].to_numpy()
    use_column = pred < threshold
    hybrid = np.where(use_column, 0.0, skill)
    return {
        "skill_median": float(np.median(hybrid)),
        "fraction_column": float(np.mean(use_column)),
        "n": int(part.shape[0]),
    }


def _validation_pool(dump: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One table per validation family. Synthetic rows come from the eval forward pass."""
    path = DUMP / "stage2_centers.csv"
    pools = {}
    if path.is_file():
        centers = pd.read_csv(path)
        centers = centers[centers["split"] == "val"].rename(columns={"skill_mid": "center_skill_mid"})
        for family in ("iid", "ood_dipping", "ood_three_layer"):
            part = centers[centers["family"] == family]
            if not part.empty:
                pools[LABEL[family]] = part
    manifold = dump[(dump["family"] == "manifold") & (dump["split"] == "val")]
    if not manifold.empty:
        pools["Manifold"] = manifold
    return pools


def _select(pools: dict[str, pd.DataFrame], thresholds: np.ndarray) -> tuple[list[dict], dict | None]:
    sweep = []
    best = None
    for threshold in thresholds:
        medians = []
        for label, part in pools.items():
            score = _hybrid(part, float(threshold))
            medians.append(score["skill_median"])
            sweep.append({"family": label, "split": "val", "threshold": float(threshold), **score, "role": "selection"})
        family_mean = float(np.mean(medians)) if medians else float("-inf")
        sweep.append(
            {
                "family": "Family mean",
                "split": "val",
                "threshold": float(threshold),
                "skill_median": family_mean,
                "fraction_column": None,
                "n": int(sum(len(part) for part in pools.values())),
                "role": "selection_score",
            }
        )
        if best is None or family_mean > best["skill_median"]:
            best = {"threshold": float(threshold), "skill_median": family_mean}
    return sweep, best


def main() -> None:
    lateral = _lateral()
    lateral["n_below_near_1d_cut"] = int(round(lateral["fraction_below_near_1d_cut"] * lateral["n"]))
    sweep = []
    chosen = None
    confirmation = []
    floor = None
    pools: dict[str, pd.DataFrame] = {}
    if FLOOR_JSON.is_file() and (DUMP / "records.csv").is_file():
        floor = float(json.loads(FLOOR_JSON.read_text())["floor_midband_median_abs_g"])
        frame_all = pd.read_csv(DUMP / "records.csv")
        shown = current_pass(frame_all)
        has_val = ((frame_all["pass"] == shown) & (frame_all["split"] == "val") & (frame_all["family"] == "iid")).any()
        if shown == "ensemble" and not has_val:
            shown = "stage2"
        frame = frame_all[frame_all["pass"] == shown]
        thresholds = np.geomspace(max(floor, 1e-4) / 10.0, max(floor, 1e-4) * 10.0, 21)
        pools = _validation_pool(frame)
        if len(pools) == 4:
            sweep, best = _select(pools, thresholds)
            if best is not None:
                per_family = {}
                for label, part in pools.items():
                    per_family[label] = _hybrid(part, best["threshold"])
                chosen = {
                    "threshold": best["threshold"],
                    "selection_score": best["skill_median"],
                    "families": per_family,
                    "objective": "Unweighted mean of the four validation-family median hybrid skills.",
                }
            report = (
                ("manifold", "blind", "Manifold blind"),
                ("iid", None, "Horizontal"),
                ("ood_dipping", None, "Dipping"),
                ("ood_three_layer", None, "Three-layer"),
            )
            if chosen is not None:
                for family, split, label in report:
                    part = _slice(frame, family, split)
                    if part.empty:
                        continue
                    score = _hybrid(part, chosen["threshold"])
                    confirmation.append({"family": label, "threshold": chosen["threshold"], **score, "role": "confirmation"})
                    for threshold in thresholds:
                        sweep.append({"family": label, "threshold": float(threshold), **_hybrid(part, float(threshold)), "role": "confirmation"})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "lateral_log_std": lateral,
                "floor": floor,
                "validation_families": list(pools),
                "chosen_on_validation": chosen,
                "confirmation": confirmation,
                "n_sweep": len(sweep),
                "rule": "The |g| threshold maximizes the unweighted mean of validation-family median hybrid skills, including three-layer. Blind and the synthetic test families do not choose it. The lateral-std cut stays the preset 0.01.",
            },
            indent=2,
        )
    )
    if sweep:
        pd.DataFrame(sweep).to_csv(OUT / "sweep.csv", index=False)
        apply_publication_style()
        fig, ax = plt.subplots(figsize=(7.48, 3.3))
        table = pd.DataFrame(sweep)
        table = table[table["role"].isin(["selection", "selection_score"])]
        shown = {"Horizontal": "#0072B2", "Dipping": "#D55E00", "Three-layer": "#009E73", "Manifold": "#000000", "Family mean": "#56B4E9"}
        for family, color in shown.items():
            part = table[table["family"] == family]
            if part.empty:
                continue
            style = "--" if family == "Family mean" else "-"
            ax.plot(part["threshold"], part["skill_median"], color=color, lw=1.2, ls=style, label=family)
        if chosen is not None:
            ax.axvline(chosen["threshold"], color="#E69F00", lw=1.0, label="Chosen on validation")
        ax.set_xscale("log")
        ax.set_xlabel(r"Predicted $|g|$ threshold")
        ax.set_ylabel("Hybrid mid-band skill")
        ax.legend(frameon=False, fontsize=6)
        fig.tight_layout()
        save_figure(fig, OUT / "central.pdf")
    print(json.dumps({"lateral": lateral, "chosen": chosen, "confirmation": confirmation}, indent=2))


if __name__ == "__main__":
    main()
