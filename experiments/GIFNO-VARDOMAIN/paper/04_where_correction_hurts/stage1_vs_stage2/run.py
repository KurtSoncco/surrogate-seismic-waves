#!/usr/bin/env python3
"""Mid-band skill and log-L1 from stage 1 to stage 2, one line per family.

Horizontal, dipping, and three-layer use the saved eval_test.json station means.
Manifold validation chose epoch 53. The blind manifold line did not enter that score.
Stage 2 gives up accuracy on the three synthetic families and gains it on the manifold.

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
import pandas as pd

from metrics import above_floor_skill
from paths import DUMP, STAGE1, STAGE2
from plotting import apply_publication_style, save_figure

OUT = _HERE.parent
FAMILIES = {
    "iid_w500": "Horizontal",
    "ood_dipping_w500": "Dipping",
    "ood_three_layer_w500": "Three-layer",
}
COLOR = {
    "Horizontal": "#0072B2",
    "Dipping": "#D55E00",
    "Three-layer": "#009E73",
    "Manifold val": "#000000",
    "Manifold blind": "#56B4E9",
}
TRADEOFF = (
    "Stage 2 raises mid-band log-L1 on the horizontal, dipping, and three-layer "
    "test families. It lowers that error on the manifold. The blind column was "
    "not used to pick epoch 53."
)


def _bin(path: Path, key: str) -> dict:
    block = json.loads(path.read_text())["bins"][key]
    return {"skill": float(block["skill_mid"]), "log_l1": float(block["log_l1_mid"])}


def _floor(frame: pd.DataFrame, stage: str, family: str, split: str | None) -> dict:
    part = frame[(frame["pass"] == stage) & (frame["family"] == family)]
    if split is not None:
        part = part[part["split"] == split]
    if family != "manifold":
        part = part[part["width"] == 500]
    if part.empty:
        return above_floor_skill([], [])
    return above_floor_skill(part["center_column_l1_mid"], part["center_skill_mid"])


def _manifold(frame: pd.DataFrame, split: str) -> dict | None:
    part = frame[(frame["family"] == "manifold") & (frame["split"] == split)]
    if part.empty:
        return None
    out = {}
    for stage, key in (("stage1", "stage1"), ("stage2", "stage2")):
        rows = part[part["pass"] == stage]
        out[key] = {
            "skill": float(rows["mean_skill_mid"].median()),
            "log_l1": float(rows["center_log_l1_mid"].mean()),
        }
    ens = part[part["pass"] == "ensemble"]
    if not ens.empty:
        out["ensemble"] = {
            "skill": float(ens["mean_skill_mid"].median()),
            "log_l1": float(ens["center_log_l1_mid"].mean()),
        }
    return out


def _with_floor(row: dict, frame: pd.DataFrame | None, family: str, split: str | None) -> dict:
    if frame is None:
        return row
    for stage, key in (("stage1", "stage1"), ("stage2", "stage2")):
        got = _floor(frame, stage, family, split)
        row[f"{key}_floor_skill"] = None if got["skill"] is None else float(got["skill"])
        row[f"{key}_n_above_floor"] = int(got["n_above_floor"])
    return row


def main() -> None:
    rows = []
    frame = pd.read_csv(DUMP / "records.csv") if (DUMP / "records.csv").is_file() else None
    family_code = {"Horizontal": "iid", "Dipping": "ood_dipping", "Three-layer": "ood_three_layer"}
    for key, label in FAMILIES.items():
        first = _bin(STAGE1 / "eval_test.json", key)
        second = _bin(STAGE2 / "eval_test.json", key)
        row = {
            "family": label,
            "split": "test",
            "stage1_skill": first["skill"],
            "stage2_skill": second["skill"],
            "stage1_log_l1": first["log_l1"],
            "stage2_log_l1": second["log_l1"],
        }
        ens_path = DUMP / "ensemble_eval_test.json"
        if ens_path.is_file():
            third = _bin(ens_path, key)
            row["ensemble_skill"] = third["skill"]
            row["ensemble_log_l1"] = third["log_l1"]
        rows.append(_with_floor(row, frame, family_code[label], "test"))
    if frame is not None:
        for split, label in (("val", "Manifold val"), ("blind", "Manifold blind")):
            got = _manifold(frame, split)
            if got is None:
                continue
            rows.append(
                _with_floor(
                    {
                        "family": label,
                        "split": split,
                        "stage1_skill": got["stage1"]["skill"],
                        "stage2_skill": got["stage2"]["skill"],
                        "stage1_log_l1": got["stage1"]["log_l1"],
                        "stage2_log_l1": got["stage2"]["log_l1"],
                        **(
                            {"ensemble_skill": got["ensemble"]["skill"], "ensemble_log_l1": got["ensemble"]["log_l1"]}
                            if "ensemble" in got
                            else {}
                        ),
                    },
                    frame,
                    "manifold",
                    split,
                )
            )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "table2.csv", index=False)
    (OUT / "results.json").write_text(json.dumps({"tradeoff": TRADEOFF, "rows": rows}, indent=2))
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.4))
    for ax, keys, ylabel in (
        (axes[0], ("stage1_skill", "stage2_skill", "ensemble_skill"), "Mid-band skill"),
        (axes[1], ("stage1_log_l1", "stage2_log_l1", "ensemble_log_l1"), "Mid-band log-L1"),
    ):
        labels = ["Stage 1", "Stage 2", "Ensemble"]
        for row in rows:
            use = [(i, row[key]) for i, key in enumerate(keys) if key in row]
            ax.plot(
                [i for i, _ in use],
                [value for _, value in use],
                color=COLOR[row["family"]],
                lw=1.2,
                marker="o",
                label=row["family"],
            )
        ax.set_xticks(range(3), labels)
        ax.set_ylabel(ylabel)
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    print(TRADEOFF)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
