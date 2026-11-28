#!/usr/bin/env python3
"""Stage-0 direct and residual mid-band scores from the saved evaluations.

The numbers are the eval_test.json files already written for those checkpoints.
This script does not load the weights and does not retrain.

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

from plotting import apply_publication_style, save_figure

OUT = _HERE.parent
FILES = {
    "direct": OUT / "stage0_direct_s0_eval_test.json",
    "residual": OUT / "stage0_residual_s0_eval_test.json",
}
BINS = (
    ("iid_w500", "Horizontal", "500 m"),
    ("ood_dipping_w500", "Dipping", "500 m"),
    ("ood_three_layer_w500", "Three-layer", "500 m"),
)
MEANS = (
    ("iid_mean", "Horizontal", "mean of widths"),
    ("ood_dipping_mean", "Dipping", "mean of widths"),
    ("ood_three_layer_mean", "Three-layer", "mean of widths"),
)


def _row(payload: dict, key: str, family: str, scope: str, arm: str) -> dict:
    block = payload["bins"][key] if scope == "500 m" else payload[key]
    return {
        "arm": arm,
        "family": family,
        "scope": scope,
        "skill_mid": float(block["skill_mid"]),
        "log_l1_mid": float(block["log_l1_mid"]),
        "best_epoch": int(payload["best_epoch"]),
        "best_val": float(payload["best_val"]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in FILES.values() if not path.is_file()]
    if missing:
        payload = {"status": "blocked", "missing": missing}
        (OUT / "results.json").write_text(json.dumps(payload, indent=2))
        print(payload)
        return
    loaded = {arm: json.loads(path.read_text()) for arm, path in FILES.items()}
    rows = []
    for arm, payload in loaded.items():
        for key, family, scope in BINS:
            rows.append(_row(payload, key, family, scope, arm))
        for key, family, scope in MEANS:
            rows.append(_row(payload, key, family, scope, arm))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table_stage0.csv", index=False)
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "status": "from_eval_json",
                "caveat": "One seed. The training budget is not matched, so this is not a matched-budget comparison.",
                "rows": rows,
            },
            indent=2,
        )
    )
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.48, 3.3))
    use = table[table["scope"] == "500 m"]
    families = ["Horizontal", "Dipping", "Three-layer"]
    x = range(len(families))
    width = 0.36
    for offset, arm, color in ((-width / 2, "residual", "#0072B2"), (width / 2, "direct", "#D55E00")):
        part = use[use["arm"] == arm]
        vals = [float(part[part["family"] == fam]["skill_mid"].iloc[0]) for fam in families]
        ax.bar([i + offset for i in x], vals, width=width, color=color, label=arm)
    ax.set_xticks(list(x), families)
    ax.set_ylabel("Mid-band skill at 500 m")
    ax.legend(frameon=False, fontsize=6)
    ax.axhline(0.0, color="#000000", lw=0.4)
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
