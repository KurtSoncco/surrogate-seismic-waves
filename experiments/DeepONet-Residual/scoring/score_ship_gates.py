#!/usr/bin/env python3
"""Compare leftover mix-test JSONs to shipped M700 GINO gates.

    python scoring/score_ship_gates.py results/arch_train/M7680_gino_rebal_ft.json \\
        results/arch_train/M7680_gino_tf_ft.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SHIP_REL_L2 = {
    "iid": 0.371,
    "ood_dipping": 0.335,
    "ood_three_layer": 0.533,
}
IID_GATE = 0.371
DIP_GATE = 0.35
TL_SHIP = 0.533


def _rel(blob: dict[str, Any], domain: str) -> float | None:
    d = (blob.get("test_by_domain") or {}).get(domain) or {}
    if "rel_l2_TF" not in d:
        return None
    return float(d["rel_l2_TF"])


def score_one(path: Path) -> dict[str, Any]:
    blob = json.loads(path.read_text())
    iid = _rel(blob, "iid")
    dip = _rel(blob, "ood_dipping")
    tl = _rel(blob, "ood_three_layer")
    iid_ok = iid is not None and iid <= IID_GATE
    dip_ok = dip is not None and dip <= DIP_GATE
    tl_win = tl is not None and tl < TL_SHIP
    ship = bool(iid_ok and dip_ok and tl_win)
    return {
        "name": blob.get("name", path.stem),
        "path": str(path),
        "iid": iid,
        "dipping": dip,
        "three_layer": tl,
        "iid_gate": iid_ok,
        "dip_gate": dip_ok,
        "beats_ship_3l": tl_win,
        "ship": ship,
        "three_layer_kill": bool(blob.get("three_layer_kill")),
        "vs_ship": {
            "iid": None if iid is None else iid - SHIP_REL_L2["iid"],
            "dipping": None if dip is None else dip - SHIP_REL_L2["ood_dipping"],
            "three_layer": None if tl is None else tl - SHIP_REL_L2["ood_three_layer"],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsons", nargs="+", type=Path)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write combined comparison JSON (default: stdout only).",
    )
    args = p.parse_args()
    rows = []
    missing = []
    for path in args.jsons:
        if not path.is_file():
            missing.append(str(path))
            continue
        rows.append(score_one(path))
    report = {
        "ship_rel_l2_TF": SHIP_REL_L2,
        "gates": {
            "iid_max": IID_GATE,
            "dipping_max": DIP_GATE,
            "three_layer_lt": TL_SHIP,
        },
        "missing": missing,
        "runs": rows,
        "winner": None,
    }
    qualified = [r for r in rows if r["iid_gate"] and r["dip_gate"]]
    if qualified:
        winner = min(
            qualified,
            key=lambda r: r["three_layer"] if r["three_layer"] is not None else 9e9,
        )
        report["winner"] = winner["name"]
        report["ship_decision"] = bool(winner["ship"])
    else:
        report["ship_decision"] = False
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    if missing:
        sys.exit(2)
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
