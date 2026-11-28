#!/usr/bin/env python3
"""Quarantine the mixed stage comparison.

Stage 1 and stage 2 skills are means of per-station skills. P128n128 skills
in single_model.py are ratios of pooled errors. Those reductions are not one
table, so this script no longer writes them side by side.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "06_reliability_and_broadband"


def main() -> None:
    superseded = OUT / "superseded"
    superseded.mkdir(parents=True, exist_ok=True)
    (superseded / "QUARANTINE.txt").write_text(
        "Quarantined. Stage 1 and stage 2 skills are means of per-station skills. "
        "The P128n128 skill is a ratio of pooled errors. The generated narrative does not read this folder.\n"
    )
    pd.DataFrame([{"status": "quarantined_mixed_aggregation"}]).to_csv(
        superseded / "family_comparison.csv", index=False,
    )
    (HERE / "split_alignment.json").write_text('{"status": "quarantined_mixed_aggregation"}\n')
    print("quarantined mixed stage comparison")


if __name__ == "__main__":
    main()
