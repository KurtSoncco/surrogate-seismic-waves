"""Sobol-sample splits and the manifold holdout."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_cache import manifold_holdout, split_by_group  # noqa: E402


def test_256_groups_are_208_24_24():
    groups = np.repeat(np.arange(256), 30)
    split = split_by_group(groups, seed=42)
    counts = {s: len({g for g, lab in zip(groups, split) if lab == s}) for s in ("train", "val", "test")}
    assert counts == {"train": 208, "val": 24, "test": 24}
    # every seed of a sample follows it
    for g in np.unique(groups):
        assert len(set(split[groups == g])) == 1


def test_blind_and_pairs_stay_together():
    rows = []
    for i, width in enumerate(("500.0", "1000.0", "1500.0", "2000.0")):
        for j in range(8):
            gid = f"g{width}-{j // 2}"  # two seeds per geology
            rows.append({
                "sample_id": f"s{i}-{j}", "status": "complete", "split": "train",
                "stratum": "low_cov", "lx_m": width, "group_id": gid,
            })
    rows.append({
        "sample_id": "blind0", "status": "complete", "split": "blind",
        "stratum": "blind", "lx_m": "1500.0", "group_id": "bg",
    })
    split = manifold_holdout(rows, seed=0)
    assert split["blind0"] == "blind"
    by_group: dict[str, set[str]] = {}
    for r in rows:
        if r["stratum"] == "blind":
            continue
        by_group.setdefault(r["group_id"], set()).add(split[r["sample_id"]])
    assert all(len(v) == 1 for v in by_group.values())
