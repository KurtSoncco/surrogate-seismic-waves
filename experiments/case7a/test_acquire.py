#!/usr/bin/env python3
"""Unit checks for LCMD / SBAL and basin / section mutation (no OpenSees, no GPU)."""

from __future__ import annotations

import numpy as np

from acquire_sections import lcmd, sbal
from basin_ensemble import mutate_basin, near_case7a, train_rows
from section_generator import mutate, near_case7, sample_rows


def test_lcmd_picks_diverse_and_avoids_train() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(8, 4))
    extra = train[:3] + 0.01 * rng.normal(size=(3, 4))
    far = rng.normal(loc=4.0, size=(5, 4))
    pool = np.vstack([extra, far])
    picks = lcmd(pool, train, k=3)
    assert len(picks) == 3 and len(set(picks)) == 3
    assert all(i >= 3 for i in picks), picks


def test_sbal_prefers_high_variance() -> None:
    rng = np.random.default_rng(1)
    pool = np.vstack([0.01 * rng.normal(size=(20, 3)), 4.0 * rng.normal(size=(5, 3))])
    picks = sbal(pool, k=4, rng=rng, beta=4.0)
    assert sum(i >= 20 for i in picks) >= 2


def test_mutations_leave_case7() -> None:
    rng = np.random.default_rng(2)
    sec = sample_rows(4, seed=3)[0]
    m = mutate(sec, rng, "m")
    assert m["sample_id"] == "m" and "mutated_key" in m and not near_case7(m)
    b = train_rows(4, seed=4)[0]
    mb = mutate_basin(b, rng, "bm")
    assert mb["sample_id"] == "bm" and not near_case7a(mb)


if __name__ == "__main__":
    test_lcmd_picks_diverse_and_avoids_train()
    test_sbal_prefers_high_variance()
    test_mutations_leave_case7()
    print("ok")
