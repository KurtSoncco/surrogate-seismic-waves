"""Stage-2 batches draw repeated-column sections at the near-1D share."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import lateral_log_std, resample_columns  # noqa: E402
from precision_query_shift import choose_carrier, power_at_period  # noqa: E402
from sampling import NEAR_1D_SHARE, QuotaBatchSampler  # noqa: E402


def test_fractional_offset_keeps_a_linear_station():
    width = 80
    start = 10
    offset = 0.5
    station = 40
    vs = np.arange(120, dtype=np.float64)[None, :].repeat(3, axis=0)
    src = np.arange(width, dtype=np.float64) + (start + offset)
    crop = resample_columns(vs, src)
    query = station - start - offset
    i0 = int(np.floor(query))
    t = query - i0
    read = (1.0 - t) * crop[0, i0] + t * crop[0, i0 + 1]
    assert abs(read - station) < 1e-8


def test_four_metre_power_and_carrier():
    x = np.arange(128)
    wave = np.sin(2 * np.pi * x / 4.0)
    assert power_at_period(wave) > power_at_period(np.sin(2 * np.pi * x / 40.0))
    assert choose_carrier({"baseline": 10.0, "hfp": 1.0, "local": 9.0, "grid": 1.0}) == "hfp"
    assert choose_carrier({"baseline": 10.0, "hfp": 9.5, "local": 9.5, "grid": 1.0}) == "both"


def test_a_repeated_column_has_no_lateral_scatter():
    column = np.linspace(140.0, 800.0, 40)[:, None]
    flat = np.repeat(column, 50, axis=1)
    ramp = flat.copy()
    ramp[:, 25:] *= 1.4
    assert lateral_log_std(flat) < 1e-8
    assert lateral_log_std(ramp) > 0.01


def _records() -> list[dict]:
    rows = [{"kind": "run", "family": "iid", "lx": 500, "near_1d": False, "stratum": "", "pair": False, "draw": i} for i in range(8)]
    for width in (500, 2000):
        for k in range(12):
            rows.append({
                "kind": "section", "family": "manifold", "lx": width, "width": width,
                "stratum": "flat" if k < 4 else "ramp",
                "near_1d": k < 3,
                "pair": False, "draw": len(rows),
                "f1_hz": 0.7, "f2_hz": 2.1,
            })
    return rows


def _section_draws(share: float, n_batches: int = 400) -> list[dict]:
    records = _records()
    sampler = QuotaBatchSampler(records, batch=4, stage=2, n_batches=n_batches, seed=0, use_pairs=False, near_1d_share=share)
    drawn = []
    for batch in sampler:
        for i in batch:
            if records[i]["kind"] == "section":
                drawn.append(records[i])
    return drawn


def test_half_the_manifold_slots_are_repeated_columns():
    drawn = _section_draws(NEAR_1D_SHARE)
    frac = np.mean([r["near_1d"] for r in drawn])
    assert 0.40 <= frac <= 0.60
    widths = {r["lx"] for r in drawn}
    assert widths == {500, 2000}
    # the Rayleigh pair on those rows stays (f1, 3 f1)
    assert all(abs(r["f2_hz"] / r["f1_hz"] - 3.0) < 1e-6 for r in drawn)


def test_zero_share_follows_the_base_rate():
    drawn = _section_draws(0.0)
    frac = np.mean([r["near_1d"] for r in drawn])
    # strata are drawn uniformly, so the repeated columns stay a minority
    assert frac < 0.45
