"""Record bootstrap stays wide when stations inside a record are copies of each other."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "paper" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from bootstrap_ci import bootstrap_median  # noqa: E402
from metrics import BANDS  # noqa: E402


def test_bands_match_the_evaluator():
    sys.path.insert(0, str(SCRIPTS.parents[2]))
    from evaluate import BANDS as EVAL_BANDS

    assert BANDS == EVAL_BANDS


def test_shuffling_stations_inside_a_record_does_not_shrink_the_interval():
    rng = np.random.default_rng(0)
    record_value = np.linspace(-2.0, 2.0, 8)
    noise = rng.normal(scale=0.05, size=(8, 40))
    values = (record_value[:, None] + noise).ravel()
    record_ids = np.repeat(np.arange(8), 40)

    record = bootstrap_median(values, record_ids, n_draw=400, rng=np.random.default_rng(1))
    shuffled = values.copy()
    for i in range(8):
        sl = slice(i * 40, (i + 1) * 40)
        shuffled[sl] = rng.permutation(shuffled[sl])
    again = bootstrap_median(shuffled, record_ids, n_draw=400, rng=np.random.default_rng(1))
    assert again["low"] == pytest.approx(record["low"])
    assert again["high"] == pytest.approx(record["high"])

    # Treating stations as independent uses the copies as if they were new records.
    station = bootstrap_median(values, np.arange(values.size), n_draw=400, rng=np.random.default_rng(2))
    assert (record["high"] - record["low"]) > (station["high"] - station["low"])
