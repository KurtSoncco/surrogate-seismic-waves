"""Collocation slots for cropped section windows."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_windows import STATION_SPACING, WINDOW_SLOTS, collocation_slots  # noqa: E402


def test_default_stride_is_the_21_stations():
    slots = collocation_slots(STATION_SPACING)
    np.testing.assert_array_equal(slots, WINDOW_SLOTS)
    assert slots.size == 21
    assert slots[10] == 250


def test_finer_stride_keeps_the_21_and_adds_columns():
    slots = collocation_slots(5)
    assert np.all(np.isin(WINDOW_SLOTS, slots))
    assert slots.size > 21
    assert 0 in slots and 495 in slots
    assert 105 in slots
