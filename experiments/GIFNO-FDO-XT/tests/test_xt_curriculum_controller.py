"""Tests for the convergence-gated band curriculum controller."""

from __future__ import annotations

import numpy as np

from curriculum import BandCurriculumController


def test_starts_in_low_only_phase():
    c = BandCurriculumController(n_bands=3, floor=0.25, patience=5, ramp_epochs=4)
    assert c.phase == 0
    assert not c.is_final_phase
    w = c.current_weights()
    assert w[0] == 1.0
    assert np.isclose(w[1], 0.25)
    assert np.isclose(w[2], 0.25)


def test_advances_after_patience_plateau_epochs():
    c = BandCurriculumController(n_bands=3, patience=5, min_delta=1e-4, ramp_epochs=1)
    # The first in-phase epoch establishes the baseline (inf -> metric), then
    # `patience` non-improving epochs trigger the advance: patience + 1 calls.
    steps = 0
    while True:
        steps += 1
        if c.step(1.0):
            break
        assert steps < 20  # guard against runaway
    assert steps == 6
    assert c.phase == 1


def test_improving_metric_does_not_advance():
    c = BandCurriculumController(n_bands=3, patience=3, min_delta=1e-4, ramp_epochs=1)
    metric = 1.0
    for _ in range(20):
        metric -= 0.01  # steady improvement
        assert c.step(metric) is False
    assert c.phase == 0


def test_final_phase_never_advances():
    c = BandCurriculumController(n_bands=3, patience=2, min_delta=1e-4, ramp_epochs=1)
    # Force to final phase.
    while not c.is_final_phase:
        c.step(1.0)
    assert c.phase == 2
    for _ in range(50):
        assert c.step(1.0) is False
    assert c.phase == 2


def test_newly_activated_band_ramps_from_floor():
    c = BandCurriculumController(n_bands=3, floor=0.2, patience=2, ramp_epochs=4)
    # Advance to phase 1 (mid activating).
    while c.phase == 0:
        c.step(1.0)
    assert c.phase == 1
    # epochs_in_phase resets to 0 on advance -> mid starts at floor.
    w0 = c.current_weights()
    assert w0[0] == 1.0  # low fully active
    assert np.isclose(w0[1], 0.2)  # mid at floor at phase entry
    assert np.isclose(w0[2], 0.2)  # high still inactive
    # March through the ramp; mid increases monotonically toward 1.
    mids = [c.current_weights()[1]]
    for _ in range(6):
        c.step(1.0)  # plateau keeps us progressing (will advance again eventually)
        mids.append(c.current_weights()[1])
    assert all(b >= a - 1e-9 for a, b in zip(mids, mids[1:]))
    assert max(mids) <= 1.0 + 1e-9
