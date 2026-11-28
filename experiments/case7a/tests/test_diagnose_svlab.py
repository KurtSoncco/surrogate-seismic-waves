"""Spatial-gate and convention-gap decisions. No checkpoint and no SVLAB cache."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnose_svlab import convention_tracks, spatial_gate_zone  # noqa: E402


def test_spatial_gate_passes_when_reads_agree():
    zone = spatial_gate_zone(0.96, 0.95, abs_log=0.01, spread=0.02)
    assert zone["passed"]
    assert zone["pearson_gap"] < 0.02


def test_spatial_gate_fails_on_pearson_gap():
    zone = spatial_gate_zone(0.96, 0.90, abs_log=0.01, spread=0.02)
    assert not zone["passed"]


def test_spatial_gate_fails_when_reads_disagree_beyond_the_overlap_spread():
    zone = spatial_gate_zone(0.96, 0.955, abs_log=0.08, spread=0.02)
    assert not zone["passed"]


def test_convention_tracks_only_a_material_gap():
    assert convention_tracks(delta_log_rms=0.20, gap_log_rms=0.12)
    assert not convention_tracks(delta_log_rms=0.20, gap_log_rms=0.04)
    assert not convention_tracks(delta_log_rms=0.20, gap_log_rms=0.02)
