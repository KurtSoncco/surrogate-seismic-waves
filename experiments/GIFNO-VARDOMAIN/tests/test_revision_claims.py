"""Provenance, aggregation, and the homogeneous-column identity for the revision."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
CASE7A = REPO / "experiments" / "case7a"
ANALYSIS = ROOT / "analysis"
for path in (CASE7A, ANALYSIS, ROOT / "paper" / "scripts"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import score_harness as harness  # noqa: E402

REQUIRED = harness.REQUIRED_PROVENANCE
HEADLINE_SIDECARS = (
    "ladder.provenance.json",
    "ladder_treasure.provenance.json",
    "baselines.provenance.json",
    "error_budget.provenance.json",
    "consistency_tests.provenance.json",
    "epoch_neighborhood.provenance.json",
    "case7_column.provenance.json",
    "cache_engine_check.provenance.json",
    "treasure_skill_identity.provenance.json",
    "dtype_stamp.provenance.json",
)


def test_homogeneous_column_outcrop_is_one():
    from section_1d import surface_over_outcrop

    freq = np.geomspace(0.1, 10.0, 48)
    vs = np.full(80, 400.0)
    got = surface_over_outcrop(freq, vs, dz=1.0, xi=np.zeros(freq.shape))
    np.testing.assert_allclose(got, 1.0, atol=1e-8)


def test_median_skill_is_not_one_minus_the_ratio_of_medians():
    model = np.array([0.01, 0.05, 0.09])
    prior = np.array([0.02, 0.04, 0.20])
    station_skill = 1.0 - model / prior
    ratio_of_medians = 1.0 - np.median(model) / np.median(prior)
    pooled = harness.skill_of_pooled(model, prior, floor=0.0)
    assert np.median(station_skill) != pytest.approx(ratio_of_medians)
    assert pooled == pytest.approx(1.0 - np.mean(model) / np.mean(prior))
    assert harness.mean_of_unit_skills(station_skill) != pytest.approx(pooled)


def test_treasure_identity_file_keeps_the_two_reductions_apart():
    path = ANALYSIS / "frozen" / "treasure_skill_identity.csv"
    if not path.is_file():
        pytest.skip("treasure identity table is not stamped")
    import pandas as pd

    frame = pd.read_csv(path).set_index("name")["value"]
    assert frame["median_station_skill"] != pytest.approx(frame["one_minus_ratio_of_medians"])
    assert frame["skill_of_mean_errors"] != pytest.approx(frame["one_minus_ratio_of_medians"])
    assert frame["median_station_skill"] != pytest.approx(frame["skill_of_mean_errors"])


def test_headline_sidecars_carry_the_required_fields():
    frozen = ANALYSIS / "frozen"
    for name in HEADLINE_SIDECARS:
        path = frozen / name
        if not path.is_file():
            pytest.skip(f"missing {name}")
        payload = json.loads(path.read_text())
        missing = [key for key in REQUIRED if key not in payload]
        assert missing == [], name


def test_case7_sidecar_does_not_claim_a_network_forward():
    path = ANALYSIS / "frozen" / "case7_column.provenance.json"
    if not path.is_file():
        pytest.skip("Case 7 column rescore has not been written")
    payload = json.loads(path.read_text())
    assert payload["network"] == "not_scored"
    assert payload["checkpoint"] is None
    assert "surface_over_outcrop" in payload["oned_engine"]


def test_parent_readme_is_marked_superseded():
    text = (ROOT / "paper" / "README.md").read_text()
    assert text.startswith("# Superseded")
    assert "geotech_v2/README.md" in text
