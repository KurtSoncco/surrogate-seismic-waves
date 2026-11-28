"""The cosine-bowl contact is not a training target."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import section_has_trough, trough_sample_ids  # noqa: E402
from generate_manifold import build  # noqa: E402


def test_trough_predicate_catches_primary_and_extra():
    assert section_has_trough("trough", "[]")
    assert not section_has_trough("ramp", "[]")
    extra = json.dumps([{"feature": 3, "feature_x_frac": 0.7, "feature_width_m": 80.0}])
    assert section_has_trough("step", extra)
    assert not section_has_trough("step", json.dumps([{"feature": 1}]))


def test_manifest_drops_the_solved_bowls():
    dropped = trough_sample_ids()
    assert "man_0001" in dropped
    assert len(dropped) >= 65


def test_new_manifold_draw_has_no_trough():
    rows = build(seed=11)
    assert rows
    assert all(row["feature_name"] != "trough" for row in rows)
    for row in rows:
        extras = json.loads(row.get("extra_features") or "[]")
        assert all(int(item["feature"]) != 3 for item in extras)
