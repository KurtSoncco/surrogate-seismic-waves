"""Phase 4 pilot checks and the Phase 6.1 length-scaling identity.

No OpenSees. Nested crops must match the master field bitwise. Scored stations
stay out of the free-field pads. Explicit and Q(Vs) rows go through
``run_sections._resolve_damping``. A homogeneous column is TF ≈ 1 on the
existing ``section_1d`` outcrop. Paired bootstrap groups by master id.
Length scaling is a column identity, not a change to training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CASE_DIR = Path(__file__).resolve().parents[1]
GD_DIR = CASE_DIR.parent / "GIFNO-VARDOMAIN"
sys.path.insert(0, str(GD_DIR))
sys.path.insert(0, str(GD_DIR / "paper" / "scripts"))
sys.path.insert(0, str(CASE_DIR))

import generate_manifold  # noqa: E402
import pilot_manifest as pilot  # noqa: E402
import section_generator as sg  # noqa: E402
from bootstrap_ci import bootstrap_cluster_statistic  # noqa: E402
from case7_crops import quality_factor, qvs_xi  # noqa: E402
from run_sections import PAD_M, _resolve_damping, pad_free_field, scored_surface_x  # noqa: E402
from section_1d import (  # noqa: E402
    outcrop_tf_rayleigh,
    rayleigh_coefficients,
    surface_over_outcrop,
)

FREQ = np.geomspace(0.1, 2.5, 48)


def test_pilot_design_and_depth_default_stays_150():
    rows = pilot.build_pilot()
    assert sg.DEPTH_M == 150.0
    assert len(rows) == 8 * 4 * 3 * 2 + 8 * 4 * 2
    assert {r["master_id"] for r in rows} == {f"p4m{i:02d}" for i in range(8)}
    assert {float(r["lx_m"]) for r in rows} == {500.0, 1000.0, 1500.0, 2000.0}
    assert {r["soil_fill"] for r in rows} == {"gradient", "layered"}
    assert all(float(r["depth_m"]) == 250.0 for r in rows)
    assert all(float(r["master_lx_m"]) == 2000.0 for r in rows)
    assert all(r["damping_family"] == "explicit" for r in rows)
    assert all("duration_s" not in r for r in rows)
    cov0 = [r for r in rows if r["cov_slice"] == 1]
    random = [r for r in rows if r["cov_slice"] == 0]
    assert len(cov0) == 8 * 4 * 2
    assert len(random) == 8 * 4 * 3 * 2
    assert all(float(r["vs_cov"]) == 0.0 for r in cov0)
    assert all(float(r["vs_cov"]) > 0.0 for r in random)
    # One zeta per master, shared by every width: damping is not recomputed from the crop.
    for master_id in {r["master_id"] for r in rows}:
        zetas = {float(r["zeta_soil"]) for r in rows if r["master_id"] == master_id}
        anchors = {float(r["rayleigh_f1_hz"]) for r in rows if r["master_id"] == master_id}
        assert len(zetas) == 1
        assert len(anchors) == 1
        f0 = max(0.2, float(next(r["f0_flat_hz"] for r in rows if r["master_id"] == master_id)))
        assert abs(anchors.pop() - f0) >= 0.05
    timing = pilot.timing_rows(rows)
    assert [t["timing_s"] for t in timing] == [200, 400]
    assert len({t["master_id"] for t in timing}) == 1
    assert len({t["manifest_index"] for t in timing}) == 1
    assert float(rows[timing[0]["manifest_index"]]["lx_m"]) == 2000.0


def test_nested_crop_matches_the_master_bitwise():
    rows = pilot.build_pilot()
    master_id = "p4m00"
    for fill in ("gradient", "layered"):
        family = [
            r for r in rows
            if r["master_id"] == master_id and r["soil_fill"] == fill and r["sample_id"].endswith("_s0")
        ]
        assert len(family) == 4
        fields = {float(r["lx_m"]): sg.section_field(r) for r in family}
        wide_vs, wide_zeta, wide_rock = fields[2000.0]
        for width, (vs, zeta, rock) in fields.items():
            i0, i1 = sg.nested_crop_bounds(2000.0, width, next(r["crop_x0_m"] for r in family if r["lx_m"] == width))
            np.testing.assert_array_equal(vs, wide_vs[:, i0:i1])
            np.testing.assert_array_equal(zeta, wide_zeta[:, i0:i1])
            np.testing.assert_array_equal(rock, wide_rock[:, i0:i1])
        short = next(r for r in family if r["lx_m"] == 500.0)
        regenerated = dict(short)
        regenerated.pop("master_lx_m")
        regenerated.pop("crop_x0_m")
        again = sg.section_field(regenerated)[0]
        assert again.shape[1] == 500
        assert not np.array_equal(again, fields[500.0][0])


def test_csv_roundtrip_keeps_the_crop_identity(tmp_path: Path):
    rows = [r for r in pilot.build_pilot() if r["sample_id"] in ("p4m03_w0500_gradient_s1", "p4m03_w2000_gradient_s1")]
    path = sg.write_rows(rows, tmp_path / "pilot.csv")
    loaded = sg.read_rows(path)
    by_id = {r["sample_id"]: r for r in loaded}
    wide = sg.section_field(by_id["p4m03_w2000_gradient_s1"])[0]
    crop = sg.section_field(by_id["p4m03_w0500_gradient_s1"])
    i0, i1 = sg.nested_crop_bounds(2000.0, 500.0, by_id["p4m03_w0500_gradient_s1"]["crop_x0_m"])
    np.testing.assert_array_equal(crop[0], wide[:, i0:i1])
    assert sg.DEPTH_M == 150.0


def test_scored_stations_are_outside_the_pads():
    assert PAD_M == 500.0
    for interior in (500.0, 1000.0, 1500.0, 2000.0):
        x = scored_surface_x(interior, PAD_M, 1.0)
        assert np.all(x >= PAD_M)
        assert np.all(x < PAD_M + interior)
        n_elem = pad_free_field(np.ones((1, int(interior))), PAD_M, 1.0).shape[1]
        nodes = np.arange(n_elem + 1, dtype=np.float64)
        inside_pad = (nodes < PAD_M) | (nodes >= PAD_M + interior)
        assert not np.intersect1d(x, nodes[inside_pad]).size
        assert x.size == int(interior)


def test_explicit_row_keeps_csv_zeta_and_qvs_row_replaces_it():
    vs = np.array([[180.0, 240.0, 1400.0], [200.0, 260.0, 1600.0]], dtype=np.float64)
    rock = np.array([[False, False, True], [False, False, True]])
    zeta = np.where(rock, 0.004, 0.02)
    explicit = {"damping_family": "explicit", "zeta_soil": 0.02, "zeta_rock": 0.004}
    qvs = dict(explicit, damping_family="qvs")
    got_row, got_zeta = _resolve_damping(explicit, vs, zeta, rock)
    assert got_row["damping_family"] == "explicit"
    assert got_row["zeta_soil"] == 0.02
    np.testing.assert_array_equal(got_zeta, zeta)
    q_row, q_zeta = _resolve_damping(qvs, vs, zeta, rock)
    xi_soil, xi_rock = qvs_xi(vs, rock)
    assert q_row["damping_family"] == "qvs"
    assert q_row["zeta_soil"] == pytest.approx(xi_soil)
    assert q_row["zeta_rock"] == pytest.approx(xi_rock)
    np.testing.assert_allclose(q_zeta, np.where(rock, xi_rock, xi_soil))
    assert abs(q_row["zeta_soil"] - explicit["zeta_soil"]) > 1e-4


def test_homogeneous_column_outcrop_is_one():
    """Undamped homogeneous column is 1 on ``surface_over_outcrop``.

    That is the corrected half-space split (up-going amplitude in the bottom
    cell). A uniform hysteretic xi is not this check: the same xi still
    attenuates over the column thickness, so |TF| falls with frequency.
    """
    freq = np.geomspace(0.1, 10.0, 40)
    vs = np.full(80, 400.0)
    undamped = surface_over_outcrop(freq, vs, dz=1.0, xi=np.zeros(freq.shape))
    np.testing.assert_allclose(undamped, 1.0, atol=1e-8)
    exact = np.abs(outcrop_tf_rayleigh(freq, vs, 0.0, 0.0, dz=1.0))
    np.testing.assert_allclose(undamped, exact, rtol=1e-8, atol=1e-8)


def test_paired_bootstrap_groups_by_master_not_width():
    rows = pilot.build_pilot()
    groups = [sg.paired_bootstrap_group(r) for r in rows]
    assert groups == [r["master_id"] for r in rows]
    one = [r for r in rows if r["master_id"] == "p4m01" and r["soil_fill"] == "gradient" and r["sample_id"].endswith("_s0")]
    assert len({float(r["lx_m"]) for r in one}) == 4
    assert len({sg.paired_bootstrap_group(r) for r in one}) == 1
    assert sg.paired_bootstrap_group(one[0]) != one[0]["sample_id"]
    with pytest.raises(ValueError):
        sg.paired_bootstrap_group({"sample_id": "crop_only", "lx_m": 500.0})
    stat = bootstrap_cluster_statistic(
        np.array([float(r["lx_m"]) for r in rows]),
        np.array(groups),
        np.mean,
        n_draw=30,
        rng=np.random.default_rng(0),
    )
    assert stat["n_clusters"] == 8
    assert stat["n_records"] == len(rows)


def test_generate_manifold_still_draws_each_width_on_its_own():
    rows = generate_manifold.build(0)
    assert rows
    assert all("master_lx_m" not in r for r in rows)
    assert {int(r["lx_m"]) for r in rows} >= {500, 1000, 1500, 2000}


def test_length_scaling_hysteretic_and_rayleigh_anchors():
    """Lengths × k, velocities unchanged: TF_k(f) = TF(k · f).

    This is a 1D column check. It is not a wider window of the same soil.
    k = 4 on a 500 m stack is a 2000 m stack that is also 4× deeper.

    Hysteretic xi that depends only on Vs, including Q(Vs) = 1 / (2 Q), is
    exact: xi does not change when lengths change.

    Rayleigh matches that identity only when the anchors scale with the site
    frequency. The thicker column is anchored at (f1 / k, f2 / k). Under
    TF_k(f) = TF(k · f) those anchors are the original (f1, f2) read on the
    scaled frequency axis, which is the plan's "scale (f1, f2) by k".
    Leaving (f1, f2) fixed on the thicker column breaks the identity.
    Multiplying the thicker column's anchors by k breaks it as well.
    """
    k = 4.0
    vs = np.r_[np.full(24, 160.0), np.full(16, 420.0), np.full(12, 1100.0)]
    xi_cells = 1.0 / (2.0 * quality_factor(vs))
    assert np.all(xi_cells > 0.0)
    xi = np.repeat(xi_cells[:, None], FREQ.size, axis=1)
    original = surface_over_outcrop(k * FREQ, vs, dz=1.0, xi=xi)
    scaled = surface_over_outcrop(FREQ, vs, dz=k, xi=xi)
    np.testing.assert_allclose(scaled, original, rtol=1e-6, atol=1e-8)

    zeta, f1, f2 = 0.03, 0.4, 1.2
    a0, a1 = rayleigh_coefficients(zeta, f1, f2)
    reference = np.abs(outcrop_tf_rayleigh(k * FREQ, vs, a0, a1, dz=1.0))
    a0_fixed, a1_fixed = rayleigh_coefficients(zeta, f1, f2)
    fixed = np.abs(outcrop_tf_rayleigh(FREQ, vs, a0_fixed, a1_fixed, dz=k))
    assert np.max(np.abs(fixed - reference)) > 1e-3
    a0_up, a1_up = rayleigh_coefficients(zeta, k * f1, k * f2)
    multiplied = np.abs(outcrop_tf_rayleigh(FREQ, vs, a0_up, a1_up, dz=k))
    assert np.max(np.abs(multiplied - reference)) > 1e-3
    a0_k, a1_k = rayleigh_coefficients(zeta, f1 / k, f2 / k)
    matched = np.abs(outcrop_tf_rayleigh(FREQ, vs, a0_k, a1_k, dz=k))
    np.testing.assert_allclose(matched, reference, rtol=1e-6, atol=1e-8)
