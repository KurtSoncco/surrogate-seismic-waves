"""Crop geometry and Q–Vs damping. No OpenSees."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from case7_crops import (  # noqa: E402
    CROPS,
    INNER_NX,
    LX_SOLVE_M,
    PAD_NX,
    cut_inner,
    extend_profile,
    prepare_crop,
    quality_factor,
    qvs_xi,
    rayleigh_pair,
)
from case7a import NX, ROCK_VS, window_column_start  # noqa: E402
from run_case7_crops import inner_surface  # noqa: E402


def test_five_crops_are_the_loglo_windows():
    assert [c["center_x_m"] for c in CROPS] == [-1500.0, 0.0, 800.0, 1175.0, 1605.0]
    vs = np.zeros((150, 4000))
    vs[:] = np.arange(4000)[None, :]
    for crop in CROPS:
        c0 = window_column_start(crop["center_x_m"])
        inner, got = cut_inner(vs, crop["center_x_m"])
        assert got == c0
        assert inner.shape == (150, NX)
        assert inner[0, 0] == c0
        assert crop["center_x_m"] - 250.0 == -2000.0 + c0


def test_extend_pads_edges_the_way_seiskit_does():
    """Left pad is n_pad // 2, matching seiskit ``_extend_profile``."""
    rng = np.random.default_rng(0)
    inner = rng.normal(size=(8, INNER_NX))
    extended = extend_profile(inner)
    assert extended.shape == (8, int(LX_SOLVE_M))
    assert np.allclose(extended[:, :PAD_NX], inner[:, :1])
    assert np.allclose(extended[:, PAD_NX : PAD_NX + INNER_NX], inner)
    assert np.allclose(extended[:, PAD_NX + INNER_NX :], inner[:, -1:])


def _damping():
    import importlib.util

    path = Path("/home/kurt-/seiskit/seiskit/damping.py")
    spec = importlib.util.spec_from_file_location("seiskit_damping_direct", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_qvs_matches_seiskit_global_avg():
    damping = _damping()
    for vs in (140.0, 400.0, 2000.0):
        assert np.isclose(quality_factor(vs), damping.compute_quality_factor(vs))
    field = np.array([[200.0, 300.0], [1800.0, 2200.0]])
    rock = np.array([[False, False], [True, True]])
    xi_soil, xi_rock = qvs_xi(field, rock)
    q_soil = [damping.compute_quality_factor(200.0), damping.compute_quality_factor(300.0)]
    assert np.isclose(xi_soil, damping.compute_average_damping_harmonic(q_soil))
    median_vs = float(np.median([1800.0, 2200.0]))
    assert np.isclose(xi_rock, damping.compute_damping_from_Q(damping.compute_quality_factor(median_vs)))


def test_prepare_uses_extended_mesh_for_xi_and_center_f0():
    vs = np.full((150, 4000), 200.0)
    vs[80:, :] = 2000.0
    pack = prepare_crop(vs, CROPS[1], ROCK_VS)
    assert pack["vs_extended"].shape == (150, 1500)
    assert pack["zeta_inner"].shape == (150, 500)
    assert np.allclose(pack["zeta_inner"][:80], pack["xi_soil"])
    assert np.allclose(pack["zeta_inner"][80:], pack["xi_rock"])
    assert pack["xi_soil"] > pack["xi_rock"]
    f0, f1, f2 = rayleigh_pair(pack["vs_inner"], ROCK_VS)
    assert np.isclose(f0, 0.25 / (80.0 / 200.0))
    assert np.isclose(pack["f0_hz"], f0)
    assert np.isclose(f2, 3.0 * f1)


def test_wider_crop_keeps_500_m_pads_and_slides_onto_the_mesh():
    from case7_crops import context_jobs, prepare_crop

    vs = np.full((150, 4000), 200.0)
    vs[80:, :] = 2000.0
    west = next(j for j in context_jobs() if j["sample_id"] == "crop_west_w1500")
    pack = prepare_crop(vs, west, ROCK_VS)
    assert pack["inner_m"] == 1500
    assert pack["pad_m"] == 500
    assert pack["lx_solve_m"] == 2500
    assert pack["vs_extended"].shape == (150, 2500)
    assert pack["crop_x0_m"] == -2000.0
    assert pack["mesh_shift_m"] == 250.0
    assert pack["crop_x0_m"] <= west["center_x_m"] < pack["crop_x1_m"]
    full = next(j for j in context_jobs() if j["sample_id"] == "full_qvs")
    pack = prepare_crop(vs, full, ROCK_VS)
    assert pack["vs_extended"].shape == (150, 5000)
    assert pack["mesh_shift_m"] == 0.0
    assert pack["crop_x0_m"] == -2000.0 and pack["crop_x1_m"] == 2000.0


def test_svlab_damping_is_the_case7_zeta_pair():
    from case7_crops import SVLAB_XI_SOIL, prepare_crop

    vs = np.full((150, 4000), 200.0)
    vs[80:, :] = 2000.0
    job = {"sample_id": "full_svlab", "name": "full_section", "center_x_m": 0.0, "inner_m": 4000.0, "damping": "svlab"}
    pack = prepare_crop(vs, job, ROCK_VS)
    assert pack["damping"] == "svlab"
    assert np.isclose(pack["xi_soil"], SVLAB_XI_SOIL)
    assert np.isclose(pack["xi_rock"], SVLAB_XI_SOIL * 0.1)
    assert np.allclose(pack["zeta_inner"][:80], pack["xi_soil"])
    assert np.allclose(pack["zeta_inner"][80:], pack["xi_rock"])


def test_inner_surface_is_the_unpadded_span():
    surface = np.arange(3 * 1501, dtype=float).reshape(3, 1501)
    inner = inner_surface(surface)
    assert inner.shape == (3, INNER_NX)
    assert np.allclose(inner, surface[:, PAD_NX : PAD_NX + INNER_NX])
