"""Stored zeta versus the Q(Vs) damping OpenSees applied on old section files."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

from applied_damping import applied_zeta  # noqa: E402
from case7_crops import qvs_xi  # noqa: E402


def _write(path: Path, *, zeta_soil: float, applied: str | None, pad: bool) -> None:
    vs = np.array([[200.0, 200.0, 800.0], [200.0, 200.0, 800.0]], dtype=np.float32)
    rock = vs > 760.0
    zeta = np.where(rock, 0.01, zeta_soil).astype(np.float32)
    with h5py.File(path, "w") as handle:
        params = handle.create_group("params")
        params.attrs["lx_solve_m"] = 3.0 if not pad else 5.0
        if applied:
            params.attrs["damping_applied"] = applied
        grid = handle.create_group("grid")
        grid.attrs["dx"] = 1.0
        grid.attrs["Lx"] = 3.0
        handle.create_dataset("Vs_realization_2D", data=vs)
        handle.create_dataset("Damping_zeta", data=zeta)
        handle.create_dataset("rock_mask", data=rock.astype(np.uint8))


def test_tagged_file_keeps_stored_zeta(tmp_path: Path):
    path = tmp_path / "new.h5"
    _write(path, zeta_soil=0.01, applied="xi_base", pad=False)
    with h5py.File(path, "r") as handle:
        got = applied_zeta(handle)
        stored = handle["Damping_zeta"][:]
    np.testing.assert_allclose(got, stored)


def test_untagged_file_is_qvs_not_the_csv(tmp_path: Path):
    path = tmp_path / "old.h5"
    _write(path, zeta_soil=0.01, applied=None, pad=False)
    with h5py.File(path, "r") as handle:
        vs = handle["Vs_realization_2D"][:]
        rock = handle["rock_mask"][:].astype(bool)
        got = applied_zeta(handle)
    xi_soil, xi_rock = qvs_xi(vs, rock)
    assert abs(xi_soil - 0.01) > 1e-4
    np.testing.assert_allclose(got[0, 0], xi_soil)
    np.testing.assert_allclose(got[0, 2], xi_rock)


def test_qvs_xi_matches_seiskit_helpers():
    vs = np.array([[150.0, 400.0, 1200.0], [180.0, 500.0, 1400.0]])
    rock = vs > 760.0
    xi_soil, xi_rock = qvs_xi(vs, rock)
    import importlib.util

    damping_py = Path.home() / "seiskit" / "seiskit" / "damping.py"
    spec = importlib.util.spec_from_file_location("seiskit_damping", damping_py)
    damping = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(damping)
    compute_average_damping_harmonic = damping.compute_average_damping_harmonic
    compute_damping_from_Q = damping.compute_damping_from_Q
    compute_quality_factor = damping.compute_quality_factor

    soil = [compute_quality_factor(float(v)) for v in vs[~rock]]
    rock_vs = float(np.median(vs[rock]))
    assert abs(xi_soil - compute_average_damping_harmonic(soil)) < 1e-9
    assert abs(xi_rock - compute_damping_from_Q(compute_quality_factor(rock_vs))) < 1e-9
