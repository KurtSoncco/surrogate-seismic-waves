#!/usr/bin/env python3
"""Damping that OpenSees actually applied, read back from a section or crop H5.

New runs set ``params.damping_applied = xi_base`` and store the zeta field that
was passed through as ``xi_soil_base`` / ``xi_rock_base``. Older section runs
stored the CSV zeta while seiskit ``global_avg`` recomputed Q(Vs) on the solved
mesh and ignored the CSV. Those files have no ``damping_applied`` attribute.
"""

from __future__ import annotations

import numpy as np

from case7_crops import qvs_xi


def _scalar(attrs, key: str, default=None):
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "item"):
        value = value.item()
    return value


def applied_zeta(handle) -> np.ndarray:
    """(nz, nx) zeta OpenSees used. Same shape as the stored ``Vs_realization_2D``."""
    zeta = np.asarray(handle["Damping_zeta"][:], dtype=np.float64)
    if _scalar(handle["params"].attrs, "damping_applied"):
        return zeta
    vs = np.asarray(handle["Vs_realization_2D"][:], dtype=np.float64)
    if "rock_mask" in handle:
        rock = np.asarray(handle["rock_mask"][:], dtype=bool)
    else:
        from case7_crops import ROCK_VS_MPS

        rock = vs > ROCK_VS_MPS
    dx = float(handle["grid"].attrs["dx"])
    stored_m = vs.shape[1] * dx
    solved_m = float(
        _scalar(handle["params"].attrs, "lx_solve_m", _scalar(handle["grid"].attrs, "solved_Lx", _scalar(handle["grid"].attrs, "Lx")))
    )
    if stored_m + dx < solved_m:
        from run_sections import pad_free_field

        pad_m = (solved_m - stored_m) / 2.0
        vs_solved = pad_free_field(vs, pad_m, dx)
        rock_solved = pad_free_field(rock.astype(np.float64), pad_m, dx) > 0.5
        xi_soil, xi_rock = qvs_xi(vs_solved, rock_solved)
    else:
        xi_soil, xi_rock = qvs_xi(vs, rock)
    return np.where(rock, xi_rock, xi_soil)
