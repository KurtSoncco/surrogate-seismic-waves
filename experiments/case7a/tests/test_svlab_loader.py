"""SVLAB Real-ESSI .out parser and Case 7 node map."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from case7a import (  # noqa: E402
    DEFAULT_SVLAB_UPLOAD,
    N_COL,
    SVLAB_CENTER_NODE,
    SVLAB_DURATION_S,
    SVLAB_WEST_NODE,
    load_svlab_motion,
)

ARRAY_S = DEFAULT_SVLAB_UPLOAD / "Acceleration_Array_Surface.0.out"
ARRAY_D = DEFAULT_SVLAB_UPLOAD / "Acceleration_Array_Downhole.0.out"
SURFACE = DEFAULT_SVLAB_UPLOAD / "Surface.0.out"


def test_array_surface_is_center_node():
    nodes, motion = load_svlab_motion(ARRAY_S)
    assert nodes.tolist() == [SVLAB_CENTER_NODE]
    assert motion.shape == (1500, 1, 2)
    assert float(np.max(np.abs(motion[:, 0, 0]))) > float(np.max(np.abs(motion[:, 0, 1])))


def test_array_downhole_header():
    nodes, motion = load_svlab_motion(ARRAY_D)
    assert nodes.shape == (1,)
    assert motion.shape == (1500, 1, 2)


def test_surface_center_matches_array():
    nodes, surf = load_svlab_motion(SURFACE)
    arr_nodes, arr = load_svlab_motion(ARRAY_S)
    assert int(nodes[0]) == SVLAB_WEST_NODE
    assert int(nodes[-1]) == 24831
    center = int(np.flatnonzero(nodes == SVLAB_CENTER_NODE)[0])
    assert center == N_COL // 2
    assert int(arr_nodes[0]) == SVLAB_CENTER_NODE
    np.testing.assert_allclose(surf[:, center, 0], arr[:, 0, 0])
    n_time = surf.shape[0]
    dt = SVLAB_DURATION_S / (n_time - 1)
    np.testing.assert_allclose(dt, 15.0 / 1499.0)
    assert surf.shape == (1500, 801, 2)
