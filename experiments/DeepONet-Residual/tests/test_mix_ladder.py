from __future__ import annotations

import numpy as np

from mix_ladder import extra_local_indices


def test_extra_local_no_parent_leak():
    parent = np.array([10, 20, 30, 40])
    child = np.array([10, 99, 20, 88, 30, 77, 40, 66])
    extra = extra_local_indices(parent, child, n_extra=3, seed=42)
    assert len(extra) == 3
    got = {int(child[i]) for i in extra}
    assert got.isdisjoint({10, 20, 30, 40})
    assert got <= {99, 88, 77, 66}


def test_extra_local_all_outside():
    parent = np.arange(4)
    child = np.arange(10)
    extra = extra_local_indices(parent, child, n_extra=6, seed=0)
    assert len(extra) == 6
    got = {int(child[i]) for i in extra}
    assert got == {4, 5, 6, 7, 8, 9}


def test_extra_local_reproducible():
    parent = np.arange(5)
    child = np.arange(15)
    a = extra_local_indices(parent, child, 4, seed=42)
    b = extra_local_indices(parent, child, 4, seed=42)
    np.testing.assert_array_equal(a, b)
