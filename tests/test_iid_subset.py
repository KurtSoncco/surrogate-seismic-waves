"""IID subset and fixed step budget used by the learning-curve arms."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "GIFNO-VARDOMAIN"))

from data import choose_groups, choose_iid_flats, manifold_folds
from sampling import batches_per_epoch


def _groups(n_groups: int = 8, n_seeds: int = 30) -> dict[int, list[int]]:
    return {g: list(range(g * n_seeds, (g + 1) * n_seeds)) for g in range(n_groups)}


def test_group_fraction_keeps_every_seed():
    kept = choose_iid_flats(_groups(), 0.25, None, seed=0)
    assert len(kept) == 2 * 30
    groups = {flat // 30 for flat in kept}
    assert len(groups) == 2


def test_seeds_per_group_keep_every_group_when_fraction_is_one():
    kept = choose_iid_flats(_groups(), 1.0, 2, seed=1)
    assert len(kept) == 8 * 2
    assert len({flat // 30 for flat in kept}) == 8


def test_draw_is_fixed_by_the_seed():
    groups = _groups()
    assert choose_iid_flats(groups, 0.5, 8, seed=3) == choose_iid_flats(groups, 0.5, 8, seed=3)
    assert choose_iid_flats(groups, 0.5, 8, seed=3) != choose_iid_flats(groups, 0.5, 8, seed=4)


def test_manifold_fraction_keeps_half_the_groups():
    kept = choose_groups(list(range(10)), 0.5, seed=0)
    assert len(kept) == 5
    assert choose_groups(list(range(10)), 1.0, seed=0) == set(range(10))


def test_five_folds_cover_every_pair_group_once():
    ids = [f"man_{i:04d}" for i in range(22)]
    folds = manifold_folds(ids, n_folds=5, seed=0)
    assert len(folds) == 5
    sizes = sorted(len(fold) for fold in folds)
    assert sizes == [4, 4, 4, 5, 5]
    flat = [gid for fold in folds for gid in fold]
    assert sorted(flat) == ids
    assert len(flat) == len(set(flat))
    assert manifold_folds(ids, 5, 0) == folds


def test_fixed_batches_ignore_the_subset_size():
    assert batches_per_epoch([], 16, 3, fixed=1560) == 1560
    assert batches_per_epoch([], 16, 3, fixed=0) == 1
