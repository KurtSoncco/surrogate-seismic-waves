from __future__ import annotations

import numpy as np

from response_variability.corner_is_design import (
    FORBIDDEN_RF_SEEDS,
    extra_rf_seeds,
    kernel_anderson,
    maximin_holdout,
)


def test_kernel_anderson_peaks_at_data():
    cov = np.array([0.12, 0.28, 0.28])
    rh = np.array([20.0, 90.0, 91.0])
    gof = np.array([0.05, 0.40, 0.42])
    hat = kernel_anderson(
        np.array([0.28, 0.12]),
        np.array([90.5, 20.0]),
        cov,
        rh,
        gof,
        bw_cov=0.02,
        bw_rh=5.0,
    )
    assert hat[0] > hat[1]


def test_maximin_holdout_unique():
    rng = np.random.default_rng(0)
    phys = np.column_stack(
        [
            rng.uniform(120, 300, 32),
            rng.uniform(20, 90, 32),
            rng.uniform(0.255, 0.3, 32),
            rng.uniform(80, 100, 32),
            rng.uniform(12, 40, 32),
            rng.uniform(800, 1400, 32),
        ]
    )
    idx = maximin_holdout(phys, 8, rng)
    assert len(idx) == 8
    assert len(np.unique(idx)) == 8
    assert idx.min() >= 0 and idx.max() < 32


def test_extra_seeds_avoid_forbidden():
    rng = np.random.default_rng(1)
    seeds = extra_rf_seeds(10, forbidden=FORBIDDEN_RF_SEEDS, rng=rng)
    assert len(seeds) == 10
    assert len(set(seeds.tolist())) == 10
    assert set(seeds.tolist()).isdisjoint(FORBIDDEN_RF_SEEDS)
