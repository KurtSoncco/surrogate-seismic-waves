from __future__ import annotations

import numpy as np
import torch

from data import (
    build_stoch_vector,
    infer_stoch_layout,
    stoch_dim,
    stoch_in_features_from_state,
)


def test_stoch_dim_xi_cov_is_17():
    assert stoch_dim() == 17
    assert stoch_dim(layout="xi_cov") == 17
    assert stoch_dim(layout="legacy20") == 20
    assert stoch_dim(layout="cov_only") == 1


def test_build_stoch_drops_rh_ahv_damp():
    xi = np.arange(16, dtype=np.float32)
    v = build_stoch_vector(
        xi_vals=xi, rH=50.0, aHV=20.0, CoV=0.22, xi_damp=0.05, layout="xi_cov"
    )
    assert v.shape == (17,)
    np.testing.assert_allclose(v[:16], xi)
    assert v[16] == np.float32(0.22)
    legacy = build_stoch_vector(
        xi_vals=xi, rH=50.0, aHV=20.0, CoV=0.22, xi_damp=0.05, layout="legacy20"
    )
    assert legacy.shape == (20,)
    np.testing.assert_allclose(legacy[16:], [50.0, 20.0, 0.22, 0.05], atol=1e-6)


def test_infer_stoch_layout_from_linear_width():
    state17 = {"base.stoch_mlp.0.weight": torch.zeros(128, 17)}
    state20 = {"inner.stoch_mlp.0.weight": torch.zeros(128, 20)}
    assert stoch_in_features_from_state(state17) == 17
    assert infer_stoch_layout(state17) == "xi_cov"
    assert infer_stoch_layout(state20) == "legacy20"
    assert infer_stoch_layout(recorded="legacy20") == "legacy20"
    state1 = {"inner.stoch_mlp.0.weight": torch.zeros(32, 1)}
    assert infer_stoch_layout(state1) == "cov_only"
