from __future__ import annotations

import numpy as np

from scoring.score_stoch_permutation import (
    apply_channel_op,
    delta_pack,
    permutation_ops,
    rel_l2_per_sample,
    stoch_channel_names,
    stoch_slices,
)


def test_stoch_slices_cover_17d():
    sl = stoch_slices(k_xi=8, layout="xi_cov")
    assert sl["xi"] == slice(0, 16)
    assert sl["CoV"] == slice(16, 17)
    assert sl["all"] == slice(0, 17)
    assert len(stoch_channel_names(8, layout="xi_cov")) == 17
    names = [op for op, _, _ in permutation_ops("xi_cov")]
    assert names[0] == "baseline"
    assert "shuffle_CoV" in names
    assert "shuffle_rH" not in names


def test_stoch_slices_cov_only():
    sl = stoch_slices(layout="cov_only")
    assert sl["CoV"] == slice(0, 1)
    assert sl["all"] == slice(0, 1)
    assert stoch_channel_names(layout="cov_only") == ["CoV"]
    names = [op for op, _, _ in permutation_ops("cov_only")]
    assert names[0] == "baseline"
    assert "shuffle_CoV" in names
    assert "shuffle_xi" not in names
    sl = stoch_slices(k_xi=8, layout="legacy20")
    assert sl["xi"] == slice(0, 16)
    assert sl["rH"] == slice(16, 17)
    assert sl["aHV"] == slice(17, 18)
    assert sl["CoV"] == slice(18, 19)
    assert sl["xi_damp"] == slice(19, 20)
    assert sl["scalars"] == slice(16, 20)
    assert sl["all"] == slice(0, 20)
    assert len(stoch_channel_names(8, layout="legacy20")) == 20
    names = [op for op, _, _ in permutation_ops("legacy20")]
    assert names[0] == "baseline"
    assert "shuffle_CoV" in names
    assert "zero_scalars" in names


def test_shuffle_breaks_pairing_keeps_marginal():
    rng = np.random.default_rng(0)
    stoch = rng.standard_normal((12, 17)).astype(np.float32)
    perm = rng.permutation(12)
    sl = stoch_slices(layout="xi_cov")["CoV"]
    out = apply_channel_op(stoch, sl, op="shuffle", perm=perm)
    np.testing.assert_allclose(out[:, :16], stoch[:, :16])
    np.testing.assert_allclose(np.sort(out[:, 16]), np.sort(stoch[:, 16]))
    assert not np.allclose(out[:, 16], stoch[:, 16])


def test_zero_is_train_mean_after_zscore():
    stoch = np.ones((4, 17), dtype=np.float32)
    out = apply_channel_op(stoch, stoch_slices(layout="xi_cov")["all"], op="zero")
    np.testing.assert_allclose(out, 0.0)


def test_identity_and_rel_l2():
    stoch = np.arange(34, dtype=np.float32).reshape(2, 17)
    out = apply_channel_op(stoch, stoch_slices(layout="xi_cov")["all"], op="identity")
    np.testing.assert_array_equal(out, stoch)
    tf = np.ones((3, 2, 5), dtype=np.float64)
    rel = rel_l2_per_sample(tf, tf)
    np.testing.assert_allclose(rel, 0.0)
    d = delta_pack(np.array([0.9, 0.8]), np.array([0.85, 0.7]), higher_better=True)
    assert d["n"] == 2
    assert d["mean_delta"] < 0
