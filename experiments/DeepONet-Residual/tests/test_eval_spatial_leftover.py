"""Smoke test for spatial leftover vs inflated TF Pearson diagnostic."""

from __future__ import annotations

import numpy as np

from response_variability.eval_spatial_leftover import (
    attach_toro,
    score_domain,
    spatial_pattern_pearson_one,
    write_markdown,
)
from response_variability.names import GINO, HASKELL_NOMINAL, PRETELL, TORO
from response_variability.plot_presentation import make_synthetic_pack


def test_spatial_pattern_undefined_when_flat_in_x():
    true = np.linspace(1.0, 2.0, 12) + np.arange(5)[:, None] * 0.05
    pred = np.broadcast_to(true[2], true.shape)
    assert np.isnan(spatial_pattern_pearson_one(pred, true))
    assert np.isfinite(spatial_pattern_pearson_one(true, true))


def test_spatial_leftover_synthetic_scores(tmp_path):
    pack = make_synthetic_pack(domain="iid", n=8, n_freq=24, n_rec=7, seed=1)
    pack["tf_toro"] = np.asarray(pack["tf_pretell"], dtype=np.float64)
    df, rec = score_domain("iid", pack)
    assert rec["n"] == 8
    assert np.isfinite(rec["opensees"]["spatial_sigma_ln"]["median"])
    assert rec["opensees"]["spatial_sigma_ln"]["median"] > 0
    assert GINO in rec["methods"]
    assert HASKELL_NOMINAL in rec["methods"]
    assert PRETELL in rec["methods"]
    assert TORO in rec["methods"]
    assert np.isfinite(rec["methods"][GINO]["central_pearson"]["median"])
    assert np.isfinite(rec["leftover"]["rel_l2"]["median"])
    assert set(df["method"].unique()) >= {GINO, HASKELL_NOMINAL, PRETELL, TORO}
    csv = tmp_path / "spatial_leftover.csv"
    df.to_csv(csv, index=False)
    assert csv.is_file()
    write_markdown({"iid": rec}, tmp_path / "SPATIAL_LEFTOVER.md")
    text = (tmp_path / "SPATIAL_LEFTOVER.md").read_text()
    assert "too smooth" in text
    assert "GINO" in text


def test_attach_toro_skips_missing(tmp_path):
    pack = make_synthetic_pack(domain="iid", n=3, n_freq=16, n_rec=5)
    out = attach_toro(pack, "iid", tmp_path)
    assert "tf_toro" not in out
    np.savez_compressed(tmp_path / "iid_toro2022.npz", tf_toro=np.ones((3, 16)))
    out2 = attach_toro(pack, "iid", tmp_path)
    assert out2["tf_toro"].shape == (3, 16)
