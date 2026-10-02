"""Smoke test for Toro 2022 frozen-H eval (synthetic pack, few seeds)."""

from __future__ import annotations

import numpy as np
import pytest

from response_variability.names import HASKELL_NOMINAL, TORO
from response_variability.plot_presentation import make_synthetic_pack
from response_variability.seiskit_arms import ensure_seiskit, hallal_config


def test_eval_toro2022_synthetic_scores(tmp_path):
    try:
        ensure_seiskit()
    except ImportError:
        pytest.skip("seiskit not installed")
    from response_variability.eval_toro2022 import add_toro_arm, run_domain, slim_score_pack

    pack = make_synthetic_pack(domain="iid", n=3, n_freq=24, n_rec=7)
    pack = add_toro_arm(pack, n_hallal_seeds=2)
    assert pack["tf_toro"].shape == (3, 24)
    assert np.all(np.isfinite(pack["tf_toro"]))
    slim = slim_score_pack(pack)
    assert "tf_pretell" not in slim
    assert "tf_gino" not in slim
    got = run_domain(
        "iid",
        pack=pack,
        out_dir=tmp_path,
        n_hallal_seeds=2,
        skip_if_present=False,
        plot=False,
    )
    rec = got["rec"]
    assert rec["n"] == 3
    assert HASKELL_NOMINAL in rec
    assert TORO in rec
    assert np.isfinite(rec[TORO]["pearson"]["median"])
    assert np.isfinite(rec["opensees_spatial_sigma_ln"]["median"])


def test_hallal_config_spid_defaults():
    try:
        ensure_seiskit()
    except ImportError:
        pytest.skip("seiskit not installed")
    cfg = hallal_config(vs1=200.0, H=40.0, cov=0.2, vs2=800.0)
    assert cfg.toro_h0 == pytest.approx(0.0)
    assert cfg.sigma_ln_vs_surface == pytest.approx(0.25)
    assert cfg.sigma_ln_vs == pytest.approx(0.15)
    assert cfg.toro_sigma_inflate == pytest.approx(1.16)
