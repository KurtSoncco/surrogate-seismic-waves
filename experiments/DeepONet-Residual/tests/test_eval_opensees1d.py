"""OpenSees 1-D shape check (no live OpenSees required)."""

from __future__ import annotations

import numpy as np

from response_variability.eval_opensees1d import (
    ARM_HASKELL_VS_2D,
    ARM_OS1D_VS_2D,
    ARM_SHAPE_OUTCROP,
    ARM_SHAPE_WITHIN,
    cache_matches,
    interp_af,
    score_pair,
    write_markdown,
)
from response_variability.metrics import odd_quarter_wave_peaks


def _median(x: float) -> dict[str, float]:
    return {"median": x, "q25": x - 0.02, "q75": x + 0.02}


def test_interp_af_identity_on_matching_grid():
    f = np.logspace(-1, 1, 32)
    y = np.linspace(1.0, 2.0, 32)
    out = interp_af(f, y, f.astype(np.float32))
    assert np.allclose(out, y, atol=1e-6)


def test_score_pair_negative_when_candidate_is_shorter():
    freq = np.logspace(-1, 1, 400)
    f0 = 1.2
    peaks = odd_quarter_wave_peaks
    ref = np.ones_like(freq)
    cand = np.ones_like(freq)
    for k, amp in enumerate((12.0, 6.0, 3.0), start=1):
        fc = (2 * k - 1) * f0
        bump = np.exp(-((np.log(freq) - np.log(fc)) ** 2) / 0.02)
        ref = ref + amp * bump
        cand = cand + 0.8 * amp * bump
    met = score_pair(freq, ref, cand, f0=f0)
    assert met["delta_ln_A_mode1"] < 0
    assert met["delta_ln_A_mode2"] < 0
    f1, _ = peaks(freq, cand, f0=f0)[0]
    assert abs(f1 - f0) < 0.1


def test_cache_matches_requires_params(tmp_path):
    spec = {
        "vs1": 200.0,
        "H": 30.0,
        "vs2": 1000.0,
        "xi_soil": 0.04,
        "dt": 0.01,
        "duration": 30.0,
        "f1": 1.5,
        "f2": 10.0,
        "hx": 1.0,
    }
    path = tmp_path / "c.npz"
    np.savez(
        path,
        freq=np.linspace(0.1, 10, 8),
        af_within_iface=np.ones(8),
        af_within_packbase=np.ones(8),
        af_outcrop=np.ones(8),
        **spec,
    )
    assert cache_matches(path, spec)
    bad = dict(spec, xi_soil=0.05)
    assert not cache_matches(path, bad)


def test_opensees1d_markdown_scopes_shape(tmp_path):
    block = {
        "pearson": _median(0.95),
        "anderson": _median(0.05),
        "delta_ln_A_peak": _median(-0.22),
        "delta_ln_A_mode1": _median(-0.22),
        "delta_ln_A_mode2": _median(-0.40),
        "delta_ln_A_mode3": _median(-0.70),
        "pearson_mode1": _median(0.99),
        "pearson_mode2": _median(0.90),
        "pearson_mode3": _median(0.70),
    }
    rec = {
        "n": 4,
        "n_scored": 4,
        "n_missing": 0,
        "methods": {
            ARM_HASKELL_VS_2D: block,
            ARM_OS1D_VS_2D: block,
            ARM_SHAPE_WITHIN: block,
            ARM_SHAPE_OUTCROP: block,
        },
    }
    dest = tmp_path / "OPENSEES1D.md"
    write_markdown({"iid": rec, "dipping": rec}, dest)
    text = dest.read_text()
    assert "model shape" in text
    assert "pack base" in text
    assert "outcrop" in text
    assert ARM_SHAPE_WITHIN in text
