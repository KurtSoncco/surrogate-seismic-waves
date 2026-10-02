"""Haskell ξ=0.05 vs soil-ζ nom (no H5 required for unit checks)."""

from __future__ import annotations

import numpy as np
import pytest

from haskell_baseline import haskell_nominal_af_within
from residual_signed import soil_mean_xi
from response_variability.eval_haskell_xi import resolve_h5
from response_variability.metrics import peak_af


def test_higher_xi_lowers_haskell_peak():
    freq = np.logspace(-1, 1, 64)
    lo = haskell_nominal_af_within(freq, vs1=200.0, H=40.0, vs2=800.0, xi=0.025)
    hi = haskell_nominal_af_within(freq, vs1=200.0, H=40.0, vs2=800.0, xi=0.05)
    _, a_lo = peak_af(freq, lo)
    _, a_hi = peak_af(freq, hi)
    assert a_hi < a_lo


def test_soil_mean_xi_uses_soil_rows_only():
    z = np.vstack([np.full((10, 4), 0.04), np.full((5, 4), 0.004)])
    assert soil_mean_xi(z, 10) == pytest.approx(0.04)


def test_resolve_h5_prefers_existing_file(tmp_path):
    fake = tmp_path / "run_0.h5"
    fake.write_bytes(b"not-a-real-h5")
    got = resolve_h5(str(fake), "iid")
    assert got == fake


def _median(x: float) -> dict[str, float]:
    return {"median": x, "q25": x - 0.01, "q75": x + 0.01}


def test_haskell_xi_markdown_names_paired_and_worse_pearson(tmp_path):
    from response_variability.eval_haskell_xi import ARM_FIXED, ARM_SAMPLE, write_markdown

    rec = {
        "n": 4,
        "xi_soil": _median(0.041),
        "xi_bedrock": _median(0.004),
        "pack_nom_rel_err_max": 3.7e-8,
        "delta_ln_A_05_vs_xi": {"median": -0.198, "q25": -0.30, "q75": -0.14},
        "methods": {
            ARM_FIXED: {
                "pearson": _median(0.772),
                "anderson": _median(0.165),
                "delta_ln_A_peak": _median(-0.086),
            },
            ARM_SAMPLE: {
                "pearson": _median(0.741),
                "anderson": _median(0.169),
                "delta_ln_A_peak": _median(0.146),
            },
        },
        "anderson_tail": [
            {
                "sample": 118,
                "vs1": 115.7,
                "H": 32.0,
                "f0": 0.90,
                "ref_f_peak": 8.5,
                "f_peak": 0.9,
                "delta_f_peak": -7.6,
                "gof_af": 1.69,
                "gof_af_soil": 1.55,
                "pearson": 0.45,
            }
        ],
    }
    dest = tmp_path / "HASKELL_XI.md"
    write_markdown({"iid": rec, "dipping": rec}, dest)
    text = dest.read_text()
    assert "paired per-realization" in text
    assert "worse" in text
    assert "Anderson" in text
    assert "Dipping Anderson tail" in text
    assert "OPENSEES1D.md" in text
    assert "118" in text
