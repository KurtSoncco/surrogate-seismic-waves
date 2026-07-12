"""Tests for seed_robustness analysis (subset logic, sigma_ln, no OpenSees)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
_SEED_DIR = _EXPERIMENT_DIR / "seed_robustness"
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

import config

config.setup_import_paths()

_spec = importlib.util.spec_from_file_location(
    "seed_robustness_check", _SEED_DIR / "seed_robustness_check.py"
)
_src = importlib.util.module_from_spec(_spec)
sys.modules["seed_robustness_check"] = _src
_spec.loader.exec_module(_src)


def test_sigma_ln_per_freq_increases_with_spread():
    rng = np.random.default_rng(0)
    tight = rng.lognormal(0, 0.05, size=(20, 50))
    wide = rng.lognormal(0, 0.3, size=(20, 50))
    s_tight = _src.sigma_ln_per_freq(tight)
    s_wide = _src.sigma_ln_per_freq(wide)
    assert float(np.mean(s_wide)) > float(np.mean(s_tight))


def test_sigma_ln_per_freq_single_seed_is_zero():
    x = np.ones((1, 10))
    out = _src.sigma_ln_per_freq(x)
    np.testing.assert_array_equal(out, 0.0)


def test_seed_cv_metrics_converges_with_n():
    rng = np.random.default_rng(42)
    n_freq = 100
    truth = rng.lognormal(0, 0.2, size=(50, n_freq))
    pred = truth * (1.0 + rng.normal(0, 0.02, size=truth.shape))
    freq = np.logspace(-1, 1, n_freq)

    rows_small = _src.seed_cv_metrics(
        truth[:10], pred[:10], freq, seed_counts=(5,), n_subsets=3, rng_seed=0
    )
    rows_full = _src.seed_cv_metrics(
        truth, pred, freq, seed_counts=(10, 30, 50), n_subsets=3, rng_seed=0
    )
    rmse_n10 = [r for r in rows_full if r["n_seeds"] == 10][0]["sigma_ln_rmse_truth"]
    rmse_n50 = [r for r in rows_full if r["n_seeds"] == 50][0]["sigma_ln_rmse_truth"]
    assert rmse_n50 <= rmse_n10 + 1e-6
    assert rows_small


def test_find_h5_searches_multiple_dirs(tmp_path: Path):
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    h5 = d2 / "run_5.h5"
    h5.write_text("stub")
    found = _src.find_h5(5, _src.resolve_h5_dirs(d1, d2))
    assert found == h5


def test_gifno_manifest_rows_for_sample(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_idx,run_index,h5_path,rf_seed,H_discretized,CoV,f0_effective,nz_actual,n_lateral\n"
        "0,0,x,1,84,0.2,3,94,21\n"
        "1,1,x,2,84,0.2,3,94,21\n"
        "29,29,x,3,84,0.2,3,94,21\n"
        "30,30,x,4,30,0.1,3,40,21\n"
    )
    rows = _src.gifno_manifest_rows_for_sample(
        manifest, sample_id=0, seeds_per_sample=30
    )
    assert len(rows) == 3
    assert rows[0].replicate_id == 0 and rows[-1].replicate_id == 29


def test_cached_truth_tf_lookup(tmp_path: Path):
    tf = np.ones((2, 21, 4), dtype=np.float32)
    freq = np.linspace(0.1, 10, 4, dtype=np.float32)
    np.save(tmp_path / "tf.npy", tf)
    np.save(tmp_path / "freq.npy", freq)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_idx,run_index,h5_path,rf_seed,H_discretized,CoV,f0_effective,nz_actual,n_lateral\n"
        "0,0,x,1,84,0.2,3,94,21\n"
        "1,5,x,2,84,0.2,3,94,21\n"
    )
    cache = _src.CachedTruthTF(tmp_path / "tf.npy", tmp_path / "freq.npy", manifest)
    np.testing.assert_array_equal(cache.truth_tf(5), tf[1])


def test_manifest_extra_seeds_extra_only():
    seiskit_data = Path.home() / "seiskit" / "neural-operator" / "data"
    if not (seiskit_data / "sobol.py").is_file():
        pytest.skip("seiskit sobol.py not available")

    _spec2 = importlib.util.spec_from_file_location(
        "manifest_extra_seeds", _SEED_DIR / "manifest_extra_seeds.py"
    )
    mod = importlib.util.module_from_spec(_spec2)
    sys.modules["manifest_extra_seeds"] = mod
    _spec2.loader.exec_module(mod)

    sobol = mod._load_sobol_module(seiskit_data)
    manifest = sobol.build_manifest(seeds_per_sample=50)
    extra = mod.entries_for_sample(manifest, 0, extra_only=True, min_replicate_id=30)
    assert len(extra) == 20
    assert extra[0].replicate_id == 30
    assert extra[-1].replicate_id == 49
