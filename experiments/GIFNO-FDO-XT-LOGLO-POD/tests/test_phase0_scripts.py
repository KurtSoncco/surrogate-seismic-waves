"""Smoke tests for Phase-0 diagnostic scripts (no GPU / no Box required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Avoid running apply_env / heavy imports by only loading helpers when possible
    return path


def test_score_ood_stratify_summary_importable():
    path = Path(__file__).resolve().parents[1] / "score_ood_campaign.py"
    assert path.is_file()
    src = path.read_text(encoding="utf-8")
    assert "def stratify_summary" in src
    assert "def score_campaign" in src


def test_pod_recon_ceiling_script_exists():
    path = Path(__file__).resolve().parents[1] / "pod_recon_ceiling.py"
    assert path.is_file()
    src = path.read_text(encoding="utf-8")
    assert "def reconstruct" in src
    assert "def eval_modes" in src


def test_prepare_ood_mix_and_eval_scripts_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "prepare_ood_mix_data.py").is_file()
    assert (root / "eval_ood_holdout.py").is_file()
    assert (root / "sanity_homog_depth.py").is_file()
    assert (root / "run_full_7680_train.sh").is_file()
    assert (root / "run_mix_finetune.sh").is_file()


def test_reconstruct_math_identity():
    """POD recon with orthonormal modes recovers centered signal when K full."""
    rng = np.random.default_rng(0)
    # One recorder, F=8, K=8 identity basis
    mean = rng.normal(size=(1, 8)).astype(np.float32)
    modes = np.eye(8, dtype=np.float32)[None, :, :]  # (1,8,8)
    tf = mean + rng.normal(size=(1, 8)).astype(np.float32)
    centered = tf - mean
    coeffs = np.einsum("rf,rkf->rk", centered, modes)
    recon = mean + np.einsum("rk,rkf->rf", coeffs, modes)
    assert np.allclose(recon, tf, atol=1e-5)
