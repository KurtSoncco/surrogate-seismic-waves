"""Tests for peak_f0_robustness peak finding."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
_SEED_DIR = _EXPERIMENT_DIR / "seed_robustness"
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

_spec = importlib.util.spec_from_file_location(
    "peak_f0_robustness", _SEED_DIR / "peak_f0_robustness.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["peak_f0_robustness"] = _mod
_spec.loader.exec_module(_mod)


def test_theoretical_f0():
    # Vs1=200, H=50 -> f0 = 1.0 Hz
    assert abs(_mod.theoretical_f0(200.0, 50.0) - 1.0) < 1e-9


def test_find_peak_in_window():
    freq = np.linspace(0.1, 5.0, 500)
    f0 = 1.0
    tf = np.exp(-((freq - 1.05) ** 2) / 0.02)
    f_peak, amp = _mod.find_peak_frequency(freq, tf, f0, window_lo=0.6, window_hi=1.4)
    assert abs(f_peak - 1.05) < 0.05
    assert amp > 0.5


def test_find_peak_outside_window_ignored():
    freq = np.linspace(0.1, 10.0, 1000)
    f0 = 1.0
    tf = np.zeros_like(freq)
    tf[np.argmin(np.abs(freq - 3.0))] = 10.0  # spike outside [0.6, 1.4]
    tf[np.argmin(np.abs(freq - 0.9))] = 1.0
    f_peak, _ = _mod.find_peak_frequency(freq, tf, f0)
    assert 0.6 <= f_peak <= 1.4


def test_split_matches_data_loader_size():
    splits = _mod.split_dataset_indices(7680, train_split=0.7, val_split=0.15, seed=42)
    assert len(splits["train"]) == 5376
    assert len(splits["val"]) == 1152
    assert len(splits["test"]) == 1152


def test_ratio_nan_when_no_peak():
    freq = np.linspace(0.1, 5.0, 100)
    f_peak, _ = _mod.find_peak_frequency(freq, np.zeros(100), f0=0.0)
    assert np.isnan(f_peak)
