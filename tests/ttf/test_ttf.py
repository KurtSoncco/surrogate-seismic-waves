import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from ttf import TTF, acceleration_to_fas, kohmachi


# Test data setup
@pytest.fixture()
def test_data():
    DT = 0.01
    N_POINTS = 1024
    T = np.arange(0, N_POINTS * DT, DT)
    FREQ = 5.0  # 5 Hz
    return DT, N_POINTS, T, FREQ


@pytest.fixture
def sample_acceleration(test_data):
    """Generate a sample sine wave acceleration signal."""
    DT, N_POINTS, T, FREQ = test_data
    return np.sin(2 * np.pi * FREQ * T)


def test_acceleration_to_fas(sample_acceleration, test_data):
    """Test the acceleration_to_fas function."""
    DT = test_data[0]
    T = test_data[2]
    FREQ = test_data[3]

    fas, freq = acceleration_to_fas(sample_acceleration, dt=DT)

    assert isinstance(fas, np.ndarray)
    assert isinstance(freq, np.ndarray)
    assert fas.shape == freq.shape

    # Check that the peak frequency is correct
    peak_freq_index = np.argmax(fas)
    assert freq[peak_freq_index] == pytest.approx(FREQ, abs=1 / T[-1])


def test_kohmachi():
    """Test the kohmachi smoothing function."""
    freq = np.linspace(0.1, 10, 100)
    signal = np.zeros_like(freq)
    signal[50] = 1.0  # A sharp peak

    smoothed_signal = kohmachi(signal, freq)

    assert isinstance(smoothed_signal, np.ndarray)
    assert smoothed_signal.shape == signal.shape
    # Check if smoothing happened: peak should be lower, and neighbors should be non-zero
    assert smoothed_signal[50] < 1.0
    assert np.all(smoothed_signal >= 0)
    assert smoothed_signal[49] > 0
    assert smoothed_signal[51] > 0


def test_ttf_constant_amplification(sample_acceleration, test_data):
    """Test TTF with a simple case where surface motion is amplified base motion."""
    base_acc = sample_acceleration
    surface_acc = 2.0 * base_acc
    DT = test_data[0]

    n_points_out = 100
    freq, tf = TTF(surface_acc, base_acc, dt=DT, n_points=n_points_out)

    assert isinstance(freq, np.ndarray)
    assert isinstance(tf, np.ndarray)
    assert freq.shape == (n_points_out,)
    assert tf.shape == (n_points_out,)

    # The transfer function should be approximately 2.0
    # We check the mean value, ignoring edges where interpolation might be less accurate
    assert np.mean(tf[10:-10]) == pytest.approx(2.0, rel=0.1)
