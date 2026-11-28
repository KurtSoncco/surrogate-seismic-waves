"""Surface / input-motion transfer functions shared by Case 7 and training runs.

Both Case 7 and the seiskit training runs inject a known velocity wavelet
through a Lysmer base, so surface / input is defined at every x. The single
Case 7 downhole is not: it is the local within motion at x = 0 only.

The injected record differs from the true outcrop motion by a constant that
depends on the injection convention (dashpot scaling, the factor of 2). That
constant is calibrated once per convention against vertical-SH theory in a
laterally uniform zone, where 2D and 1D must agree.
"""

from __future__ import annotations

import math

import numpy as np

N_FREQ = 1000
SMOOTH_COEFF = 500
TRUSTED_BAND_HZ = (0.1, 2.5)
CORE_BAND_HZ = (0.4, 2.5)
CALIBRATION_BAND_HZ = (0.4, 2.0)
# One FFT window for every vardomain label source, longer than any record (2^17 samples at 0.01 s).
LABEL_WINDOW_S = 1310.72


def case7_input_acceleration(ricker_velocity: np.ndarray, time_out: np.ndarray, dt_in: float = 0.01) -> np.ndarray:
    """Case 7 ``ricker.in`` is a velocity at ``dt_in``. Returns acceleration on ``time_out``."""
    t_in = np.arange(ricker_velocity.size, dtype=np.float64) * dt_in
    vel = np.interp(np.asarray(time_out, dtype=np.float64), t_in, ricker_velocity.astype(np.float64))
    return np.gradient(vel, np.asarray(time_out, dtype=np.float64))


def seiskit_input_acceleration(
    motion_freq: float, t_shift: float, time_out: np.ndarray, ricker_is: str = "acceleration"
) -> np.ndarray:
    """Input acceleration of a seiskit run, sampled on ``time_out``.

    Current seiskit integrates the Ricker acceleration to the injected
    velocity (``ricker_is='acceleration'``). The original in-distribution
    campaign injected the Ricker itself as velocity (``'velocity'``).
    """
    t = np.asarray(time_out, dtype=np.float64)
    arg = (math.pi * float(motion_freq) * (t - float(t_shift))) ** 2
    ricker = (1.0 - 2.0 * arg) * np.exp(-arg)
    if ricker_is == "acceleration":
        return ricker
    if ricker_is == "velocity":
        return np.gradient(ricker, t)
    raise ValueError(f"ricker_is must be 'acceleration' or 'velocity', got {ricker_is!r}")


def tf_over_input(
    surface: np.ndarray, input_acc: np.ndarray, dt: float, ttf_batch, window_s: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Konno-Ohmachi |FAS(surface)| / |FAS(input)|. ``surface`` is (n_time, n_chan).

    ``window_s`` fixes the FFT length in seconds, so records of any length and
    time step share one bin spacing. None keeps seiskit's
    ``next_fast_len(max(n, 8192))``, which depends on the record.
    """
    surf = np.ascontiguousarray(np.asarray(surface, dtype=np.float64).T)
    base = np.repeat(np.asarray(input_acc, dtype=np.float64)[None, :], surf.shape[0], axis=0)
    nfreq = None
    if window_s is not None:
        nfreq = int(round(float(window_s) / float(dt)))
        if nfreq < surf.shape[1]:
            raise ValueError(f"record of {surf.shape[1] * dt:.1f} s exceeds the {window_s} s analysis window")
    freq, mag = ttf_batch(base, surf, dt=float(dt), n_points=N_FREQ, smooth_coeff=SMOOTH_COEFF, dz=1.0, nfreq=nfreq)
    return freq.astype(np.float64), mag.astype(np.float64)


def band_mask(freq: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (freq >= band[0]) & (freq <= band[1])


def calibrate_input_scale(
    tf_ratio: np.ndarray,
    tf_1d: np.ndarray,
    freq: np.ndarray,
    band: tuple[float, float] = CALIBRATION_BAND_HZ,
) -> float:
    """Median of ratio / 1D over the band and all supplied (laterally uniform) columns."""
    m = band_mask(freq, band)
    return float(np.median(np.asarray(tf_ratio)[..., m] / np.asarray(tf_1d)[..., m]))
