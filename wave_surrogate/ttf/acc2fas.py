from typing import Optional, Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()


def acceleration_to_fas(
    acc: np.ndarray, dt: float, n: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert acceleration time history to Fourier Amplitude Spectrum (FAS).

    Parameters
    ----------
    acc : np.ndarray
        Acceleration time history (1D array).
    dt : float
        Time step of the acceleration time history.
    n : int, optional
        Number of points for FFT. If None, uses the length of `acc`.
        If `n` is smaller than `len(acc)`, the signal is truncated.
        If `n` is larger, it's zero-padded.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - FAS (np.ndarray): Single-sided Fourier amplitude spectrum.
        - freq (np.ndarray): Frequency vector corresponding to FAS.
    """
    numpts = len(acc)
    if n is None:
        n = numpts

    if numpts > n:
        logger.warning(f"Signal truncated from {numpts} to {n} points.")

    freq = rfftfreq(n, d=dt)
    fas = np.abs(rfft(acc, n=n)) * (2.0 / numpts)  # type: ignore

    return fas, freq
