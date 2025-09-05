import numpy as np
from typing import Union

def kohmachi(
    signal: Union[np.ndarray, list],
    freq_array: Union[np.ndarray, list],
    smooth_coeff: float = 50.0,
) -> np.ndarray:
    """
    Smooth a frequency-domain signal using the Konno & Ohmachi (1998) method.

    Original paper:
        K. Konno & T. Ohmachi (1998) "Ground-motion characteristics estimated
        from spectral ratio between horizontal and vertical components of
        microtremor." Bulletin of the Seismological Society of America.
        Vol.88, No.1, 228-241.

    Parameters
    ----------
    signal : Union[np.ndarray, list]
        Signal to be smoothed in the frequency domain.
    freq_array : Union[np.ndarray, list]
        Frequency array corresponding to the signal.
    smooth_coeff : float, optional
        Smoothing coefficient. A smaller value increases smoothing. Default is 50.0.

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    x = np.asarray(signal)
    f = np.asarray(freq_array)

    if len(x) < 3:
        return x.copy()

    # Vectorized computation
    f_shifted = f / (1 + 1e-4)
    log_z = np.log10(f_shifted[:, np.newaxis] / f[1:-1])
    
    # sinc function: sin(x)/x
    sinc_arg = smooth_coeff * log_z
    # Use np.sinc for stability at x=0
    w = np.sinc(sinc_arg / np.pi) ** 4

    y_middle = np.dot(x, w) / np.sum(w, axis=0)

    y = np.zeros_like(x)
    y[1:-1] = y_middle
    y[0] = y[1]
    y[-1] = y[-2]

    return y
