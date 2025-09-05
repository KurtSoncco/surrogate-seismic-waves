import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Optional

from .acc2fas import acceleration_to_fas
from .kohmachi import kohmachi

def TTF(
    surface_acc: np.ndarray,
    base_acc: np.ndarray,
    dt: float = 1e-4,
    n_points: int = 1000,
    Vsmin: Optional[float] = None,
    dz: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Transfer Function (TF) between surface and base acceleration.

    Args:
    surface_acc : np.ndarray
        Surface acceleration time history.
    base_acc : np.ndarray
        Base acceleration time history.
    dt : float, optional
        Time step of acceleration histories, by default 1e-4.
    n_points : int, optional
        Number of points for frequency vector, by default 1000.
    Vsmin : Optional[float], optional
        Minimum shear wave velocity for fmax calculation, by default None.
    dz : float, optional
        Layer thickness for fmax calculation, by default 5.0.

    Returns:
    Tuple[np.ndarray, np.ndarray]
        - freq (np.ndarray): Downsampled frequency vector.
        - TF (np.ndarray): Transfer function.
    """
    fmax = Vsmin / (15 * dz) if Vsmin is not None else 2.5
    freq_out = np.logspace(np.log10(0.1), np.log10(fmax), n_points)

    def get_resampled_fas(acc: np.ndarray) -> np.ndarray:
        fas, freq = acceleration_to_fas(acc, dt)
        # Use bounds_error=False and fill_value=0 for robustness
        interp_func = interp1d(freq, fas, bounds_error=False, fill_value=0.0)
        return interp_func(freq_out)

    fas_s_resampled = get_resampled_fas(surface_acc)
    fas_b_resampled = get_resampled_fas(base_acc)

    # Smooth the resampled FAS
    fas_s_smooth = kohmachi(fas_s_resampled, freq_out, 150)
    fas_b_smooth = kohmachi(fas_b_resampled, freq_out, 150)

    # Calculate Transfer Function, avoid division by zero
    tf = np.divide(
        fas_s_smooth, fas_b_smooth, out=np.zeros_like(fas_s_smooth), where=fas_b_smooth != 0
    )

    return freq_out, tf
