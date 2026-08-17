"""Seiskit Response_Variability arms on GIFNO/seiskit IID H5s.

Hallal Toro / Passeri use seiskit's simplified 1-D randomization (same flags as
comparison/Response_Variability). Pretell is the geometric mean of Thomson–Haskell
|TF| on 200 columns across the 500 m variability strip (OpenSees 1-D Pretell is
not rerun).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_SEISKIT_CANDIDATES = (
    os.environ.get("SEISKIT_ROOT"),
    str(Path.home() / "seiskit"),
    "/tmp/seiskit",
)


def ensure_seiskit() -> Path:
    for raw in _SEISKIT_CANDIDATES:
        if not raw:
            continue
        root = Path(raw).expanduser()
        if (root / "seiskit" / "profile_randomization").is_dir():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    raise ImportError(
        "seiskit not found. Clone https://github.com/KurtSoncco/seiskit "
        "and set SEISKIT_ROOT, or place it at ~/seiskit."
    )


def pretell_strip_columns(n_samples: int = 200, n_strip: int = 500) -> np.ndarray:
    """Evenly spaced columns on the cropped 500 m variability strip."""
    n = max(1, int(n_samples))
    return np.linspace(0, n_strip - 1, n, dtype=int)


def hallal_config(*, vs1: float, H: float, cov: float, vs2: float, dz: float = 0.5):
    from seiskit.profile_randomization import ProfileRandomizationConfig

    return ProfileRandomizationConfig(
        vs_mean=float(vs1),
        thickness=float(H),
        dz=float(dz),
        vs_bedrock=float(vs2),
        bedrock_thickness=10.0,
        cov=float(cov),
        use_full_model=False,
        randomize_layer_thickness=False,
        randomize_bedrock_depth=False,
        vary_bedrock_vs=False,
    )


def _geomean(stack: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(stack, dtype=np.float64), 1e-12, None)
    return np.exp(np.mean(np.log(clipped), axis=0))


def hallal_geomean_tf(
    *,
    freq: np.ndarray,
    vs1: float,
    H: float,
    cov: float,
    vs2: float,
    xi: float,
    n_seeds: int,
    kind: str,
    dz: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (geomean |TF|, σ_ln across seeds), both (n_freq,)."""
    ensure_seiskit()
    from seiskit.profile_randomization import (
        generate_tts_randomized_profile,
        generate_vs_randomized_profile,
    )

    from haskell_baseline import haskell_af_within

    gen = (
        generate_vs_randomized_profile
        if kind == "toro"
        else generate_tts_randomized_profile
    )
    cfg = hallal_config(vs1=vs1, H=H, cov=cov, vs2=vs2, dz=dz)
    rows = []
    for seed in range(1, int(n_seeds) + 1):
        rng = np.random.default_rng(seed)
        vs = np.asarray(gen(cfg, rng), dtype=float)
        zeta = np.full_like(vs, float(xi))
        soil_nz = int(round(float(H) / dz))
        soil_nz = max(1, min(soil_nz, len(vs)))
        rows.append(
            haskell_af_within(
                freq, vs, zeta, dz=dz, vs_rock=float(vs2), soil_nz=soil_nz
            )
        )
    stack = np.vstack(rows)
    from response_variability.metrics import spatial_sigma_ln

    return _geomean(stack), spatial_sigma_ln(stack)


def pretell_haskell_tf(
    *,
    freq: np.ndarray,
    vs_strip: np.ndarray,
    zeta_strip: np.ndarray,
    vs2: float,
    soil_nz: int,
    dz: float = 1.0,
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Geomean Haskell |TF| over Pretell columns. Returns (geomean, σ_ln)."""
    from haskell_baseline import haskell_at_columns

    from response_variability.metrics import spatial_sigma_ln

    cols = pretell_strip_columns(n_samples, n_strip=vs_strip.shape[1])
    stack = haskell_at_columns(
        freq,
        vs_strip,
        zeta_strip,
        cols,
        dz=dz,
        vs_rock=float(vs2),
        soil_nz=int(soil_nz),
    )
    return _geomean(stack), spatial_sigma_ln(stack)
