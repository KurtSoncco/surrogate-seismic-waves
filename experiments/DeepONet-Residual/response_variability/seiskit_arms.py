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


LN_P84_Z = 1.0  # Φ(1) ≈ 0.8413; GMPE / site-response "84th percentile"


def lognormal_upper(
    geomean: np.ndarray, sigma_ln: np.ndarray, *, z: float = LN_P84_Z
) -> np.ndarray:
    """Upper lognormal percentile: geomean × exp(z σ_ln). Default z=1 ≈ 84th."""
    geo = np.clip(np.asarray(geomean, dtype=np.float64), 1e-12, None)
    sig = np.asarray(sigma_ln, dtype=np.float64)
    if sig.shape == geo.shape:
        pass
    elif sig.ndim == 0:
        sig = np.full(geo.shape, float(sig))
    elif sig.ndim == 1 and geo.ndim >= 2 and sig.shape[0] == geo.shape[0]:
        sig = sig.reshape((geo.shape[0],) + (1,) * (geo.ndim - 1))
    elif sig.ndim == 1 and geo.ndim >= 1 and sig.shape[0] == geo.shape[-1]:
        sig = sig.reshape((1,) * (geo.ndim - 1) + (sig.shape[0],))
    else:
        sig = np.broadcast_to(sig, geo.shape)
    return geo * np.exp(float(z) * np.clip(sig, 0.0, None))


def upgrade_sigma_ln_from_presentation(
    pack: dict,
    source: Path | None = None,
    *,
    key: str = "sigma_ln_pretell",
) -> dict:
    """Replace collapsed mean(σ_ln) with per-frequency σ_ln when sample_idx matches.

    Seiskit ``predictions.npz`` stored ``mean(σ_ln(f))`` per case. That makes
    p84 a uniform scale of the geomean (Pearson unchanged). Presentation packs
    keep σ_ln(f), which is the GMPE-style 84th-percentile |TF|(f) envelope.
    """
    sig = pack.get(key)
    if sig is None:
        return pack
    sig = np.asarray(sig)
    if sig.ndim >= 2:
        return pack
    if source is None:
        import config as _config

        source = _config.RESULTS_DIR / "presentation" / "iid_pack.npz"
    source = Path(source)
    if not source.is_file():
        return pack
    other = np.load(source, allow_pickle=True)
    if key not in other.files:
        return pack
    other_sig = np.asarray(other[key], dtype=np.float64)
    if other_sig.ndim < 2:
        return pack
    if "sample_idx" in pack and "sample_idx" in other.files:
        if not np.array_equal(
            np.asarray(pack["sample_idx"]), np.asarray(other["sample_idx"])
        ):
            return pack
    n = int(np.asarray(pack["tf_pretell"]).shape[0])
    if other_sig.shape[0] != n:
        return pack
    out = dict(pack)
    out[key] = other_sig
    return out


def attach_pretell_p84(pack: dict) -> dict:
    """Add ``tf_pretell_p84`` from stored Pretell geomean and σ_ln (no Haskell rerun)."""
    if "tf_pretell" not in pack or "sigma_ln_pretell" not in pack:
        return pack
    out = dict(pack)
    out["tf_pretell_p84"] = lognormal_upper(
        pack["tf_pretell"], pack["sigma_ln_pretell"]
    )
    return out


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
