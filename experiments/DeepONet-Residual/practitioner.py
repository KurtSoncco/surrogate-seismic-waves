"""Practitioner leftover adapter: Vs 2D + layered profile + CoV → GINO inputs.

Required from the user: a 2D Vs section, nominal layers ``{H_i, Vs_i, ζ_i}``,
CoV, recorder x, and a frequency grid. Damping is **provided** — no Darendeli
and no ``DEFAULT_XI_TREND = 0.05``.

``rf_seed`` / ``rH`` / ``aHV`` are **not** network inputs. They belong only in
the GRF ensemble sampler that emits Vs sections. Training support for that
sampler is ``rH ∈ {10, 30, 50}`` m, ``aHV ∈ {10, 20}``, ``CoV ∈ {0.1, 0.2, 0.3}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from features import exponential_psd, fourier_freq_features
from haskell_baseline import haskell_nominal_layered_af_within

from data import (  # noqa: E402
    append_serial_tf1d,
    build_stoch_vector,
    build_trunk_queries,
    f0_quarter_wavelength,
    multiscale_freq_features,
    normalize_vs_surface,
    normalize_zeta_max,
    pad_depth,
    travel_time_s,
    trunk_feature_names,
    vs_eq_travel_time,
)

_EPS = 1e-12
TRAIN_RH_M = (10.0, 30.0, 50.0)
TRAIN_AHV = (10.0, 20.0)
TRAIN_COV = (0.1, 0.2, 0.3)


@dataclass(frozen=True)
class NominalLayers:
    H: np.ndarray
    Vs: np.ndarray
    zeta: np.ndarray
    vs_rock: float

    def __post_init__(self) -> None:
        h = np.asarray(self.H, dtype=np.float64).ravel()
        vs = np.asarray(self.Vs, dtype=np.float64).ravel()
        z = np.asarray(self.zeta, dtype=np.float64).ravel()
        if not (h.size == vs.size == z.size) or h.size == 0:
            raise ValueError("H, Vs, and zeta must be non-empty and the same length")
        object.__setattr__(self, "H", h)
        object.__setattr__(self, "Vs", vs)
        object.__setattr__(self, "zeta", z)


def zeta_field_from_layers(
    layers: NominalLayers,
    *,
    nx: int,
    dz: float = config.DZ,
    nz: int | None = None,
) -> np.ndarray:
    """Layered ζ broadcast across x. Shape ``(nz, nx)``."""
    counts = [max(1, int(round(float(h) / max(float(dz), _EPS)))) for h in layers.H]
    soil_nz = int(sum(counts))
    nz_out = int(nz) if nz is not None else soil_nz
    col = np.empty(nz_out, dtype=np.float64)
    i0 = 0
    for n, zi in zip(counts, layers.zeta):
        i1 = min(nz_out, i0 + n)
        col[i0:i1] = float(zi)
        i0 = i1
        if i0 >= nz_out:
            break
    if i0 < nz_out:
        col[i0:] = float(layers.zeta[-1])
    return np.broadcast_to(col[:, None], (nz_out, nx)).copy()


def fields_from_section(
    vs: np.ndarray,
    layers: NominalLayers,
    recorder_x: np.ndarray,
    *,
    zeta: np.ndarray | None = None,
    rho: float = config.RHO,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``fields (3, Nz_max, n_rec)`` and ``vs_col (n_rec,)``.

    Channel order matches training: ``(Vs_norm, ζ_norm, Z=ρ·Vs)``.
    """
    vs = np.asarray(vs, dtype=np.float64)
    if vs.ndim != 2:
        raise ValueError(f"Vs must be (nz, nx); got {vs.shape}")
    if zeta is None:
        zeta = zeta_field_from_layers(layers, nx=vs.shape[1], nz=vs.shape[0])
    else:
        zeta = np.asarray(zeta, dtype=np.float64)
        if zeta.shape != vs.shape:
            raise ValueError("zeta field must match Vs shape")
    soil_nz = max(1, min(int(round(float(np.sum(layers.H) / config.DZ))), vs.shape[0]))
    vs_pad = pad_depth(vs, config.NZ_MAX)
    zeta_pad = pad_depth(zeta, config.NZ_MAX)
    vs_n = normalize_vs_surface(vs_pad)
    zeta_n = normalize_zeta_max(zeta_pad, soil_nz)
    z_imp = (rho * vs_pad).astype(np.float32)
    z_imp = z_imp / max(float(z_imp.max()), _EPS)
    cols = np.clip(np.asarray(recorder_x, dtype=int), 0, vs.shape[1] - 1)
    fields = np.stack(
        [vs_n[:, cols], zeta_n[:, cols], z_imp[:, cols]], axis=0
    ).astype(np.float32)
    vs_col = vs[:soil_nz, cols].mean(axis=0).astype(np.float32)
    return fields, vs_col


def cov_from_section(
    vs: np.ndarray,
    layers: NominalLayers,
) -> float:
    """CoV of soil Vs on the provided section (std / mean)."""
    vs = np.asarray(vs, dtype=np.float64)
    n = max(1, min(int(round(float(np.sum(layers.H) / config.DZ))), vs.shape[0]))
    soil = vs[:n]
    mu = float(np.mean(soil))
    return float(np.std(soil, ddof=0) / max(mu, _EPS))


def tf1d_nom_from_layers(
    freq: np.ndarray,
    layers: NominalLayers,
    *,
    n_rec: int,
    rho: float = config.RHO,
) -> np.ndarray:
    """Practitioner layered Haskell with **their** ζ. Shape ``(n_rec, n_freq)``."""
    tf = haskell_nominal_layered_af_within(
        np.asarray(freq, dtype=np.float64),
        H=layers.H,
        Vs=layers.Vs,
        vs_rock=float(layers.vs_rock),
        xi=layers.zeta,
        rho=rho,
    ).astype(np.float32)
    return np.broadcast_to(tf[None, :], (int(n_rec), tf.size)).copy()


def practitioner_trunk(
    *,
    vs_col: np.ndarray,
    layers: NominalLayers,
    recorder_x: np.ndarray,
    freq: np.ndarray,
    tf1d: np.ndarray | None = None,
    serial_tf1d: bool = True,
    trunk_scales: int = 4,
) -> np.ndarray:
    """Travel-time ``f/f0`` + ``x/H`` trunk (opt-in practitioner physics)."""
    sin_f, cos_f = fourier_freq_features(
        freq, f_min=config.FREQ_START_HZ, f_max=config.FREQ_END_HZ
    )
    names = trunk_feature_names("full", int(trunk_scales), x_coord="xH")
    extra = multiscale_freq_features(
        freq,
        n_scales=int(trunk_scales),
        f_min=config.FREQ_START_HZ,
        f_max=config.FREQ_END_HZ,
    )
    trunk = build_trunk_queries(
        vs_col=vs_col,
        H=float(np.sum(layers.H)),
        recorder_x=np.asarray(recorder_x, dtype=np.float64),
        freq_s=np.asarray(freq, dtype=np.float64),
        sin_f=sin_f,
        cos_f=cos_f,
        trunk_names=names,
        extra_freq_feats=extra,
        fstar_kind="tts",
        layer_H=layers.H,
        layer_Vs=layers.Vs,
        vs_rock=float(layers.vs_rock),
    )
    if serial_tf1d and tf1d is not None:
        trunk = append_serial_tf1d(trunk, tf1d)
    return trunk


def practitioner_stoch(cov: float) -> np.ndarray:
    return build_stoch_vector(
        xi_vals=np.zeros(0, dtype=np.float32),
        rH=0.0,
        aHV=1.0,
        CoV=float(cov),
        xi_damp=0.0,
        layout="cov_only",
    )


def pack_practitioner_inputs(
    *,
    vs: np.ndarray,
    layers: NominalLayers,
    cov: float,
    recorder_x: np.ndarray,
    freq: np.ndarray,
    zeta: np.ndarray | None = None,
    serial_tf1d: bool = True,
    trunk_scales: int = 4,
) -> dict[str, np.ndarray | float]:
    """Build network tensors from practitioner-typed inputs (no seed / rH / aHV)."""
    fields, vs_col = fields_from_section(vs, layers, recorder_x, zeta=zeta)
    n_rec = int(np.asarray(recorder_x).ravel().size)
    tf1d = tf1d_nom_from_layers(freq, layers, n_rec=n_rec)
    trunk = practitioner_trunk(
        vs_col=vs_col,
        layers=layers,
        recorder_x=recorder_x,
        freq=freq,
        tf1d=tf1d,
        serial_tf1d=serial_tf1d,
        trunk_scales=trunk_scales,
    )
    section_cov = cov_from_section(vs, layers)
    vs_arr = np.asarray(vs, dtype=np.float64)
    rec = np.asarray(recorder_x, dtype=int).ravel()
    c = int(np.clip(rec[len(rec) // 2], 0, vs_arr.shape[1] - 1))
    vs_1d = vs_arr[:, c]
    return {
        "fields": fields,
        "stoch": practitioner_stoch(cov),
        "trunk_y": trunk,
        "tf1d": tf1d,
        "vs_col": vs_col,
        "f0": float(
            f0_quarter_wavelength(
                layers.H, layers.Vs, vs_rock=layers.vs_rock, vs_1d=vs_1d
            )
        ),
        "vs_eq": float(
            vs_eq_travel_time(
                layers.H, layers.Vs, vs_rock=layers.vs_rock, vs_1d=vs_1d
            )
        ),
        "travel_time_s": float(
            travel_time_s(layers.H, layers.Vs, vs_rock=layers.vs_rock, vs_1d=vs_1d)
        ),
        "cov_user": float(cov),
        "cov_from_section": float(section_cov),
        "cov_mismatch": float(section_cov - cov),
    }


def _color_grf(
    *,
    rf_seed: int,
    rH: float,
    aHV: float,
    nx: int,
    nz: int,
    dx: float = config.DX,
    dz: float = config.DZ,
) -> np.ndarray:
    """Unit-variance exponential GRF matching the training FFT coloring."""
    rng = np.random.default_rng(int(rf_seed))
    noise = rng.standard_normal((nz, nx)) + 1j * rng.standard_normal((nz, nx))
    psd = exponential_psd(nx, nz, dx, dz, float(rH), float(aHV))
    field = np.fft.ifft2(noise * np.sqrt(np.maximum(psd, 0.0))).real
    field = field - field.mean()
    std = float(field.std())
    if std > _EPS:
        field = field / std
    return field.astype(np.float64)


def sample_vs_ensemble(
    layers: NominalLayers,
    *,
    cov: float,
    rf_seed: int,
    rH: float,
    aHV: float,
    nx: int = config.NX,
    dz: float = config.DZ,
    warn_support: bool = True,
) -> np.ndarray:
    """GRF ensemble wrapper: seed / rH / aHV live **here**, not in the network.

    Returns ``(nz, nx)`` Vs with soil ln-fluctuations of CoV and bedrock ``vs_rock``.
    """
    if warn_support:
        if float(rH) not in TRAIN_RH_M or float(aHV) not in TRAIN_AHV:
            print(
                f"[practitioner] rH={rH} aHV={aHV} outside training support "
                f"rH∈{TRAIN_RH_M} aHV∈{TRAIN_AHV}",
                flush=True,
            )
        if float(cov) not in TRAIN_COV:
            print(
                f"[practitioner] CoV={cov} outside training support CoV∈{TRAIN_COV}",
                flush=True,
            )
    counts = [max(1, int(round(float(h) / max(float(dz), _EPS)))) for h in layers.H]
    soil_nz = int(sum(counts))
    nz = soil_nz + 8
    grf = _color_grf(rf_seed=rf_seed, rH=rH, aHV=aHV, nx=nx, nz=nz, dz=dz)
    vs = np.full((nz, nx), float(layers.vs_rock), dtype=np.float64)
    i0 = 0
    for n, vs_i in zip(counts, layers.Vs):
        trend = np.log(max(float(vs_i), _EPS))
        vs[i0 : i0 + n] = np.exp(trend + float(cov) * grf[i0 : i0 + n])
        i0 += n
    return vs
