"""Dataset: material fields + stochastic branch, nondimensional trunk queries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import config
import h5py

from features import (
    fourier_freq_features,
    log_freq_hat,
    spectral_kl_coefficients,
)

_EPS = 1e-12
TargetName = Literal["R_col", "R_nom"]
TrunkSet = Literal["fstar", "fstar_fourier", "xL", "full"]
StochLayout = Literal["xi_cov", "legacy20", "cov_only"]
FstarKind = Literal["legacy", "tts"]
XCoordKind = Literal["x_over_lambda", "xH"]
STOCH_LAYOUT_DEFAULT: StochLayout = "xi_cov"
FSTAR_KIND_DEFAULT: FstarKind = "legacy"
X_COORD_DEFAULT: XCoordKind = "x_over_lambda"


def freq_screen_indices(freq: np.ndarray, n: int) -> np.ndarray:
    if n >= len(freq):
        return np.arange(len(freq))
    targets = np.logspace(np.log10(freq[0]), np.log10(freq[-1]), n)
    idx = np.unique([int(np.argmin(np.abs(freq - t))) for t in targets])
    if len(idx) < n:
        extra = [i for i in range(len(freq)) if i not in set(idx)]
        idx = np.concatenate([idx, extra[: n - len(idx)]])
    return np.sort(idx[:n])


def freq_band_balanced_indices(
    freq: np.ndarray,
    n: int,
    bands: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Equal query counts in low / mid / high Hz (Wave A arm F)."""
    f = np.asarray(freq, dtype=np.float64).ravel()
    if n >= len(f):
        return np.arange(len(f))
    if bands is None:
        bands = [
            config.FREQ_BAND_LOW,
            config.FREQ_BAND_MID,
            config.FREQ_BAND_HIGH,
        ]
    n_bands = len(bands)
    base, extra = divmod(int(n), n_bands)
    chosen: list[int] = []
    used: set[int] = set()
    for i, (lo, hi) in enumerate(bands):
        last = i == n_bands - 1
        mask = (f >= lo) & (f <= hi if last else f < hi)
        cand = np.where(mask)[0]
        if cand.size == 0:
            continue
        want = base + (1 if i < extra else 0)
        want = min(want, int(cand.size))
        sub = freq_screen_indices(f[cand], want)
        for j in cand[sub]:
            ij = int(j)
            if ij not in used:
                used.add(ij)
                chosen.append(ij)
    if len(chosen) < n:
        for j in range(len(f)):
            if j not in used:
                chosen.append(j)
                used.add(j)
            if len(chosen) >= n:
                break
    return np.sort(np.asarray(chosen[:n], dtype=int))


def pad_depth(arr: np.ndarray, nz_max: int) -> np.ndarray:
    nz, nx = arr.shape
    if nz == nz_max:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((nz_max, nx), dtype=np.float32)
    n = min(nz, nz_max)
    out[:n] = arr[:n]
    if nz < nz_max and nz > 0:
        out[nz:] = arr[-1]
    return out


def normalize_vs_surface(vs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    surface = np.maximum(vs[0:1, :], eps)
    return (vs / surface).astype(np.float32)


def normalize_zeta_max(zeta: np.ndarray, nz: int, eps: float = 1e-12) -> np.ndarray:
    n = max(1, min(int(nz), zeta.shape[0]))
    zmax = float(np.max(zeta[:n]))
    if zmax < eps:
        return zeta.astype(np.float32)
    return (zeta / zmax).astype(np.float32)


def stoch_dim(
    k_xi: int = config.K_XI, layout: str = STOCH_LAYOUT_DEFAULT
) -> int:
    """Length of the stochastic branch vector.

    ``xi_cov`` (default) is ξ (2 K_XI real/imag KL coeffs) plus CoV.
    ``legacy20`` is the original concat ξ + [rH, aHV, CoV, ξ_damp].
    ``cov_only`` is the practitioner scalar CoV (no ξ).
    rH/aHV still enter ξ via PSD mode ranking in ``xi_cov`` / ``legacy20``.
    """
    k2 = 2 * int(k_xi)
    if layout == "legacy20":
        return k2 + 4
    if layout == "xi_cov":
        return k2 + 1
    if layout == "cov_only":
        return 1
    raise ValueError(f"unknown stoch layout {layout!r}")


def build_stoch_vector(
    *,
    xi_vals: np.ndarray,
    rH: float,
    aHV: float,
    CoV: float,
    xi_damp: float,
    layout: str = STOCH_LAYOUT_DEFAULT,
) -> np.ndarray:
    """Assemble the branch stochastic vector. rH/aHV are unused in ``xi_cov`` / ``cov_only``."""
    xi = np.asarray(xi_vals, dtype=np.float32).ravel()
    if layout == "legacy20":
        tail = np.array([rH, aHV, CoV, xi_damp], dtype=np.float32)
        return np.concatenate([xi, tail]).astype(np.float32)
    if layout == "xi_cov":
        tail = np.array([CoV], dtype=np.float32)
        return np.concatenate([xi, tail]).astype(np.float32)
    if layout == "cov_only":
        return np.array([CoV], dtype=np.float32)
    raise ValueError(f"unknown stoch layout {layout!r}")


def stoch_in_features_from_state(state: dict) -> int | None:
    """``stoch_mlp`` Linear in-features from a ``state_dict``-like mapping."""
    for key, value in state.items():
        name = str(key)
        if "stoch_mlp" not in name or not name.endswith("weight"):
            continue
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) == 2:
            return int(shape[1])
    return None


def infer_stoch_layout(
    state: dict | None = None,
    *,
    recorded: str | None = None,
    in_features: int | None = None,
) -> StochLayout:
    """Recover layout from the stoch Linear width, then the checkpoint field."""
    dim = in_features
    if dim is None and state is not None:
        dim = stoch_in_features_from_state(state)
    k2 = 2 * int(config.K_XI)
    if dim == k2 + 4:
        return "legacy20"
    if dim == k2 + 1:
        return "xi_cov"
    if dim == 1:
        return "cov_only"
    if recorded in ("legacy20", "xi_cov", "cov_only"):
        return recorded  # type: ignore[return-value]
    return STOCH_LAYOUT_DEFAULT


def stoch_layout_from_blob(blob: dict) -> StochLayout:
    return infer_stoch_layout(
        blob.get("model") or {},
        recorded=blob.get("stoch_layout"),
    )


def stoch_dim_from_dataset(ds: object) -> int:
    return stoch_dim(layout=str(getattr(ds, "stoch_layout", STOCH_LAYOUT_DEFAULT)))


def dataset_kwargs_from_blob(blob: dict) -> dict[str, object]:
    """Opt-in physics flags stored on a leftover checkpoint (defaults = shipped)."""
    return {
        "stoch_layout": str(blob.get("stoch_layout", STOCH_LAYOUT_DEFAULT)),
        "fstar_kind": str(blob.get("fstar_kind", FSTAR_KIND_DEFAULT)),
        "x_coord": str(blob.get("x_coord", X_COORD_DEFAULT)),
        "nom_variant": str(blob.get("nom_variant", "default")),
        "trunk_scales": int(blob.get("trunk_scales", 1) or 1),
    }


def repeat_center_column(fields: np.ndarray) -> np.ndarray:
    """Tile the central recorder column across x (GNO/FNO see no lateral Vs)."""
    f = np.asarray(fields)
    if f.ndim != 4:
        raise ValueError(f"fields must be (B, C, Nz, n_rec); got {f.shape}")
    c = f.shape[-1] // 2
    col = f[..., c : c + 1]
    return np.broadcast_to(col, f.shape).copy()


def resolve_bedrock_vs(
    *,
    vs_1d: Sequence[float] | np.ndarray | None = None,
    vs_rock: float | None = None,
    layer_Vs: Sequence[float] | np.ndarray | None = None,
) -> list[float]:
    """Bedrock velocities to exclude from T.

    Prefer the bottom of the 1D profile ``Vs_array_1D[-1]`` (true halfspace).
    ``Vs2`` is a fallback and can be wrong on three-layer / dipping, so any
    layered stack with 3+ entries always treats ``layer_Vs[-1]`` as rock too.
    """
    out: list[float] = []
    if vs_1d is not None:
        arr = np.asarray(vs_1d, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            out.append(float(arr[-1]))
    if vs_rock is not None:
        r = float(vs_rock)
        if np.isfinite(r) and r > 0:
            out.append(r)
    if layer_Vs is not None:
        lv = np.asarray(layer_Vs, dtype=np.float64).ravel()
        if lv.size >= 3:
            out.append(float(lv[-1]))
    return out


def _is_bedrock_vs(vs_i: float, rocks: Sequence[float]) -> bool:
    return any(np.isclose(vs_i, r, rtol=1e-3, atol=1.0) for r in rocks)


def soil_layers_excluding_bedrock(
    layer_H: Sequence[float] | np.ndarray,
    layer_Vs: Sequence[float] | np.ndarray,
    *,
    vs_rock: float | None = None,
    vs_1d: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Soil ``(H_i, Vs_i)`` only. Bedrock is Vs2 and/or ``Vs_1d[-1]``, never in T."""
    h = np.asarray(layer_H, dtype=np.float64).ravel()
    vs = np.asarray(layer_Vs, dtype=np.float64).ravel()
    if h.size != vs.size:
        raise ValueError("layer_H and layer_Vs must match")
    if h.size == 0:
        return h, vs
    rocks = resolve_bedrock_vs(vs_1d=vs_1d, vs_rock=vs_rock, layer_Vs=vs)
    if not rocks or h.size == 1:
        return h, vs
    keep = np.ones(h.size, dtype=bool)
    i = int(h.size - 1)
    while i >= 1 and _is_bedrock_vs(float(vs[i]), rocks):
        keep[i] = False
        i -= 1
    return h[keep], vs[keep]


def travel_time_s(
    layer_H: Sequence[float] | np.ndarray,
    layer_Vs: Sequence[float] | np.ndarray,
    *,
    vs_rock: float | None = None,
    vs_1d: Sequence[float] | np.ndarray | None = None,
) -> float:
    """One-way soil travel time T = Σ H_i / Vs_i [s]. Bedrock Vs is not included."""
    h, vs = soil_layers_excluding_bedrock(
        layer_H, layer_Vs, vs_rock=vs_rock, vs_1d=vs_1d
    )
    vs = np.maximum(vs, _EPS)
    return float(np.sum(h / vs))


def vs_eq_travel_time(
    layer_H: Sequence[float] | np.ndarray,
    layer_Vs: Sequence[float] | np.ndarray,
    *,
    vs_rock: float | None = None,
    vs_1d: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Travel-time equivalent soil velocity Vs_eq = H / T (no bedrock)."""
    h, vs = soil_layers_excluding_bedrock(
        layer_H, layer_Vs, vs_rock=vs_rock, vs_1d=vs_1d
    )
    t = travel_time_s(h, vs)
    return float(np.sum(h) / max(t, _EPS))


def f0_quarter_wavelength(
    layer_H: Sequence[float] | np.ndarray,
    layer_Vs: Sequence[float] | np.ndarray,
    *,
    vs_rock: float | None = None,
    vs_1d: Sequence[float] | np.ndarray | None = None,
) -> float:
    """f0 = 1/(4T) on soil layers. Bedrock is Vs2 or Vs_1d[-1], not in T."""
    return 1.0 / (
        4.0
        * max(travel_time_s(layer_H, layer_Vs, vs_rock=vs_rock, vs_1d=vs_1d), _EPS)
    )


def _meta_float_at(meta: dict, local_i: int, *keys: str) -> float | None:
    for key in keys:
        if key not in meta:
            continue
        arr = np.asarray(meta[key])
        try:
            if arr.ndim == 0:
                val = float(arr)
            else:
                flat = arr.ravel()
                val = float(flat[local_i] if flat.size > local_i else flat[0])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(val) and val > 0.0:
            return val
    return None


def nom_layers_from_meta(
    meta: dict,
    local_i: int,
    *,
    vs_1d: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nominal **soil** layers for f0. Bedrock is Vs2 and/or Vs_1d[-1]."""
    vs_rock = _meta_float_at(meta, local_i, "vs_rock", "Vs2", "Vs_bedrock")
    if "layer_H" in meta and "layer_Vs" in meta:
        h = np.asarray(meta["layer_H"][local_i], dtype=np.float64).ravel()
        vs = np.asarray(meta["layer_Vs"][local_i], dtype=np.float64).ravel()
        if h.size and vs.size and h.size == vs.size:
            return soil_layers_excluding_bedrock(
                h, vs, vs_rock=vs_rock, vs_1d=vs_1d
            )
    h = float(meta["H"][local_i])
    vs1 = float(meta["Vs1"][local_i])
    return np.array([h], dtype=np.float64), np.array([vs1], dtype=np.float64)


def vs_1d_from_h5(
    h5_path: Path | str, *, col: int | None = None
) -> np.ndarray | None:
    """Central (or ``col``) 1D Vs column, bedrock at ``[-1]``."""
    path = Path(h5_path)
    if not path.is_file():
        return None
    try:
        with h5py.File(path, "r") as f:
            vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
    except OSError:
        return None
    vs = vs[:, config.X_SLICE_START : config.X_SLICE_END]
    if vs.size == 0:
        return None
    j = vs.shape[1] // 2 if col is None else int(col)
    j = min(max(j, 0), vs.shape[1] - 1)
    return vs[:, j]


def append_serial_tf1d(trunk: np.ndarray, tf1d: np.ndarray) -> np.ndarray:
    """Concatenate log(TF_1D) onto trunk queries (serial / discrepancy operator)."""
    extra = np.log(np.maximum(np.asarray(tf1d).reshape(-1, 1), _EPS)).astype(np.float32)
    return np.concatenate([np.asarray(trunk, dtype=np.float32), extra], axis=-1)


def multiscale_freq_names(n_scales: int) -> list[str]:
    """Names for the extra octave-spaced log-frequency harmonics (k = 2, 4, ...)."""
    return [
        f"{p}_f_k{2 ** k}"
        for k in range(1, max(int(n_scales), 1))
        for p in ("sin", "cos")
    ]


def multiscale_freq_features(
    freq: np.ndarray, *, n_scales: int, f_min: float = 0.1, f_max: float = 10.0
) -> dict[str, np.ndarray]:
    """sin/cos(2 pi k f_hat) for k = 2, 4, ..., 2**(n_scales-1) on log-scaled f_hat.

    k = 1 is already supplied by `fourier_freq_features`; the shipped trunk stops
    there, so every structure narrower than one full log-frequency period has to
    be synthesised by the MLP. These harmonics give the trunk an explicit basis
    down to a fractional bandwidth of (log10 f_max - log10 f_min) / k decades.
    """
    if int(n_scales) <= 1:
        return {}
    fhat = log_freq_hat(np.asarray(freq, dtype=np.float64), f_min=f_min, f_max=f_max)
    out: dict[str, np.ndarray] = {}
    for k_i in range(1, int(n_scales)):
        k = 2**k_i
        ang = 2.0 * np.pi * float(k) * fhat
        out[f"sin_f_k{k}"] = np.sin(ang).astype(np.float32)
        out[f"cos_f_k{k}"] = np.cos(ang).astype(np.float32)
    return out


def infer_trunk_scales(
    *,
    trunk_set: TrunkSet,
    serial: bool,
    recorded: int = 1,
    trunk_in_features: int | None = None,
) -> int:
    """Fourier-harmonic count from a checkpoint's trunk input width.

    CombinedResidualDataset used to drop ``trunk_scales``, so some run JSONs
    record 1 even when the trunk was trained with extra octave harmonics.
    Prefer the Linear in-features on ``trunk*.net.0.weight`` when it matches
    ``len(names) + serial``.
    """
    recorded = max(int(recorded or 1), 1)
    if trunk_in_features is None:
        return recorded
    base = len(trunk_feature_names(trunk_set, 1)) + (1 if serial else 0)
    extra = int(trunk_in_features) - base
    if extra < 0 or extra % 2 != 0:
        return recorded
    return extra // 2 + 1


def trunk_in_features_from_state(state: dict) -> int | None:
    """First trunk Linear in-features from a ``state_dict``-like mapping."""
    for key, value in state.items():
        name = str(key)
        if "trunk" not in name or not name.endswith("net.0.weight"):
            continue
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) == 2:
            return int(shape[1])
    return None


def trunk_feature_names(
    trunk_set: TrunkSet,
    trunk_scales: int = 1,
    *,
    x_coord: str = X_COORD_DEFAULT,
) -> list[str]:
    if trunk_set == "fstar":
        base = ["f_star"]
    elif trunk_set == "fstar_fourier":
        base = ["f_star", "sin_f", "cos_f"]
    elif trunk_set == "xL":
        base = ["f_star", "sin_f", "cos_f", "x_over_L"]
    else:
        xname = "x_over_H" if str(x_coord) == "xH" else "x_over_lambda"
        base = ["f_star", "sin_f", "cos_f", xname]
    return base + multiscale_freq_names(trunk_scales)


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_splits(
    n: int,
    *,
    seed: int = config.SEED,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
) -> SplitIndices:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return SplitIndices(
        train=perm[:n_train],
        val=perm[n_train : n_train + n_val],
        test=perm[n_train + n_val :],
    )


def build_trunk_queries(
    *,
    vs_col: np.ndarray,
    H: float,
    recorder_x: np.ndarray,
    freq_s: np.ndarray,
    sin_f: np.ndarray,
    cos_f: np.ndarray,
    trunk_names: Sequence[str],
    extra_freq_feats: dict[str, np.ndarray] | None = None,
    fstar_kind: str = FSTAR_KIND_DEFAULT,
    layer_H: Sequence[float] | np.ndarray | None = None,
    layer_Vs: Sequence[float] | np.ndarray | None = None,
    vs_rock: float | None = None,
    vs_1d: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized trunk features, shape (n_rec * n_freq, n_feat).

    ``fstar_kind='legacy'`` is ``f H / vs_col`` (shipped). ``'tts'`` is
    ``f / f0`` with ``f0 = 1/(4T)`` on soil layers (bedrock = Vs2 or Vs_1d[-1]).
    """
    vs_c = np.maximum(np.asarray(vs_col, dtype=np.float64).ravel(), _EPS)
    f = np.asarray(freq_s, dtype=np.float64).ravel()
    cols = np.asarray(recorder_x, dtype=np.float64).ravel()
    x_m = (cols + 0.5) * config.DX
    n_rec = vs_c.size
    n_f = f.size
    h_tot = float(H)
    if str(fstar_kind) == "tts":
        lh = layer_H if layer_H is not None else [h_tot]
        lv = layer_Vs if layer_Vs is not None else [float(np.mean(vs_c))]
        t_s = travel_time_s(lh, lv, vs_rock=vs_rock, vs_1d=vs_1d)
        f_star = np.broadcast_to(
            (4.0 * f * t_s).astype(np.float32)[None, :], (n_rec, n_f)
        ).copy()
    else:
        f_star = (f[None, :] * h_tot / vs_c[:, None]).astype(np.float32)
    lam = vs_c[:, None] / np.maximum(f[None, :], _EPS)
    x_over_lambda = (x_m[:, None] / np.maximum(lam, _EPS)).astype(np.float32)
    x_over_H = np.broadcast_to(
        (x_m / max(h_tot, _EPS)).astype(np.float32)[:, None],
        (n_rec, n_f),
    ).copy()
    x_over_L = np.broadcast_to(
        (x_m / float(config.LX_VARIABILITY)).astype(np.float32)[:, None],
        (n_rec, n_f),
    )
    sin_b = np.broadcast_to(np.asarray(sin_f, dtype=np.float32)[None, :], (n_rec, n_f))
    cos_b = np.broadcast_to(np.asarray(cos_f, dtype=np.float32)[None, :], (n_rec, n_f))
    feat_map = {
        "f_star": f_star,
        "sin_f": sin_b,
        "cos_f": cos_b,
        "x_over_L": x_over_L,
        "x_over_lambda": x_over_lambda,
        "x_over_H": x_over_H,
    }
    for name, vec in (extra_freq_feats or {}).items():
        feat_map[name] = np.broadcast_to(
            np.asarray(vec, dtype=np.float32)[None, :], (n_rec, n_f)
        )
    stacked = np.stack([feat_map[name] for name in trunk_names], axis=-1)
    return stacked.reshape(n_rec * n_f, -1).astype(np.float32)


class ResidualDeepONetDataset(Dataset):
    """One item = one realization; queries all recorders × selected freqs."""

    def __init__(
        self,
        cache_dir: Path,
        indices: Sequence[int],
        *,
        target: TargetName = "R_col",
        trunk_set: TrunkSet = "full",
        n_freq: int = config.N_FREQ_TRAIN,
        n_freq_train: int | None = None,
        serial_tf1d: bool = False,
        freq_sample: str = "log",
        trunk_scales: int = 1,
        stoch_layout: str = STOCH_LAYOUT_DEFAULT,
        fstar_kind: str = FSTAR_KIND_DEFAULT,
        x_coord: str = X_COORD_DEFAULT,
        nom_variant: str = "default",
    ):
        if n_freq_train is not None:
            n_freq = n_freq_train
        self.cache_dir = Path(cache_dir)
        self.indices = np.asarray(indices, dtype=int)
        self.target = target
        self.trunk_set = trunk_set
        self.n_freq_requested = int(n_freq)
        self.serial_tf1d = bool(serial_tf1d)
        self.freq_sample = str(freq_sample)
        self.trunk_scales = max(int(trunk_scales), 1)
        self.stoch_layout: StochLayout = infer_stoch_layout(recorded=stoch_layout)
        self.fstar_kind: FstarKind = "tts" if str(fstar_kind) == "tts" else "legacy"
        self.x_coord: XCoordKind = "xH" if str(x_coord) == "xH" else "x_over_lambda"
        self.nom_variant = str(nom_variant)

        self.meta = dict(np.load(self.cache_dir / "meta.npz", allow_pickle=True))
        key = "r_col_signed.npy" if target == "R_col" else "r_nom_signed.npy"
        tf_key = "tf1d_col.npy" if target == "R_col" else "tf1d_nom.npy"
        if target == "R_nom" and self.nom_variant == "sample_xi":
            key = "r_nom_xi_signed.npy"
            tf_key = "tf1d_nom_xi.npy"
            if not (self.cache_dir / key).is_file() or not (self.cache_dir / tf_key).is_file():
                raise FileNotFoundError(
                    f"nom_variant=sample_xi needs {key} and {tf_key} in {self.cache_dir}"
                )
        self.r = np.load(self.cache_dir / key, mmap_mode="r")
        self.tf1d = np.load(self.cache_dir / tf_key, mmap_mode="r")
        tf2d_path = self.cache_dir / "tf2d.npy"
        self.tf2d_local = (
            np.load(tf2d_path, mmap_mode="r") if tf2d_path.is_file() else None
        )
        self.tf_all = None
        if self.tf2d_local is None and config.TF_PER_SAMPLE_PATH.is_file():
            try:
                self.tf_all = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
            except OSError:
                self.tf_all = None
        self.sample_indices = np.load(self.cache_dir / "sample_indices.npy")
        fields_path = self.cache_dir / "fields.npy"
        vs_col_path = self.cache_dir / "vs_col.npy"
        self._fields_all = (
            np.load(fields_path, mmap_mode="r") if fields_path.is_file() else None
        )
        self._vs_col_all = (
            np.load(vs_col_path, mmap_mode="r") if vs_col_path.is_file() else None
        )

        self.freq = np.load(
            self.cache_dir / "freq.npy"
            if (self.cache_dir / "freq.npy").is_file()
            else config.TF_FREQ_PATH
        )
        rec_path = self.cache_dir / "recorder_x.npy"
        if rec_path.is_file():
            self.recorder_x = np.load(rec_path)
        elif config.RECORDER_X_IDX_PATH.is_file():
            self.recorder_x = np.load(config.RECORDER_X_IDX_PATH)
        else:
            self.recorder_x = np.arange(self.r.shape[1])
        if str(freq_sample) == "band":
            self.f_idx = freq_band_balanced_indices(self.freq, n_freq)
        else:
            self.f_idx = freq_screen_indices(self.freq, n_freq)
        self.freq_s = self.freq[self.f_idx]
        self.sin_f, self.cos_f = fourier_freq_features(
            self.freq_s,
            f_min=config.FREQ_START_HZ,
            f_max=config.FREQ_END_HZ,
        )
        self.extra_freq_feats = multiscale_freq_features(
            self.freq_s,
            n_scales=self.trunk_scales,
            f_min=config.FREQ_START_HZ,
            f_max=config.FREQ_END_HZ,
        )
        self.n_rec = len(self.recorder_x)
        self.n_q = self.n_rec * len(self.f_idx)
        self.trunk_names = trunk_feature_names(
            trunk_set, self.trunk_scales, x_coord=self.x_coord
        )
        self.stoch_dim = stoch_dim(layout=self.stoch_layout)
        # Preload tensors for this split (n=100 ablation is H5-bound otherwise).
        self._cache: list[dict[str, torch.Tensor]] = []
        desc = f"dataset {target}/{trunk_set}/nf={len(self.f_idx)}"
        for local_i in tqdm(self.indices, desc=desc, leave=False):
            item = self._load_item(int(local_i))
            self._cache.append({k: torch.from_numpy(v) for k, v in item.items()})

    def __len__(self) -> int:
        return len(self.indices)

    def _stoch(self, local_i: int) -> np.ndarray:
        CoV = float(self.meta["CoV"][local_i])
        if self.stoch_layout == "cov_only":
            return build_stoch_vector(
                xi_vals=np.zeros(0, dtype=np.float32),
                rH=0.0,
                aHV=1.0,
                CoV=CoV,
                xi_damp=0.0,
                layout="cov_only",
            )
        rf_seed = int(self.meta["rf_seed"][local_i])
        rH = float(self.meta["rH"][local_i])
        aHV = float(self.meta["aHV"][local_i])
        nz = int(self.meta["nz"][local_i])
        xi_damp = float(
            self.meta["xi_damp"][local_i]
            if "xi_damp" in self.meta
            else config.DEFAULT_XI_TREND
        )
        xi_vals, _ = spectral_kl_coefficients(
            rf_seed=rf_seed,
            rH=rH,
            aHV=aHV,
            nx=config.NX,
            nz=nz,
            dx=config.DX,
            dz=config.DZ,
            k=config.K_XI,
        )
        return build_stoch_vector(
            xi_vals=xi_vals,
            rH=rH,
            aHV=aHV,
            CoV=CoV,
            xi_damp=xi_damp,
            layout=self.stoch_layout,
        )

    def _load_item(self, local_i: int) -> dict[str, np.ndarray]:
        h5_path = Path(str(self.meta["h5_path"][local_i]))
        nz = int(self.meta["nz"][local_i])
        H = float(self.meta["H"][local_i])
        soil_nz = int(self.meta["soil_nz"][local_i])
        vs_1d: np.ndarray | None = None
        if self._fields_all is not None and self._vs_col_all is not None:
            fields = np.asarray(self._fields_all[local_i], dtype=np.float32)
            vs_col = np.asarray(self._vs_col_all[local_i], dtype=np.float64)
        else:
            with h5py.File(h5_path, "r") as f:
                vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
                zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
            vs = vs[:, config.X_SLICE_START : config.X_SLICE_END]
            zeta = zeta[:, config.X_SLICE_START : config.X_SLICE_END]
            c = int(self.recorder_x[len(self.recorder_x) // 2])
            c = min(max(c, 0), vs.shape[1] - 1)
            vs_1d = vs[:, c]
            vs_pad = pad_depth(vs, config.NZ_MAX)
            zeta_pad = pad_depth(zeta, config.NZ_MAX)
            vs_n = normalize_vs_surface(vs_pad)
            zeta_n = normalize_zeta_max(zeta_pad, nz)
            z_imp = (config.RHO * vs_pad).astype(np.float32)
            z_imp = z_imp / max(float(z_imp.max()), _EPS)
            cols = self.recorder_x.astype(int)
            fields = np.stack(
                [vs_n[:, cols], zeta_n[:, cols], z_imp[:, cols]], axis=0
            ).astype(np.float32)
            n = max(1, min(soil_nz, vs.shape[0]))
            vs_col = vs[:n, cols].mean(axis=0)

        r = np.asarray(self.r[local_i][:, self.f_idx], dtype=np.float32)
        tf1d = np.asarray(self.tf1d[local_i][:, self.f_idx], dtype=np.float32)
        if self.tf2d_local is not None:
            tf2d = np.asarray(self.tf2d_local[local_i][:, self.f_idx], dtype=np.float32)
        elif self.tf_all is not None:
            sidx = int(self.sample_indices[local_i])
            if 0 <= sidx < len(self.tf_all):
                tf2d = np.asarray(self.tf_all[sidx][:, self.f_idx], dtype=np.float32)
            else:
                tf2d = (tf1d + r).astype(np.float32)
        else:
            tf2d = (tf1d + r).astype(np.float32)
        layer_H, layer_Vs = nom_layers_from_meta(self.meta, local_i, vs_1d=vs_1d)
        vs_rock = _meta_float_at(self.meta, local_i, "vs_rock", "Vs2", "Vs_bedrock")
        f0_tts = f0_quarter_wavelength(
            layer_H, layer_Vs, vs_rock=vs_rock, vs_1d=vs_1d
        )
        trunk_y = build_trunk_queries(
            vs_col=vs_col,
            H=H,
            recorder_x=self.recorder_x,
            freq_s=self.freq_s,
            sin_f=self.sin_f,
            cos_f=self.cos_f,
            trunk_names=self.trunk_names,
            extra_freq_feats=self.extra_freq_feats,
            fstar_kind=self.fstar_kind,
            layer_H=layer_H,
            layer_Vs=layer_Vs,
            vs_rock=vs_rock,
            vs_1d=vs_1d,
        )
        if self.serial_tf1d:
            trunk_y = append_serial_tf1d(trunk_y, tf1d)
        return {
            "fields": fields,
            "stoch": self._stoch(local_i),
            "trunk_y": trunk_y,
            "target": r.reshape(-1),
            "tf1d": tf1d.reshape(-1),
            "tf2d": tf2d.reshape(-1),
            "f0_tts": np.array(float(f0_tts), dtype=np.float64),
            "geom_flags": np.zeros(2, dtype=np.float32),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._cache[idx]


def make_loaders(
    cache_dir: Path,
    splits: SplitIndices,
    *,
    target: TargetName,
    trunk_set: TrunkSet = "full",
    batch_size: int = config.BATCH_SIZE,
    n_freq: int = config.N_FREQ_TRAIN,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def _loader(idxs: np.ndarray, shuffle: bool) -> DataLoader:
        ds = ResidualDeepONetDataset(
            cache_dir, idxs, target=target, trunk_set=trunk_set, n_freq=n_freq
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.NUM_WORKERS,
            drop_last=False,
        )

    return (
        _loader(splits.train, True),
        _loader(splits.val, False),
        _loader(splits.test, False),
    )


def geom_flags_from_name(name: str) -> torch.Tensor:
    """[dip, three_layer] flags from a mix-domain label."""
    n = str(name).lower()
    dip = 1.0 if ("dip" in n or "dipping" in n) else 0.0
    three = 1.0 if ("three" in n or "3l" in n or "three_layer" in n) else 0.0
    return torch.tensor([dip, three], dtype=torch.float32)


class CombinedResidualDataset(Dataset):
    """Concat of preloaded ResidualDeepONetDataset caches (for multi-domain train)."""

    def __init__(
        self,
        datasets: Sequence[ResidualDeepONetDataset],
        domain_names: Sequence[str] | None = None,
    ):
        if not datasets:
            raise ValueError("need at least one dataset")
        self._parts = list(datasets)
        names = (
            list(domain_names)
            if domain_names is not None
            else ["unk"] * len(self._parts)
        )
        if len(names) != len(self._parts):
            raise ValueError("domain_names must match datasets")
        self._cache: list[dict[str, torch.Tensor]] = []
        self.domain_names_per_item: list[str] = []
        for ds, name in zip(self._parts, names):
            self._cache.extend(ds._cache)
            self.domain_names_per_item.extend([str(name)] * len(ds._cache))
        self.n_rec = self._parts[0].n_rec
        self.f_idx = self._parts[0].f_idx
        self.freq_s = getattr(self._parts[0], "freq_s", None)
        self.trunk_names = self._parts[0].trunk_names
        self.trunk_scales = int(getattr(self._parts[0], "trunk_scales", 1) or 1)
        self.stoch_layout = getattr(
            self._parts[0], "stoch_layout", STOCH_LAYOUT_DEFAULT
        )
        self.fstar_kind = getattr(self._parts[0], "fstar_kind", FSTAR_KIND_DEFAULT)
        self.x_coord = getattr(self._parts[0], "x_coord", X_COORD_DEFAULT)
        self.nom_variant = getattr(self._parts[0], "nom_variant", "default")
        self.serial_tf1d = self._parts[0].serial_tf1d
        self._geom_flags = torch.stack(
            [geom_flags_from_name(n) for n in self.domain_names_per_item], dim=0
        )

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = dict(self._cache[idx])
        item["geom_flags"] = self._geom_flags[idx]
        return item


def iid_resample_sampler(
    ds: CombinedResidualDataset,
    iid_frac: float,
) -> WeightedRandomSampler:
    """Weighted sampler so an expected ``iid_frac`` of each epoch is IID."""
    is_iid = np.array(
        [name.startswith("iid") for name in ds.domain_names_per_item], dtype=bool
    )
    n_iid = int(is_iid.sum())
    n_ood = int((~is_iid).sum())
    if n_iid == 0 or n_ood == 0:
        raise ValueError("iid resampling needs both IID and OOD items")
    weights = np.zeros(len(ds), dtype=np.float64)
    weights[is_iid] = float(iid_frac) / n_iid
    weights[~is_iid] = (1.0 - float(iid_frac)) / n_ood
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(ds),
        replacement=True,
    )
