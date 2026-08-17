"""Box OOD corpus I/O: discover H5, TF cache / accel TF, Haskell-nom params."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import config
import h5py

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_RUN_RE = re.compile(r"(?:run|case)_(\d+)\.h5$", re.IGNORECASE)
_EPS = 1e-12

# GIFNO / seiskit: first n_lateral channels are base when row_y_m is present
# (ascending elevation: deeper then shallower). Without row_y_m, GIFNO TF
# preprocess swaps that order. We follow compute_transfer_function.py.


def corpus_root(name: str) -> Path:
    key = name.lower().replace("-", "_")
    if key in ("dipping", "ood_dipping"):
        return config.ood_dipping_root()
    if key in ("three_layer", "ood_three_layer", "threelayer"):
        return config.ood_three_layer_root()
    raise ValueError(f"Unknown OOD corpus {name!r}")


def default_ood_roots() -> dict[str, Path]:
    return {
        "ood_dipping": config.ood_dipping_root(),
        "ood_three_layer": config.ood_three_layer_root(),
    }


def discover_h5_files(root: Path) -> list[Path]:
    """Accept GIFNO mini-corpus (`h5/run_*.h5`) or flat seiskit-style trees.

    Prefer ``manifest.csv`` (Box OOD packs) so we do not glob 960 FUSE entries.
    """
    root = Path(root)
    if not root.exists():
        return []
    man = root / "manifest.csv"
    mini = root / "h5"
    if man.is_file() and mini.is_dir():
        with open(man, newline="") as f:
            rows = list(csv.DictReader(f))
        files: list[Path] = []
        for row in rows:
            if row.get("h5_path"):
                cand = Path(row["h5_path"])
                if not cand.is_file():
                    cand = mini / cand.name
            elif "index" in row:
                cand = mini / f"run_{int(row['index'])}.h5"
                if not cand.is_file():
                    cand = mini / f"case_{int(row['index'])}.h5"
            else:
                continue
            if cand.is_file():
                files.append(cand)
        if files:
            return files

    candidates: list[Path] = []
    search_roots = [mini] if mini.is_dir() else [root]
    for sr in search_roots:
        for pat in ("run_*.h5", "case_*.h5"):
            candidates.extend(sr.glob(pat) if sr == mini else sr.rglob(pat))
        if sr == mini and not candidates:
            candidates.extend(mini.glob("*.h5"))
    if not candidates and not mini.is_dir():
        for pat in ("run_*.h5", "case_*.h5"):
            candidates.extend(root.rglob(pat))
    uniq = sorted({p.resolve() for p in candidates if p.is_file()}, key=_h5_sort_key)
    return uniq


def _h5_sort_key(p: Path) -> tuple[str, int, str]:
    m = _RUN_RE.search(p.name)
    idx = int(m.group(1)) if m else 10**9
    return (p.parent.as_posix(), idx, p.name)


def tf_cache_dir(root: Path) -> Path | None:
    d = Path(root) / "transfer_function"
    if (d / "tf_per_sample.npy").is_file():
        return d
    return None


def recorder_x_indices(root: Path | None = None) -> np.ndarray:
    if root is not None:
        cached = Path(root) / "transfer_function" / "recorder_x_idx.npy"
        if cached.is_file():
            return np.load(cached)
    iid = config.RECORDER_X_IDX_PATH
    if iid.is_file():
        return np.load(iid)
    # GIFNO default: 21 recorders on cropped NX=500, 15 m spacing
    center = config.NX // 2
    step = 15
    lo = center - 10 * step
    hi = center + 10 * step
    return np.arange(lo, hi + 1, step, dtype=np.int64)


def _attr_float(
    attrs: dict[str, Any], *keys: str, default: float | None = None
) -> float:
    for k in keys:
        if k in attrs and attrs[k] is not None:
            try:
                return float(attrs[k])
            except (TypeError, ValueError):
                continue
    if default is not None:
        return float(default)
    raise KeyError(f"none of {keys} in attrs")


def _attr_int(attrs: dict[str, Any], *keys: str, default: int | None = None) -> int:
    for k in keys:
        if k in attrs and attrs[k] is not None:
            try:
                return round(float(attrs[k]))
            except (TypeError, ValueError):
                continue
    if default is not None:
        return int(default)
    raise KeyError(f"none of {keys} in attrs")


def read_h5_sample(
    h5_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Return Vs, zeta, params attrs, extra (grid, accel shape)."""
    extra: dict[str, Any] = {}
    with h5py.File(h5_path, "r") as f:
        vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
        zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
        params = {k: f["params"].attrs[k] for k in f["params"].attrs}
        extra["vs_shape"] = tuple(int(x) for x in vs.shape)
        extra["param_keys"] = sorted(str(k) for k in params)
        if "grid" in f:
            extra["grid_attrs"] = {k: f["grid"].attrs[k] for k in f["grid"].attrs}
            extra["dt"] = float(f["grid"].attrs.get("dt", np.nan))
        if "recorders/accel/data" in f:
            d = f["recorders/accel/data"]
            extra["accel_shape"] = tuple(int(x) for x in d.shape)
            extra["accel_n_channels"] = int(d.shape[1]) if d.ndim >= 2 else 0
            extra["accel_attrs"] = {k: d.attrs[k] for k in d.attrs}
    return vs, zeta, params, extra


def nominal_layer_params(params: dict[str, Any]) -> dict[str, Any]:
    """Single-layer Haskell-nom (Vs1, H, Vs2) plus a provenance string.

    IID / dipping: attrs already have (Vs1, H_discretized|H, Vs2).
    three_layer: no meaningful one-layer (Vs1, H, Vs2). Nom is **misspecified
    by design**: top-layer Vs1, total soil H = H1+H2, bedrock Vs. Do not treat
    this as an equivalent 1-layer fit.
    """
    keys = set(params)

    has_iid = (
        "Vs1" in keys
        and ("Vs2" in keys or "vs2" in keys)
        and ("H_discretized" in keys or "H" in keys)
        and "H1_discretized" not in keys
        and "H1" not in keys
        and "Vs_bedrock" not in keys
        and "Vs_mid" not in keys
    )
    if has_iid:
        vs1 = _attr_float(params, "Vs1")
        H = _attr_float(params, "H_discretized", "H")
        vs2 = _attr_float(params, "Vs2")
        return {
            "vs1": vs1,
            "H": H,
            "vs2": vs2,
            "source": "attrs_Vs1_H_Vs2",
            "misspecified": False,
        }

    three = (
        "H1_discretized" in keys
        or "H2_discretized" in keys
        or "H1" in keys
        or "Vs_bedrock" in keys
    )
    if three or "Vs_mid" in keys:
        vs1 = _attr_float(params, "Vs1")
        h1 = _attr_float(params, "H1_discretized", "H1", "H1_requested", default=0.0)
        h2 = _attr_float(params, "H2_discretized", "H2", "H2_requested", default=0.0)
        vs2 = _attr_float(params, "Vs_bedrock", "Vs2")
        vs_mid = _attr_float(params, "Vs_mid") if "Vs_mid" in params else None
        out = {
            "vs1": vs1,
            "H": float(h1 + h2),
            "H1": h1,
            "H2": h2,
            "vs2": vs2,
            "vs_mid": vs_mid,
            "source": "three_layer_topVs1_totalH_bedrock",
            "misspecified": True,
        }
        if vs_mid is not None and h1 > 0.0 and h2 > 0.0:
            out["true_layers"] = {
                "H": [h1, h2],
                "Vs": [vs1, vs_mid],
                "vs_rock": vs2,
                "source": "three_layer_Vs1_H1_Vsmid_H2_bedrock",
            }
        return out

    # Last resort: whatever looks like a 1-layer stack
    vs1 = _attr_float(params, "Vs1")
    H = _attr_float(params, "H_discretized", "H", "H1_discretized")
    vs2 = _attr_float(params, "Vs2", "Vs_bedrock")
    return {
        "vs1": vs1,
        "H": H,
        "vs2": vs2,
        "source": "fallback_attrs",
        "misspecified": True,
    }


def soil_nz_from_params(params: dict[str, Any], vs_nz: int) -> int:
    if "soil_layer_count" in params:
        return max(1, min(_attr_int(params, "soil_layer_count"), vs_nz))
    if (
        "H1_discretized" in params
        or "layer1_count" in params
        or "H1" in params
        or "H2" in params
    ):
        n1 = _attr_int(params, "layer1_count", "H1_discretized", "H1", default=0)
        n2 = _attr_int(params, "layer2_count", "H2_discretized", "H2", default=0)
        n = n1 + n2
        if n > 0:
            return max(1, min(n, vs_nz))
    if "H_discretized" in params or "H" in params:
        return max(1, min(_attr_int(params, "H_discretized", "H"), vs_nz))
    return vs_nz


def crop_variability(field: np.ndarray) -> np.ndarray:
    if field.shape[1] >= config.X_SLICE_END:
        return field[:, config.X_SLICE_START : config.X_SLICE_END]
    return field


def _split_base_surf(
    data: np.ndarray, row_y_m: Any, n_lateral: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (base_2d, surf_2d) each (n_lateral, n_time).

    Box OOD and seiskit capability-check H5s store ``[base | surface]`` even
    when ``row_y_m`` is absent. GIFNO IID TF preprocess only swaps that order
    when ``row_y_m is None``; those IID files typically set ``row_y_m``.
    """
    if data.shape[1] != 2 * n_lateral:
        raise ValueError(
            f"Expected {2 * n_lateral} accel channels, got {data.shape[1]}"
        )
    if row_y_m is None:
        base_2d = data[:, :n_lateral].T
        surf_2d = data[:, n_lateral : 2 * n_lateral].T
    else:
        row_y = np.asarray(row_y_m, dtype=np.float64).ravel()
        # Ascending elevation: deeper (base) then shallower (surface).
        if row_y.size == 2 and row_y[0] > row_y[1]:
            surf_2d = data[:, :n_lateral].T
            base_2d = data[:, n_lateral : 2 * n_lateral].T
        else:
            base_2d = data[:, :n_lateral].T
            surf_2d = data[:, n_lateral : 2 * n_lateral].T
    return base_2d, surf_2d


def _ttf_batch_seiskit(
    base_2d: np.ndarray,
    surf_2d: np.ndarray,
    *,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    from seiskit.ttf.TTF import TTF_batch_fast  # noqa: WPS433

    freq, mags = TTF_batch_fast(
        base_2d,
        surf_2d,
        dt=dt,
        dz=config.DZ,
        smooth_coeff=config.SMOOTH_COEFF,
        Vsmin=None,
        n_points=config.N_FREQ,
    )
    return np.asarray(mags, dtype=np.float32), np.asarray(freq, dtype=np.float32)


_KO_WEIGHTS: tuple[np.ndarray, np.ndarray] | None = None  # (freq, W)


def _kohmachi_weights(freq: np.ndarray, smooth_coeff: float) -> np.ndarray:
    """Return W such that smoothed[1:-1] = signal @ W, shape (n_freq, n_freq-2)."""
    f = np.asarray(freq, dtype=np.float64)
    f_shifted = f / (1 + 1e-4)
    log_z = np.log10(f_shifted[:, np.newaxis] / f[1:-1])
    w = np.sinc(smooth_coeff * log_z / np.pi) ** 4
    denom = np.sum(w, axis=0, keepdims=True)
    denom = np.maximum(denom, _EPS)
    return w / denom


def _ttf_batch_local(
    base_2d: np.ndarray,
    surf_2d: np.ndarray,
    *,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Konno–Ohmachi |TF| on the GIFNO 0.1–10 Hz log grid (seiskit fallback)."""
    from scipy.fft import rfft, rfftfreq

    global _KO_WEIGHTS
    n_lat = int(surf_2d.shape[0])
    n_time = int(surf_2d.shape[1])
    freq_out = np.logspace(
        np.log10(config.FREQ_START_HZ), np.log10(config.FREQ_END_HZ), config.N_FREQ
    ).astype(np.float64)
    freq_fft = rfftfreq(n_time, d=dt)
    scale = 2.0 / n_time
    fas_s = np.abs(rfft(np.asarray(surf_2d, dtype=np.float64), axis=1)) * scale
    fas_b = np.abs(rfft(np.asarray(base_2d, dtype=np.float64), axis=1)) * scale
    fas_s_i = np.vstack(
        [
            np.interp(freq_out, freq_fft, fas_s[i], left=0.0, right=0.0)
            for i in range(n_lat)
        ]
    )
    fas_b_i = np.vstack(
        [
            np.interp(freq_out, freq_fft, fas_b[i], left=0.0, right=0.0)
            for i in range(n_lat)
        ]
    )
    if _KO_WEIGHTS is None or _KO_WEIGHTS[0].shape != freq_out.shape:
        _KO_WEIGHTS = (freq_out, _kohmachi_weights(freq_out, config.SMOOTH_COEFF))
    W = _KO_WEIGHTS[1]
    fas_s_s = np.zeros_like(fas_s_i)
    fas_b_s = np.zeros_like(fas_b_i)
    mid_s = fas_s_i @ W
    mid_b = fas_b_i @ W
    fas_s_s[:, 1:-1] = mid_s
    fas_b_s[:, 1:-1] = mid_b
    fas_s_s[:, 0] = fas_s_s[:, 1]
    fas_s_s[:, -1] = fas_s_s[:, -2]
    fas_b_s[:, 0] = fas_b_s[:, 1]
    fas_b_s[:, -1] = fas_b_s[:, -2]
    mags = np.divide(
        fas_s_s,
        fas_b_s,
        out=np.zeros_like(fas_s_s),
        where=np.abs(fas_b_s) > _EPS,
    ).astype(np.float32)
    return mags, freq_out.astype(np.float32)


def compute_tf_from_accel(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """OpenSees accel → (n_recorders, n_freq) linear |TF|."""
    n_lateral = config.N_LATERAL
    with h5py.File(h5_path, "r") as f:
        data = np.asarray(f["recorders/accel/data"][:])
        dt = float(f["grid"].attrs["dt"])
        row_y_m = f["recorders/accel/data"].attrs.get("row_y_m")
    base_2d, surf_2d = _split_base_surf(data, row_y_m, n_lateral)
    try:
        return _ttf_batch_seiskit(base_2d, surf_2d, dt=dt)
    except ImportError:
        return _ttf_batch_local(base_2d, surf_2d, dt=dt)


def load_corpus_tf_cache(
    root: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]] | None:
    d = tf_cache_dir(root)
    if d is None:
        return None
    tf = np.load(d / "tf_per_sample.npy", mmap_mode="r")
    freq = np.load(d / "freq.npy")
    manifest: list[dict[str, str]] = []
    man_path = d / "manifest.csv"
    if man_path.is_file():
        with open(man_path, newline="") as f:
            manifest = list(csv.DictReader(f))
    return tf, freq, manifest


def per_file_tf_cache_path(cache_dir: Path, h5_path: Path) -> Path:
    slug = f"{h5_path.parent.parent.name}_{h5_path.stem}"
    return Path(cache_dir) / slug / "tf_true.npy"


def load_or_compute_tf(
    h5_path: Path,
    cache_dir: Path,
    *,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = Path(cache_dir)
    case_dir = cache_dir / f"{h5_path.parent.parent.name}_{h5_path.stem}"
    tf_path = case_dir / "tf_true.npy"
    freq_path = case_dir / "freq.npy"
    meta_path = case_dir / "gt_meta.json"
    if not force and tf_path.is_file() and freq_path.is_file():
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                if (
                    meta.get("h5_path") == str(h5_path.resolve())
                    and meta.get("h5_size") == h5_path.stat().st_size
                ):
                    return np.load(tf_path), np.load(freq_path)
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        else:
            return np.load(tf_path), np.load(freq_path)
    tf, freq = compute_tf_from_accel(h5_path)
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(tf_path, tf)
    np.save(freq_path, freq)
    meta_path.write_text(
        json.dumps(
            {
                "h5_path": str(h5_path.resolve()),
                "h5_size": h5_path.stat().st_size,
                "n_recorders": int(tf.shape[0]),
                "n_freq": int(tf.shape[1]),
            },
            indent=2,
        )
    )
    return tf, freq


def depth2_listing(root: Path, *, max_files: int = 8) -> list[str]:
    lines: list[str] = []
    root = Path(root)
    if not root.exists():
        return [f"{root}  [missing]"]

    def _walk(p: Path, depth: int, prefix: str) -> None:
        try:
            kids = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except OSError as exc:
            lines.append(f"{prefix}{p.name}/  ERR {exc}")
            return
        dirs = [k for k in kids if k.is_dir()]
        files = [k for k in kids if k.is_file()]
        lines.append(f"{prefix}{p.name}/  dirs={len(dirs)} files={len(files)}")
        for f in files[:max_files]:
            try:
                sz = f.stat().st_size
            except OSError:
                sz = -1
            lines.append(f"{prefix}  {f.name}  {sz}")
        if len(files) > max_files:
            lines.append(f"{prefix}  ... +{len(files) - max_files} files")
        if depth > 0:
            for d in dirs:
                _walk(d, depth - 1, prefix + "  ")

    _walk(root, 2, "")
    return lines


def probe_corpus(root: Path) -> dict[str, Any]:
    root = Path(root)
    h5s = discover_h5_files(root)
    tf_dir = tf_cache_dir(root)
    sample: dict[str, Any] | None = None
    if h5s:
        vs, zeta, params, extra = read_h5_sample(h5s[0])
        nom = nominal_layer_params(params)
        sample = {
            "path": str(h5s[0]),
            "Vs_realization_2D": extra["vs_shape"],
            "Damping_zeta": tuple(int(x) for x in zeta.shape),
            "params": {k: _jsonable(v) for k, v in params.items()},
            "param_keys": extra["param_keys"],
            "grid_attrs": {
                k: _jsonable(v) for k, v in extra.get("grid_attrs", {}).items()
            },
            "accel_n_channels": extra.get("accel_n_channels"),
            "accel_shape": extra.get("accel_shape"),
            "nominal": nom,
            "soil_nz": soil_nz_from_params(params, vs.shape[0]),
        }
    man_root = root / "manifest.csv"
    return {
        "root": str(root),
        "exists": root.exists(),
        "tree": depth2_listing(root),
        "n_h5": len(h5s),
        "tf_cache": str(tf_dir) if tf_dir else None,
        "root_manifest": str(man_root) if man_root.is_file() else None,
        "sample": sample,
    }


def _jsonable(v: Any) -> Any:
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def parse_cache_tag(cache_tag: str) -> tuple[int, int]:
    """Parse ``n2000_seed42`` → (2000, 42)."""
    tag = cache_tag.strip()
    n = None
    seed = config.SEED
    for part in tag.replace("-", "_").split("_"):
        if part.startswith("n") and part[1:].isdigit():
            n = int(part[1:])
        if part.startswith("seed") and part[4:].isdigit():
            seed = int(part[4:])
    if n is None:
        raise ValueError(f"Cannot parse n from cache tag {cache_tag!r}")
    return n, seed


def clamp_residual(
    r_hat: np.ndarray,
    mode: str = "none",
    *,
    gain: float = 1.0,
) -> np.ndarray:
    """Fail-soft residual. ``tanh`` uses Δ_eff = ln3 · tanh(g Δ / ln3)."""
    r = np.asarray(r_hat, dtype=np.float64)
    if mode in ("none", "off", ""):
        return r.astype(np.float32, copy=False)
    if mode in ("zero", "clamp0"):
        return np.zeros_like(r, dtype=np.float32)
    if mode == "tanh":
        ln3 = float(np.log(3.0))
        return (ln3 * np.tanh(gain * r / ln3)).astype(np.float32)
    raise ValueError(f"Unknown clamp mode {mode!r}")
