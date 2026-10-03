#!/usr/bin/env python3
"""Nature-style presentation figures for the shipped residual GINO.

2×3 case comparisons (3 pages × 2 cases per domain), a Pearson histogram, and
a Vs mosaic, one column per domain (default: iid, dipping, three_layer; use
--domain to restrict). Packs cache GINO + Pretell geomean so plots can rerun
without a GPU.

    uv run python experiments/DeepONet-Residual/response_variability/plots/plot_presentation.py
    uv run python .../plot_presentation.py --skip-predict
    uv run python .../plot_presentation.py --domain iid --domain dipping --skip-predict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from response_variability.metrics import central_recorder  # noqa: E402
from response_variability.names import (  # noqa: E402
    DMULT,
    GINO,
    HASKELL_NOMINAL,
    METHOD_COLORS,
    METHOD_LINESTYLES,
    METHOD_ZORDER,
    OPENSEES,
    PASSERI,
    PRETELL,
    PRETELL_P84,
    TORO,
)
from response_variability.style import (  # noqa: E402
    apply_nature_style,
    figsize,
    panel_letter,
    savefig,
)

OUT_DIR = config.RESULTS_DIR / "presentation"
CENTRAL_REC = config.N_LATERAL // 2
PEARSON_QUANTILES = (0.10, 0.30, 0.50, 0.70, 0.85, 0.95)
MOSAIC_QUANTILES = (0.50, 0.85)
N_PRETELL_DEFAULT = 200
_PAGE_LETTERS = "abcdef"

DOMAIN_SPECS: dict[str, dict[str, str]] = {
    "iid": {
        "mix_key": "iid",
        "title": "IID (2-layer)",
        "pack_name": "iid",
    },
    "dipping": {
        "mix_key": "ood_dipping",
        "title": "dipping OOD",
        "pack_name": "dipping",
    },
    "three_layer": {
        "mix_key": "ood_three_layer",
        "title": "three-layer OOD",
        "pack_name": "three_layer",
    },
}


def pack_path(out_dir: Path, domain: str) -> Path:
    return Path(out_dir) / f"{DOMAIN_SPECS[domain]['pack_name']}_pack.npz"


def caches_ready() -> bool:
    """True when mix test caches exist (H5 / ckpt checked separately)."""
    try:
        from mix_ladder import mix_test_parts
    except Exception:
        return False
    try:
        parts = mix_test_parts()
    except FileNotFoundError:
        return False
    need = ("tf1d_nom.npy", "meta.npz", "r_nom_signed.npy", "fields.npy")
    for _name, (cache, _idx) in parts.items():
        cache = Path(cache)
        if not all((cache / k).is_file() for k in need):
            return False
        if (
            not (cache / "tf2d.npy").is_file()
            and not config.TF_PER_SAMPLE_PATH.is_file()
        ):
            return False
    return True


def pearson_tf_freq_per_sample(tf_true: np.ndarray, tf_pred: np.ndarray) -> np.ndarray:
    """Mean Pearson-of-|TF| along frequency over recorders (same idea as pearson_TF_freq)."""
    y = np.asarray(tf_true, dtype=np.float64)
    p = np.asarray(tf_pred, dtype=np.float64)
    if p.ndim == 2:
        p = np.broadcast_to(p[:, None, :], y.shape)
    n, n_rec, _n_freq = y.shape
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        cors: list[float] = []
        for r in range(n_rec):
            a, b = y[i, r], p[i, r]
            if not np.isfinite(a).any() or not np.isfinite(b).any():
                continue
            if float(np.nanstd(a)) < 1e-12 or float(np.nanstd(b)) < 1e-12:
                continue
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.sum() < 2:
                continue
            cors.append(float(np.corrcoef(a[finite], b[finite])[0, 1]))
        if cors:
            out[i] = float(np.mean(cors))
    return out


def pick_pearson_quantile_indices(
    pearson: np.ndarray,
    quantiles: tuple[float, ...] = PEARSON_QUANTILES,
) -> np.ndarray:
    """Unique samples nearest the Pearson quantiles (easy→hard spread, not cherry-picked)."""
    p = np.asarray(pearson, dtype=np.float64)
    finite = np.isfinite(p)
    if int(finite.sum()) == 0:
        raise ValueError("no finite Pearson scores")
    chosen: list[int] = []
    used: set[int] = set()
    for q in quantiles:
        target = float(np.quantile(p[finite], q))
        for i in np.argsort(np.abs(p - target)):
            i = int(i)
            if i not in used and finite[i]:
                chosen.append(i)
                used.add(i)
                break
    return np.asarray(chosen, dtype=int)


def nominal_vs_profile(
    *,
    vs1: float,
    H: float,
    vs2: float,
    nz: int,
    dz: float = 1.0,
    h1: float | None = None,
    h2: float | None = None,
    vs_mid: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """1D stack vs depth: two-layer Vs1–H–Vs2, or three-layer Vs1/H1/Vs_mid/H2/bedrock."""
    z = (np.arange(int(nz), dtype=np.float64) + 0.5) * float(dz)
    vs = np.full(int(nz), float(vs2), dtype=np.float64)
    three = (
        h1 is not None
        and h2 is not None
        and vs_mid is not None
        and np.isfinite(h1)
        and np.isfinite(h2)
        and np.isfinite(vs_mid)
        and float(h1) > 0.0
        and float(h2) > 0.0
    )
    if three:
        vs[z <= float(h1)] = float(vs1)
        vs[(z > float(h1)) & (z <= float(h1) + float(h2))] = float(vs_mid)
    else:
        vs[z <= float(H)] = float(vs1)
    return z, vs


def nominal_vs_stairs(
    *,
    vs1: float,
    H: float,
    vs2: float,
    z_max: float,
    h1: float | None = None,
    h2: float | None = None,
    vs_mid: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Polyline (Vs, depth) for the TF_1D stack, including a bedrock interval."""
    z_max = max(float(z_max), float(H) + 1.0)
    three = (
        h1 is not None
        and h2 is not None
        and vs_mid is not None
        and np.isfinite(h1)
        and np.isfinite(h2)
        and np.isfinite(vs_mid)
        and float(h1) > 0.0
        and float(h2) > 0.0
    )
    if three:
        bounds = [0.0, float(h1), float(h1) + float(h2), z_max]
        vals = [float(vs1), float(vs_mid), float(vs2)]
    else:
        bounds = [0.0, float(H), z_max]
        vals = [float(vs1), float(vs2)]
    z: list[float] = []
    vs: list[float] = []
    for lo, hi, val in zip(bounds[:-1], bounds[1:], vals):
        z.extend([lo, hi])
        vs.extend([val, val])
    return np.asarray(vs, dtype=np.float64), np.asarray(z, dtype=np.float64)


def _meta_col(
    meta: dict[str, Any], key: str, idx: np.ndarray, default: float | None = None
):
    if key not in meta:
        if default is None:
            raise KeyError(key)
        return np.full(len(idx), default)
    return np.asarray(meta[key])[idx]


def load_domain_arrays(cache_dir: Path, test_idx: np.ndarray) -> dict[str, np.ndarray]:
    cache_dir = Path(cache_dir)
    meta = dict(np.load(cache_dir / "meta.npz", allow_pickle=True))
    tf2d_path = cache_dir / "tf2d.npy"
    if tf2d_path.is_file():
        tf_ops = np.asarray(
            np.load(tf2d_path, mmap_mode="r")[test_idx], dtype=np.float64
        )
    else:
        tf_all = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
        sidx = np.load(cache_dir / "sample_indices.npy")[test_idx]
        tf_ops = np.asarray(tf_all[sidx], dtype=np.float64)
    freq_path = cache_dir / "freq.npy"
    freq = np.load(freq_path if freq_path.is_file() else config.TF_FREQ_PATH)
    rec_path = cache_dir / "recorder_x.npy"
    if rec_path.is_file():
        recorder_x = np.load(rec_path)
    elif config.RECORDER_X_IDX_PATH.is_file():
        recorder_x = np.load(config.RECORDER_X_IDX_PATH)
    else:
        recorder_x = np.arange(tf_ops.shape[1])
    return {
        "tf_opensees": tf_ops,
        "tf_haskell_nominal": np.asarray(
            np.load(cache_dir / "tf1d_nom.npy", mmap_mode="r")[test_idx],
            dtype=np.float64,
        ),
        "freq": np.asarray(freq, dtype=float),
        "vs1": np.asarray(_meta_col(meta, "Vs1", test_idx), dtype=float),
        "H": np.asarray(_meta_col(meta, "H", test_idx), dtype=float),
        "vs2": np.asarray(_meta_col(meta, "Vs2", test_idx), dtype=float),
        "cov": np.asarray(_meta_col(meta, "CoV", test_idx, 0.0), dtype=float),
        "rH": np.asarray(_meta_col(meta, "rH", test_idx, float("nan")), dtype=float),
        "aHV": np.asarray(_meta_col(meta, "aHV", test_idx, float("nan")), dtype=float),
        "soil_nz": np.asarray(
            _meta_col(meta, "soil_nz", test_idx, config.NZ_MAX), dtype=int
        ),
        "sample_idx": np.asarray(_meta_col(meta, "sample_idx", test_idx), dtype=int),
        "rf_seed": np.asarray(_meta_col(meta, "rf_seed", test_idx, -1), dtype=int),
        "local_idx": np.asarray(test_idx, dtype=int),
        "h5_path": np.asarray(_meta_col(meta, "h5_path", test_idx)),
        "recorder_x": np.asarray(recorder_x, dtype=int),
    }


def resolve_sample_h5(stored: str, domain: str) -> Path | None:
    p = Path(str(stored))
    if p.is_file():
        return p
    from residual_signed import resolve_h5_path

    cand = resolve_h5_path(str(stored))
    if cand.is_file():
        return cand
    name = p.name
    iid = config.H5_DIR / name
    if iid.is_file():
        return iid
    from ood_io import default_ood_roots

    mix = DOMAIN_SPECS[domain]["mix_key"]
    roots = default_ood_roots()
    root = roots.get(mix)
    if root is not None:
        for hit in (root / "h5" / name, root / name):
            if hit.is_file():
                return hit
    return None


def _pad_col(col: np.ndarray, nz_max: int) -> np.ndarray:
    out = np.full(nz_max, np.nan, dtype=np.float32)
    n = min(int(col.shape[0]), nz_max)
    out[:n] = np.asarray(col[:n], dtype=np.float32)
    return out


def _pad_field(field: np.ndarray, nz_max: int, nx: int) -> np.ndarray:
    out = np.full((nz_max, nx), np.nan, dtype=np.float32)
    nz = min(int(field.shape[0]), nz_max)
    nx_use = min(int(field.shape[1]), nx)
    out[:nz, :nx_use] = np.asarray(field[:nz, :nx_use], dtype=np.float32)
    return out


def attach_vs_and_pretell(
    pack: dict[str, np.ndarray],
    *,
    domain: str,
    n_pretell: int = N_PRETELL_DEFAULT,
) -> dict[str, np.ndarray]:
    """Load cropped Vs/ζ from H5; optional Pretell geomean × exp(±σ_ln)."""
    from ood_io import (
        crop_variability,
        nominal_layer_params,
        read_h5_sample,
        soil_nz_from_params,
    )
    from response_variability.seiskit_arms import pretell_haskell_tf
    from tqdm import tqdm

    n = int(pack["tf_opensees"].shape[0])
    n_freq = int(pack["freq"].shape[0])
    nz_max = int(config.NZ_MAX)
    nx = int(config.LX_VARIABILITY)
    rec_x = np.asarray(pack["recorder_x"], dtype=int)
    rec_i = int(min(CENTRAL_REC, rec_x.size - 1))
    vs_column = np.full((n, nz_max), np.nan, dtype=np.float32)
    vs_2d = np.full((n, nz_max, nx), np.nan, dtype=np.float32)
    h1 = np.full(n, np.nan, dtype=np.float64)
    h2 = np.full(n, np.nan, dtype=np.float64)
    vs_mid = np.full(n, np.nan, dtype=np.float64)
    tf_pr = np.full((n, n_freq), np.nan, dtype=np.float64)
    sig_pr = np.full((n, n_freq), np.nan, dtype=np.float64)
    do_pretell = int(n_pretell) > 0

    for i in tqdm(range(n), desc=f"Vs/Pretell {domain}", leave=False):
        h5 = resolve_sample_h5(str(pack["h5_path"][i]), domain)
        if h5 is None:
            continue
        vs, zeta, params, _extra = read_h5_sample(h5)
        vs_c = crop_variability(vs)
        zeta_c = crop_variability(zeta)
        nom = nominal_layer_params(params)
        soil_nz = soil_nz_from_params(params, vs_c.shape[0])
        pack["soil_nz"][i] = int(soil_nz)
        pack["vs1"][i] = float(nom["vs1"])
        pack["H"][i] = float(nom["H"])
        pack["vs2"][i] = float(nom["vs2"])
        if nom.get("H1") is not None:
            h1[i] = float(nom["H1"])
        if nom.get("H2") is not None:
            h2[i] = float(nom["H2"])
        if nom.get("vs_mid") is not None and nom["vs_mid"] is not None:
            vs_mid[i] = float(nom["vs_mid"])
        col = int(np.clip(rec_x[rec_i], 0, vs_c.shape[1] - 1))
        vs_column[i] = _pad_col(vs_c[:, col], nz_max)
        vs_2d[i] = _pad_field(vs_c, nz_max, nx)
        if not do_pretell:
            continue
        geo, sig = pretell_haskell_tf(
            freq=pack["freq"],
            vs_strip=vs_c,
            zeta_strip=zeta_c,
            vs2=float(nom["vs2"]),
            soil_nz=int(soil_nz),
            dz=config.DZ,
            n_samples=int(n_pretell),
        )
        tf_pr[i] = np.asarray(geo, dtype=np.float64)
        sig_pr[i] = np.asarray(sig, dtype=np.float64)

    pack["vs_column"] = vs_column
    pack["vs_2d"] = vs_2d
    pack["H1"] = h1
    pack["H2"] = h2
    pack["vs_mid"] = vs_mid
    if do_pretell and np.isfinite(tf_pr).any():
        pack["tf_pretell"] = tf_pr
        pack["sigma_ln_pretell"] = sig_pr
    return pack


def score_gino(
    cache_dir: Path,
    test_idx: np.ndarray,
    ckpt_path: Path,
    batch_size: int,
) -> np.ndarray:
    from response_variability.evals.eval_iid import predict_gino

    return predict_gino(
        cache_dir=cache_dir,
        test_idx=np.asarray(test_idx, dtype=int),
        ckpt_path=ckpt_path,
        batch_size=batch_size,
    )


def finalize_pearson(pack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from response_variability.seiskit_arms import attach_pretell_p84

    pack = attach_pretell_p84(pack)
    ops = pack["tf_opensees"]
    pack["pearson_gino"] = pearson_tf_freq_per_sample(ops, pack["tf_gino"])
    pack["pearson_1d"] = pearson_tf_freq_per_sample(ops, pack["tf_haskell_nominal"])
    if "tf_pretell" in pack:
        pack["pearson_pretell"] = pearson_tf_freq_per_sample(ops, pack["tf_pretell"])
    if "tf_pretell_p84" in pack:
        pack["pearson_pretell_p84"] = pearson_tf_freq_per_sample(
            ops, pack["tf_pretell_p84"]
        )
    return pack


def save_pack(pack: dict[str, np.ndarray], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pack)
    return path


def load_pack(path: Path) -> dict[str, np.ndarray]:
    blob = np.load(path, allow_pickle=True)
    return {k: blob[k] for k in blob.files}


def _merge_sota_arms(pack: dict[str, np.ndarray], domain: str) -> dict[str, np.ndarray]:
    """Overlay Toro / Passeri / Dmult when those NPZs exist (IID SOTA pages)."""
    from response_variability.evals.eval_classical import merge_classical_into_pack

    return merge_classical_into_pack(pack, domain)


def build_domain_pack(
    domain: str,
    *,
    ckpt_path: Path,
    out_dir: Path,
    batch_size: int,
    n_pretell: int,
    skip_predict: bool,
) -> dict[str, np.ndarray]:
    path = pack_path(out_dir, domain)
    if skip_predict:
        if not path.is_file():
            raise FileNotFoundError(f"--skip-predict needs {path}")
        pack = load_pack(path)
        pack = _merge_sota_arms(pack, domain)
        return finalize_pearson(pack)

    from mix_ladder import mix_test_parts

    mix_key = DOMAIN_SPECS[domain]["mix_key"]
    cache_dir, test_idx = mix_test_parts()[mix_key]
    pack = load_domain_arrays(cache_dir, test_idx)
    pack["tf_gino"] = score_gino(cache_dir, test_idx, ckpt_path, batch_size)
    pack = attach_vs_and_pretell(pack, domain=domain, n_pretell=n_pretell)
    pack["domain"] = np.array(domain)
    pack = finalize_pearson(pack)
    save_pack(pack, path)
    print(f"Wrote {path}", flush=True)
    return pack


def make_synthetic_pack(
    *,
    domain: str = "iid",
    n: int = 12,
    n_rec: int = 21,
    n_freq: int = 32,
    nz: int = 40,
    nx: int = 48,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Deterministic pack for layout tests (no H5, no GPU)."""
    rng = np.random.default_rng(seed)
    freq = np.logspace(-1, 1, n_freq)
    t = np.linspace(0.0, 3.0, n_freq)
    ops = np.empty((n, n_rec, n_freq), dtype=np.float64)
    gino = np.empty_like(ops)
    nom = np.empty_like(ops)
    pretell = np.empty((n, n_freq), dtype=np.float64)
    sig = np.empty((n, n_freq), dtype=np.float64)
    for i in range(n):
        shape = 1.4 + 0.35 * np.sin(t + 0.15 * i)
        rec_scale = 1.0 + 0.04 * rng.standard_normal(n_rec)
        noise = 0.04 * rng.standard_normal((n_rec, n_freq))
        ops[i] = rec_scale[:, None] * shape + noise
        # Higher Pearson index → closer GINO (easy cases at high quantiles).
        err = (0.45 - 0.35 * i / max(n - 1, 1)) * rng.standard_normal((n_rec, n_freq))
        gino[i] = ops[i] + err
        nom[i] = 0.65 * ops[i] + 0.55
        pretell[i] = ops[i].mean(axis=0) * 0.9 + 0.12
        sig[i] = 0.08 + 0.02 * rng.random(n_freq)
    vs_column = np.full((n, config.NZ_MAX), np.nan, dtype=np.float32)
    vs_2d = np.full((n, config.NZ_MAX, nx), np.nan, dtype=np.float32)
    z = np.arange(nz, dtype=np.float32)
    three = domain == "three_layer"
    vs1 = np.linspace(150.0, 240.0, n)
    H = np.linspace(22.0, 50.0, n) if not three else np.linspace(40.0, 70.0, n)
    vs2 = np.linspace(700.0, 1100.0, n)
    h1 = np.linspace(12.0, 28.0, n) if three else np.full(n, np.nan)
    h2 = np.linspace(18.0, 42.0, n) if three else np.full(n, np.nan)
    vs_mid = np.linspace(280.0, 480.0, n) if three else np.full(n, np.nan)
    if three:
        H = h1 + h2
    for i in range(n):
        col = 150.0 + 40.0 * np.exp(-z / 18.0) + 8.0 * rng.standard_normal(nz)
        vs_column[i, :nz] = col
        n_bed = min(nz + 12, config.NZ_MAX)
        vs_column[i, int(H[i]) : n_bed] = float(vs2[i])
        field = col[:, None] + 12.0 * rng.standard_normal((nz, nx))
        if domain == "dipping":
            field += 0.4 * np.arange(nx)[None, :]
        vs_2d[i, :nz] = field
        vs_2d[i, int(H[i]) : n_bed] = float(vs2[i])
    pack = {
        "tf_opensees": ops,
        "tf_gino": gino,
        "tf_haskell_nominal": nom,
        "tf_pretell": pretell,
        "sigma_ln_pretell": sig,
        "freq": freq,
        "vs1": vs1,
        "H": H,
        "vs2": vs2,
        "H1": h1,
        "H2": h2,
        "vs_mid": vs_mid,
        "cov": np.linspace(0.12, 0.32, n),
        "rH": np.linspace(20.0, 80.0, n),
        "aHV": np.linspace(12.0, 40.0, n),
        "soil_nz": np.full(n, nz, dtype=int),
        "sample_idx": np.arange(n, dtype=int),
        "local_idx": np.arange(n, dtype=int),
        "rf_seed": np.arange(n, dtype=int),
        "vs_column": vs_column,
        "vs_2d": vs_2d,
        "recorder_x": np.linspace(0, nx - 1, n_rec, dtype=int),
        "h5_path": np.array([f"synthetic_{i}.h5" for i in range(n)]),
        "domain": np.array(domain),
    }
    if domain == "dipping":
        pack["dip_angle_deg"] = np.linspace(8.0, 36.0, n)
        pack["dip_span"] = np.linspace(50.0, 140.0, n)
        pack["bedrock_H"] = np.linspace(6.0, 18.0, n)
        pack["dip_direction"] = np.linspace(-1.0, 1.0, n)
    return finalize_pearson(pack)


def _stored_nz(field: np.ndarray) -> int:
    """Last depth with finite Vs, including bedrock below ``soil_nz``."""
    a = np.asarray(field)
    if a.ndim == 1:
        finite = np.isfinite(a)
    else:
        finite = np.isfinite(a).any(axis=tuple(range(1, a.ndim)))
    if not bool(finite.any()):
        return 1
    return int(np.where(finite)[0][-1]) + 1


def _plot_vs_panel(ax, pack: dict[str, np.ndarray], i: int) -> None:
    """1D nominal column used for TF_1D (not the 2-D Vs realization)."""
    H = float(pack["H"][i])
    if "vs_column" in pack:
        nz = _stored_nz(pack["vs_column"][i])
    else:
        nz = 0
    nz = max(nz, int(np.ceil(H / float(config.DZ))) + 12, 16)
    z_max = nz * float(config.DZ)
    h1 = float(pack["H1"][i]) if "H1" in pack else float("nan")
    h2 = float(pack["H2"][i]) if "H2" in pack else float("nan")
    vs_mid = float(pack["vs_mid"][i]) if "vs_mid" in pack else float("nan")
    vs_nom, z_nom = nominal_vs_stairs(
        vs1=float(pack["vs1"][i]),
        H=H,
        vs2=float(pack["vs2"][i]),
        z_max=z_max,
        h1=h1,
        h2=h2,
        vs_mid=vs_mid,
    )
    ax.plot(
        vs_nom,
        z_nom,
        color=METHOD_COLORS[HASKELL_NOMINAL],
        ls="-",
        lw=1.35,
        zorder=4,
        label=HASKELL_NOMINAL,
    )
    ax.invert_yaxis()
    ax.set_xlabel(r"$V_s$ (m s$^{-1}$)")
    ax.set_ylabel("depth (m)")


def _plot_tf_panel(ax, pack: dict[str, np.ndarray], i: int) -> None:
    freq = pack["freq"]
    ops = central_recorder(pack["tf_opensees"][i])
    nom = central_recorder(pack["tf_haskell_nominal"][i])
    gino = central_recorder(pack["tf_gino"][i])
    ax.plot(
        freq,
        ops,
        color=METHOD_COLORS[OPENSEES],
        ls=METHOD_LINESTYLES[OPENSEES],
        lw=1.45,
        zorder=METHOD_ZORDER[OPENSEES],
        label=OPENSEES,
    )
    ax.plot(
        freq,
        nom,
        color=METHOD_COLORS[HASKELL_NOMINAL],
        ls=METHOD_LINESTYLES[HASKELL_NOMINAL],
        lw=1.05,
        zorder=METHOD_ZORDER[HASKELL_NOMINAL],
        label=HASKELL_NOMINAL,
    )
    ax.plot(
        freq,
        gino,
        color=METHOD_COLORS[GINO],
        ls=METHOD_LINESTYLES[GINO],
        lw=1.45,
        zorder=METHOD_ZORDER[GINO],
        label=rf"{GINO} $\widehat{{\mathrm{{TF}}}}=\mathrm{{TF}}_{{1D}}+\hat R$",
    )
    if "tf_pretell" in pack and np.isfinite(pack["tf_pretell"][i]).any():
        geo = np.asarray(pack["tf_pretell"][i], dtype=np.float64)
        sig = np.asarray(pack["sigma_ln_pretell"][i], dtype=np.float64)
        lo = np.maximum(geo * np.exp(-sig), 1e-6)
        p84 = np.maximum(geo * np.exp(sig), 1e-6)
        geo_c = np.maximum(geo, 1e-6)
        ax.fill_between(
            freq,
            lo,
            geo_c,
            color=METHOD_COLORS[PRETELL],
            alpha=0.20,
            linewidth=0,
            zorder=1,
        )
        ax.fill_between(
            freq,
            geo_c,
            p84,
            color=METHOD_COLORS[PRETELL_P84],
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            freq,
            geo_c,
            color=METHOD_COLORS[PRETELL],
            ls=METHOD_LINESTYLES[PRETELL],
            lw=1.2,
            zorder=METHOD_ZORDER[PRETELL],
            label=PRETELL,
        )
        ax.plot(
            freq,
            p84,
            color=METHOD_COLORS[PRETELL_P84],
            ls=METHOD_LINESTYLES[PRETELL_P84],
            lw=1.55,
            zorder=METHOD_ZORDER[PRETELL_P84],
            label=PRETELL_P84,
        )
    if "tf_dmult" in pack and np.isfinite(pack["tf_dmult"][i]).any():
        from response_variability.plots.tf_atlas import ATLAS_CURVE_STYLE

        dstyle = ATLAS_CURVE_STYLE[DMULT]
        ax.plot(
            freq,
            np.maximum(np.asarray(pack["tf_dmult"][i], dtype=np.float64), 1e-6),
            color=dstyle["color"],
            ls=dstyle["ls"],
            lw=dstyle["lw"],
            zorder=5,
            label=DMULT,
        )
    if "tf_toro" in pack and np.isfinite(pack["tf_toro"][i]).any():
        ax.plot(
            freq,
            np.maximum(np.asarray(pack["tf_toro"][i], dtype=np.float64), 1e-6),
            color=METHOD_COLORS[TORO],
            ls=METHOD_LINESTYLES[TORO],
            lw=0.9,
            zorder=METHOD_ZORDER[TORO],
            alpha=0.85,
            label="Toro geomean",
        )
    if "tf_passeri" in pack and np.isfinite(pack["tf_passeri"][i]).any():
        ax.plot(
            freq,
            np.maximum(np.asarray(pack["tf_passeri"][i], dtype=np.float64), 1e-6),
            color=METHOD_COLORS[PASSERI],
            ls=METHOD_LINESTYLES[PASSERI],
            lw=0.9,
            zorder=METHOD_ZORDER[PASSERI],
            alpha=0.85,
            label="Passeri geomean",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(freq.min()), float(freq.max()))
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$|\mathrm{TF}|$")


def leftover_central(
    pack: dict[str, np.ndarray], i: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_true, R_hat) on the central recorder: TF_2D − TF_1D vs GINO − TF_1D."""
    ops = central_recorder(pack["tf_opensees"][i])
    nom = central_recorder(pack["tf_haskell_nominal"][i])
    gino = central_recorder(pack["tf_gino"][i])
    return ops - nom, gino - nom


def _plot_diff_panel(ax, pack: dict[str, np.ndarray], i: int) -> None:
    freq = pack["freq"]
    r_true, r_hat = leftover_central(pack, i)
    ax.axhline(0.0, color="0.55", lw=0.5, zorder=0)
    ax.plot(
        freq,
        r_true,
        color=METHOD_COLORS[OPENSEES],
        ls=METHOD_LINESTYLES[OPENSEES],
        lw=1.2,
        zorder=METHOD_ZORDER[OPENSEES],
        label=r"$R=\mathrm{TF}_{2D}-\mathrm{TF}_{1D}$",
    )
    ax.plot(
        freq,
        r_hat,
        color=METHOD_COLORS[GINO],
        ls=METHOD_LINESTYLES[GINO],
        lw=1.2,
        zorder=METHOD_ZORDER[GINO],
        label=r"$\hat R$",
    )
    ax.set_xscale("log")
    ax.set_xlim(float(freq.min()), float(freq.max()))
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$R$")


def attach_stoch_from_cache(
    pack: dict[str, np.ndarray], domain: str
) -> dict[str, np.ndarray]:
    """Fill rH / aHV from the signed-cache meta when a saved pack omitted them."""
    if "rH" in pack and "aHV" in pack:
        return pack
    if "local_idx" not in pack:
        return pack
    try:
        from mix_ladder import mix_test_parts
    except Exception:
        return pack
    try:
        cache_dir, _idx = mix_test_parts()[DOMAIN_SPECS[domain]["mix_key"]]
        meta_path = Path(cache_dir) / "meta.npz"
        if not meta_path.is_file():
            return pack
        meta = dict(np.load(meta_path, allow_pickle=True))
    except FileNotFoundError:
        return pack
    loc = np.asarray(pack["local_idx"], dtype=int)
    out = dict(pack)
    if "rH" not in out and "rH" in meta:
        out["rH"] = np.asarray(meta["rH"][loc], dtype=float)
    if "aHV" not in out and "aHV" in meta:
        out["aHV"] = np.asarray(meta["aHV"][loc], dtype=float)
    return out


def _case_title(pack: dict[str, np.ndarray], i: int, q: float, domain: str) -> str:
    title = DOMAIN_SPECS[domain]["title"]
    pg = float(pack["pearson_gino"][i])
    line1 = rf"{title}, $q$={100 * q:.0f}%  Pearson={pg:.2f}"
    vs1 = float(pack["vs1"][i])
    H = float(pack["H"][i])
    vs2 = float(pack["vs2"][i])
    h1 = float(pack["H1"][i]) if "H1" in pack else float("nan")
    h2 = float(pack["H2"][i]) if "H2" in pack else float("nan")
    vs_mid = float(pack["vs_mid"][i]) if "vs_mid" in pack else float("nan")
    if (
        np.isfinite(h1)
        and np.isfinite(h2)
        and np.isfinite(vs_mid)
        and h1 > 0.0
        and h2 > 0.0
    ):
        line2 = (
            rf"$V_{{s1}}$={vs1:.0f}, $H_1$={h1:.0f} m, "
            rf"$V_{{s,\mathrm{{mid}}}}$={vs_mid:.0f}, $H_2$={h2:.0f} m, "
            rf"$V_{{s2}}$={vs2:.0f} m s$^{{-1}}$"
        )
    else:
        line2 = (
            rf"$V_{{s1}}$={vs1:.0f} m s$^{{-1}}$, $H$={H:.0f} m, "
            rf"$V_{{s2}}$={vs2:.0f} m s$^{{-1}}$"
        )
    bits: list[str] = []
    if "rH" in pack and np.isfinite(pack["rH"][i]):
        bits.append(rf"$r_H$={pack['rH'][i]:.0f}")
    if "aHV" in pack and np.isfinite(pack["aHV"][i]):
        bits.append(rf"$a_{{HV}}$={pack['aHV'][i]:.0f}")
    if "cov" in pack and np.isfinite(pack["cov"][i]):
        bits.append(rf"CoV={pack['cov'][i]:.2f}")
    lines = [line1, line2]
    if bits:
        lines.append(", ".join(bits))
    return "\n".join(lines)


def _compare_legend(pack: dict[str, np.ndarray] | None = None) -> list:
    handles: list = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[OPENSEES],
            ls=METHOD_LINESTYLES[OPENSEES],
            lw=1.5,
            label=OPENSEES,
        ),
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[HASKELL_NOMINAL],
            ls=METHOD_LINESTYLES[HASKELL_NOMINAL],
            lw=1.1,
            label=HASKELL_NOMINAL,
        ),
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[GINO],
            ls=METHOD_LINESTYLES[GINO],
            lw=1.5,
            label=GINO,
        ),
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[PRETELL],
            ls=METHOD_LINESTYLES[PRETELL],
            lw=1.1,
            label=PRETELL,
        ),
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[PRETELL_P84],
            ls=METHOD_LINESTYLES[PRETELL_P84],
            lw=1.55,
            label=PRETELL_P84,
        ),
        Patch(
            facecolor=METHOD_COLORS[PRETELL_P84],
            edgecolor="none",
            alpha=0.25,
            label=r"Pretell percentile band ($\mathrm{median}\times e^{\pm\sigma_{\ln}}$)",
        ),
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[OPENSEES],
            ls=METHOD_LINESTYLES[OPENSEES],
            lw=1.1,
            label=r"$R=\mathrm{TF}_{2D}-\mathrm{TF}_{1D}$",
        ),
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[GINO],
            ls=METHOD_LINESTYLES[GINO],
            lw=1.1,
            label=r"$\hat R$",
        ),
    ]
    if pack is not None:
        extras = (
            (DMULT, "tf_dmult"),
            (TORO, "tf_toro"),
            (PASSERI, "tf_passeri"),
        )
        from response_variability.plots.tf_atlas import ATLAS_CURVE_STYLE

        for method, key in extras:
            if key in pack and np.isfinite(np.asarray(pack[key])).any():
                style = ATLAS_CURVE_STYLE.get(method)
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=style["color"] if style else METHOD_COLORS[method],
                        ls=style["ls"] if style else METHOD_LINESTYLES[method],
                        lw=style["lw"] if style else 0.9,
                        alpha=0.85,
                        label={
                            "Toro Vs": "Toro geomean",
                            "Passeri tts": "Passeri geomean",
                        }.get(method, method),
                    )
                )
    return handles


def plot_compare_page(
    pack: dict[str, np.ndarray],
    idx_pair: tuple[int, int],
    quantiles: tuple[float, float],
    out_path: Path,
    *,
    domain: str,
) -> Path:
    apply_nature_style()
    pack = attach_stoch_from_cache(pack, domain)
    fig, axes = plt_subplots_2x3()
    for row, (i, q, letters) in enumerate(
        zip(idx_pair, quantiles, (_PAGE_LETTERS[0:3], _PAGE_LETTERS[3:6]))
    ):
        _plot_vs_panel(axes[row, 0], pack, int(i))
        _plot_tf_panel(axes[row, 1], pack, int(i))
        _plot_diff_panel(axes[row, 2], pack, int(i))
        axes[row, 1].set_title(
            _case_title(pack, int(i), q, domain),
            fontsize=6.0,
            pad=5,
            linespacing=1.2,
        )
        for col, letter in enumerate(letters):
            panel_letter(axes[row, col], letter, x=0.02, y=0.98)
    fig.legend(
        handles=_compare_legend(pack),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
        fontsize=6.5,
    )
    fig.tight_layout(h_pad=1.1, w_pad=0.55, rect=(0, 0.12, 1, 1.0))
    return savefig(fig, out_path)


def plt_subplots_2x3():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=figsize("double", height_mm=175))
    return fig, axes


def plot_domain_compares(
    pack: dict[str, np.ndarray],
    out_dir: Path,
    *,
    domain: str,
) -> list[Path]:
    idx = pick_pearson_quantile_indices(pack["pearson_gino"])
    if len(idx) < 6:
        raise ValueError(f"{domain}: need 6 quantile cases, got {len(idx)}")
    pairs = ((idx[0], idx[1]), (idx[2], idx[3]), (idx[4], idx[5]))
    q_pairs = (
        (PEARSON_QUANTILES[0], PEARSON_QUANTILES[1]),
        (PEARSON_QUANTILES[2], PEARSON_QUANTILES[3]),
        (PEARSON_QUANTILES[4], PEARSON_QUANTILES[5]),
    )
    paths: list[Path] = []
    for page, (pair, qs) in enumerate(zip(pairs, q_pairs), start=1):
        name = f"compare_{DOMAIN_SPECS[domain]['pack_name']}_page{page}.png"
        paths.append(
            plot_compare_page(pack, pair, qs, Path(out_dir) / name, domain=domain)
        )
    return paths


def plot_pearson_histograms(
    packs: dict[str, dict[str, np.ndarray]],
    out_path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    domains = list(packs)
    fig, axes = plt.subplots(
        1, len(domains), figsize=figsize("double", height_mm=72), sharey=True
    )
    bins = np.linspace(0.0, 1.0, 21)
    for ax, domain in zip(axes, domains):
        pack = packs[domain]
        ax.hist(
            pack["pearson_gino"],
            bins=bins,
            color=METHOD_COLORS[GINO],
            alpha=0.55,
            density=True,
            label=GINO,
            histtype="stepfilled",
        )
        ax.hist(
            pack["pearson_1d"],
            bins=bins,
            color=METHOD_COLORS[HASKELL_NOMINAL],
            alpha=0.85,
            density=True,
            label=HASKELL_NOMINAL,
            histtype="step",
            lw=1.2,
        )
        if "pearson_pretell" in pack:
            ax.hist(
                pack["pearson_pretell"],
                bins=bins,
                color=METHOD_COLORS[PRETELL],
                alpha=0.85,
                density=True,
                label=PRETELL,
                histtype="step",
                lw=1.1,
                ls=METHOD_LINESTYLES[PRETELL],
            )
        if "pearson_pretell_p84" in pack:
            ax.hist(
                pack["pearson_pretell_p84"],
                bins=bins,
                color=METHOD_COLORS[PRETELL_P84],
                alpha=0.85,
                density=True,
                label=PRETELL_P84,
                histtype="step",
                lw=1.1,
                ls=METHOD_LINESTYLES[PRETELL_P84],
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"Pearson of $|\mathrm{TF}|$ along $f$")
        ax.set_title(DOMAIN_SPECS[domain]["title"])
        panel_letter(ax, _PAGE_LETTERS[domains.index(domain)], x=0.02, y=0.98)
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout(w_pad=0.6)
    return savefig(fig, out_path)


def plot_vs_mosaic(
    packs: dict[str, dict[str, np.ndarray]],
    out_path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    domains = list(packs)
    fig, axes = plt.subplots(2, len(domains), figsize=figsize("double", height_mm=110))
    row_labels = ("median Pearson", r"$\approx$85th-pct Pearson")
    vmin, vmax = np.inf, -np.inf
    tiles: list[tuple[Any, np.ndarray, int]] = []
    for col, domain in enumerate(domains):
        pack = packs[domain]
        idx = pick_pearson_quantile_indices(pack["pearson_gino"])
        q_idx = pick_pearson_quantile_indices(pack["pearson_gino"], MOSAIC_QUANTILES)
        # Prefer the same cases as the 2×3 pages (q=50 and q=85 → slots 2 and 4).
        picks = (
            (int(idx[2]), int(idx[4]))
            if len(idx) >= 5
            else (int(q_idx[0]), int(q_idx[-1]))
        )
        for row, i in enumerate(picks):
            nz = _stored_nz(pack["vs_2d"][i])
            img = np.asarray(pack["vs_2d"][i, :nz], dtype=np.float64)
            finite = img[np.isfinite(img)]
            if finite.size:
                vmin = min(vmin, float(finite.min()))
                vmax = max(vmax, float(finite.max()))
            tiles.append((axes[row, col], img, i))
            if row == 0:
                axes[row, col].set_title(DOMAIN_SPECS[domain]["title"])
            if col == 0:
                axes[row, col].set_ylabel(row_labels[row] + "\ndepth (m)")
            axes[row, col].set_xlabel(r"$x$ (m)")
    if not np.isfinite(vmin):
        vmin, vmax = 100.0, 800.0
    im = None
    for ax, img, _i in tiles:
        nz, nx = img.shape
        im = ax.imshow(
            img,
            origin="upper",
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=(0.0, nx * float(config.DX), nz * float(config.DZ), 0.0),
        )
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(r"$V_s$ (m s$^{-1}$)")
    fig.subplots_adjust(left=0.08, right=0.86, hspace=0.38, wspace=0.28)
    return savefig(fig, out_path)


def plot_all_from_packs(
    packs: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for domain in packs:
        paths.extend(plot_domain_compares(packs[domain], out_dir, domain=domain))
    paths.append(plot_pearson_histograms(packs, out_dir / "pearson_histograms.png"))
    paths.append(plot_vs_mosaic(packs, out_dir / "vs_mosaic.png"))
    return paths


def run(
    *,
    ckpt: Path,
    out_dir: Path,
    batch_size: int,
    n_pretell: int,
    skip_predict: bool,
    domains: tuple[str, ...] | None = None,
) -> list[Path]:
    domains = tuple(domains or DOMAIN_SPECS)
    packs: dict[str, dict[str, np.ndarray]] = {}
    for domain in domains:
        packs[domain] = build_domain_pack(
            domain,
            ckpt_path=ckpt,
            out_dir=out_dir,
            batch_size=batch_size,
            n_pretell=n_pretell,
            skip_predict=skip_predict,
        )
    if set(packs) != set(domains):
        # Partial rebuild: still try to plot whatever is on disk for the rest.
        for domain in domains:
            if domain in packs:
                continue
            path = pack_path(out_dir, domain)
            if path.is_file():
                packs[domain] = finalize_pearson(
                    _merge_sota_arms(load_pack(path), domain)
                )
    if set(packs) != set(domains):
        missing = [d for d in domains if d not in packs]
        raise FileNotFoundError(f"need packs for {missing} to write the gallery")
    paths = plot_all_from_packs(packs, out_dir)
    for p in paths:
        print(f"Wrote {p}", flush=True)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        type=Path,
        default=config.DEFAULT_CHECKPOINT,
        help="Shipped residual GINO (default M7680_gino_rebal_ft.pt).",
    )
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--n-pretell", type=int, default=N_PRETELL_DEFAULT)
    p.add_argument(
        "--skip-predict",
        action="store_true",
        help="Reuse results/presentation/{iid,dipping,three_layer}_pack.npz.",
    )
    p.add_argument(
        "--domain",
        action="append",
        choices=list(DOMAIN_SPECS),
        default=None,
        help="Restrict scoring and plotting to these domains (repeatable). Default: all.",
    )
    args = p.parse_args()
    run(
        ckpt=args.ckpt,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        n_pretell=args.n_pretell,
        skip_predict=args.skip_predict,
        domains=tuple(args.domain) if args.domain else None,
    )


if __name__ == "__main__":
    main()
