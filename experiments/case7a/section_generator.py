#!/usr/bin/env python3
"""Case-7-like whole-section generator (plan step 6).

Training sections are 2000 m long and 150 m deep on 1 m cells, with one
feature (flat, ramp, step or trough soil/rock contact) flanked by flat zones,
so each run yields about 100 windows of 500 m at 15 m spacing. A few 4000 m
sections with several features are generated for validation only.

A nested row sets ``master_lx_m``. ``section_field`` builds that master once,
including its random Vs field, and returns the ``lx_m`` crop. Shorter interiors
are slices of the same array, not new draws. ``DEPTH_M`` stays 150 m; a deeper
mesh is the CSV column ``depth_m``.

Soil is gradational, Vs(z) = Vs_s + (Vs_c - Vs_s) (z / H(x))^p. Rock below the
contact is two to five horizontal bands at 900-2600 m/s, increasing with
depth. Damping is 0.5-5 % in soil and between 0.25 % and the soil value in
rock, applied as seiskit ``global_avg`` Rayleigh at (f0, 3 f0) like Case 7.

``mutate`` makes ACCEL-style single-parameter edits for the acquisition rounds,
and ``near_case7`` keeps Case 7 and its close neighbours out of training.

    python section_generator.py --n 64 --out sections/round0.csv
    python section_generator.py --validation 4 --out sections/validation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEPTH_M = 150.0
DZ_M = 1.0
TRAIN_LX_M = 2000.0
VALIDATION_LX_M = 4000.0
FEATURES = ("flat", "ramp", "step", "trough")
MOTION_FREQ_HZ = 3.0
RAYLEIGH_F2_OVER_F1 = 3.0
STEP_WIDTH_M = 10.0

# name: (low, high). Sampled uniformly (Sobol); mutation steps are a fraction of the range.
RANGES = {
    "soil_thickness_m": (60.0, 120.0),
    "vs_surface_mps": (100.0, 200.0),
    "vs_contact_mps": (250.0, 600.0),
    "gradient_power": (0.5, 1.5),
    "rock_vs_min_mps": (900.0, 1500.0),
    "rock_vs_max_mps": (1600.0, 2600.0),
    "n_rock_bands": (2, 5),
    "rock_top_m": (40.0, 120.0),
    "zeta_soil": (0.005, 0.05),
    "zeta_rock_frac": (0.05, 1.0),
    "feature": (0, len(FEATURES) - 1),
    "feature_x_frac": (0.4, 0.6),
    "feature_width_m": (100.0, 600.0),
    "feature_dh_m": (-50.0, 50.0),
    "trough_depth_m": (20.0, 60.0),
}
INTEGER_KEYS = ("n_rock_bands", "feature")
ZETA_ROCK_MIN = 0.0025
MASTER_LX_M = TRAIN_LX_M
NESTED_WIDTHS_M = (500.0, 1000.0, 1500.0, 2000.0)

# Case 7 center column (basin_ensemble.py) and the exclusion radius in
# range-normalized units over these keys.
CASE7 = {"soil_thickness_m": 85.0, "vs_surface_mps": 140.0, "vs_contact_mps": 353.0, "zeta_soil": 0.0075}
CASE7_EXCLUSION_RADIUS = 0.08


def _scale(name: str, u: float):
    lo, hi = RANGES[name]
    if name in INTEGER_KEYS:
        return int(min(hi, lo + int(np.floor(u * (hi - lo + 1)))))
    return float(lo + u * (hi - lo))


def nested_crop_bounds(
    master_lx_m: float,
    width_m: float,
    crop_x0_m: float | None = None,
    dx_m: float = DZ_M,
) -> tuple[int, int]:
    """Column slice ``[i0, i1)`` of an interior inside one master field.

    The default origin centers the crop. ``crop_x0_m`` places it explicitly.
    Both edges are whole cells of ``dx_m``. The slice is the only width
    operation: the random field is built on the master grid and then cropped.
    """
    dx = float(dx_m)
    master_n = int(round(float(master_lx_m) / dx))
    width_n = int(round(float(width_m) / dx))
    if master_n < 1 or width_n < 1:
        raise ValueError(f"non-positive length master={master_lx_m} width={width_m}")
    if crop_x0_m is None:
        if width_n > master_n:
            raise ValueError(f"width {width_m} m does not fit in master {master_lx_m} m")
        i0 = (master_n - width_n) // 2
    else:
        i0 = int(round(float(crop_x0_m) / dx))
    i1 = i0 + width_n
    if i0 < 0 or i1 > master_n:
        raise ValueError(f"crop columns [{i0}, {i1}) outside master of {master_n}")
    return i0, i1


def paired_bootstrap_group(row: dict) -> str:
    """Cluster id for a paired width contrast.

    The resampled unit is the master geology. Width crops of that master are
    one draw; ``sample_id`` and ``lx_m`` are not.
    """
    master = row.get("master_id")
    if master in (None, ""):
        raise ValueError("paired width bootstrap requires master_id")
    return str(master)


def _travel_time_s(h: float, vs_s: float, vs_c: float, p: float, n: int = 400) -> float:
    z = (np.arange(n) + 0.5) / n
    return float(np.sum(h / n / (vs_s + (vs_c - vs_s) * z**p)))


def finalize(row: dict, sample_id: str, lx_m: float = TRAIN_LX_M, split: str = "train") -> dict:
    """Derived fields (Rayleigh pair, rock zeta, flat f0) for a sampled or mutated row."""
    out = {k: row[k] for k in RANGES}
    for k in INTEGER_KEYS:
        out[k] = int(out[k])
    f0 = 0.25 / _travel_time_s(out["soil_thickness_m"], out["vs_surface_mps"], out["vs_contact_mps"], out["gradient_power"])
    out.update(
        sample_id=sample_id,
        split=split,
        lx_m=float(lx_m),
        depth_m=DEPTH_M,
        feature_name=FEATURES[out["feature"]],
        zeta_rock=max(ZETA_ROCK_MIN, out["zeta_soil"] * out["zeta_rock_frac"]),
        f0_flat_hz=f0,
        rayleigh_f1_hz=max(0.2, f0),
        rayleigh_f2_hz=RAYLEIGH_F2_OVER_F1 * max(0.2, f0),
        motion_freq_hz=MOTION_FREQ_HZ,
        extra_features=row.get("extra_features", "[]"),
    )
    return out


def near_case7(row: dict) -> bool:
    d = [
        (float(row[k]) - v) / (RANGES[k][1] - RANGES[k][0])
        for k, v in CASE7.items()
    ]
    return bool(np.linalg.norm(d) < CASE7_EXCLUSION_RADIUS and row.get("feature_name", FEATURES[int(row["feature"])]) == "trough")


def sample_rows(n: int, seed: int = 0, prefix: str = "sec", start: int = 0) -> list[dict]:
    """Scrambled-Sobol draw over RANGES; rows near Case 7 are redrawn."""
    from scipy.stats import qmc

    keys = list(RANGES)
    sampler = qmc.Sobol(d=len(keys), scramble=True, seed=seed)
    rows: list[dict] = []
    while len(rows) < n:
        for u in sampler.random(max(8, n)):
            row = finalize({k: _scale(k, float(x)) for k, x in zip(keys, u)}, f"{prefix}_{start + len(rows):04d}")
            if not near_case7(row):
                rows.append(row)
            if len(rows) == n:
                break
    return rows


def mutate(row: dict, rng: np.random.Generator, sample_id: str, step: float = 0.15) -> dict:
    """ACCEL edit: change one parameter by up to +-step of its range (or switch feature)."""
    for _ in range(100):
        key = str(rng.choice(list(RANGES)))
        lo, hi = RANGES[key]
        new = dict(row)
        if key in INTEGER_KEYS:
            new[key] = int(np.clip(int(row[key]) + int(rng.choice([-1, 1])), lo, hi))
        else:
            new[key] = float(np.clip(float(row[key]) + rng.uniform(-step, step) * (hi - lo), lo, hi))
        if new[key] == row[key]:
            continue
        out = finalize(new, sample_id, float(row.get("lx_m", TRAIN_LX_M)))
        out["parent_id"] = row["sample_id"]
        out["mutated_key"] = key
        if not near_case7(out):
            return out
    raise RuntimeError(f"could not mutate {row['sample_id']} away from Case 7")


def validation_rows(n: int, seed: int = 1000) -> list[dict]:
    """4000 m sections with two or three features each (validation only)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, base in enumerate(sample_rows(n, seed=seed, prefix="val")):
        extra = []
        for _ in range(int(rng.integers(1, 3))):
            extra.append(
                {
                    "feature": int(rng.integers(1, len(FEATURES))),
                    "feature_x_frac": float(rng.uniform(0.1, 0.9)),
                    "feature_width_m": float(rng.uniform(*RANGES["feature_width_m"])),
                    "feature_dh_m": float(rng.uniform(*RANGES["feature_dh_m"])),
                    "trough_depth_m": float(rng.uniform(*RANGES["trough_depth_m"])),
                }
            )
        base["extra_features"] = json.dumps(extra)
        rows.append(finalize(base, f"val_{i:04d}", VALIDATION_LX_M, "validation"))
    return rows


def _smooth1d(h: np.ndarray, sigma_m: float, dx_m: float = DZ_M) -> np.ndarray:
    """Edge-padded Gaussian along x; sigma_m <= 0 leaves h unchanged."""
    sig = float(sigma_m) / float(dx_m)
    if sig < 0.5:
        return h
    r = int(np.ceil(3.0 * sig))
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sig) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(h.astype(np.float64), r, mode="edge"), k, mode="valid")


def _feature_offset(
    x: np.ndarray, lx: float, kind: str, xf: float, width: float, dh: float, trough: float,
    step_width: float = STEP_WIDTH_M,
) -> np.ndarray:
    """Change of contact depth (m, positive = deeper) produced by one feature."""
    xc = xf * lx
    if kind == "flat":
        return np.zeros_like(x)
    if kind == "ramp":
        return dh * np.clip((x - (xc - width / 2)) / width, 0.0, 1.0)
    if kind == "step":
        return dh * 0.5 * (1.0 + np.tanh((x - xc) / max(float(step_width), 1.0)))
    if kind == "trough":
        s = np.clip(np.abs(x - xc) / (width / 2), 0.0, 1.0)
        return trough * 0.5 * (1.0 + np.cos(np.pi * s))
    raise ValueError(kind)


def contact_depth(row: dict, x: np.ndarray) -> np.ndarray:
    lx = float(row["lx_m"])
    depth = float(row.get("depth_m", DEPTH_M))
    step_w = float(row.get("step_width_m", STEP_WIDTH_M))
    feats = [
        {k: row[k] for k in ("feature", "feature_x_frac", "feature_width_m", "feature_dh_m", "trough_depth_m")}
    ] + json.loads(row.get("extra_features") or "[]")
    h = np.full_like(x, float(row["soil_thickness_m"]))
    for f in feats:
        h += _feature_offset(
            x, lx, FEATURES[int(f["feature"])], float(f["feature_x_frac"]), float(f["feature_width_m"]),
            float(f["feature_dh_m"]), float(f["trough_depth_m"]),
            step_width=float(f.get("step_width_m", step_w)),
        )
    blend = float(row.get("contact_blend_m", 0.0) or 0.0)
    if blend > 0:
        h = _smooth1d(h, blend, DZ_M)
    return np.clip(h, 15.0, depth - 5.0)


def section_field(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vs, zeta, rock_mask), surface at row 0, element centers.

    Shape is ``(depth_m / DZ_M, lx_m / DZ_M)``. Depth defaults to ``DEPTH_M``
    (150 m); CSV ``depth_m`` overrides it, as the 250 m manifests do.

    ``soil_fill`` is ``gradient`` (default power-law Vs) or ``layered`` /
    ``soilgen`` (one 1D layered column stretched to H(x); see
    ``soilgen_columns.py``). A positive ``vs_cov`` multiplies soil Vs by a
    random field (``soil_variability``); 0 or absent leaves the base field
    bit-identical.

    A row with ``master_lx_m`` is a crop of one field built at that length.
    The random field is drawn on the master grid only. ``crop_x0_m`` is the
    left edge in metres; omitted, the crop is centered.
    """
    master_lx = row.get("master_lx_m")
    if master_lx in (None, ""):
        return _build_section_field(row)
    x0 = row.get("crop_x0_m")
    x0_m = None if x0 in (None, "") else float(x0)
    i0, i1 = nested_crop_bounds(float(master_lx), float(row["lx_m"]), x0_m)
    master = dict(row)
    master["lx_m"] = float(master_lx)
    master.pop("master_lx_m", None)
    master.pop("crop_x0_m", None)
    vs, zeta, rock = _build_section_field(master)
    return vs[:, i0:i1].copy(), zeta[:, i0:i1].copy(), rock[:, i0:i1].copy()


def _build_section_field(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the row's own grid. Callers that need a nested crop use ``section_field``."""
    lx = float(row["lx_m"])
    depth = float(row.get("depth_m", DEPTH_M))
    nz, nx = int(depth / DZ_M), int(round(lx / DZ_M))
    z = (np.arange(nz) + 0.5) * DZ_M
    x = (np.arange(nx) + 0.5) * DZ_M
    h = contact_depth(row, x)
    soil = z[:, None] < h[None, :]
    vs_s, vs_c = float(row["vs_surface_mps"]), float(row["vs_contact_mps"])
    fill = str(row.get("soil_fill", "gradient"))
    if fill == "layered":
        fill = "soilgen"
    if fill == "soilgen":
        from soilgen_columns import column_for_row, paint_soil

        bounds, vs_col = column_for_row(row)
        vs_soil = paint_soil(z, h, bounds, vs_col)
    elif fill == "gradient":
        frac = np.clip(z[:, None] / h[None, :], 0.0, 1.0) ** float(row["gradient_power"])
        vs_soil = vs_s + (vs_c - vs_s) * frac
    else:
        raise ValueError(f"unknown soil_fill {fill!r}")

    n_band = int(row["n_rock_bands"])
    v_lo = float(row["rock_vs_min_mps"])
    v_hi = max(float(row["rock_vs_max_mps"]), v_lo)
    band_vs = v_lo * (v_hi / v_lo) ** (np.arange(n_band) / max(n_band - 1, 1))
    top = float(row["rock_top_m"])
    edges = np.linspace(top, depth, n_band + 1)[1:-1]
    band = np.searchsorted(edges, z)
    vs_rock = np.broadcast_to(band_vs[band][:, None], (nz, nx)).copy()
    max_ratio = float(row.get("max_vs_ratio", 0.0) or 0.0)
    if max_ratio > 0:
        vs_rock = np.clip(vs_rock, vs_c, vs_c * max_ratio)

    vs = np.where(soil, vs_soil, np.maximum(vs_rock, vs_c))
    zeta = np.where(soil, float(row["zeta_soil"]), float(row["zeta_rock"]))
    smooth_m = float(row.get("vs_smooth_m", 0.0) or 0.0)
    if smooth_m > 0:
        vs = np.stack([_smooth1d(vs[i], smooth_m) for i in range(vs.shape[0])])
    factor = soil_variability(row, nz, nx)
    if factor is not None:
        vs = np.where(soil, vs * factor, vs)
    return vs.astype(np.float64), zeta.astype(np.float64), ~soil


def soil_variability(row: dict, nz: int, nx: int) -> np.ndarray | None:
    """Mean-one lognormal factor on soil Vs, or None when ``vs_cov`` is 0 or absent.

    Same field and transform as seiskit ``generate_gaussian_field_fft`` and
    ``_generate_vs_variability_field``, so a seed draws the same field: exponential
    correlation (``vs_corr_h_m``, anisotropy ``vs_corr_aniso`` = rH / rV),
    clipped at two standard deviations, seeded by ``field_seed``.
    """
    cov = float(row.get("vs_cov", 0.0) or 0.0)
    if cov <= 0.0:
        return None
    rng = np.random.default_rng(int(float(row["field_seed"])))
    r_h = float(row.get("vs_corr_h_m", 50.0) or 50.0)
    r_v = r_h / float(row.get("vs_corr_aniso", 10.0) or 10.0)
    kx, kz = np.meshgrid(2 * np.pi * np.fft.fftfreq(nx, d=DZ_M), 2 * np.pi * np.fft.fftfreq(nz, d=DZ_M))
    psd = (2 * np.pi * r_h * r_v) / (1 + (r_h * kx) ** 2 + (r_v * kz) ** 2) ** 1.5
    noise = rng.standard_normal((nz, nx)) + 1j * rng.standard_normal((nz, nx))
    g = np.fft.ifft2(np.sqrt(psd / DZ_M**2) * noise).real
    g = (g - g.mean()) / g.std()
    s = np.sqrt(np.log1p(cov**2))
    return np.exp(-0.5 * s**2 + s * np.clip(g, -2.0, 2.0))


def write_rows(rows: list[dict], path: Path) -> Path:
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def read_rows(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path)):
        out = dict(r)
        for k in RANGES:
            out[k] = int(float(r[k])) if k in INTEGER_KEYS else float(r[k])
        for k in ("lx_m", "zeta_rock", "rayleigh_f1_hz", "rayleigh_f2_hz", "motion_freq_hz", "f0_flat_hz"):
            out[k] = float(r[k])
        if "n_soil_layers" in out and out["n_soil_layers"] not in ("", None):
            out["n_soil_layers"] = int(float(out["n_soil_layers"]))
        if "soilgen_seed" in out and out["soilgen_seed"] not in ("", None):
            out["soilgen_seed"] = int(float(out["soilgen_seed"]))
        for k in (
            "duration_s", "vs_cov", "vs_corr_h_m", "vs_corr_aniso", "hx_m", "dt_s",
            "depth_m", "master_lx_m", "crop_x0_m", "step_width_m", "contact_blend_m",
            "vs_smooth_m", "max_vs_ratio",
        ):
            if out.get(k) not in ("", None):
                out[k] = float(out[k])
        if out.get("field_seed") not in ("", None):
            out["field_seed"] = int(float(out["field_seed"]))
        rows.append(out)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=64, help="training sections (Sobol)")
    ap.add_argument("--validation", type=int, default=0, help="write this many 4000 m validation sections instead")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--soil",
        choices=("gradient", "soilgen"),
        default="gradient",
        help="gradient: power-law Vs(z). soilgen: one 1D layered column stretched to H(x)",
    )
    ap.add_argument("--plot", action="store_true", help="save a Vs image of the first rows next to --out")
    args = ap.parse_args()
    rows = validation_rows(args.validation, args.seed + 1000) if args.validation else sample_rows(args.n, args.seed)
    if args.soil == "soilgen":
        rng = np.random.default_rng(args.seed + 17)
        for i, row in enumerate(rows):
            row["soil_fill"] = "soilgen"
            row["n_soil_layers"] = int(rng.integers(4, 13))
            row["soilgen_seed"] = int(args.seed) * 1000 + i
    write_rows(rows, args.out)
    print(f"wrote {len(rows)} rows -> {args.out}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(6, len(rows))
        fig, axes = plt.subplots(n, 1, figsize=(10, 1.6 * n), constrained_layout=True)
        for ax, r in zip(np.atleast_1d(axes), rows[:n]):
            vs, _, _ = section_field(r)
            im = ax.imshow(vs, aspect="auto", cmap="viridis", extent=(0, r["lx_m"], DEPTH_M, 0), vmin=100, vmax=2600)
            ax.set_title(f"{r['sample_id']}: {r['feature_name']}", fontsize=8)
        fig.colorbar(im, ax=axes, label="Vs (m/s)")
        fig.savefig(args.out.with_suffix(".png"), dpi=120)


if __name__ == "__main__":
    main()
