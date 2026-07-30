#!/usr/bin/env python3
"""Diagnostics D1–D3 (AGENTS §4). Cheap, no training.

D1: EV of H_1D(trend) per corner
D2: ΔH vs H_2D roughness / effective rank along f
D3: bake-off trend vs realization-geomean vs Pretell vs GIFNO
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP))
import config

config.setup_import_paths()

from haskell_baseline import (  # noqa: E402
    H_1D_trend,
    haskell_af_within,
    haskell_realization_geomean,
)

_EPS = 1e-12


def _rel_l2(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + _EPS))


def _pearson(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _explained_variance(h2d: np.ndarray, h1d: np.ndarray) -> float:
    """EV = 1 - Var(Δ) / Var(H_2D) over concatenated samples × freq."""
    d = (h2d - h1d).ravel()
    t = h2d.ravel()
    vt = float(np.var(t))
    if vt < _EPS:
        return float("nan")
    return float(1.0 - np.var(d) / vt)


def _participation_ratio(mat: np.ndarray) -> float:
    """Effective rank via singular-value participation ratio. mat (n_samples, n_freq)."""
    if mat.shape[0] < 2:
        return float("nan")
    x = mat - mat.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    s2 = s**2
    if s2.sum() < _EPS:
        return 0.0
    return float((s2.sum() ** 2) / ((s2**2).sum() + _EPS))


def _high_freq_energy_ratio(mat: np.ndarray, frac: float = 0.5) -> float:
    """Fraction of power in upper half of frequency indices (proxy roughness)."""
    spec = np.mean(np.abs(np.fft.rfft(mat, axis=1)) ** 2, axis=0)
    n = len(spec)
    cut = max(1, int(n * frac))
    return float(spec[cut:].sum() / (spec.sum() + _EPS))


def _load_rv_lookup():
    sys.path.insert(0, str(config.RV_ROOT))
    os.environ.pop("RV_SMOKE", None)
    from manifest import (  # noqa: WPS433
        _hallal_block_size,
        active_rf_seeds,
        active_sobol_count,
        index_to_params,
        total_combinations,
    )

    hallal0 = _hallal_block_size(active_sobol_count())
    lookup = {}
    for i in range(hallal0, total_combinations()):
        p = index_to_params(i)
        if p.method in ("opensees_2d", "pretell", "grf_2d"):
            lookup[(p.sobol_id, p.method, p.seed)] = i
    return lookup, active_sobol_count(), active_rf_seeds()


def collect_rv_corner(h5_dir: Path, max_seeds: int | None = 10, max_sobol: int | None = None):
    lookup, n_sobol, seeds = _load_rv_lookup()
    if max_seeds is not None:
        seeds = seeds[:max_seeds]
    if max_sobol is not None:
        n_sobol = min(n_sobol, max_sobol)

    rows = []
    h2d_stack, trend_stack, local_stack, geo_stack = [], [], [], []
    pretell_stack, gifno_stack = [], []

    center_full = config.BC_WIDTH + config.NX // 2
    # columns for realization geomean (21 strip recorders mapped to full grid)
    rec_strip = config.recorder_x_indices()
    rec_full = config.BC_WIDTH + rec_strip

    for sid in range(n_sobol):
        for seed in seeds:
            io = lookup.get((sid, "opensees_2d", seed))
            if io is None:
                continue
            path = h5_dir / f"run_{io}.h5"
            if not path.is_file():
                continue
            with h5py.File(path, "r") as f:
                vs = f["Vs_field"][:]
                zeta = f["Damping_zeta"][:]
                dz = float(f["grid"].attrs.get("dz", 1.0))
                vs1 = float(f["params"].attrs["Vs1"])
                H = float(f["params"].attrs["H"])
                vs2 = float(f["params"].attrs["Vs2"])
                cov = float(f["params"].attrs["CoV"])
                af = f["transfer_function"]["AF"][:].astype(np.float64)
                freq = f["transfer_function"]["freq"][:].astype(np.float64)

            soil_nz = max(1, min(int(round(H / dz)), vs.shape[0]))
            trend = H_1D_trend(
                freq, vs1=vs1, H=H, vs2=vs2, xi=config.DEFAULT_XI_TREND
            )
            local = haskell_af_within(
                freq,
                vs[:, min(center_full, vs.shape[1] - 1)],
                zeta[:, min(center_full, vs.shape[1] - 1)],
                dz=dz,
                vs_rock=vs2,
                soil_nz=soil_nz,
            )
            cols = [int(c) for c in rec_full if 0 <= int(c) < vs.shape[1]]
            geo = haskell_realization_geomean(
                freq, vs, zeta, cols, dz=dz, vs_rock=vs2, soil_nz=soil_nz
            )

            af_p = af_g = None
            ip = lookup.get((sid, "pretell", seed))
            if ip is not None and (h5_dir / f"run_{ip}.h5").is_file():
                with h5py.File(h5_dir / f"run_{ip}.h5", "r") as f:
                    af_p = f["transfer_function"]["AF"][:].astype(np.float64)
                    fp = f["transfer_function"]["freq"][:].astype(np.float64)
                if fp.shape != freq.shape:
                    af_p = np.interp(freq, fp, af_p)
            ig = lookup.get((sid, "grf_2d", seed))
            if ig is not None and (h5_dir / f"run_{ig}.h5").is_file():
                with h5py.File(h5_dir / f"run_{ig}.h5", "r") as f:
                    af_g = f["transfer_function"]["AF"][:].astype(np.float64)
                    fg = f["transfer_function"]["freq"][:].astype(np.float64)
                if fg.shape != freq.shape:
                    af_g = np.interp(freq, fg, af_g)

            rows.append(
                {
                    "corner": "rv_pancake",
                    "sobol_id": sid,
                    "seed": seed,
                    "cov": cov,
                    "rel_trend": _rel_l2(trend, af),
                    "pear_trend": _pearson(trend, af),
                    "rel_local": _rel_l2(local, af),
                    "rel_geo": _rel_l2(geo, af),
                    "rel_pretell": _rel_l2(af_p, af) if af_p is not None else float("nan"),
                    "rel_gifno": _rel_l2(af_g, af) if af_g is not None else float("nan"),
                }
            )
            h2d_stack.append(af)
            trend_stack.append(trend)
            local_stack.append(local)
            geo_stack.append(geo)
            if af_p is not None:
                pretell_stack.append(af_p)
            if af_g is not None:
                gifno_stack.append(af_g)

    return {
        "rows": rows,
        "h2d": np.stack(h2d_stack) if h2d_stack else None,
        "trend": np.stack(trend_stack) if trend_stack else None,
        "local": np.stack(local_stack) if local_stack else None,
        "geo": np.stack(geo_stack) if geo_stack else None,
        "pretell": np.stack(pretell_stack) if pretell_stack else None,
        "gifno": np.stack(gifno_stack) if gifno_stack else None,
    }


def collect_nofield_flat():
    """Synthetic no-field flat: H_2D ≃ H_1D(trend) by construction."""
    freq = np.logspace(-1, 1, 1000)
    vs1, H, vs2 = 200.0, 40.0, 900.0
    trend = H_1D_trend(freq, vs1=vs1, H=H, vs2=vs2, xi=0.05)
    # identically the "truth"
    return {
        "h2d": trend[None, :],
        "trend": trend[None, :],
        "corner": "nofield_flat",
    }


def summarize_corner(name: str, h2d, trend) -> dict:
    delta = h2d - trend
    return {
        "corner": name,
        "n": int(h2d.shape[0]),
        "EV_trend": _explained_variance(h2d, trend),
        "rel_l2_trend_mean": float(
            np.mean([_rel_l2(trend[i], h2d[i]) for i in range(len(h2d))])
        ),
        "rank_H2D": _participation_ratio(h2d),
        "rank_delta": _participation_ratio(delta),
        "hf_energy_H2D": _high_freq_energy_ratio(h2d),
        "hf_energy_delta": _high_freq_energy_ratio(delta),
        "delta_easier": bool(
            _participation_ratio(delta) <= _participation_ratio(h2d) * 1.05
            and _high_freq_energy_ratio(delta) <= _high_freq_energy_ratio(h2d) * 1.05
        ),
    }


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-seeds", type=int, default=10)
    p.add_argument("--max-sobol", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=config.DIAGNOSTICS_DIR)
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("[D1–D3] Collecting RV pancake corner...")
    rv = collect_rv_corner(
        config.RV_H5_DIR, max_seeds=args.max_seeds, max_sobol=args.max_sobol
    )
    summaries = []
    if rv["h2d"] is not None:
        summaries.append(summarize_corner("rv_pancake", rv["h2d"], rv["trend"]))

    nf = collect_nofield_flat()
    summaries.append(summarize_corner("nofield_flat", nf["h2d"], nf["trend"]))

    # D3 bake-off means
    bake = {"corner": "rv_pancake"}
    if rv["rows"]:
        for key in ("rel_trend", "rel_local", "rel_geo", "rel_pretell", "rel_gifno"):
            vals = [r[key] for r in rv["rows"] if np.isfinite(r[key])]
            bake[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")

    # Write tables
    with (out / "d1_d2_per_corner.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    with (out / "d3_bakeoff_rv.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bake.keys()))
        w.writeheader()
        w.writerow(bake)

    if rv["rows"]:
        with (out / "d3_per_seed.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rv["rows"][0].keys()))
            w.writeheader()
            w.writerows(rv["rows"])

    # D2 gate
    rv_sum = next((s for s in summaries if s["corner"] == "rv_pancake"), None)
    d2_pass = True
    if rv_sum is not None:
        d2_pass = bool(rv_sum["delta_easier"])
        # also fail if delta rank much higher
        if rv_sum["rank_delta"] > rv_sum["rank_H2D"] * 1.1:
            d2_pass = False

    go_path = out / "GO_NO_GO.md"
    lines = [
        "# GO / NO-GO (diagnostics D1–D3)",
        "",
        f"Generated by `diagnostics/run_d1_d3.py`. D4 pending separately.",
        "",
        "## D1 — Explained variance of H_1D(trend)",
        "",
        "| corner | n | EV_trend | rel_l2_trend_mean |",
        "|--------|---|----------|-------------------|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['corner']} | {s['n']} | {s['EV_trend']:.4f} | {s['rel_l2_trend_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## D2 — Is ΔH easier than H_2D?",
            "",
            "| corner | rank_H2D | rank_delta | hf_H2D | hf_delta | delta_easier |",
            "|--------|----------|------------|--------|----------|--------------|",
        ]
    )
    for s in summaries:
        lines.append(
            f"| {s['corner']} | {s['rank_H2D']:.2f} | {s['rank_delta']:.2f} | "
            f"{s['hf_energy_H2D']:.4f} | {s['hf_energy_delta']:.4f} | {s['delta_easier']} |"
        )
    lines.extend(
        [
            "",
            f"**D2 gate (rv_pancake): `{'PASS' if d2_pass else 'FAIL'}`**",
            "",
            "## D3 — Bake-off on RV (mean rel L2 vs OpenSees-2D)",
            "",
            f"- H_1D(trend): **{bake.get('rel_trend_mean', float('nan')):.4f}**",
            f"- Realization local column: {bake.get('rel_local_mean', float('nan')):.4f}",
            f"- Realization geomean (opponent, not baseline): {bake.get('rel_geo_mean', float('nan')):.4f}",
            f"- Pretell: {bake.get('rel_pretell_mean', float('nan')):.4f}",
            f"- GIFNO grf_2d: {bake.get('rel_gifno_mean', float('nan')):.4f}",
            "",
            f"**Number to beat (best physics on this corner):** Pretell "
            f"{bake.get('rel_pretell_mean', float('nan')):.4f} "
            f"(geomean opponent {bake.get('rel_geo_mean', float('nan')):.4f} is near-exact "
            f"on pancake fields for reasons AGENTS forbids as training baseline).",
            "",
            "## Decision",
            "",
        ]
    )
    if d2_pass:
        lines.append(
            "GO for residual architecture **pending D4**. "
            "ΔH appears no harder than H_2D on RV pancake by rank/HF energy proxies."
        )
    else:
        lines.append(
            "NO-GO for delta learning on this evidence: ΔH is rougher/higher-rank "
            "than H_2D. Prefer direct `|TF|` prediction. Do not implement §2 residual stack."
        )

    go_path.write_text("\n".join(lines) + "\n")
    (out / "d2_pass.json").write_text(json.dumps({"d2_pass": d2_pass, "summaries": summaries}, indent=2))
    print(f"Wrote {go_path}")
    print(f"D2 pass={d2_pass}")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
