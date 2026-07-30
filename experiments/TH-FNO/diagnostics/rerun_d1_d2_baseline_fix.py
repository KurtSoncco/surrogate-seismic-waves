#!/usr/bin/env python3
"""Baseline-fix D1/D2 decision tree (pancake corner).

Tests whether the original D2 FAIL was an artifact of a coarse 2-layer
attr baseline {Vs1,H,Vs2,ξ=0.05} vs the real depth-averaged trend profile.

Decision tree:
  1. Rebuild H_1D from lateral-mean Vs/ζ columns (fitted ζ).
  2. Recompute D1 EV. If EV → 90%+, original D2 is void.
  3. Peak-alignment histogram (first resonance offset).
  4. D2 on linear ΔH and log-ratio ΔH with centered participation-ratio rank.
"""

from __future__ import annotations

import argparse
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
    haskell_trend_from_mean_profile,
)

_EPS = 1e-12
_LOG_EPS = float(getattr(config, "TF_LOG_EPS", 1e-3))


def _rel_l2(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + _EPS))


def _explained_variance(h2d: np.ndarray, h1d: np.ndarray) -> float:
    d = (h2d - h1d).ravel()
    t = h2d.ravel()
    vt = float(np.var(t))
    if vt < _EPS:
        return float("nan")
    return float(1.0 - np.var(d) / vt)


def _participation_ratio_centered(mat: np.ndarray) -> float:
    """SVD participation ratio after subtracting mean spectrum (variation-about-mean)."""
    if mat.shape[0] < 2:
        return float("nan")
    x = mat - mat.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    s2 = s**2
    if s2.sum() < _EPS:
        return 0.0
    return float((s2.sum() ** 2) / ((s2**2).sum() + _EPS))


def _participation_ratio_raw(mat: np.ndarray) -> float:
    """Mean-included rank (for comparison only)."""
    if mat.shape[0] < 2:
        return float("nan")
    s = np.linalg.svd(mat, compute_uv=False)
    s2 = s**2
    if s2.sum() < _EPS:
        return 0.0
    return float((s2.sum() ** 2) / ((s2**2).sum() + _EPS))


def _hf_energy_ratio(mat: np.ndarray, frac: float = 0.5) -> float:
    spec = np.mean(np.abs(np.fft.rfft(mat, axis=1)) ** 2, axis=0)
    n = len(spec)
    cut = max(1, int(n * frac))
    return float(spec[cut:].sum() / (spec.sum() + _EPS))


def _first_resonance_f(freq: np.ndarray, af: np.ndarray) -> float:
    """First prominent peak frequency; fallback to global argmax."""
    af = np.asarray(af, float).ravel()
    freq = np.asarray(freq, float).ravel()
    try:
        from scipy.signal import find_peaks

        peaks, props = find_peaks(af, prominence=max(0.05 * (af.max() - af.min()), 1e-6))
        if peaks.size:
            return float(freq[int(peaks[0])])
    except Exception:
        pass
    return float(freq[int(np.argmax(af))])


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
        if p.method == "opensees_2d":
            lookup[(p.sobol_id, p.seed)] = i
    return lookup, active_sobol_count(), active_rf_seeds()


def collect(h5_dir: Path, max_seeds: int, max_sobol: int | None):
    lookup, n_sobol, seeds = _load_rv_lookup()
    seeds = seeds[:max_seeds]
    if max_sobol is not None:
        n_sobol = min(n_sobol, max_sobol)

    # Lateral mean over full mesh or central variability strip
    strip = slice(config.X_SLICE_START, config.X_SLICE_END)

    h2d_list, attr_list, mean_list = [], [], []
    peak_rows = []

    for sid in range(n_sobol):
        for seed in seeds:
            idx = lookup.get((sid, seed))
            if idx is None:
                continue
            path = h5_dir / f"run_{idx}.h5"
            if not path.is_file():
                continue
            with h5py.File(path, "r") as f:
                vs = f["Vs_field"][:]
                zeta = f["Damping_zeta"][:]
                dz = float(f["grid"].attrs.get("dz", 1.0))
                vs1 = float(f["params"].attrs["Vs1"])
                H = float(f["params"].attrs["H"])
                vs2 = float(f["params"].attrs["Vs2"])
                af = f["transfer_function"]["AF"][:].astype(np.float64)
                freq = f["transfer_function"]["freq"][:].astype(np.float64)

            soil_nz = max(1, min(int(round(H / dz)), vs.shape[0] - 1))
            attr = H_1D_trend(
                freq, vs1=vs1, H=H, vs2=vs2, xi=config.DEFAULT_XI_TREND
            )
            # Mean profile over central strip (IID train domain); ζ from field
            mean_p = haskell_trend_from_mean_profile(
                freq,
                vs,
                zeta,
                dz=dz,
                vs_rock=vs2,
                soil_nz=soil_nz,
                x_slice=strip,
            )

            h2d_list.append(af)
            attr_list.append(attr)
            mean_list.append(mean_p)

            f2 = _first_resonance_f(freq, af)
            fa = _first_resonance_f(freq, attr)
            fm = _first_resonance_f(freq, mean_p)
            peak_rows.append(
                {
                    "sobol_id": sid,
                    "seed": seed,
                    "f_H2D": f2,
                    "f_attr": fa,
                    "f_meanprof": fm,
                    "df_attr": fa - f2,
                    "df_meanprof": fm - f2,
                    "df_rel_attr": (fa - f2) / (f2 + _EPS),
                    "df_rel_meanprof": (fm - f2) / (f2 + _EPS),
                    "rel_attr": _rel_l2(attr, af),
                    "rel_meanprof": _rel_l2(mean_p, af),
                }
            )

    return {
        "h2d": np.stack(h2d_list),
        "attr": np.stack(attr_list),
        "meanprof": np.stack(mean_list),
        "peak_rows": peak_rows,
    }


def d2_bundle(h2d: np.ndarray, h1d: np.ndarray, name: str) -> dict:
    delta_lin = h2d - h1d
    log_h2 = np.log(np.maximum(h2d, _LOG_EPS))
    log_h1 = np.log(np.maximum(h1d, _LOG_EPS))
    delta_log = log_h2 - log_h1

    rank_h2_c = _participation_ratio_centered(h2d)
    rank_dlin_c = _participation_ratio_centered(delta_lin)
    rank_dlog_c = _participation_ratio_centered(delta_log)
    rank_h2_raw = _participation_ratio_raw(h2d)
    rank_dlin_raw = _participation_ratio_raw(delta_lin)

    easier_lin = rank_dlin_c <= rank_h2_c * 1.05
    easier_log = rank_dlog_c <= rank_h2_c * 1.05
    return {
        "baseline": name,
        "n": int(h2d.shape[0]),
        "EV": _explained_variance(h2d, h1d),
        "rel_l2_mean": float(np.mean([_rel_l2(h1d[i], h2d[i]) for i in range(len(h2d))])),
        "rank_H2D_centered": rank_h2_c,
        "rank_delta_lin_centered": rank_dlin_c,
        "rank_delta_log_centered": rank_dlog_c,
        "rank_H2D_raw": rank_h2_raw,
        "rank_delta_lin_raw": rank_dlin_raw,
        "hf_H2D": _hf_energy_ratio(h2d),
        "hf_delta_lin": _hf_energy_ratio(delta_lin),
        "hf_delta_log": _hf_energy_ratio(delta_log),
        "delta_lin_easier_centered": bool(easier_lin),
        "delta_log_easier_centered": bool(easier_log),
    }


def decide(attr_d1: dict, mean_d1: dict, peaks: list[dict]) -> dict:
    ev_mean = mean_d1["EV"]
    ev_attr = attr_d1["EV"]
    df_attr = np.array([r["df_rel_attr"] for r in peaks], float)
    df_mean = np.array([r["df_rel_meanprof"] for r in peaks], float)

    verdict = {
        "EV_attr": ev_attr,
        "EV_meanprof": ev_mean,
        "df_rel_attr_median": float(np.median(df_attr)),
        "df_rel_meanprof_median": float(np.median(df_mean)),
        "df_rel_attr_abs_median": float(np.median(np.abs(df_attr))),
        "df_rel_meanprof_abs_median": float(np.median(np.abs(df_mean))),
        "original_d2_void": bool(ev_mean >= 0.90),
        "meanprof_d2_lin_pass": bool(mean_d1["delta_lin_easier_centered"]),
        "meanprof_d2_log_pass": bool(mean_d1["delta_log_easier_centered"]),
    }

    if ev_mean >= 0.90:
        verdict["branch"] = "A_baseline_was_misbuilt"
        verdict["action"] = (
            "Original attr-baseline D2 is VOID. Adopt mean-profile H_1D. "
            f"Rerun D2: lin={'PASS' if verdict['meanprof_d2_lin_pass'] else 'FAIL'}, "
            f"log-ratio={'PASS' if verdict['meanprof_d2_log_pass'] else 'FAIL'}."
        )
        if verdict["meanprof_d2_log_pass"] or verdict["meanprof_d2_lin_pass"]:
            verdict["prefer"] = "residual_on_meanprof_trend"
        else:
            verdict["prefer"] = "still_direct_on_this_corner_despite_good_EV"
    elif np.median(np.abs(df_mean)) > 0.02:
        verdict["branch"] = "B_peak_offset_remains"
        verdict["action"] = (
            "EV still low OR peak offsets persist after mean-profile rebuild — "
            "further baseline work (layering / damping) before trusting D2."
        )
        verdict["prefer"] = "fix_baseline_further"
    else:
        verdict["branch"] = "C_genuine_pancake_complexity"
        verdict["action"] = (
            "Mean-profile EV still ~low and peaks aligned → residual on this "
            "pancake corner is genuinely hard; direct |TF| OK *for this corner*. "
            "Still run D1–D2 on short-rH / steep-dip (GIFNO) where delta should win."
        )
        verdict["prefer"] = "direct_for_pancake_only"
    return verdict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-seeds", type=int, default=5)
    p.add_argument("--max-sobol", type=int, default=32)
    p.add_argument("--out-dir", type=Path, default=config.DIAGNOSTICS_DIR)
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("[baseline-fix] collecting RV pancake...", flush=True)
    data = collect(config.RV_H5_DIR, args.max_seeds, args.max_sobol)
    h2d, attr, meanp = data["h2d"], data["attr"], data["meanprof"]
    print(f"  n={len(h2d)}", flush=True)

    attr_s = d2_bundle(h2d, attr, "attr_Vs1_H_Vs2_xi0.05")
    mean_s = d2_bundle(h2d, meanp, "mean_profile_fitted_zeta")
    verdict = decide(attr_s, mean_s, data["peak_rows"])

    # CSV tables
    with (out / "baseline_fix_d1_d2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(attr_s.keys()))
        w.writeheader()
        w.writerow(attr_s)
        w.writerow(mean_s)

    with (out / "baseline_fix_peak_align.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data["peak_rows"][0].keys()))
        w.writeheader()
        w.writerows(data["peak_rows"])

    (out / "baseline_fix_verdict.json").write_text(json.dumps(verdict, indent=2))

    md = [
        "# Baseline-fix D1/D2 (RV pancake)",
        "",
        "Critique: attr-based 2-layer `H_1D(Vs1,H,Vs2,ξ=0.05)` may misbuild the trend,",
        "so EV~58% and high-rank linear ΔH can be artifacts. This re-runs the decision tree.",
        "",
        f"n = {len(h2d)} (max_sobol={args.max_sobol}, max_seeds={args.max_seeds})",
        "",
        "## D1 — EV comparison",
        "",
        "| baseline | EV | mean rel L2 |",
        "|----------|----|-------------|",
        f"| attr {{Vs1,H,Vs2,ξ=0.05}} | {attr_s['EV']:.4f} | {attr_s['rel_l2_mean']:.4f} |",
        f"| **mean profile (fitted ζ)** | **{mean_s['EV']:.4f}** | **{mean_s['rel_l2_mean']:.4f}** |",
        "",
        "## Peak alignment (first resonance)",
        "",
        f"- median Δf/f (attr − H₂D): {verdict['df_rel_attr_median']:.4f} "
        f"(abs median {verdict['df_rel_attr_abs_median']:.4f})",
        f"- median Δf/f (meanprof − H₂D): {verdict['df_rel_meanprof_median']:.4f} "
        f"(abs median {verdict['df_rel_meanprof_abs_median']:.4f})",
        "",
        "## D2 — centered rank + log-ratio",
        "",
        "| baseline | rank H₂D (ctr) | rank Δ_lin (ctr) | rank Δ_log (ctr) | lin easier? | log easier? |",
        "|----------|----------------|------------------|------------------|-------------|-------------|",
        (
            f"| attr | {attr_s['rank_H2D_centered']:.2f} | {attr_s['rank_delta_lin_centered']:.2f} | "
            f"{attr_s['rank_delta_log_centered']:.2f} | {attr_s['delta_lin_easier_centered']} | "
            f"{attr_s['delta_log_easier_centered']} |"
        ),
        (
            f"| meanprof | {mean_s['rank_H2D_centered']:.2f} | {mean_s['rank_delta_lin_centered']:.2f} | "
            f"{mean_s['rank_delta_log_centered']:.2f} | {mean_s['delta_lin_easier_centered']} | "
            f"{mean_s['delta_log_easier_centered']} |"
        ),
        "",
        "Raw (mean-included) ranks for reference:",
        f"- attr: H₂D={attr_s['rank_H2D_raw']:.2f}, Δ_lin={attr_s['rank_delta_lin_raw']:.2f}",
        f"- meanprof: H₂D={mean_s['rank_H2D_raw']:.2f}, Δ_lin={mean_s['rank_delta_lin_raw']:.2f}",
        "",
        "## Decision",
        "",
        f"**Branch:** `{verdict['branch']}`",
        "",
        f"**Original attr D2 void?** `{verdict['original_d2_void']}` "
        f"(criterion: meanprof EV ≥ 0.90)",
        "",
        f"**Prefer:** `{verdict['prefer']}`",
        "",
        verdict["action"],
        "",
        "## Caveat (unchanged)",
        "",
        "This is still the RV pancake OOD corner. Mount `GIFNO_DATA_ROOT` and run",
        "D1–D2 on short-rH / steep-dip hold-outs — that is where delta-learning",
        "is supposed to win.",
        "",
    ]
    path = out / "GO_NO_GO_BASELINE_FIX.md"
    path.write_text("\n".join(md) + "\n")

    # Append pointer to main GO_NO_GO
    go = out / "GO_NO_GO.md"
    if go.is_file():
        note = (
            "\n---\n\n## Addendum: baseline-fix decision tree\n\n"
            f"See [`GO_NO_GO_BASELINE_FIX.md`](GO_NO_GO_BASELINE_FIX.md). "
            f"Branch `{verdict['branch']}`; original D2 void={verdict['original_d2_void']}; "
            f"prefer `{verdict['prefer']}`.\n"
        )
        text = go.read_text()
        if "GO_NO_GO_BASELINE_FIX" not in text:
            go.write_text(text.rstrip() + note)

    print(json.dumps(verdict, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
