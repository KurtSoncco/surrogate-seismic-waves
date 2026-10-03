#!/usr/bin/env python3
"""Score 1-D Haskell nom and Toro 2022 (frozen-H AR(1)) vs OpenSees 2-D.

Recomputes ``tf_toro`` only. Does not overwrite ``*_classical.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from response_variability.evals.eval_iid import (  # noqa: E402
    _as_central,
    aggregate_json,
    band_misfit_table,
    summarize_methods,
)
from response_variability.metrics import fmt_iqr, median_iqr, spatial_sigma_ln  # noqa: E402
from response_variability.names import (  # noqa: E402
    HASKELL_NOMINAL,
    METHOD_COLORS,
    METHOD_LINESTYLES,
    OPENSEES,
    TORO,
)
from response_variability.plots.plot_presentation import (  # noqa: E402
    DOMAIN_SPECS,
    load_pack,
    make_synthetic_pack,
    pack_path,
)
from response_variability.seiskit_arms import (  # noqa: E402
    hallal_geomean_tf,
    lognormal_upper,
)
from response_variability.style import apply_nature_style, figsize, savefig  # noqa: E402

OUT_DIR = config.RESULTS_DIR / "response_variability" / "eval_bias"
PACK_DIR = config.RESULTS_DIR / "presentation"
SCORE_METHODS = (HASKELL_NOMINAL, TORO)
SCORE_DOMAINS = ("iid", "dipping")
KEEP_KEYS = (
    "tf_opensees",
    "tf_haskell_nominal",
    "tf_toro",
    "sigma_ln_toro",
    "freq",
    "vs1",
    "H",
    "cov",
    "vs2",
    "xi_mean",
    "local_idx",
    "sample_idx",
    "rf_seed",
    "f0",
    "f0_calc",
    "n_hallal_seeds",
)


def toro2022_path(out_dir: Path, domain: str) -> Path:
    return Path(out_dir) / f"{DOMAIN_SPECS[domain]['pack_name']}_toro2022.npz"


def add_toro_arm(
    pack: dict[str, np.ndarray],
    *,
    n_hallal_seeds: int = 40,
) -> dict[str, np.ndarray]:
    freq = pack["freq"]
    n = int(np.asarray(pack["tf_opensees"]).shape[0])
    n_freq = int(np.asarray(freq).shape[0])
    tf_toro = np.empty((n, n_freq), dtype=np.float64)
    sig_toro = np.empty((n, n_freq), dtype=np.float64)
    for i in tqdm(range(n), desc="Toro 2022", leave=False):
        vs1 = float(pack["vs1"][i])
        H = float(pack["H"][i])
        cov = float(pack["cov"][i])
        vs2 = float(pack["vs2"][i])
        xi = float(pack["xi_mean"][i]) if "xi_mean" in pack else config.DEFAULT_XI_TREND
        if not np.isfinite(xi) or xi <= 0:
            xi = config.DEFAULT_XI_TREND
        geo, sig = hallal_geomean_tf(
            freq=freq,
            vs1=vs1,
            H=H,
            cov=cov,
            vs2=vs2,
            xi=xi,
            n_seeds=n_hallal_seeds,
            kind="toro",
        )
        tf_toro[i] = np.asarray(geo, dtype=np.float64)
        sig_toro[i] = np.asarray(sig, dtype=np.float64)
    out = dict(pack)
    out["tf_toro"] = tf_toro
    out["sigma_ln_toro"] = sig_toro
    out["n_hallal_seeds"] = np.array(n_hallal_seeds)
    return out


def slim_score_pack(pack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop GINO / Pretell / etc. so summarize_methods only sees 1-D nom and Toro."""
    out = {k: pack[k] for k in pack if k in KEEP_KEYS or k.startswith("tf_")}
    drop = [
        k
        for k in list(out)
        if k.startswith("tf_")
        and k not in ("tf_opensees", "tf_haskell_nominal", "tf_toro")
    ]
    for k in drop:
        out.pop(k, None)
    return out


def _opensees_spatial_sigma_ln(pack: dict[str, np.ndarray]) -> np.ndarray:
    """Mean-over-frequency spatial σ_ln for each case (OpenSees recorder axis)."""
    tf_ops = np.asarray(pack["tf_opensees"], dtype=np.float64)
    if tf_ops.ndim != 3:
        return np.full(tf_ops.shape[0], np.nan)
    return np.array(
        [float(np.mean(spatial_sigma_ln(tf_ops[i]))) for i in range(tf_ops.shape[0])],
        dtype=np.float64,
    )


def domain_summary(
    summary: pd.DataFrame, misfit: pd.DataFrame, pack: dict
) -> dict[str, Any]:
    rec: dict[str, Any] = {"n": int(summary["sample"].nunique())}
    rec["opensees_spatial_sigma_ln"] = median_iqr(_opensees_spatial_sigma_ln(pack))
    if "sigma_ln_toro" in pack:
        rec["toro_ensemble_sigma_ln"] = median_iqr(
            np.mean(np.asarray(pack["sigma_ln_toro"], dtype=np.float64), axis=1)
        )
    for method in SCORE_METHODS:
        sub = summary[summary["method"] == method]
        mis = misfit[misfit["method"] == method]
        rec[method] = {
            "pearson": median_iqr(sub["pearson"].to_numpy()),
            "anderson": median_iqr(sub["gof_af"].to_numpy()),
            "delta_ln_A_peak": median_iqr(sub["delta_ln_A_peak"].to_numpy()),
            "delta_f_peak": median_iqr(sub["delta_f_peak"].to_numpy()),
            "gof_low": median_iqr(mis["gof_low"].to_numpy())
            if "gof_low" in mis
            else {},
            "gof_mid": median_iqr(mis["gof_mid"].to_numpy())
            if "gof_mid" in mis
            else {},
            "gof_high": median_iqr(mis["gof_high"].to_numpy())
            if "gof_high" in mis
            else {},
        }
    return rec


def plot_overlays(pack: dict[str, np.ndarray], dest: Path, *, n_panel: int = 4) -> None:
    import matplotlib.pyplot as plt

    apply_nature_style()
    freq = np.asarray(pack["freq"], dtype=float)
    n = int(pack["tf_opensees"].shape[0])
    idx = np.unique(np.linspace(0, n - 1, n_panel, dtype=int))
    fig, axes = plt.subplots(1, len(idx), figsize=figsize("double"), sharey=True)
    if len(idx) == 1:
        axes = [axes]
    p84 = lognormal_upper(pack["tf_toro"], pack["sigma_ln_toro"])
    for ax, i in zip(axes, idx):
        ops = _as_central(pack["tf_opensees"][i])
        nom = _as_central(pack["tf_haskell_nominal"][i])
        ax.loglog(
            freq,
            np.maximum(ops, 1e-6),
            color=METHOD_COLORS[OPENSEES],
            lw=1.2,
            label=OPENSEES,
        )
        ax.loglog(
            freq,
            np.maximum(nom, 1e-6),
            color=METHOD_COLORS[HASKELL_NOMINAL],
            ls=METHOD_LINESTYLES[HASKELL_NOMINAL],
            lw=1.0,
            label=HASKELL_NOMINAL,
        )
        ax.loglog(
            freq,
            np.maximum(pack["tf_toro"][i], 1e-6),
            color=METHOD_COLORS[TORO],
            ls=METHOD_LINESTYLES[TORO],
            lw=1.0,
            label=TORO,
        )
        ax.loglog(
            freq,
            np.maximum(p84[i], 1e-6),
            color=METHOD_COLORS[TORO],
            ls=":",
            lw=0.8,
            alpha=0.8,
            label=f"{TORO} p84",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"case {int(i)}")
    axes[0].set_ylabel(r"$|\mathrm{TF}|$")
    axes[-1].legend(loc="lower left", fontsize=6)
    fig.tight_layout()
    savefig(fig, dest)
    plt.close(fig)


def write_markdown(agg: dict[str, Any], dest: Path, *, synthetic: bool) -> None:
    lines = [
        "# Toro 2022 vs OpenSees 2-D",
        "",
        "Frozen-H AR(1) on `dz` (Toro 2022 Sec. 4), SPID $\\sigma_{\\ln V}(z)$, "
        "Thomson–Haskell geomean over Hallal seeds. Scored against OpenSees 2-D "
        "on the central recorder. 1-D Base Case is the same nominal Haskell column "
        "GINO residualizes.",
        "",
    ]
    if synthetic:
        lines += [
            "**Universe:** synthetic packs (presentation NPZs were not on disk). "
            "Re-run without `--synthetic` on a machine with `results/presentation/`.",
            "",
        ]
    lines += [
        "| Domain | Arm | Pearson | Anderson | $\\Delta\\ln A_{\\mathrm{peak}}$ | $\\Delta f_{\\mathrm{peak}}$ |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for domain, rec in agg.items():
        if domain.startswith("_"):
            continue
        for method in SCORE_METHODS:
            m = rec[method]
            lines.append(
                f"| {domain} | {method} | {fmt_iqr(m['pearson'])} | "
                f"{fmt_iqr(m['anderson'])} | {fmt_iqr(m['delta_ln_A_peak'])} | "
                f"{fmt_iqr(m['delta_f_peak'])} |"
            )
    lines += [
        "",
        "## Reading",
        "",
        "- **1-D vs 2-D** is the geometry leftover: nominal Haskell vs OpenSees 2-D.",
        "- **Toro 2022 vs 2-D** uses the same 1-D solver on AR(1)-grid $V_s$ draws. "
        "If the geomean stays next to the 1-D nom, randomization is not capturing 2-D shape.",
        "- **Toro ensemble $\\sigma_{\\ln}$** vs OpenSees spatial $\\sigma_{\\ln}$ "
        "(across recorders) compares each method's own variability estimate.",
        "",
    ]
    for domain, rec in agg.items():
        if domain.startswith("_"):
            continue
        lines.append(
            f"- **{domain}** (n={rec['n']}): OpenSees spatial "
            f"$\\sigma_{{\\ln}}$ {fmt_iqr(rec.get('opensees_spatial_sigma_ln', {}))}; "
            f"Toro ensemble {fmt_iqr(rec.get('toro_ensemble_sigma_ln', {}))}."
        )
    lines += ["", "## Findings", ""]
    for domain, rec in agg.items():
        if domain.startswith("_"):
            continue
        nom = rec[HASKELL_NOMINAL]
        toro = rec[TORO]
        d_a = toro["delta_ln_A_peak"]["median"] - nom["delta_ln_A_peak"]["median"]
        d_r = toro["pearson"]["median"] - nom["pearson"]["median"]
        d_g = toro["anderson"]["median"] - nom["anderson"]["median"]
        ops_s = rec.get("opensees_spatial_sigma_ln", {}).get("median", float("nan"))
        tor_s = rec.get("toro_ensemble_sigma_ln", {}).get("median", float("nan"))
        peak_note = (
            "geomean lowers the 1-D peak (Toro 2022 soften-the-peaks)"
            if d_a < -0.05
            else "geomean stays near the 1-D nom peak amplitude"
        )
        disp_note = (
            "still over-disperses vs 2-D spatial"
            if np.isfinite(ops_s) and np.isfinite(tor_s) and tor_s > ops_s
            else "ensemble spread is not larger than 2-D spatial"
        )
        lines.append(
            f"- **{domain}:** {peak_note}; Pearson "
            f"{d_r:+.3f}, Anderson {d_g:+.3f} vs 1-D nom. Toro {disp_note} "
            f"(ensemble {fmt_iqr(rec.get('toro_ensemble_sigma_ln', {}))} vs "
            f"OpenSees {fmt_iqr(rec.get('opensees_spatial_sigma_ln', {}))})."
        )
    dest.write_text("\n".join(lines) + "\n")


def run_domain(
    domain: str,
    *,
    pack: dict[str, np.ndarray],
    out_dir: Path,
    n_hallal_seeds: int,
    skip_if_present: bool,
    plot: bool,
) -> dict[str, Any]:
    dest = toro2022_path(out_dir, domain)
    out_dir.mkdir(parents=True, exist_ok=True)
    if skip_if_present and dest.is_file():
        pack = load_pack(dest)
    else:
        pack = add_toro_arm(pack, n_hallal_seeds=n_hallal_seeds)
        keep = {k: pack[k] for k in pack}
        np.savez_compressed(dest, **keep)
        print(f"Wrote {dest}", flush=True)
    slim = slim_score_pack(pack)
    summary, peaks = summarize_methods(slim)
    misfit = band_misfit_table(slim)
    summary["domain"] = domain
    misfit["domain"] = domain
    peaks["domain"] = domain
    rec = domain_summary(summary, misfit, pack)
    rec["aggregate"] = aggregate_json(summary, misfit)
    if plot and domain == "iid":
        plot_overlays(pack, out_dir / "toro2022_tf_overlay.png")
    return {
        "summary": summary,
        "misfit": misfit,
        "peaks": peaks,
        "rec": rec,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-hallal-seeds", type=int, default=40)
    p.add_argument(
        "--domains", nargs="+", default=list(SCORE_DOMAINS), choices=list(DOMAIN_SPECS)
    )
    p.add_argument("--skip-if-present", action="store_true")
    p.add_argument(
        "--synthetic", action="store_true", help="Use make_synthetic_pack (no H5)."
    )
    p.add_argument("--synthetic-n", type=int, default=8)
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    rows_s: list[pd.DataFrame] = []
    rows_m: list[pd.DataFrame] = []
    agg: dict[str, Any] = {"_synthetic": bool(args.synthetic)}
    for domain in args.domains:
        if args.synthetic:
            pack = make_synthetic_pack(domain=domain, n=args.synthetic_n)
        else:
            src = pack_path(args.pack_dir, domain)
            if not src.is_file():
                raise FileNotFoundError(
                    f"Missing presentation pack {src}. Pass --synthetic for a smoke run."
                )
            pack = load_pack(src)
        got = run_domain(
            domain,
            pack=pack,
            out_dir=args.out_dir,
            n_hallal_seeds=args.n_hallal_seeds,
            skip_if_present=args.skip_if_present,
            plot=args.plot,
        )
        rows_s.append(got["summary"])
        rows_m.append(got["misfit"])
        agg[domain] = got["rec"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "toro2022_vs_2d.csv"
    pd.concat(rows_s, ignore_index=True).to_csv(csv_path, index=False)
    pd.concat(rows_m, ignore_index=True).to_csv(
        args.out_dir / "toro2022_vs_2d_bands.csv", index=False
    )
    json_path = args.out_dir / "toro2022_vs_2d_summary.json"
    json_path.write_text(json.dumps(agg, indent=2))
    write_markdown(agg, args.out_dir / "TORO2022.md", synthetic=bool(args.synthetic))
    print(json.dumps(agg, indent=2), flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
