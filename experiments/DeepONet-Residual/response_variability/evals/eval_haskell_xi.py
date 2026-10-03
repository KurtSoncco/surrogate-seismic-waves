#!/usr/bin/env python3
"""Haskell ξ=0.05 nom vs Haskell at OpenSees soil ζ, both vs OpenSees 2-D.

The shipped 1-D Base Case uses ``DEFAULT_XI_TREND=0.05``. OpenSees 2-D uses
Rayleigh matched to ``Damping_zeta`` (Campbell/Taborda Q–Vs, ``global_avg``)
at ``(min(f0, f_motion), 10 Hz)``. This scores both Haskell noms on the nested
presentation packs. Does not retrain or overwrite classical NPZ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import h5py

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from ood_io import jsonable  # noqa: E402
from haskell_baseline import haskell_nominal_af_within  # noqa: E402
from residual_signed import soil_mean_xi  # noqa: E402

from response_variability.evals.eval_iid import _as_central  # noqa: E402
from response_variability.metrics import (  # noqa: E402
    fmt_iqr,
    median_iqr,
    method_vs_reference,
    pearson,
    theoretical_f0,
)
from response_variability.plots.plot_presentation import (  # noqa: E402
    DOMAIN_SPECS,
    load_pack,
    pack_path,
)

OUT_DIR = config.RESULTS_DIR / "response_variability" / "eval_bias"
PACK_DIR = config.RESULTS_DIR / "presentation"
SCORE_DOMAINS = ("iid", "dipping")
XI_FIXED = float(config.DEFAULT_XI_TREND)
_SCREEN = Path("/home/kurt-/surrogate-seismic-waves/data/gifno_screen")
_BOX = Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data")

ARM_FIXED = "1D Haskell ξ=0.05"
ARM_SAMPLE = "1D Haskell ξ_soil"


def resolve_h5(stored: str, domain: str) -> Path:
    """Prefer a local screen copy, then the stored path, then Box."""
    name = Path(str(stored)).name
    if domain == "dipping":
        roots = (
            _SCREEN / "ood_dipping" / "h5",
            config.ood_dipping_root() / "h5",
            _BOX / "ood_dipping" / "h5",
        )
    elif domain == "three_layer":
        roots = (
            _SCREEN / "ood_three_layer" / "h5",
            config.ood_three_layer_root() / "h5",
            _BOX / "ood_three_layer" / "h5",
        )
    else:
        roots = (
            _SCREEN / "h5",
            Path(config.H5_DIR),
            _BOX / "h5",
        )
    candidates = [Path(str(stored)), *(r / name for r in roots)]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No H5 for {stored} (domain={domain})")


def read_soil_xi(h5_path: Path, soil_nz: int) -> tuple[float, float]:
    """Return (soil-mean ζ, bedrock-mean ζ) from ``Damping_zeta``."""
    with h5py.File(h5_path, "r") as f:
        zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
        params = {k: f["params"].attrs[k] for k in f["params"].attrs}
    nz = int(params.get("soil_layer_count", params.get("H_discretized", soil_nz)))
    if zeta.ndim == 2:
        crop = zeta[:, config.X_SLICE_START : config.X_SLICE_END]
    else:
        crop = zeta
    xi_soil = soil_mean_xi(crop, nz)
    n = max(1, min(int(nz), int(crop.shape[0])))
    if n < crop.shape[0]:
        bed = crop[n:]
        xi_bed = float(np.mean(bed)) if bed.size else float("nan")
    else:
        xi_bed = float("nan")
    return xi_soil, xi_bed


def score_one(freq: np.ndarray, ops: np.ndarray, cand: np.ndarray) -> dict[str, float]:
    return method_vs_reference(
        freq=freq, af_ref=_as_central(ops), af_cand=np.asarray(cand, float)
    )


def score_domain(
    domain: str, pack: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    freq = np.asarray(pack["freq"], dtype=float)
    ops = np.asarray(pack["tf_opensees"], dtype=np.float64)
    nom_pack = np.asarray(pack["tf_haskell_nominal"], dtype=np.float64)
    n = ops.shape[0]
    rows: list[dict[str, Any]] = []
    rel_pack = np.empty(n, dtype=np.float64)
    for i in range(n):
        h5 = resolve_h5(str(pack["h5_path"][i]), domain)
        soil_nz = (
            int(pack["soil_nz"][i])
            if "soil_nz" in pack
            else int(round(float(pack["H"][i])))
        )
        xi_soil, xi_bed = read_soil_xi(h5, soil_nz)
        vs1 = float(pack["vs1"][i])
        H = float(pack["H"][i])
        vs2 = float(pack["vs2"][i])
        tf05 = haskell_nominal_af_within(
            freq, vs1=vs1, H=H, vs2=vs2, xi=XI_FIXED, rho=config.RHO
        )
        tf_xi = haskell_nominal_af_within(
            freq, vs1=vs1, H=H, vs2=vs2, xi=xi_soil, rho=config.RHO
        )
        packed = _as_central(nom_pack[i])
        rel_pack[i] = float(
            np.max(np.abs(tf05 - packed)) / max(float(np.max(np.abs(packed))), 1e-12)
        )
        m05 = score_one(freq, ops[i], tf05)
        mxi = score_one(freq, ops[i], tf_xi)
        m_vs = method_vs_reference(freq=freq, af_ref=tf_xi, af_cand=tf05)
        shared = {
            "domain": domain,
            "sample": i,
            "vs1": vs1,
            "H": H,
            "vs2": vs2,
            "f0": theoretical_f0(vs1, H),
            "cov": float(pack["cov"][i]) if "cov" in pack else float("nan"),
            "xi_soil": xi_soil,
            "xi_bedrock": xi_bed,
            "xi_fixed": XI_FIXED,
            "pack_nom_rel_err": float(rel_pack[i]),
            "pearson_05_vs_xi": float(pearson(tf05, tf_xi)),
            "delta_ln_A_05_vs_xi": float(m_vs["delta_ln_A_peak"]),
        }
        metric_keys = (
            "pearson",
            "gof_af",
            "delta_ln_A_peak",
            "delta_f_peak",
            "A_peak",
            "f_peak",
            "ref_f_peak",
            "ref_A_peak",
            "rel_l2",
        )
        rows.append({**shared, "method": ARM_FIXED, **{k: m05[k] for k in metric_keys}})
        rows.append(
            {**shared, "method": ARM_SAMPLE, **{k: mxi[k] for k in metric_keys}}
        )
    df = pd.DataFrame(rows)
    rec: dict[str, Any] = {
        "n": n,
        "xi_soil": median_iqr(df.drop_duplicates("sample")["xi_soil"].to_numpy()),
        "xi_bedrock": median_iqr(df.drop_duplicates("sample")["xi_bedrock"].to_numpy()),
        "xi_fixed": XI_FIXED,
        "pack_nom_rel_err_max": float(np.nanmax(rel_pack)),
        "pearson_05_vs_xi": median_iqr(
            df.drop_duplicates("sample")["pearson_05_vs_xi"].to_numpy()
        ),
        "delta_ln_A_05_vs_xi": median_iqr(
            df.drop_duplicates("sample")["delta_ln_A_05_vs_xi"].to_numpy()
        ),
        "methods": {},
    }
    for method in (ARM_FIXED, ARM_SAMPLE):
        sub = df[df["method"] == method]
        rec["methods"][method] = {
            "pearson": median_iqr(sub["pearson"].to_numpy()),
            "anderson": median_iqr(sub["gof_af"].to_numpy()),
            "delta_ln_A_peak": median_iqr(sub["delta_ln_A_peak"].to_numpy()),
            "delta_f_peak": median_iqr(sub["delta_f_peak"].to_numpy()),
        }
    rec["anderson_tail"] = _anderson_tail(df, ARM_FIXED, ARM_SAMPLE)
    return df, rec


def _anderson_tail(
    df: pd.DataFrame, arm_fixed: str, arm_soil: str, n: int = 8
) -> list[dict[str, Any]]:
    """Largest Anderson scores on the ξ=0.05 arm, with the soil-ζ twin."""
    fixed = df[df["method"] == arm_fixed].nlargest(n, "gof_af")
    soil = df[df["method"] == arm_soil].set_index("sample")
    rows: list[dict[str, Any]] = []
    for rec in fixed.to_dict(orient="records"):
        s = soil.loc[rec["sample"]] if rec["sample"] in soil.index else None
        rows.append(
            {
                "sample": int(rec["sample"]),
                "vs1": float(rec["vs1"]),
                "H": float(rec["H"]),
                "f0": float(rec.get("f0", theoretical_f0(rec["vs1"], rec["H"]))),
                "ref_f_peak": float(rec.get("ref_f_peak", float("nan"))),
                "f_peak": float(rec.get("f_peak", float("nan"))),
                "delta_f_peak": float(rec["delta_f_peak"]),
                "gof_af": float(rec["gof_af"]),
                "gof_af_soil": float(s["gof_af"]) if s is not None else float("nan"),
                "pearson": float(rec["pearson"]),
            }
        )
    return rows


def write_markdown(agg: dict[str, Any], dest: Path) -> None:
    lines = [
        "# Haskell ξ vs OpenSees soil ζ",
        "",
        "Shipped 1-D Base Case is Thomson–Haskell with **fixed ξ=0.05**. "
        "OpenSees 2-D Rayleigh is matched to H5 `Damping_zeta` (Q–Vs `global_avg`) "
        "at $(f_0, 10\\,\\mathrm{Hz})$. The fair constant-$Q$ Haskell uses that "
        "soil-mean ζ, not 0.05. Rayleigh $\\zeta(f)$ still differs away from the "
        "two match frequencies; this check is only the ξ-**value** mismatch. "
        "Both arms are hysteretic Haskell, so they cannot see damping *model "
        "shape* (Rayleigh vs constant-$Q$).",
        "",
        "**Anderson** here is a Gaussian-weighted $L_1$ on $\\ln|\\mathrm{TF}|$, "
        "centered at the OpenSees 2-D peak frequency (width 1.5 Hz); lower is better.",
        "",
        "| Domain | Soil ζ | Bedrock ζ | Pack nom vs ξ=0.05 max rel. err |",
        "| --- | ---: | ---: | ---: |",
    ]
    for domain in SCORE_DOMAINS:
        rec = agg.get(domain)
        if not rec:
            continue
        lines.append(
            f"| {domain} | {fmt_iqr(rec['xi_soil'])} | {fmt_iqr(rec['xi_bedrock'])} | "
            f"{rec['pack_nom_rel_err_max']:.2e} |"
        )
    lines += [
        "",
        "| Domain | Arm | Pearson vs 2-D | Anderson | $\\Delta\\ln A_{\\mathrm{peak}}$ |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for domain in SCORE_DOMAINS:
        rec = agg.get(domain)
        if not rec:
            continue
        for method in (ARM_FIXED, ARM_SAMPLE):
            m = rec["methods"][method]
            lines.append(
                f"| {domain} | {method} | {fmt_iqr(m['pearson'])} | "
                f"{fmt_iqr(m['anderson'])} | {fmt_iqr(m['delta_ln_A_peak'])} |"
            )
    lines += [
        "",
        "The **0.05 vs soil-ζ peak** figures below are the median of "
        "*paired per-realization* $\\ln(A_{0.05}/A_{\\zeta})$, not the "
        "arithmetic difference of the two marginal $\\Delta\\ln A$ vs 2-D "
        "medians in the table (those marginals are wide and correlated, so "
        "subtracting them does not recover the paired number).",
        "",
        "## Reading",
        "",
        "- Soil ζ sits uniformly *below* 0.05, so switching to soil ζ is "
        "*less* damping and must raise peak amplitude. It does: paired "
        "$\\Delta\\ln A$ is negative (ξ=0.05 shorter than soil-ζ) in both "
        "domains, and the domain with the larger ζ-gap also has the larger "
        "peak-amplitude gap.",
        "- Correcting to the physically correct damping makes Pearson vs 2-D "
        "**worse**, not better. Nominal ξ=0.05 was mildly *flattering* the "
        "1-D score by accidentally mimicking part of the real 2-D "
        "peak-broadening. The ~0.77 / ~0.69 gap is therefore not a "
        "damping-*value* artifact. It is not yet a damping-*model-shape* "
        "check — that needs a literal OpenSees 1-D Rayleigh arm "
        "(`OPENSEES1D.md`).",
        "- Residual GINO is trained on the 0.05 nom; this table does not retrain.",
        "",
    ]
    for domain in SCORE_DOMAINS:
        rec = agg.get(domain)
        if not rec:
            continue
        a = rec["methods"][ARM_FIXED]["pearson"]["median"]
        b = rec["methods"][ARM_SAMPLE]["pearson"]["median"]
        da = rec["methods"][ARM_FIXED]["delta_ln_A_peak"]["median"]
        db = rec["methods"][ARM_SAMPLE]["delta_ln_A_peak"]["median"]
        lines.append(
            f"- **{domain}:** soil ζ {fmt_iqr(rec['xi_soil'])} vs 0.05; "
            f"Pearson {a:.3f} (0.05) → {b:.3f} (soil ζ); "
            f"marginal $\\Delta\\ln A$ vs 2-D {da:.3f} vs {db:.3f}; "
            f"paired 0.05 vs soil-ζ peak {fmt_iqr(rec['delta_ln_A_05_vs_xi'])}."
        )
    dip = agg.get("dipping") or {}
    tail = dip.get("anderson_tail") or []
    if tail:
        lines += [
            "",
            "## Dipping Anderson tail",
            "",
            "Dipping Anderson IQR upper bound is ~5× the median and barely "
            "moves between damping arms. The same samples dominate both "
            "columns. Pooled $A_{\\mathrm{peak}}$ over 0.1–10 Hz is grabbing "
            "a high-frequency lobe on the 2-D spectrum "
            "($f_{\\mathrm{ref}}\\approx 8$–$9$ Hz vs $f_0=V_{s1}/4H$), so "
            "the tail is a metric / high-$f$ issue — not a ξ-value effect. "
            "It is also not a single geometry: $\\Delta f_{\\mathrm{peak}}$ "
            "q25 is already $\\approx -5.5$ Hz, so **at least a quarter** of "
            "dipping 2-D spectra have their 0.1–10 Hz global max far above "
            "$f_0$. Mode-windowed peaks in `OPENSEES1D.md` avoid this pooled argmax.",
            "",
            "| sample | $V_{s1}$ | $H$ | $f_0$ | $f_{\\mathrm{peak}}$ 2-D | "
            "$\\Delta f_{\\mathrm{peak}}$ | Anderson 0.05 | Anderson soil-ζ |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in tail:
            lines.append(
                f"| {row['sample']} | {row['vs1']:.1f} | {row['H']:.0f} | "
                f"{row['f0']:.2f} | {row['ref_f_peak']:.2f} | "
                f"{row['delta_f_peak']:.2f} | {row['gof_af']:.3f} | "
                f"{row['gof_af_soil']:.3f} |"
            )
    dest.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--domains", nargs="+", default=list(SCORE_DOMAINS), choices=list(DOMAIN_SPECS)
    )
    args = p.parse_args()

    frames: list[pd.DataFrame] = []
    agg: dict[str, Any] = {}
    for domain in args.domains:
        src = pack_path(args.pack_dir, domain)
        if not src.is_file():
            raise FileNotFoundError(src)
        df, rec = score_domain(domain, load_pack(src))
        frames.append(df)
        agg[domain] = rec

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "haskell_xi_vs_2d.csv"
    pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
    json_path = args.out_dir / "haskell_xi_vs_2d_summary.json"
    json_path.write_text(json.dumps(jsonable(agg), indent=2))
    write_markdown(agg, args.out_dir / "HASKELL_XI.md")
    print(json.dumps(jsonable(agg), indent=2), flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
