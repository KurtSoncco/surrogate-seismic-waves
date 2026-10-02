#!/usr/bin/env python3
"""Spatial leftover vs inflated |TF| Pearson (IID + dipping, pack-only).

Scores OpenSees spatial scatter, leftover R, and GINO / Pretell / 1-D nom /
Toro 2022 on the same nested tests. Does not retrain or overwrite classical NPZ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from response_variability.evals.eval_toro2022 import toro2022_path  # noqa: E402
from response_variability.gino_bias import leftover, ols_ab  # noqa: E402
from response_variability.metrics import fmt_iqr, median_iqr  # noqa: E402
from response_variability.names import (  # noqa: E402
    GINO,
    HASKELL_NOMINAL,
    METHOD_COLORS,
    PRETELL,
    TORO,
)
from response_variability.plots.plot_presentation import (  # noqa: E402
    DOMAIN_SPECS,
    load_pack,
    make_synthetic_pack,
    pack_path,
)
from response_variability.style import (  # noqa: E402
    apply_nature_style,
    figsize,
    panel_letter,
    savefig,
)

OUT_DIR = config.RESULTS_DIR / "response_variability" / "eval_bias"
PACK_DIR = config.RESULTS_DIR / "presentation"
SCORE_DOMAINS = ("iid", "dipping")
ARMS = (HASKELL_NOMINAL, TORO, PRETELL, GINO)
_EPS = 1e-12
_STD_MIN = 1e-12


def _as_3d(tf: np.ndarray, n_rec: int) -> np.ndarray:
    x = np.asarray(tf, dtype=np.float64)
    if x.ndim == 2:
        return np.broadcast_to(x[:, None, :], (x.shape[0], n_rec, x.shape[1])).copy()
    if x.ndim != 3:
        raise ValueError(f"expected (n, n_freq) or (n, n_rec, n_freq), got {x.shape}")
    if x.shape[1] == 1 and n_rec > 1:
        return np.broadcast_to(x, (x.shape[0], n_rec, x.shape[2])).copy()
    return x


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson along the last axis; ``a``/``b`` are (n, n_freq)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - np.mean(a, axis=1, keepdims=True)
    bc = b - np.mean(b, axis=1, keepdims=True)
    num = np.sum(ac * bc, axis=1)
    den = np.sqrt(np.sum(ac**2, axis=1) * np.sum(bc**2, axis=1))
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    ok = np.isfinite(den) & (den > _EPS)
    out[ok] = num[ok] / den[ok]
    return out


def mean_spatial_sigma_ln(tf_i: np.ndarray) -> float:
    """Mean-over-frequency spatial σ_ln. Zero if there is no recorder axis."""
    a = np.asarray(tf_i, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2:
        return 0.0
    if float(np.std(a)) < _STD_MIN:
        return 0.0
    lx = np.log(np.clip(a, _EPS, None))
    sig = float(np.mean(np.std(lx, axis=0, ddof=1)))
    return 0.0 if (not np.isfinite(sig) or sig < 1e-8) else sig


def spatial_pattern_pearson_one(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Pearson across recorders at each frequency. NaN if pred is flat in x."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    if p.ndim != 2 or t.ndim != 2 or p.shape[0] < 2:
        return float("nan")
    p_std = np.std(p, axis=0)
    t_std = np.std(t, axis=0)
    ok = (p_std > _STD_MIN) & (t_std > _STD_MIN)
    if not np.any(ok):
        return float("nan")
    pc = p - np.mean(p, axis=0, keepdims=True)
    tc = t - np.mean(t, axis=0, keepdims=True)
    num = np.mean(pc * tc, axis=0)
    den = p_std * t_std
    cor = np.full(p.shape[1], np.nan, dtype=np.float64)
    cor[ok] = num[ok] / den[ok]
    return float(np.nanmean(cor))


def attach_toro(
    pack: dict[str, np.ndarray], domain: str, out_dir: Path
) -> dict[str, np.ndarray]:
    src = toro2022_path(out_dir, domain)
    if not src.is_file():
        return pack
    blob = load_pack(src)
    if "tf_toro" not in blob:
        return pack
    out = dict(pack)
    out["tf_toro"] = np.asarray(blob["tf_toro"], dtype=np.float64)
    return out


def leftover_metrics(
    ops: np.ndarray, nom: np.ndarray, gino: np.ndarray | None
) -> dict[str, np.ndarray]:
    n = ops.shape[0]
    nan = np.full(n, np.nan, dtype=np.float64)
    if gino is None:
        return {
            "leftover_rel_l2": nan,
            "leftover_pearson": nan,
            "leftover_r2": nan,
        }
    r = leftover(ops, nom)
    r_hat = leftover(gino, nom)
    mid = r.shape[1] // 2
    R = r[:, mid]
    Rh = r_hat[:, mid]
    ops_c = ops[:, mid]
    rel = np.linalg.norm(R, axis=1) / np.clip(np.linalg.norm(ops_c, axis=1), _EPS, None)
    ss_res = np.sum((Rh - R) ** 2, axis=1)
    ss_tot = np.sum((R - np.mean(R, axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1.0 - ss_res / np.clip(ss_tot, _EPS, None)
    return {
        "leftover_rel_l2": rel,
        "leftover_pearson": pearson_rows(Rh, R),
        "leftover_r2": r2,
    }


def score_arm(
    pred: np.ndarray,
    ops: np.ndarray,
) -> dict[str, np.ndarray]:
    n, n_rec, n_freq = ops.shape
    p3 = _as_3d(pred, n_rec)
    mid = n_rec // 2
    central = pearson_rows(p3[:, mid], ops[:, mid])
    array = pearson_rows(p3.reshape(-1, n_freq), ops.reshape(-1, n_freq)).reshape(
        n, n_rec
    )
    spatial_sig = np.array(
        [mean_spatial_sigma_ln(p3[i]) for i in range(n)], dtype=np.float64
    )
    spat_pat = np.array(
        [spatial_pattern_pearson_one(p3[i], ops[i]) for i in range(n)], dtype=np.float64
    )
    return {
        "central_pearson": central,
        "array_pearson": np.nanmean(array, axis=1),
        "spatial_sigma_ln": spatial_sig,
        "spatial_pattern_pearson": spat_pat,
    }


def _quartile_table(cov: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    c = np.asarray(cov, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(c) & np.isfinite(v)
    out: dict[str, Any] = {}
    if int(finite.sum()) < 8:
        return out
    try:
        labels = pd.qcut(
            c[finite], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
        )
    except ValueError:
        return out
    sub_v = v[finite]
    for q in ("Q1", "Q2", "Q3", "Q4"):
        mask = np.asarray(labels) == q
        if not np.any(mask):
            continue
        rec = median_iqr(sub_v[mask])
        rec["n"] = int(mask.sum())
        out[q] = rec
    return out


def score_domain(
    domain: str, pack: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ops = np.asarray(pack["tf_opensees"], dtype=np.float64)
    n, n_rec, _n_freq = ops.shape
    mid = n_rec // 2
    edge = 0
    adj = min(mid + 1, n_rec - 1)
    cov = (
        np.asarray(pack["cov"], dtype=np.float64)
        if "cov" in pack
        else np.full(n, np.nan)
    )
    rH = (
        np.asarray(pack["rH"], dtype=np.float64) if "rH" in pack else np.full(n, np.nan)
    )
    nom = np.asarray(pack["tf_haskell_nominal"], dtype=np.float64)
    gino = np.asarray(pack["tf_gino"], dtype=np.float64) if "tf_gino" in pack else None
    left = leftover_metrics(ops, nom, gino)

    ops_sig = np.array(
        [mean_spatial_sigma_ln(ops[i]) for i in range(n)], dtype=np.float64
    )
    ops_edge = pearson_rows(ops[:, edge], ops[:, mid])
    ops_adj = pearson_rows(ops[:, mid], ops[:, adj])

    present: dict[str, np.ndarray] = {}
    if "tf_haskell_nominal" in pack:
        present[HASKELL_NOMINAL] = nom
    if "tf_toro" in pack:
        present[TORO] = np.asarray(pack["tf_toro"], dtype=np.float64)
    if "tf_pretell" in pack:
        present[PRETELL] = np.asarray(pack["tf_pretell"], dtype=np.float64)
    if gino is not None:
        present[GINO] = gino

    pooled_r2 = float("nan")
    pooled_b = float("nan")
    if gino is not None:
        r = leftover(ops, nom)[:, mid].ravel()
        rh = leftover(gino, nom)[:, mid].ravel()
        fit = ols_ab(r, rh)
        pooled_r2 = float(fit["r2"])
        pooled_b = float(fit["b"])

    rows: list[dict[str, Any]] = []
    method_rec: dict[str, Any] = {}
    for method, tf in present.items():
        scored = score_arm(tf, ops)
        method_rec[method] = {
            "central_pearson": median_iqr(scored["central_pearson"]),
            "array_pearson": median_iqr(scored["array_pearson"]),
            "spatial_sigma_ln": median_iqr(scored["spatial_sigma_ln"]),
            "spatial_pattern_pearson": median_iqr(scored["spatial_pattern_pearson"]),
        }
        for i in range(n):
            rows.append(
                {
                    "domain": domain,
                    "sample": i,
                    "cov": float(cov[i]),
                    "rH": float(rH[i]),
                    "method": method,
                    "central_pearson": float(scored["central_pearson"][i]),
                    "array_pearson": float(scored["array_pearson"][i]),
                    "spatial_sigma_ln": float(scored["spatial_sigma_ln"][i]),
                    "spatial_pattern_pearson": float(
                        scored["spatial_pattern_pearson"][i]
                    ),
                    "ops_spatial_sigma_ln": float(ops_sig[i]),
                    "ops_edge_center_pearson": float(ops_edge[i]),
                    "ops_adjacent_pearson": float(ops_adj[i]),
                    "leftover_rel_l2": float(left["leftover_rel_l2"][i]),
                    "leftover_pearson": float(left["leftover_pearson"][i]),
                    "leftover_r2": float(left["leftover_r2"][i]),
                }
            )

    rec: dict[str, Any] = {
        "n": n,
        "n_rec": n_rec,
        "opensees": {
            "spatial_sigma_ln": median_iqr(ops_sig),
            "edge_center_pearson": median_iqr(ops_edge),
            "adjacent_pearson": median_iqr(ops_adj),
        },
        "leftover": {
            "rel_l2": median_iqr(left["leftover_rel_l2"]),
            "pearson_R": median_iqr(left["leftover_pearson"]),
            "r2_per_sample": median_iqr(left["leftover_r2"]),
            "r2_pooled": pooled_r2,
            "b_pooled": pooled_b,
        },
        "methods": method_rec,
        "cov_quartile": {
            "opensees_spatial_sigma_ln": _quartile_table(cov, ops_sig),
            "one_d_central_pearson": _quartile_table(
                cov,
                score_arm(nom, ops)["central_pearson"]
                if HASKELL_NOMINAL in present
                else np.array([]),
            ),
            "gino_central_pearson": _quartile_table(
                cov,
                score_arm(gino, ops)["central_pearson"]
                if gino is not None
                else np.array([]),
            ),
        },
    }
    if np.isfinite(rH).sum() >= 8:
        rec["rH_quartile"] = {
            "opensees_spatial_sigma_ln": _quartile_table(rH, ops_sig),
            "one_d_central_pearson": _quartile_table(
                rH, score_arm(nom, ops)["central_pearson"]
            ),
        }
    return pd.DataFrame(rows), rec


def plot_cov(
    df: pd.DataFrame,
    agg: dict[str, Any],
    dest: Path,
) -> None:
    import matplotlib.pyplot as plt

    apply_nature_style()
    domains = [d for d in SCORE_DOMAINS if d in agg and not str(d).startswith("_")]
    if not domains:
        return
    fig, axes = plt.subplots(len(domains), 2, figsize=figsize("double", height_mm=95))
    axes = np.atleast_2d(axes)
    for row, domain in enumerate(domains):
        sub = df[df["domain"] == domain]
        ops = sub.drop_duplicates("sample")
        ax0 = axes[row, 0]
        ax0.scatter(
            ops["cov"],
            ops["ops_spatial_sigma_ln"],
            s=8,
            c="0.25",
            alpha=0.7,
            linewidths=0,
        )
        ax0.set_xlabel("CoV")
        ax0.set_ylabel(r"OpenSees spatial $\sigma_{\ln}$")
        ax0.set_title(domain)
        panel_letter(ax0, "ab"[row] if len(domains) == 1 else "ac"[row])

        ax1 = axes[row, 1]
        gino = sub[sub["method"] == GINO]
        nom = sub[sub["method"] == HASKELL_NOMINAL]
        if not nom.empty and nom["cov"].notna().sum() >= 8:
            try:
                q = pd.qcut(
                    nom["cov"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
                )
            except ValueError:
                q = None
        else:
            q = None
        if q is not None:
            labels = list(q.cat.categories)
            data_1d = [
                nom.loc[q == lab, "central_pearson"].to_numpy() for lab in labels
            ]
            if not gino.empty:
                gq = pd.qcut(
                    gino["cov"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
                )
                data_g = [
                    gino.loc[gq == lab, "central_pearson"].to_numpy() for lab in labels
                ]
            else:
                data_g = None
            pos = np.arange(len(labels), dtype=float)
            bp1 = ax1.boxplot(
                data_1d,
                positions=pos - 0.18,
                widths=0.32,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp1["boxes"]:
                patch.set_facecolor(METHOD_COLORS[HASKELL_NOMINAL])
                patch.set_alpha(0.7)
            if data_g is not None:
                bp2 = ax1.boxplot(
                    data_g,
                    positions=pos + 0.18,
                    widths=0.32,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch in bp2["boxes"]:
                    patch.set_facecolor(METHOD_COLORS[GINO])
                    patch.set_alpha(0.7)
            ax1.set_xticks(pos, labels)
        ax1.set_ylabel("Central Pearson")
        ax1.set_ylim(0.4, 1.02)
        ax1.set_title(domain)
        panel_letter(ax1, "bd"[row] if len(domains) == 2 else "b")
        ax1.plot(
            [], [], color=METHOD_COLORS[HASKELL_NOMINAL], lw=4, label=HASKELL_NOMINAL
        )
        ax1.plot([], [], color=METHOD_COLORS[GINO], lw=4, label=GINO)
        if row == 0:
            ax1.legend(loc="lower left")
    fig.tight_layout()
    savefig(fig, dest)


def write_markdown(agg: dict[str, Any], dest: Path) -> None:
    lines = [
        "# Spatial leftover vs inflated TF Pearson",
        "",
        "Pack-only diagnostic: nested IID and dipping. Pearson of "
        r"$|\mathrm{TF}|(f)$ is affine-invariant, so a 1-D resonance can look "
        "strong even when the predictor has no lateral field.",
        "",
        "## 1. Is the 2-D field too smooth?",
        "",
        "| Domain | OpenSees spatial $\\sigma_{\\ln}$ | Edge vs center Pearson | Adjacent Pearson |",
        "| --- | ---: | ---: | ---: |",
    ]
    for domain in SCORE_DOMAINS:
        rec = agg.get(domain)
        if not rec:
            continue
        ops = rec["opensees"]
        lines.append(
            f"| {domain} | {fmt_iqr(ops['spatial_sigma_ln'])} | "
            f"{fmt_iqr(ops['edge_center_pearson'])} | "
            f"{fmt_iqr(ops['adjacent_pearson'])} |"
        )
    lines += [
        "",
        "IID stations share almost the same spectrum (adjacent Pearson near 1). "
        "That is lateral smoothness of *shape*, not a missing random field: "
        "spatial $\\sigma_{\\ln}$ is positive, and CoV quartiles below still move it. "
        "Dipping is the geometry leftover (edge vs center drops).",
        "",
        "CoV quartile of OpenSees spatial $\\sigma_{\\ln}$ / 1-D Pearson:",
        "",
    ]
    for domain in SCORE_DOMAINS:
        rec = agg.get(domain)
        if not rec:
            continue
        qsig = rec.get("cov_quartile", {}).get("opensees_spatial_sigma_ln", {})
        q1d = rec.get("cov_quartile", {}).get("one_d_central_pearson", {})
        bits = []
        for q in ("Q1", "Q2", "Q3", "Q4"):
            if q not in qsig:
                continue
            bits.append(
                f"{q}: $\\sigma_{{\\ln}}$ {fmt_iqr(qsig[q])}, "
                f"1-D Pearson {fmt_iqr(q1d.get(q, {}))}"
            )
        if bits:
            lines.append(f"- **{domain}:** " + "; ".join(bits) + ".")
    lines += [
        "",
        "## 2. Why SOTA looks good",
        "",
        "| Domain | Arm | Central Pearson | Array-mean Pearson | Spatial $\\sigma_{\\ln}$ | Spatial-pattern Pearson |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for domain in SCORE_DOMAINS:
        rec = agg.get(domain)
        if not rec:
            continue
        for method in ARMS:
            m = rec.get("methods", {}).get(method)
            if not m:
                continue
            lines.append(
                f"| {domain} | {method} | {fmt_iqr(m['central_pearson'])} | "
                f"{fmt_iqr(m['array_pearson'])} | {fmt_iqr(m['spatial_sigma_ln'])} | "
                f"{fmt_iqr(m['spatial_pattern_pearson'])} |"
            )
    lines += [
        "",
        "- **1-D Base Case** already owns the resonance; array-mean Pearson barely "
        "moves on IID because every station looks like the same column.",
        "- **Pretell median** is still 1-D wave physics on the *true* 2-D $V_s$ "
        "strip (200-column geomean). That lifts IID shape; it does not invent "
        "dipping/scattering, so the OOD gap vs GINO remains.",
        "- **Toro** is a broadcast 1-D geomean (spatial $\\sigma_{\\ln}=0$, "
        "spatial-pattern Pearson undefined).",
        "",
        "## 3. GINO is good on TF Pearson because it starts from 1-D",
        "",
        r"GINO reconstructs $\widehat{\mathrm{TF}}=\mathrm{TF}_{1D}+\hat R$. "
        "TF Pearson inherits the 1-D backbone. Leftover $R$ is the part that is "
        "actually 2-D:",
        "",
        "| Domain | TF Pearson (GINO) | Pearson of $R$ | $R^2$ pooled | $\\|R\\|/\\|\\mathrm{TF}\\|$ | Spatial-pattern Pearson |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain in SCORE_DOMAINS:
        rec = agg.get(domain)
        if not rec:
            continue
        g = rec.get("methods", {}).get(GINO, {})
        left = rec.get("leftover", {})
        lines.append(
            f"| {domain} | {fmt_iqr(g.get('central_pearson', {}))} | "
            f"{fmt_iqr(left.get('pearson_R', {}))} | "
            f"{left.get('r2_pooled', float('nan')):.3f} | "
            f"{fmt_iqr(left.get('rel_l2', {}))} | "
            f"{fmt_iqr(g.get('spatial_pattern_pearson', {}))} |"
        )
    lines += [
        "",
        "If TF Pearson is high while Pearson of $R$ and spatial-pattern Pearson "
        "are modest, GINO is under-correcting a real leftover — not saturating 2-D. "
        "The OOD Pearson gap vs Pretell is the part that is *not* metric inflation: "
        "geometry the 1-D arms cannot invent.",
        "",
    ]
    dest.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--domains", nargs="+", default=list(SCORE_DOMAINS), choices=list(DOMAIN_SPECS)
    )
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic-n", type=int, default=8)
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    frames: list[pd.DataFrame] = []
    agg: dict[str, Any] = {"_synthetic": bool(args.synthetic)}
    for domain in args.domains:
        if args.synthetic:
            pack = make_synthetic_pack(domain=domain, n=args.synthetic_n)
            pack["tf_toro"] = np.asarray(pack["tf_pretell"], dtype=np.float64)
        else:
            src = pack_path(args.pack_dir, domain)
            if not src.is_file():
                raise FileNotFoundError(
                    f"Missing presentation pack {src}. Pass --synthetic for a smoke run."
                )
            pack = attach_toro(load_pack(src), domain, args.out_dir)
        df, rec = score_domain(domain, pack)
        frames.append(df)
        agg[domain] = rec

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.concat(frames, ignore_index=True)
    csv_path = args.out_dir / "spatial_leftover.csv"
    table.to_csv(csv_path, index=False)
    json_path = args.out_dir / "spatial_leftover_summary.json"
    json_path.write_text(json.dumps(_jsonable(agg), indent=2))
    if args.plot:
        plot_cov(table, agg, args.out_dir / "spatial_leftover_cov.png")
    write_markdown(agg, args.out_dir / "SPATIAL_LEFTOVER.md")
    print(json.dumps(_jsonable(agg), indent=2), flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
