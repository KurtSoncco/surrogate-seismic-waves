#!/usr/bin/env python3
"""Summarize Session N+1 §B1 seeded A/B and apply the gate.

Gate: if the interval on (direct − log_mult) rel_c excludes 0 → log_mult wins.
If it includes 0 → keep direct (simpler). Single-run numbers are not winners.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_arm(dir_path: Path, arm: str) -> list[dict]:
    rows = []
    for p in sorted(dir_path.glob(f"{arm}_s*.json")):
        rows.append(json.loads(p.read_text()))
    return rows


def _stats(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Gate JSON path (default: <dir>/gate_summary.json)",
    )
    args = p.parse_args()
    out = args.out or (args.dir / "gate_summary.json")

    direct = _load_arm(args.dir, "direct")
    residual = _load_arm(args.dir, "residual")
    if not direct or not residual:
        raise SystemExit(
            f"Need both arms under {args.dir} (got direct={len(direct)} residual={len(residual)})"
        )

    # Pair by seed when possible
    d_by_seed = {int(r["seed"]): r for r in direct if "seed" in r}
    r_by_seed = {int(r["seed"]): r for r in residual if "seed" in r}
    seeds = sorted(set(d_by_seed) & set(r_by_seed))
    if not seeds:
        # fall back to list order
        n = min(len(direct), len(residual))
        seeds = list(range(n))
        pairs = [
            (direct[i]["rel_l2_center_mean"], residual[i]["rel_l2_center_mean"])
            for i in range(n)
        ]
    else:
        pairs = [
            (
                d_by_seed[s]["rel_l2_center_mean"],
                r_by_seed[s]["rel_l2_center_mean"],
            )
            for s in seeds
        ]

    d_vals = [a for a, _ in pairs]
    r_vals = [b for _, b in pairs]
    deltas = [a - b for a, b in pairs]  # direct − residual (>0 ⇒ residual better)

    d_s = _stats(d_vals)
    r_s = _stats(r_vals)
    delta_s = _stats(deltas)

    # Mean ± 1.96 * se (approx 95% CI); with n=3–5 this is indicative.
    n = delta_s["n"]
    se = delta_s["std"] / math.sqrt(n) if n > 0 else float("nan")
    ci_lo = delta_s["mean"] - 1.96 * se
    ci_hi = delta_s["mean"] + 1.96 * se
    excludes_zero = bool(ci_lo > 0 or ci_hi < 0)
    # Positive delta mean ⇒ residual lower rel_c ⇒ residual wins when CI excludes 0
    residual_wins = bool(excludes_zero and delta_s["mean"] > 0)

    # Per-curve distribution (mean across seeds of each arm's med/p10/p90)
    def _curve_agg(rows: list[dict], key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r and r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "n_seeds": n,
        "seeds": seeds,
        "direct_rel_c": d_s,
        "residual_rel_c": r_s,
        "delta_direct_minus_residual_rel_c": {
            **delta_s,
            "ci95_lo": float(ci_lo),
            "ci95_hi": float(ci_hi),
            "se": float(se),
        },
        "gate_excludes_zero": excludes_zero,
        "residual_wins": residual_wins,
        "keep": "log_mult" if residual_wins else "direct",
        "curve_direct": {
            "median": _curve_agg(direct, "rel_l2_curve_median"),
            "p10": _curve_agg(direct, "rel_l2_curve_p10"),
            "p90": _curve_agg(direct, "rel_l2_curve_p90"),
        },
        "curve_residual": {
            "median": _curve_agg(residual, "rel_l2_curve_median"),
            "p10": _curve_agg(residual, "rel_l2_curve_p10"),
            "p90": _curve_agg(residual, "rel_l2_curve_p90"),
        },
        "note": (
            "Winner only if CI on (direct−residual) rel_c excludes 0. "
            "Otherwise keep direct (simpler)."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print()
    print(
        f"GATE: delta(direct−residual) rel_c = "
        f"{delta_s['mean']:.4f} ± {delta_s['std']:.4f} "
        f"CI95=[{ci_lo:.4f}, {ci_hi:.4f}]  "
        f"excludes_0={excludes_zero} → keep={summary['keep']}"
    )


if __name__ == "__main__":
    main()
