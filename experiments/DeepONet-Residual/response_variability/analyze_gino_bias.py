#!/usr/bin/env python3
"""Wave 0 GINO inductive-bias audit on nested presentation packs.

    uv run python experiments/DeepONet-Residual/response_variability/analyze_gino_bias.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from response_variability.gino_bias import (  # noqa: E402
    N_BOOT,
    analyze_pack,
    crossed_f0_cov,
    f0_impedance,
    phase3_protocol,
    to_jsonable,
)
from response_variability.plot_presentation import (  # noqa: E402
    DOMAIN_SPECS,
    load_pack,
    pack_path,
)

PACK_DIR = config.RESULTS_DIR / "presentation"
OUT_DIR = config.RESULTS_DIR / "response_variability" / "gino_bias"
KNN_CSV = config.RESULTS_DIR / "response_variability" / "sobol_probe" / "test_error_vs_knn.csv"


def _load_inside_hull(domain: str, n: int) -> np.ndarray | None:
    if not KNN_CSV.is_file():
        return None
    import pandas as pd

    df = pd.read_csv(KNN_CSV)
    sub = df[df["domain"] == domain]
    if "inside_aabb_4d" not in sub.columns or len(sub) != n:
        return None
    return sub.sort_values("sample")["inside_aabb_4d"].to_numpy(dtype=bool)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Wave 0 GINO bias audit on presentation packs")
    p.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing = [d for d in DOMAIN_SPECS if not pack_path(args.pack_dir, d).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing presentation packs for: " + ", ".join(missing)
        )
    domains: dict[str, Any] = {}
    bin_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for domain in DOMAIN_SPECS:
        pack = load_pack(pack_path(args.pack_dir, domain))
        n = int(np.asarray(pack["tf_opensees"]).shape[0])
        hull = _load_inside_hull(domain, n)
        blob = analyze_pack(
            pack, domain=domain, n_boot=args.n_boot, seed=args.seed, inside_hull=hull
        )
        per = blob.pop("per_sample")
        dct_ratio = blob.pop("dct_ratio")
        f0, _z = f0_impedance(pack)
        blob["f0_cov_crossed"] = crossed_f0_cov(
            f0, pack["cov"], per["rel_l2"], n_boot=args.n_boot, seed=args.seed
        )
        domains[domain] = blob
        np.save(args.out_dir / f"{domain}_dct_ratio.npy", dct_ratio)
        for qname, rows in blob["quartile_rel_l2"].items():
            for row in rows:
                bin_rows.append({"domain": domain, "axis": qname, **row})
        for i in range(n):
            sample_rows.append(
                {
                    "domain": domain,
                    "sample": i,
                    "rel_l2": float(per["rel_l2"][i]),
                    "delta_ln_A_peak": float(per["delta_ln_A_peak"][i]),
                    "log_bias_trough_safe": float(per["log_bias_trough_safe"][i]),
                    "slope_b": float(per["slope_b"][i]),
                    "Vs1": float(pack["vs1"][i]),
                    "H": float(pack["H"][i]),
                    "CoV": float(pack["cov"][i]),
                    "Vs2": float(pack["vs2"][i]),
                    "f0": float(f0[i]),
                }
            )
        print(
            f"{domain}: n={n}  b={blob['slope']['b']:.3f} "
            f"[{blob['slope']['b_lo']:.3f},{blob['slope']['b_hi']:.3f}]  "
            f"relL2={blob['rel_l2_mean']:.3f}  "
            f"collapse={blob['scalar_floor']['collapse_ratio']}",
            flush=True,
        )
    floor_rows = []
    for domain, blob in domains.items():
        s = blob["scalar_floor"]
        floor_rows.append(
            {
                "domain": domain,
                "n": blob["n"],
                "n_unique_4d": s["n_unique_4d"],
                "n_exact_cells_ge2": s["n_exact_cells_ge2"],
                "n_exact_replicates": s["n_exact_replicates"],
                "r2_scalar_exact": s["r2_scalar_exact"],
                "collapse_ratio": s["collapse_ratio"],
                "mse_gino_exact": s["mse_gino_exact"],
                "var_ops_exact": s["var_ops_exact"],
                "var_gino_exact": s["var_gino_exact"],
                "r2_scalar_neighborhood_ub": s["r2_scalar_neighborhood_ub"],
            }
        )
    summary = {
        "product": "characterization of shipped residual GINO, not a gated checkpoint",
        "seed": args.seed,
        "n_boot": args.n_boot,
        "exploratory_single_seed": True,
        "domains": domains,
    }
    summary["phase3"] = phase3_protocol(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2))
    (args.out_dir / "phase3_protocol.json").write_text(
        json.dumps(to_jsonable(summary["phase3"]), indent=2)
    )
    _write_csv(args.out_dir / "sobol_bins.csv", bin_rows)
    _write_csv(args.out_dir / "per_sample.csv", sample_rows)
    _write_csv(args.out_dir / "aleatoric_floor.csv", floor_rows)
    print(f"Wrote {args.out_dir / 'summary.json'}", flush=True)
    print(f"Phase 3 primary: {summary['phase3']['primary']} ({summary['phase3']['kind']})")


if __name__ == "__main__":
    main()
