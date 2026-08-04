"""DeepONet feature ablation: OrbitAll vs GIFNO-XT inputs via MI and RF only.

Reuses residual cache (default n1000). Rebuilds feature table with XT columns.

Usage:
  PYTHONUNBUFFERED=1 .venv/bin/python experiments/Residual/run_deeponet_screen.py --cache n1000_seed42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config
from build_table import build_feature_table, feature_set
from screen_compare import plot_set_comparison, run_mi_rf, save_comparison


SETS = ["orbitall", "gifno_xt", "gifno_xt_full", "gifno_xt_plus", "combined"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, default="n1000_seed42")
    p.add_argument("--force-table", action="store_true")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--max-rows", type=int, default=30_000)
    p.add_argument(
        "--sets",
        nargs="+",
        default=SETS,
        help=f"Feature sets (default: {SETS})",
    )
    args = p.parse_args()

    cache_dir = config.CACHE_DIR / args.cache
    if not cache_dir.exists():
        raise SystemExit(f"Missing residual cache: {cache_dir}")

    print(f"Building / loading XT feature table from {cache_dir}", flush=True)
    df = build_feature_table(cache_dir, force=args.force_table)
    print(f"Table shape={df.shape}", flush=True)

    summaries = []
    for target in ("R_col", "R_nom"):
        for set_name in args.sets:
            feats = feature_set(set_name)
            print(f"\n=== {set_name} → {target}  (n_feat={len(feats)}) ===", flush=True)
            result = run_mi_rf(
                df,
                target,
                set_name,
                feats,
                seed=args.seed,
                max_rows=args.max_rows,
            )
            path = save_comparison(result)
            m = result["metrics"]
            print(
                f"  test_R2={m['test_r2']:.3f}  train_R2={m['train_r2']:.3f}  "
                f"MI top={result['mi'].head(3)['feature'].tolist()}",
                flush=True,
            )
            summaries.append(json.loads(path.read_text()))

        plot_set_comparison(summaries, target)
        print(f"Wrote compare_r2_{target}.png", flush=True)

    out = config.RESULTS_DIR / "deeponet_feature_compare.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"\nSummary → {out}", flush=True)


if __name__ == "__main__":
    main()
