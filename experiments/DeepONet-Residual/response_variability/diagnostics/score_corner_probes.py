#!/usr/bin/env python3
"""Score Part 0 probe cells (6/76, 29/112, 50/148) for named leftover ckpts.

Rebuilds presentation TFs when the checkpoint exists; otherwise writes pending
rows from the shipped pack / seed_ceiling.csv.

    uv run python experiments/DeepONet-Residual/response_variability/diagnostics/score_corner_probes.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from response_variability.gino_bias import central_slice  # noqa: E402
from response_variability.metrics import band_pearson  # noqa: E402
from response_variability.plots.plot_presentation import (  # noqa: E402
    build_domain_pack,
    load_pack,
    pack_path,
)
from response_variability.diagnostics.tail_a_vs_b import (  # noqa: E402
    cell_ids,
    param_matrix,
    pairwise_ops_pearson,
)

PROBE_IDX = (6, 76, 29, 112, 50, 148)
OPS_OPS_CELLS = {
    (6, 76): 0.726,
    (29, 112): 0.694,
    (50, 148): 0.652,
}
RUNS = (
    ("ship", "M7680_gino_rebal_ft.pt"),
    ("0a_xi_field_acf", "M7680_xi_field_acf_ft.pt"),
    ("0b_rh_dilate", "M7680_gno_rh_dilate_ft.pt"),
    ("0c_fno_modes832", "M7680_fno_modes832_ft.pt"),
)
SHIP_PACK_DIR = config.RESULTS_DIR / "presentation"
OUT_DIR = config.RESULTS_DIR / "response_variability" / "eval_bias"
CEILING_CSV = OUT_DIR / "seed_ceiling.csv"


def _central_pearson(gino: np.ndarray, ops: np.ndarray, freq: np.ndarray) -> np.ndarray:
    gc = central_slice(gino)
    oc = central_slice(ops)
    n = int(oc.shape[0])
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        out[i] = band_pearson(gc[i], oc[i], freq, lo=0.1, hi=10.0)
    return out


def _ops_ops_for_idx(pack: dict[str, np.ndarray], idx: tuple[int, int]) -> float:
    x = param_matrix(pack)
    ids = cell_ids(x)
    a, b = idx
    if max(a, b) >= len(ids) or ids[a] != ids[b]:
        return float("nan")
    members = np.array([a, b], dtype=int)
    pair = pairwise_ops_pearson(pack["tf_opensees"], pack["freq"], members)
    return float(pair["ops_ops_pearson_median_central"])


def _load_ceiling_fallback() -> dict[str, float]:
    out: dict[str, float] = {}
    if not CEILING_CSV.is_file():
        return out
    with CEILING_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("domain")) != "iid":
                continue
            key = str(row.get("pack_indices", "")).replace(" ", "")
            try:
                out[key] = float(row["gino_ops_pearson_min_central"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def score_ckpt(
    name: str,
    ckpt: Path,
    *,
    batch_size: int,
    skip_predict: bool,
) -> dict[str, Any]:
    pack_dir = config.RESULTS_DIR / "presentation" / name
    iid_path = pack_path(pack_dir, "iid")
    status = "pending"
    pack: dict[str, np.ndarray] | None = None
    if name == "ship" and pack_path(SHIP_PACK_DIR, "iid").is_file():
        pack = load_pack(pack_path(SHIP_PACK_DIR, "iid"))
        status = "ship_pack"
    elif iid_path.is_file() and skip_predict:
        pack = load_pack(iid_path)
        status = "cached_pack"
    elif ckpt.is_file() and not skip_predict:
        print(f"=== rebuild iid pack for {name} from {ckpt.name} ===", flush=True)
        pack = build_domain_pack(
            "iid",
            ckpt_path=ckpt,
            out_dir=pack_dir,
            batch_size=batch_size,
            n_pretell=0,
            skip_predict=False,
        )
        status = "scored"
    rec: dict[str, Any] = {"run": name, "ckpt": ckpt.name, "status": status}
    ceiling = _load_ceiling_fallback()
    if pack is None:
        for i in PROBE_IDX:
            rec[f"pearson_{i}"] = ""
        rec["ops_ops_50_148"] = OPS_OPS_CELLS[(50, 148)]
        rec["sample50_below_ceiling"] = ""
        rec["note"] = "checkpoint or pack missing"
        if name == "ship":
            rec["pearson_50"] = ceiling.get("50,148", "")
        return rec
    freq = np.asarray(pack["freq"], dtype=float)
    pearson = _central_pearson(pack["tf_gino"], pack["tf_opensees"], freq)
    for i in PROBE_IDX:
        rec[f"pearson_{i}"] = float(pearson[i]) if i < len(pearson) else ""
    rec["ops_ops_6_76"] = _ops_ops_for_idx(pack, (6, 76))
    rec["ops_ops_29_112"] = _ops_ops_for_idx(pack, (29, 112))
    rec["ops_ops_50_148"] = _ops_ops_for_idx(pack, (50, 148))
    p50 = rec.get("pearson_50")
    rec["sample50_below_ceiling"] = (
        bool(float(p50) < float(rec["ops_ops_50_148"]))
        if p50 not in ("", None)
        else ""
    )
    rec["note"] = ""
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-predict", action="store_true")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, ckpt_name in RUNS:
        rows.append(
            score_ckpt(
                name,
                config.CHECKPOINT_DIR / ckpt_name,
                batch_size=int(args.batch_size),
                skip_predict=bool(args.skip_predict),
            )
        )
    out_csv = OUT_DIR / "part0_probes.csv"
    fields = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "part0_probes.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"Wrote {out_csv}", flush=True)
    for rec in rows:
        print(
            f"  {rec['run']:20s} {rec['status']:12s} "
            f"p50={rec.get('pearson_50', '')} below={rec.get('sample50_below_ceiling', '')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
