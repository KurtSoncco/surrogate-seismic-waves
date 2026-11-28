#!/usr/bin/env python3
"""One blind variable-width section for the slide-12 example.

The network emits a natural-log residual, and g in this file is that residual:

    tf_pred = tf_1d * exp(g)

g_true is the same ratio for the OpenSees surface transfer function:

    tf_2d = tf_1d * exp(g_true)

Blind sections were reserved before training and are not in the loss or the
early-stopping set. The checkpoint on disk is stage 0: 500 m runs only.
Variable-width sections enter at stage 2, which has not been trained.

    python export_holdout_section.py
    python export_holdout_section.py --section man_0307
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from data import Cache  # noqa: E402
from encoding import encode, query_columns  # noqa: E402
from evaluate import load_model  # noqa: E402

CACHE = REPO / "data" / "vardomain" / "cache"
MODEL = REPO / "checkpoints" / "vardomain" / "stage0_residual_s0"
MANIFEST = Path(
    "/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data/vardomain/manifold150/manifest.csv"
)
MID = (0.4, 2.5)
# Widths at which a centered station sees hundreds of metres of lateral structure.
SLIDE_WIDTHS = (1500, 2000)
SLIDE_FEATURES = ("trough", "ramp", "step")


def _manifest() -> dict[str, dict]:
    with MANIFEST.open() as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


def _predict(model, device, sec) -> dict[str, np.ndarray]:
    vs = np.asarray(sec.vs, dtype=np.float64)
    zeta = np.asarray(sec.zeta, dtype=np.float64)
    x_m = np.asarray(sec.x_m, dtype=np.float64)
    enc = encode("metric", vs, zeta, vs.shape[0], f1_hz=sec.f1_hz, f2_hz=sec.f2_hz)
    query = query_columns("metric", x_m, vs.shape[1]).astype(np.float32)
    with torch.no_grad():
        g = model(
            torch.from_numpy(np.ascontiguousarray(enc)).unsqueeze(0).to(device),
            torch.from_numpy(query).unsqueeze(0).to(device),
        ).float().cpu().numpy()[0]
    tf_1d = np.exp(np.asarray(sec.log_tf1d, dtype=np.float64))
    tf_2d = np.exp(np.asarray(sec.log_tf, dtype=np.float64))
    g64 = g.astype(np.float64)
    tf_pred = tf_1d * np.exp(g64)
    return {
        "x_station_m": x_m.astype(np.float32),
        "tf_1d": tf_1d.astype(np.float32),
        "tf_2d": tf_2d.astype(np.float32),
        "tf_pred": tf_pred.astype(np.float32),
        "g": g64.astype(np.float32),
        "g_true": (np.log(tf_2d) - np.log(tf_1d)).astype(np.float32),
    }


def _band_stats(freq: np.ndarray, pred: dict) -> dict[str, float]:
    band = (freq >= MID[0]) & (freq <= MID[1])
    log_p = np.log(np.maximum(pred["tf_pred"][:, band], 1e-6))
    log_t = np.log(np.maximum(pred["tf_2d"][:, band], 1e-6))
    log_d = np.log(np.maximum(pred["tf_1d"][:, band], 1e-6))
    err = np.abs(log_p - log_t).mean()
    base = np.abs(log_d - log_t).mean()
    gap = np.abs(pred["g_true"][:, band]).mean()
    return {
        "gap_mid": float(gap),
        "log_l1_mid": float(err),
        "log_l1_1d_mid": float(base),
        "skill_mid": float(1.0 - err / max(base, 1e-6)),
    }


def _choose(rows: list[dict], forced: str | None) -> dict:
    if forced:
        hit = [r for r in rows if r["sample_id"] == forced]
        if not hit:
            raise SystemExit(f"{forced} is not a blind section in the cache")
        return hit[0]
    pool = [r for r in rows if r["lx_m"] in SLIDE_WIDTHS and r["feature_name"] in SLIDE_FEATURES]
    if not pool:
        pool = rows
    skilled = [r for r in pool if r["skill_mid"] > 0]
    return max(skilled or pool, key=lambda r: r["gap_mid"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", default=None, help="sample id, e.g. man_0307; default picks a blind 1500–2000 m section")
    ap.add_argument("--model-dir", type=Path, default=MODEL)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()

    torch.set_num_threads(8)
    device = torch.device("cpu")
    model, cfg = load_model(args.model_dir, device)
    model.eval()
    cache = Cache(args.cache)
    freq = np.asarray(cache.freq, dtype=np.float64)
    meta_rows = _manifest()

    scored = []
    by_id: dict[str, tuple] = {}
    for i, sec in enumerate(cache.sections):
        if sec.split != "blind":
            continue
        sid = sec.name.split("__", 1)[-1]
        pred = _predict(model, device, sec)
        row = meta_rows[sid]
        stats = _band_stats(freq, pred)
        rec = {
            "sample_id": sid,
            "index": i,
            "lx_m": int(sec.lx),
            "feature_name": row["feature_name"],
            "n_stations": int(pred["x_station_m"].size),
            **stats,
        }
        scored.append(rec)
        by_id[sid] = (sec, pred, row)
        print(
            f"{sid}  {rec['feature_name']:7} {rec['lx_m']:4} m  "
            f"gap {rec['gap_mid']:.3f}  skill {rec['skill_mid']:+.3f}",
            flush=True,
        )

    pick = _choose(scored, args.section)
    sec, pred, row = by_id[pick["sample_id"]]
    # The stored labels and the cache frequency are the same grid.
    if pred["tf_2d"].shape[-1] != freq.size:
        raise SystemExit(f"frequency length {freq.size} != TF length {pred['tf_2d'].shape[-1]}")
    ratio = pred["tf_pred"] / np.maximum(pred["tf_1d"], 1e-6)
    err = np.max(np.abs(ratio - np.exp(pred["g"].astype(np.float64))))
    if err > 1e-4:
        raise SystemExit(f"exp(g) does not rebuild the prediction (max abs {err})")

    nz, nx = sec.vs.shape
    dz = 1.0
    meta = {
        "sample_id": pick["sample_id"],
        "split": "blind",
        "feature_name": row["feature_name"],
        "lx_m": int(sec.lx),
        "depth_m": float(row["depth_m"]),
        "dz_m": dz,
        "dx_m": 1.0,
        "z_axis": "row 0 is the ground surface; z_m is the layer-center depth, positive downward",
        "x_axis": "column j of vs is x = j metres on the interior; pads were dropped before this grid",
        "stations": "every 5 m of the interior surface, including the edges",
        "n_stations": int(pred["x_station_m"].size),
        "n_freq": int(freq.size),
        "formula": "tf_pred = tf_1d * exp(g) ; tf_2d = tf_1d * exp(g_true)",
        "g_base": "e",
        "quantities": "linear |surface / outcrop|, one acceleration Ricker, fixed 1310.72 s window",
        "f1_hz": float(sec.f1_hz),
        "f2_hz": float(sec.f2_hz),
        "damping_family": str(sec.damping),
        "feature_x_frac": float(row["feature_x_frac"]),
        "feature_width_m": float(row["feature_width_m"]),
        "feature_dh_m": float(row["feature_dh_m"]),
        "trough_depth_m": float(row["trough_depth_m"]) if row["trough_depth_m"] else None,
        "model_dir": str(args.model_dir),
        "model_stage": int(cfg["stage"]),
        "model_target": cfg["target"],
        "model_epoch": None,
        "skill_mid_0.4_2.5Hz": pick["skill_mid"],
        "log_l1_mid": pick["log_l1_mid"],
        "log_l1_1d_mid": pick["log_l1_1d_mid"],
        "gap_mid": pick["gap_mid"],
        "note": (
            "Held out: blind stratum, excluded from training and from early stopping. "
            "This checkpoint is stage 0 (500 m IID and OOD runs). "
            "It has not been trained on variable-width sections."
        ),
    }
    ckpt = torch.load(args.model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    meta["model_epoch"] = int(ckpt.get("epoch", -1))

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"{pick['sample_id']}_blind.npz"
    np.savez_compressed(
        dest,
        vs=np.asarray(sec.vs, dtype=np.float32),
        z_m=((np.arange(nz) + 0.5) * dz).astype(np.float32),
        x_grid_m=np.arange(nx, dtype=np.float32),
        freq_hz=freq.astype(np.float32),
        x_station_m=pred["x_station_m"],
        tf_1d=pred["tf_1d"],
        tf_2d=pred["tf_2d"],
        tf_pred=pred["tf_pred"],
        g=pred["g"],
        g_true=pred["g_true"].astype(np.float32),
        meta_json=np.array(json.dumps(meta)),
    )
    (args.out / f"{pick['sample_id']}_blind.json").write_text(json.dumps(meta, indent=2))
    (args.out / "blind_scores.json").write_text(json.dumps(scored, indent=2))
    print(f"wrote {dest}", flush=True)
    print(json.dumps({k: meta[k] for k in ("sample_id", "feature_name", "lx_m", "formula", "skill_mid_0.4_2.5Hz", "gap_mid")}, indent=2))


if __name__ == "__main__":
    main()
