#!/usr/bin/env python3
"""Dense lateral acceptance test.

Query g every 1 m. At 2.5 Hz, the range of model |TF| inside each gap between
adjacent 15 m recorders must not exceed the OpenSees |TF| difference of those
two recorders. A field that only interpolates the stations ties the bound.

    python lateral_field.py --checkpoint checkpoints/vardomain/coord_reseed/stage2 --cache data/vardomain/cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import Cache, WindowSet, eval_records  # noqa: E402
from encoding import CONTEXT_M  # noqa: E402
from precision_diagnose import load_raw  # noqa: E402

STATION_SPACING_M = 15.0


def _gap_ranges(x_m: np.ndarray, tf_model: np.ndarray, tf_osees: np.ndarray) -> list[dict]:
    rows = []
    for i in range(x_m.size - 1):
        lo, hi = float(x_m[i]), float(x_m[i + 1])
        if hi - lo > STATION_SPACING_M + 1.0:
            continue
        inside = (np.arange(tf_model.size) >= lo) & (np.arange(tf_model.size) <= hi)
        if int(inside.sum()) < 2:
            continue
        span = tf_model[inside]
        model_range = float(span.max() - span.min())
        osees_range = float(abs(tf_osees[i + 1] - tf_osees[i]))
        rows.append({
            "x0_m": lo,
            "x1_m": hi,
            "model_range": model_range,
            "opensees_range": osees_range,
            "ok": bool(model_range <= osees_range + 1e-6),
        })
    return rows


def _dense_tf(model, x: torch.Tensor, prior_station: np.ndarray, x_local: np.ndarray, freq: np.ndarray, k25: int):
    width = int(round(float(x_local.max()) + 5))
    cols = torch.arange(CONTEXT_M, CONTEXT_M + width, device=x.device, dtype=torch.float32)
    with torch.no_grad():
        g = model(x, cols.unsqueeze(0))[0, :, k25].float().cpu().numpy()
    metres = np.arange(width, dtype=np.float64)
    order = np.argsort(x_local)
    prior = np.interp(metres, x_local[order], prior_station[order])
    return metres, np.exp(prior + g)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--dense-npz", type=Path, default=None, help="optional dense_labels npz to overlay")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Cache(args.cache)
    freq = np.asarray(cache.freq, dtype=np.float64)
    k25 = int(np.argmin(np.abs(freq - 2.5)))
    model, _cfg = load_raw(args.checkpoint, device)
    records = [
        row for row in eval_records(cache, "test", families=("iid",), positions=1) if int(row["width"]) == 500
    ][: args.limit]
    windows = WindowSet(cache, "metric", records, stage=0, seed=0, mirror=False)
    reports = []
    plotted = False
    for i in range(len(windows)):
        x, query, tf, prior, valid, info, _xa = windows[i]
        idx = np.flatnonzero(valid.numpy())
        order = np.argsort(query.numpy()[idx])
        idx = idx[order]
        x_local = query.numpy()[idx] - CONTEXT_M
        x = x.unsqueeze(0).to(device)
        metres, tf_model = _dense_tf(model, x, prior.numpy()[idx, k25], x_local, freq, k25)
        tf_osees = np.exp(tf.numpy()[idx, k25])
        gaps = _gap_ranges(x_local, tf_model, tf_osees)
        reports.append({
            "index": int(info[1]),
            "n_gaps": len(gaps),
            "n_fail": int(sum(not row["ok"] for row in gaps)),
            "max_model_range": max((row["model_range"] for row in gaps), default=0.0),
            "max_opensees_range": max((row["opensees_range"] for row in gaps), default=0.0),
            "gaps": gaps,
        })
        if not plotted:
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            ax.plot(metres, tf_model, color="#4C72B0", lw=1.0, label="model, 1 m")
            ax.plot(x_local, tf_osees, "o", color="#D55E00", ms=4, label="OpenSees, 15 m")
            if args.dense_npz is not None and args.dense_npz.is_file():
                dense = np.load(args.dense_npz)
                f_dense = np.asarray(dense["freq"], dtype=np.float64)
                kd = int(np.argmin(np.abs(f_dense - 2.5)))
                g_dense = np.log(np.maximum(dense["tf"][:, kd], 1e-8)) - np.log(np.maximum(dense["tf1d"][:, kd], 1e-8))
                ax.plot(np.asarray(dense["x_m"]), np.exp(np.log(np.maximum(dense["tf1d"][:, kd], 1e-8)) + g_dense),
                        color="#009E73", lw=1.0, label="OpenSees, dense")
            ax.set_xlabel("Metres from the window origin")
            ax.set_ylabel("|TF| at 2.5 Hz")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(args.out / "lateral_field.png", dpi=160)
            plt.close(fig)
            plotted = True
    n_fail = sum(row["n_fail"] for row in reports)
    summary = {"n_runs": len(reports), "n_fail_gaps": n_fail, "passed": n_fail == 0, "runs": reports}
    path = args.out / "lateral_field.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"n_runs": summary["n_runs"], "n_fail_gaps": n_fail, "passed": summary["passed"]}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
