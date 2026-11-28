#!/usr/bin/env python3
"""Center-station scores from evaluate.predict, not the saved |TF| curves.

Test windows at 500 m for the horizontal and dipping families supply the
0.4–2.5 Hz and 0.4–3.3 Hz log-L1. Validation windows at 500 m for those
families and three-layer supply predicted |g| and mid-band skill for the
threshold. One stage-2 load covers both.

    python stage2_centers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE))
from _entry import enter

enter(_HERE)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import FAMILY_NAME, Cache, WindowSet, eval_records
from evaluate import load_model, predict
from metrics import pearson
from paths import CACHE, DUMP, FIG, STAGE2
from provenance import full_stamp, write_sidecar

OUT = DUMP / "stage2_centers.csv"
JOBS = (
    ("test", ("iid", "ood_dipping")),
    ("val", ("iid", "ood_dipping", "ood_three_layer")),
)


def _records(cache: Cache, split: str, families: tuple[str, ...]) -> list[dict]:
    rows = eval_records(cache, split, families=families, positions=1)
    return [row for row in rows if int(row["width"]) == 500]


def _band_mean(err: np.ndarray, freq: np.ndarray, lo: float, hi: float) -> float:
    mask = (freq >= lo) & (freq <= hi)
    return float(np.mean(np.abs(err[mask])))


def _centers(model, cache: Cache, records: list[dict], device, amp: bool, target: str, split: str) -> tuple[list[dict], dict[str, dict[int, np.ndarray]]]:
    loader = DataLoader(
        WindowSet(cache, "metric", records, stage=0, seed=0, mirror=False),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    freq = np.asarray(cache.freq, dtype=np.float64)
    best: dict[tuple[int, int], tuple[float, dict]] = {}
    spectra: dict[str, dict[int, np.ndarray]] = {}
    seen = 0
    for log_pred, log_true, log_prior, info, _x_abs, x_local in predict(model, loader, device, amp, target):
        for i in range(log_pred.shape[0]):
            width = int(info[i, 2])
            index = int(info[i, 1])
            local = float(x_local[i])
            dist = abs(local - 0.5 * width)
            key = (index, width)
            if key in best and dist >= best[key][0]:
                continue
            pred = np.asarray(log_pred[i], dtype=np.float64)
            true = np.asarray(log_true[i], dtype=np.float64)
            prior = np.asarray(log_prior[i], dtype=np.float64)
            g = pred - prior
            mid = (freq >= 0.4) & (freq <= 2.5)
            model_mid = _band_mean(pred - true, freq, 0.4, 2.5)
            column_mid = _band_mean(prior - true, freq, 0.4, 2.5)
            _camp, local_run = cache.campaign(index)
            if split == "test":
                spectra.setdefault(FAMILY_NAME.get(int(info[i, 4]), "section"), {})[int(local_run)] = np.exp(pred).astype(np.float32)
            best[key] = (
                dist,
                {
                    "split": split,
                    "family": FAMILY_NAME.get(int(info[i, 4]), "section"),
                    "index": index,
                    "local_run": int(local_run),
                    "group": int(info[i, 5]),
                    "width": width,
                    "model_log_l1_mid": model_mid,
                    "column_log_l1_mid": column_mid,
                    "skill_mid": float(1.0 - model_mid / max(column_mid, 1e-6)),
                    "model_log_l1_0p4_3p3": _band_mean(pred - true, freq, 0.4, 3.3),
                    "column_log_l1_0p4_3p3": _band_mean(prior - true, freq, 0.4, 3.3),
                    "g_pred_abs_mid": float(np.mean(np.abs(g[mid]))),
                    "pearson": pearson(np.exp(pred), np.exp(true)),
                },
            )
        seen += log_pred.shape[0]
        if seen % 4000 < log_pred.shape[0]:
            print(f"  {split} stations {seen}", flush=True)
    return [row for _dist, row in best.values()], spectra


def _saved_pearson(frame: pd.DataFrame) -> dict[str, float]:
    from center_curves import FAMILIES, test_frame

    out = {}
    for name, family in FAMILIES.items():
        saved = test_frame(family)
        npy = np.asarray(np.load(FIG / "stage2" / f"test_model_{family}.npy"), dtype=np.float64)
        got = frame[(frame["split"] == "test") & (frame["family"] == family)]
        merged = got.merge(
            pd.DataFrame({"local_run": saved["run"], "saved_pearson": [
                pearson(npy[i], saved["opensees"][i]) for i in range(saved["run"].size)
            ]}),
            on="local_run",
            how="inner",
        )
        out[name] = {
            "n": int(len(merged)),
            "eval_median": float(merged["pearson"].median()) if len(merged) else None,
            "saved_median": float(merged["saved_pearson"].median()) if len(merged) else None,
        }
        if len(merged):
            out[name]["delta"] = float(out[name]["eval_median"] - out[name]["saved_median"])
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("all", "test", "val"), default="all")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    model, cfg = load_model(STAGE2, device)
    stamp = full_stamp(STAGE2, device, bool(cfg.get("amp", True)))
    cache = Cache(CACHE)
    rows: list[dict] = []
    jobs = JOBS if args.split == "all" else tuple(job for job in JOBS if job[0] == args.split)
    for split, families in jobs:
        records = _records(cache, split, families)
        print(f"{split} windows {len(records)}", flush=True)
        got, spectra = _centers(model, cache, records, device, bool(cfg.get("amp", True)), cfg.get("target", "residual"), split)
        rows.extend(got)
        for family, curves in spectra.items():
            if not curves:
                continue
            runs = np.array(sorted(curves), dtype=np.int64)
            stack = np.stack([curves[int(run)] for run in runs])
            dest = DUMP / f"stage2_center_tf_{family}.npz"
            np.savez_compressed(dest, run=runs, tf=stack)
            write_sidecar(dest, {**stamp, "family": family, "n": int(runs.size), "rows": "local run id"})
    DUMP.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if args.split != "all" and OUT.is_file():
        old = pd.read_csv(OUT)
        old = old[old["split"] != args.split]
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(OUT, index=False)
    summary = []
    for family in ("iid", "ood_dipping"):
        part = frame[(frame["split"] == "test") & (frame["family"] == family)]
        summary.append(
            {
                "family": family,
                "n": int(len(part)),
                "model_mid_mean": float(part["model_log_l1_mid"].mean()),
                "model_mid_median": float(part["model_log_l1_mid"].median()),
                "column_mid_mean": float(part["column_log_l1_mid"].mean()),
                "column_mid_median": float(part["column_log_l1_mid"].median()),
                "model_wide_mean": float(part["model_log_l1_0p4_3p3"].mean()),
                "model_wide_median": float(part["model_log_l1_0p4_3p3"].median()),
                "column_wide_mean": float(part["column_log_l1_0p4_3p3"].mean()),
                "column_wide_median": float(part["column_log_l1_0p4_3p3"].median()),
            }
        )
    pear = _saved_pearson(frame) if (FIG / "stage2" / "test_model_iid.npy").is_file() else {}
    (DUMP / "stage2_centers_summary.json").write_text(
        __import__("json").dumps({"provenance": stamp, "rows": summary, "pearson_vs_saved_spectra": pear}, indent=2)
    )
    print(__import__("json").dumps({"rows": summary, "pearson_vs_saved_spectra": pear}, indent=2))


if __name__ == "__main__":
    main()
