#!/usr/bin/env python3
"""Score the four-checkpoint mean and write the dumps the paper figures read.

The prediction is the mean of g from P128, P128s1, P128n128, and P128c, in
float32. P128c is the epoch-30 checkpoint. Stage 1 and stage 2 rows already in
records.csv are left in place. This adds pass ``ensemble``.

Cosine-bowl troughs are not in this pass. ``Cache`` drops them in
``trough_sample_ids`` before ``eval_records`` builds the windows, so the
ensemble rows are the scored set. Stage 1 and stage 2 dumps still contain
those troughs from the earlier cache; compare the three passes on the shared
record ids.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE))
from _entry import enter

enter(_HERE)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import KIND_SECTION, Cache, WindowSet
from diag_forward import ROOT, load_fp32
from dump_eval import COLUMNS, _describe, _jobs, _manifest
from metrics import band_mask, pearson
from paths import CACHE, DUMP

MODELS = ("P128", "P128s1", "P128n128", "P128c")
sys.path.insert(0, str(_HERE.parents[1]))


def _load_all(device: torch.device):
    models = []
    stamps = []
    for name in MODELS:
        model, stamp = load_fp32(ROOT / name, device)
        models.append(model)
        stamps.append({
            "name": name,
            "epoch": int(stamp["epoch"]),
            "narrow": None if stamp.get("narrow") is None else int(stamp["narrow"]),
            "contact_features": bool(stamp.get("contact_features")),
            "seed": None if stamp.get("seed") is None else int(stamp["seed"]),
        })
        print(f"loaded {name} epoch {stamp['epoch']}", flush=True)
    return models, stamps


def _mean_g(models, x, query, device) -> torch.Tensor:
    total = None
    for model in models:
        pred = model(x, query)
        total = pred if total is None else total + pred
    return total / len(models)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Cache(CACHE)
    manifest = _manifest()
    jobs = _jobs(cache, None)
    described = {}
    for rec in jobs:
        described.setdefault((rec["kind"], int(rec["index"])), _describe(cache, rec, manifest))
    print(f"device {device} windows {len(jobs)}", flush=True)
    models, stamps = _load_all(device)
    freq = np.asarray(cache.freq, dtype=np.float64)
    masks = {band: band_mask(freq, band) for band in ("low", "mid", "high")}
    loader = DataLoader(
        WindowSet(cache, "metric", jobs, stage=0, seed=0, mirror=False),
        batch_size=8, shuffle=False, num_workers=8, pin_memory=device.type == "cuda",
        persistent_workers=True, prefetch_factor=2,
    )
    buckets: dict[tuple, dict] = {}
    curves: dict[str, dict] = {}
    centers: dict[str, dict[int, np.ndarray]] = {}
    dist, skill, family, width = [], [], [], []
    bin_sum: dict[str, dict[str, float]] = {}
    bin_n: dict[str, int] = {}
    t0 = time.perf_counter()
    seen = 0
    with torch.no_grad():
        for step, (x, query, tf, prior, valid, info, x_abs) in enumerate(loader):
            raw = _mean_g(models, x.to(device, non_blocking=True), query.to(device, non_blocking=True), device).float()
            log_pred = (raw + prior.to(device)).cpu().numpy()
            g = raw.cpu().numpy()
            log_true = tf.numpy()
            log_prior = prior.numpy()
            info_np = info.numpy()
            valid_np = valid.numpy()
            x_local = x_abs.numpy() - info_np[:, 3:4]
            flat_pred = log_pred.reshape(-1, log_pred.shape[-1])
            flat_true = log_true.reshape(-1, log_true.shape[-1])
            flat_prior = log_prior.reshape(-1, log_prior.shape[-1])
            flat_g = g.reshape(-1, g.shape[-1])
            flat_info = np.repeat(info_np, valid_np.shape[1], axis=0)
            flat_local = x_local.reshape(-1)
            keep = np.flatnonzero(valid_np.ravel())
            for src in keep:
                inf = flat_info[src]
                key = (int(inf[0]), int(inf[1]), int(inf[2]), int(inf[3]))
                bucket = buckets.setdefault(key, {"center_dist": 1e9})
                pred_i = flat_pred[src]
                true_i = flat_true[src]
                prior_i = flat_prior[src]
                g_i = flat_g[src]
                local = float(flat_local[src])
                w = int(inf[2])
                fam_code = int(inf[4])
                fam_name = described[("section" if int(inf[0]) == KIND_SECTION else "run", int(inf[1]))]["family"]
                bin_key = f"{fam_name}_w{w}"
                slot = bin_sum.setdefault(bin_key, {})
                for band in ("low", "mid", "high"):
                    model_l1 = float(np.mean(np.abs(pred_i[masks[band]] - true_i[masks[band]])))
                    column_l1 = float(np.mean(np.abs(prior_i[masks[band]] - true_i[masks[band]])))
                    bucket.setdefault(f"{band}_model", []).append(model_l1)
                    bucket.setdefault(f"{band}_column", []).append(column_l1)
                    bucket.setdefault(f"{band}_skill", []).append(1.0 - model_l1 / max(column_l1, 1e-6))
                    slot[f"log_l1_{band}"] = slot.get(f"log_l1_{band}", 0.0) + model_l1
                    slot[f"column_{band}"] = slot.get(f"column_{band}", 0.0) + column_l1
                mid = masks["mid"]
                pear = pearson(np.exp(pred_i[mid]), np.exp(true_i[mid]))
                bucket.setdefault("pearson", []).append(pear)
                slot["pearson_tr"] = slot.get("pearson_tr", 0.0) + pear
                bin_n[bin_key] = bin_n.get(bin_key, 0) + 1
                dist_edge = min(local, w - local)
                dist.append(dist_edge)
                skill.append(bucket["mid_skill"][-1])
                family.append(fam_code)
                width.append(w)
                center_dist = abs(local - 0.5 * w)
                if center_dist < bucket["center_dist"]:
                    bucket["center_dist"] = center_dist
                    bucket["center"] = {
                        "model": {band: bucket[f"{band}_model"][-1] for band in ("low", "mid", "high")},
                        "column": {band: bucket[f"{band}_column"][-1] for band in ("low", "mid", "high")},
                        "g_pred": float(np.mean(np.abs(g_i[mid]))),
                        "g_true": float(np.mean(np.abs(true_i[mid] - prior_i[mid]))),
                        "g": g_i.astype(np.float32),
                        "log_true": true_i.astype(np.float32),
                        "log_prior": prior_i.astype(np.float32),
                        "tf": np.exp(pred_i).astype(np.float32),
                    }
            seen += int(keep.size)
            if step % 40 == 0:
                print(f"  batch {step} stations {seen} {time.perf_counter() - t0:.0f}s", flush=True)
    rows = []
    for key, bucket in buckets.items():
        kind, index, w, _start = key
        rec_kind = "section" if kind == KIND_SECTION else "run"
        meta = described[(rec_kind, index)]
        center = bucket["center"]
        row = dict(meta)
        row.update({
            "pass": "ensemble",
            "width": int(w),
            "mean_skill_low": float(np.mean(bucket["low_skill"])),
            "mean_skill_mid": float(np.mean(bucket["mid_skill"])),
            "mean_skill_high": float(np.mean(bucket["high_skill"])),
            "mean_pearson_mid": float(np.mean(bucket["pearson"])),
            "center_log_l1_low": center["model"]["low"],
            "center_log_l1_mid": center["model"]["mid"],
            "center_log_l1_high": center["model"]["high"],
            "center_column_l1_low": center["column"]["low"],
            "center_column_l1_mid": center["column"]["mid"],
            "center_column_l1_high": center["column"]["high"],
            "center_skill_mid": float(1.0 - center["model"]["mid"] / max(center["column"]["mid"], 1e-6)),
            "g_pred_abs_mid": center["g_pred"],
            "g_true_abs_mid": center["g_true"],
        })
        rows.append(row)
        if meta["family"] == "ood_three_layer" and int(w) == 500:
            curves[meta["record_id"]] = center
        if meta["family"] in ("iid", "ood_dipping") and int(w) == 500 and meta["split"] == "test":
            _camp, local_run = cache.campaign(index)
            centers.setdefault(meta["family"], {})[int(local_run)] = center["tf"]
    elapsed = time.perf_counter() - t0
    DUMP.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(DUMP / "records.csv") if (DUMP / "records.csv").is_file() else pd.DataFrame(columns=list(COLUMNS))
    previous = previous[previous["pass"] != "ensemble"]
    frame = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame[list(COLUMNS)].to_csv(DUMP / "records.csv", index=False, na_rep="")
    np.savez(DUMP / "ensemble_stations.npz", dist_edge=np.asarray(dist), skill_mid=np.asarray(skill), family=np.asarray(family), width=np.asarray(width))
    for family_name, spec in centers.items():
        runs = np.array(sorted(spec), dtype=np.int64)
        np.savez(DUMP / f"ensemble_center_tf_{family_name}.npz", run=runs, tf=np.stack([spec[int(r)] for r in runs]))
    bins = {}
    for key, slot in bin_sum.items():
        n = bin_n[key]
        block = {"n_stations": n}
        for band in ("low", "mid", "high"):
            log_l1 = slot[f"log_l1_{band}"] / n
            column = slot[f"column_{band}"] / n
            block[f"log_l1_{band}"] = log_l1
            block[f"skill_{band}"] = 1.0 - log_l1 / max(column, 1e-6)
        block["pearson_tr"] = slot["pearson_tr"] / n
        bins[key] = block
    (DUMP / "ensemble_eval_test.json").write_text(json.dumps({"bins": bins, "models": stamps, "seconds": elapsed}, indent=2))
    example = DUMP / "three_layer_example.npz"
    if example.is_file() and curves:
        blob = dict(np.load(example))
        rid = str(blob["record_id"])
        if rid in curves:
            prior = curves[rid]["log_prior"]
            blob["ensemble"] = np.exp(prior + curves[rid]["g"])
            blob["g_ensemble"] = curves[rid]["g"]
            np.savez(example, **blob)
    g_path = DUMP / "g_mean.npz"
    if g_path.is_file() and curves:
        blob = dict(np.load(g_path))
        ids = [str(i) for i in np.asarray(blob["record_id"])]
        if all(i in curves for i in ids):
            blob["ensemble"] = np.stack([curves[i]["g"] for i in ids]).mean(0)
            np.savez(g_path, **blob)
    (DUMP / "ensemble_runtime.json").write_text(json.dumps({
        "models": stamps,
        "windows": len(jobs),
        "stations": seen,
        "seconds": elapsed,
        "seconds_per_window": elapsed / max(len(jobs), 1),
        "forward": "float32 mean of g",
    }, indent=2))
    print(f"ensemble rows {len(rows)} in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
