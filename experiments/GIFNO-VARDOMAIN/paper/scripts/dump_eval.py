#!/usr/bin/env python3
"""Score stage 1, stage 2, and stage 2 with modes 64:80 zeroed.

Writes paper/dump/records.csv, stations.npz, g_mean.npz, and
three_layer_example.npz. The zeroed pass is three-layer only: that is the
ablation the drift figure uses. Train-split manifold rows are not scored.

    python dump_eval.py
    python dump_eval.py --limit 2
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from _entry import enter

enter(_HERE)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import KIND_SECTION, NEAR_1D_LOG_STD, Cache, WindowSet, eval_records, lateral_log_std
from evaluate import load_model
from provenance import full_stamp
from metrics import ROCK_VS, band_mask, pearson, soil_rows
from paths import CACHE, DUMP, MANIFEST, STAGE1, STAGE2

COLUMNS = (
    "pass", "family", "width", "record_id", "split", "stratum", "damping", "group_id",
    "near_1d", "cov", "lateral_log_std", "log_impedance", "log_thickness", "log_f0", "log_f2",
    "mean_skill_low", "mean_skill_mid", "mean_skill_high", "mean_pearson_mid",
    "center_log_l1_low", "center_log_l1_mid", "center_log_l1_high",
    "center_column_l1_low", "center_column_l1_mid", "center_column_l1_high",
    "center_skill_mid", "g_pred_abs_mid", "g_true_abs_mid",
)


def _manifest() -> dict[str, dict]:
    with MANIFEST.open() as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def _cov(family: str, index: int, manifest: dict[str, dict], sample: str) -> float:
    if sample in manifest:
        return float(manifest[sample]["vs_cov"])
    return float("nan")


def _describe(cache: Cache, rec: dict, manifest: dict[str, dict]) -> dict:
    if rec["kind"] == "section":
        sec = cache.sections[int(rec["index"])]
        vs = np.asarray(sec.vs, dtype=np.float64)
        sample = sec.name.split("manifold__")[-1]
        row = manifest.get(sample, {})
        f0 = float(row.get("f0_flat_hz") or sec.f1_hz)
        f2 = float(row.get("rayleigh_f2_hz") or sec.f2_hz)
        meta = {
            "record_id": sample,
            "split": sec.split,
            "stratum": sec.stratum,
            "damping": sec.damping or row.get("damping_family", ""),
            "group_id": sec.group_id,
            "family": "manifold",
        }
    else:
        camp, j = cache.campaign(int(rec["index"]))
        n = int(camp.n_active[j])
        vs = np.asarray(camp.vs[j, :n], dtype=np.float64)
        f0 = float(camp.f1[j])
        f2 = float(camp.f2[j])
        meta = {
            "record_id": f"{camp.name}:{int(j)}",
            "split": str(camp.split[j]),
            "stratum": "",
            "damping": "",
            "group_id": str(int(camp.group[j])),
            "family": camp.name,
        }
    soil = soil_rows(vs)
    col = vs[:, vs.shape[1] // 2]
    rock = np.flatnonzero(col >= ROCK_VS)
    contact = int(rock[0]) if rock.size else max(col.size - 1, 1)
    meta.update(
        {
            "near_1d": int(lateral_log_std(soil) < NEAR_1D_LOG_STD),
            "cov": _cov(meta["family"], int(rec["index"]), manifest, meta["record_id"]),
            "lateral_log_std": float(lateral_log_std(soil)),
            "log_impedance": float(np.log(col[min(contact, col.size - 1)] / max(col[0], 1.0))),
            "log_thickness": float(np.log(max(contact, 1))),
            "log_f0": float(np.log(max(f0, 1e-6))),
            "log_f2": float(np.log(max(f2, 1e-6))),
        }
    )
    return meta


def _jobs(cache: Cache, limit: int | None) -> list[dict]:
    runs = eval_records(cache, "test", families=("iid", "ood_dipping", "ood_three_layer"), positions=1)
    sections = []
    for split in ("val", "blind"):
        sections.extend(eval_records(cache, split, families=("manifold",), positions=1))
    if limit is None:
        return runs + sections
    kept = []
    seen: dict[str, int] = {}
    for rec in runs + sections:
        family = rec.get("family", "manifold")
        if seen.get(family, 0) >= limit:
            continue
        seen[family] = seen.get(family, 0) + 1
        kept.append(rec)
    return kept


def _pass(
    name: str,
    model,
    jobs: list[dict],
    described: dict,
    cache: Cache,
    device,
    target: str,
    *,
    n_keep: int | None,
    save_stations: bool,
    amp: bool,
) -> tuple[list[dict], dict, dict]:
    freq = np.asarray(cache.freq, dtype=np.float64)
    masks = {band: band_mask(freq, band) for band in ("low", "mid", "high")}
    loader = DataLoader(
        WindowSet(cache, "metric", jobs, stage=0, seed=0, mirror=False),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    buckets: dict[tuple, dict] = {}
    curves: dict[str, dict] = {}
    dist, skill, family, width = [], [], [], []
    model.eval()
    with torch.no_grad():
        for step, (x, query, tf, prior, valid, info, x_abs) in enumerate(loader):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                raw = model(x.to(device), query.to(device), n_keep=n_keep).float()
            prior_d = prior.to(device)
            log_pred = (raw + prior_d).cpu().numpy() if target == "residual" else raw.cpu().numpy()
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
            flat_valid = valid_np.ravel()
            keep = np.flatnonzero(flat_valid)
            for src in keep:
                inf = flat_info[src]
                key = (int(inf[0]), int(inf[1]), int(inf[2]), int(inf[3]))
                bucket = buckets.setdefault(key, {"mid_skill": [], "low_skill": [], "high_skill": [], "pearson": [], "center_dist": 1e9})
                pred_i = flat_pred[src]
                true_i = flat_true[src]
                prior_i = flat_prior[src]
                g_i = flat_g[src]
                local = float(flat_local[src])
                w = int(inf[2])
                for band in ("low", "mid", "high"):
                    model_l1 = float(np.mean(np.abs(pred_i[masks[band]] - true_i[masks[band]])))
                    column_l1 = float(np.mean(np.abs(prior_i[masks[band]] - true_i[masks[band]])))
                    bucket[f"{band}_model"] = bucket.get(f"{band}_model", [])
                    bucket[f"{band}_column"] = bucket.get(f"{band}_column", [])
                    bucket[f"{band}_model"].append(model_l1)
                    bucket[f"{band}_column"].append(column_l1)
                    bucket[f"{band}_skill"].append(1.0 - model_l1 / max(column_l1, 1e-6))
                mid = masks["mid"]
                bucket["pearson"].append(pearson(np.exp(pred_i[mid]), np.exp(true_i[mid])))
                dist_edge = min(local, w - local)
                if save_stations:
                    dist.append(dist_edge)
                    skill.append(bucket["mid_skill"][-1])
                    family.append(int(inf[4]))
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
                    }
            if step % 50 == 0:
                print(f"  {name} batch {step}", flush=True)
    rows = []
    for key, bucket in buckets.items():
        kind, index, w, _start = key
        rec_kind = "section" if kind == KIND_SECTION else "run"
        meta = described[(rec_kind, index)]
        center = bucket["center"]
        row = dict(meta)
        row.update(
            {
                "pass": name,
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
            }
        )
        rows.append(row)
        if meta["family"] == "ood_three_layer" and int(w) == 500:
            curves[meta["record_id"]] = center
    stations = {"dist_edge": np.asarray(dist), "skill_mid": np.asarray(skill), "family": np.asarray(family), "width": np.asarray(width)}
    return rows, curves, stations


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Cache(args.cache)
    manifest = _manifest()
    jobs = _jobs(cache, args.limit)
    described = {}
    for rec in jobs:
        described.setdefault((rec["kind"], int(rec["index"])), _describe(cache, rec, manifest))
    print(f"device {device} windows {len(jobs)}", flush=True)
    specs = (
        ("stage2", STAGE2, None, True),
        ("stage1", STAGE1, None, False),
        ("stage2_zeroed", STAGE2, 64, False),
    )
    rows = []
    curves = {}
    stations = None
    stamps = {}
    freq = np.asarray(cache.freq, dtype=np.float64)
    for name, model_dir, n_keep, save_stations in specs:
        model, cfg = load_model(model_dir, device)
        if name in ("stage1", "stage2"):
            stamps[name] = full_stamp(model_dir, device, bool(cfg.get("amp", True)))
        use = jobs
        if name == "stage2_zeroed":
            use = [rec for rec in jobs if rec.get("family") == "ood_three_layer"]
        print(f"{name} windows {len(use)} n_keep {n_keep}", flush=True)
        got, got_curves, got_stations = _pass(
            name, model, use, described, cache, device, cfg.get("target", "residual"),
            n_keep=n_keep, save_stations=save_stations, amp=bool(cfg.get("amp", True)),
        )
        rows.extend(got)
        curves[name] = got_curves
        if save_stations:
            stations = got_stations
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    DUMP.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame[list(COLUMNS)].to_csv(DUMP / "records.csv", index=False, na_rep="")
    if stations is not None:
        np.savez(DUMP / "stations.npz", **stations)
    ids = sorted(set(curves.get("stage2", {})) & set(curves.get("stage1", {})) & set(curves.get("stage2_zeroed", {})))
    if ids:
        def _stack(pass_name: str) -> np.ndarray:
            return np.stack([curves[pass_name][i]["g"] for i in ids])

        np.savez(
            DUMP / "g_mean.npz",
            freq=freq,
            stage1=_stack("stage1").mean(0),
            stage2=_stack("stage2").mean(0),
            stage2_zeroed=_stack("stage2_zeroed").mean(0),
            record_id=np.array(ids),
        )
        stage2_rows = frame[(frame["pass"] == "stage2") & (frame["record_id"].isin(ids)) & (frame["width"] == 500)]
        worst = stage2_rows.sort_values("center_skill_mid").iloc[0]
        rid = str(worst["record_id"])
        prior = curves["stage2"][rid]["log_prior"]
        true = curves["stage2"][rid]["log_true"]
        np.savez(
            DUMP / "three_layer_example.npz",
            freq=freq,
            record_id=np.array(rid),
            opensees=np.exp(true),
            column=np.exp(prior),
            stage1=np.exp(prior + curves["stage1"][rid]["g"]),
            stage2=np.exp(prior + curves["stage2"][rid]["g"]),
            g_true=true - prior,
            g_stage1=curves["stage1"][rid]["g"],
            g_stage2=curves["stage2"][rid]["g"],
        )
    (DUMP / "meta.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "n_rows": int(len(frame)),
                "limit": args.limit,
                "stage2": stamps["stage2"],
                "stage1": stamps["stage1"],
            },
            indent=2,
        )
    )
    print(f"wrote {DUMP} rows {len(frame)}", flush=True)


if __name__ == "__main__":
    main()
