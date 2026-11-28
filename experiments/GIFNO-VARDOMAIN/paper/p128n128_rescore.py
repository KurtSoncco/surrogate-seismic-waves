#!/usr/bin/env python3
"""Paper box and band figures for P128n128.

The published files stay on stage 2, epoch 53. This writes a new box, the
0.1–10 Hz center scores, and the test-station band and width tables under
paper/p128n128/. The forward is the checkpoint's bfloat16 autocast, which is
the forward the paper figures use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
VARDOMAIN = HERE.parent
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(VARDOMAIN))

from bootstrap_ci import bootstrap_cluster_statistic
from center_curves import _f0
from data import FAMILY_NAME, STAGE_UNIFIED, Cache, WindowSet, eval_records
from evaluate import evaluate, load_model, predict
from metrics import anderson, pearson
from paths import CACHE, FIG
from plotting import apply_publication_style, panel_letter, save_figure

MODEL = VARDOMAIN.parents[1] / "checkpoints" / "vardomain" / "v2_ablations" / "P128n128"
OUT = HERE / "p128n128"
METHODS = ("Column", "Stage 2", "P128n128", "Pretell", "Passeri", "Dmult", "Toro")
BOX_COLOR = {
    "Column": "#0072B2",
    "Stage 2": "#E69F00",
    "P128n128": "#D55E00",
    "Pretell": "#009E73",
    "Passeri": "#CC79A7",
    "Dmult": "#332288",
    "Toro": "#F0E442",
}
FAMILIES = {"Horizontal": "iid", "Dipping": "ood_dipping"}


def _band_mean(err: np.ndarray, freq: np.ndarray, lo: float, hi: float) -> float:
    mask = (freq >= lo) & (freq <= hi)
    return float(np.mean(np.abs(err[mask])))


def _centers(model, cache: Cache, device) -> dict[str, dict[int, np.ndarray]]:
    records = [
        row for row in eval_records(cache, "test", families=("iid", "ood_dipping"), positions=1)
        if int(row["width"]) == 500
    ]
    loader = DataLoader(
        WindowSet(cache, "metric", records, stage=STAGE_UNIFIED, seed=0, mirror=False),
        batch_size=8, shuffle=False, num_workers=0,
    )
    best: dict[tuple[int, int], tuple[float, str, int, np.ndarray]] = {}
    seen = 0
    for log_pred, _true, _prior, info, _x_abs, x_local in predict(model, loader, device, True, "residual"):
        for i in range(log_pred.shape[0]):
            width = int(info[i, 2])
            index = int(info[i, 1])
            dist = abs(float(x_local[i]) - 0.5 * width)
            key = (index, width)
            if key in best and dist >= best[key][0]:
                continue
            family = FAMILY_NAME.get(int(info[i, 4]), "section")
            _camp, local_run = cache.campaign(index)
            best[key] = (dist, family, int(local_run), np.exp(np.asarray(log_pred[i], dtype=np.float64)).astype(np.float32))
        seen += log_pred.shape[0]
        if seen % 4000 < log_pred.shape[0]:
            print(f"  center stations {seen}", flush=True)
    spectra: dict[str, dict[int, np.ndarray]] = {}
    for _dist, family, run, curve in best.values():
        spectra.setdefault(family, {})[run] = curve
    return spectra


def _align(curves: dict[int, np.ndarray], runs: np.ndarray) -> np.ndarray:
    missing = [int(run) for run in runs if int(run) not in curves]
    if missing:
        raise ValueError(f"missing center curves for runs {missing[:5]}")
    return np.stack([np.asarray(curves[int(run)], dtype=np.float64) for run in runs])


def _site_median(values: np.ndarray, groups: np.ndarray) -> dict:
    ci = bootstrap_cluster_statistic(values, groups, np.median, rng=np.random.default_rng(0))
    site = pd.DataFrame({"value": values, "group": groups}).groupby("group")["value"].median()
    return {"record": float(ci["point"]), "low": float(ci["low"]), "high": float(ci["high"]), "site": float(site.median())}


def _frame(family: str) -> dict[str, np.ndarray]:
    split = np.load(CACHE / family / "split.npy")
    runs = np.flatnonzero(np.asarray(split).astype(str) == "test").astype(np.int64)
    log_tf = np.asarray(np.load(CACHE / family / "log_tf.npy", mmap_mode="r")[runs, 10], dtype=np.float64)
    log_col = np.asarray(np.load(CACHE / family / "log_tf1d.npy", mmap_mode="r")[runs, 10], dtype=np.float64)
    f1 = np.load(CACHE / family / "rayleigh_f1.npy")[runs]
    arms = np.load(FIG / f"test_arms_{family}.npz")
    if not np.array_equal(np.asarray(arms["run"]), runs):
        raise ValueError(f"test arms run ids do not match {family}")
    passeri = np.load(FIG / f"test_passeri_{family}.npz")
    zeta = np.load(FIG / f"test_zeta_arms_{family}.npz")
    stage2 = np.load(HERE / "dump" / f"stage2_center_tf_{family}.npz")
    order = {int(run): i for i, run in enumerate(np.asarray(stage2["run"]))}
    stage2_tf = np.stack([np.asarray(stage2["tf"][order[int(run)]], dtype=np.float64) for run in runs])
    return {
        "run": runs,
        "group": np.asarray(np.load(CACHE / family / "group.npy")[runs]),
        "freq": np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64),
        "f0": np.array([_f0(family, int(run), float(fallback)) for run, fallback in zip(runs, f1)]),
        "opensees": np.exp(log_tf),
        "log_opensees": log_tf,
        "log_column": log_col,
        "Column": np.exp(log_col),
        "Stage 2": stage2_tf,
        "Pretell": np.asarray(arms["pretell"], dtype=np.float64),
        "Toro": np.asarray(arms["toro"], dtype=np.float64),
        "Dmult": np.asarray(arms["dmult"], dtype=np.float64),
        "Passeri": np.asarray(passeri["passeri"], dtype=np.float64),
        "Toro (record zeta)": np.asarray(zeta["toro"], dtype=np.float64),
        "Passeri (record zeta)": np.asarray(zeta["passeri"], dtype=np.float64),
    }


def _full_band(log_model: np.ndarray, log_true: np.ndarray, log_column: np.ndarray, freq: np.ndarray) -> dict:
    mask = (freq >= 0.1) & (freq <= 10.0)
    model = np.mean(np.abs(log_model[:, mask] - log_true[:, mask]), axis=1)
    column = np.mean(np.abs(log_column[:, mask] - log_true[:, mask]), axis=1)
    mid = (freq >= 0.4) & (freq <= 2.5)
    model_mid = np.mean(np.abs(log_model[:, mid] - log_true[:, mid]), axis=1)
    column_mid = np.mean(np.abs(log_column[:, mid] - log_true[:, mid]), axis=1)
    return {
        "n": int(model.size),
        "model_0p1_10_mean": float(model.mean()),
        "model_0p1_10_median": float(np.median(model)),
        "column_0p1_10_mean": float(column.mean()),
        "skill_0p1_10_median": float(np.median(1.0 - model / np.maximum(column, 1e-6))),
        "model_mid_mean": float(model_mid.mean()),
        "model_mid_median": float(np.median(model_mid)),
        "column_mid_mean": float(column_mid.mean()),
        "skill_mid_median": float(np.median(1.0 - model_mid / np.maximum(column_mid, 1e-6))),
    }


def _box(box: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.6))
    panels = (
        (0, 0, "Horizontal", "pearson", r"Pearson of $|$TF$|$"),
        (0, 1, "Horizontal", "anderson", "Anderson misfit"),
        (1, 0, "Dipping", "pearson", r"Pearson of $|$TF$|$"),
        (1, 1, "Dipping", "anderson", "Anderson misfit"),
    )
    for letter, (row, col, name, metric, ylabel) in enumerate(panels):
        ax = axes[row, col]
        methods = [method for method in METHODS if method in box[name][metric]]
        drawn = ax.boxplot(
            [box[name][metric][method] for method in methods],
            widths=0.62, patch_artist=True, showfliers=False,
            medianprops={"color": "black", "lw": 0.8},
            whiskerprops={"lw": 0.6}, capprops={"lw": 0.6}, boxprops={"lw": 0.6},
        )
        for patch, method in zip(drawn["boxes"], methods):
            patch.set_facecolor(BOX_COLOR[method])
            patch.set_alpha(0.55)
            patch.set_edgecolor("#333333")
        ax.set_xticks(range(1, len(methods) + 1), methods, fontsize=6, rotation=20, ha="right")
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        panel_letter(ax, "abcd"[letter])
    fig.tight_layout()
    save_figure(fig, OUT / "fig_box_test_methods.pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    model, cfg = load_model(MODEL, device)
    cache = Cache(CACHE)
    print("test evaluate", flush=True)
    res = evaluate(
        model, cache, cfg["encoding"], "test", device, batch=8, workers=0,
        iid_limit=None, positions=1, amp=True, target=cfg.get("target", "residual"),
        families=("iid", "ood_dipping", "ood_three_layer"), stage=int(cfg.get("stage", STAGE_UNIFIED)),
    )
    (OUT / "eval_test.json").write_text(json.dumps(res))
    print("center curves", flush=True)
    spectra = _centers(model, cache, device)
    for family, curves in spectra.items():
        runs = np.array(sorted(curves), dtype=np.int64)
        np.savez_compressed(OUT / f"center_tf_{family}.npz", run=runs, tf=np.stack([curves[int(run)] for run in runs]))
        print(f"  saved {family} {runs.size}", flush=True)

    rows = []
    centers = []
    harm = []
    box = {}
    for name, family in FAMILIES.items():
        frame = _frame(family)
        fresh = _align(spectra[family], frame["run"])
        curves = {method: frame[method] for method in frame if method not in ("run", "group", "freq", "f0", "opensees", "log_opensees", "log_column")}
        curves["P128n128"] = fresh
        pear = {
            method: np.array([pearson(curves[method][i], frame["opensees"][i]) for i in range(frame["run"].size)])
            for method in curves
        }
        gof = {
            method: np.array([
                anderson(frame["freq"], frame["opensees"][i], curves[method][i], frame["f0"][i])
                for i in range(frame["run"].size)
            ])
            for method in pear
        }
        for method in list(METHODS) + ["Toro (record zeta)", "Passeri (record zeta)"]:
            if method not in pear:
                continue
            for metric, values in (("pearson", pear[method]), ("anderson", gof[method])):
                stat = _site_median(values, frame["group"])
                rows.append({"domain": name, "method": method, "metric": metric, **stat, "n": int(frame["run"].size)})
        worse_p = pear["P128n128"] < pear["Column"]
        worse_a = gof["P128n128"] > gof["Column"]
        harm.append({
            "domain": name,
            "pearson_worse_than_column": float(worse_p.mean()),
            "anderson_worse_than_column": float(worse_a.mean()),
            "n": int(frame["run"].size),
        })
        box[name] = {"pearson": {m: pear[m] for m in METHODS if m in pear}, "anderson": {m: gof[m] for m in METHODS if m in gof}}
        for label, spec in (("Stage 2", curves["Stage 2"]), ("P128n128", fresh)):
            log_model = np.log(np.clip(spec, 1e-12, None))
            centers.append({"domain": name, "model": label, **_full_band(log_model, frame["log_opensees"], frame["log_column"], frame["freq"])})

    bands = []
    labels = {"iid": "Horizontal", "ood_dipping": "Dipping", "ood_three_layer": "Three-layer"}
    for family, label in labels.items():
        for width in (500, 1000, 1500):
            block = res["bins"].get(f"{family}_w{width}")
            if block is None:
                continue
            bands.append({
                "family": label,
                "width": width,
                "n": block["n_stations"],
                "log_l1_0p1_10": block.get("log_l1"),
                "skill_0p1_10": block.get("skill_full"),
                "log_l1_low": block.get("log_l1_low"),
                "log_l1_mid": block.get("log_l1_mid"),
                "skill_mid": block.get("skill_mid"),
                "log_l1_high": block.get("log_l1_high"),
                "skill_high": block.get("skill_high"),
                "pearson": block.get("pearson"),
            })
    pd.DataFrame(rows).to_csv(OUT / "table1_medians.csv", index=False)
    pd.DataFrame(harm).to_csv(OUT / "harm_rate.csv", index=False)
    pd.DataFrame(centers).to_csv(OUT / "center_bands.csv", index=False)
    pd.DataFrame(bands).to_csv(OUT / "station_bands.csv", index=False)
    apply_publication_style()
    _box(box)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(pd.DataFrame(centers).to_string(index=False), flush=True)
    print(pd.DataFrame(bands).to_string(index=False), flush=True)
    print(pd.DataFrame(harm).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
