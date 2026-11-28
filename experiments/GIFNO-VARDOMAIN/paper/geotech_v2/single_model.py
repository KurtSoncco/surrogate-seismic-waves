#!/usr/bin/env python3
"""Score one checkpoint, P128n128, on the same windows as the four-model mean.

Validation mid-band skill picks this checkpoint over the other three on
horizontal ground, dipping ground, and the realistic sections. The plain
P128 checkpoint is a little higher on the three-layer validation set. The
choice is that validation table, not the test set.
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
VARDOMAIN = HERE.parent.parent
sys.path.insert(0, str(VARDOMAIN))
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

from data import KIND_SECTION, Cache, WindowSet  # noqa: E402
from diag_forward import ROOT, load_fp32  # noqa: E402
from dump_eval import _jobs  # noqa: E402
from metrics import band_mask  # noqa: E402
from paths import CACHE  # noqa: E402
from plotting import apply_publication_style, panel_letter, save_figure  # noqa: E402

NAME = "P128n128"
OUT = HERE / "07_one_model"
LABEL = {
    "iid": "Horizontal",
    "ood_dipping": "Dipping",
    "ood_three_layer": "Three-layer",
    "manifold": "Realistic sections",
}


def _score(model, cache: Cache, device) -> tuple[pd.DataFrame, dict[str, dict[int, np.ndarray]]]:
    jobs = _jobs(cache, None)
    freq = np.asarray(cache.freq, dtype=np.float64)
    masks = {band: band_mask(freq, band) for band in ("low", "mid", "high")}
    loader = DataLoader(
        WindowSet(cache, "metric", jobs, stage=0, seed=0, mirror=False),
        batch_size=8, shuffle=False, num_workers=4, pin_memory=device.type == "cuda",
    )
    bins: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    centers: dict[str, dict[int, tuple[float, np.ndarray]]] = {}
    seen = 0
    with torch.inference_mode():
        for step, (x, query, tf, prior, valid, info, x_abs) in enumerate(loader):
            g = model(x.to(device), query.to(device)).float().cpu().numpy()
            log_pred = g + prior.numpy()
            log_true = tf.numpy()
            log_prior = prior.numpy()
            info_np = info.numpy()
            valid_np = valid.numpy()
            local_x = x_abs.numpy() - info_np[:, 3:4]
            for b in range(valid_np.shape[0]):
                stations = np.flatnonzero(valid_np[b])
                if stations.size == 0:
                    continue
                kind, index, width, _start = (int(v) for v in info_np[b, :4])
                if kind == KIND_SECTION:
                    family = "manifold"
                    run_id = None
                else:
                    camp, j = cache.campaign(index)
                    family = camp.name
                    run_id = int(j)
                key = f"{family}_w{width}"
                slot = bins.setdefault(key, {})
                for band, mask in masks.items():
                    model_l1 = np.abs(log_pred[b, stations][:, mask] - log_true[b, stations][:, mask]).mean(axis=1)
                    column_l1 = np.abs(log_prior[b, stations][:, mask] - log_true[b, stations][:, mask]).mean(axis=1)
                    slot[f"log_l1_{band}"] = slot.get(f"log_l1_{band}", 0.0) + float(model_l1.sum())
                    slot[f"column_{band}"] = slot.get(f"column_{band}", 0.0) + float(column_l1.sum())
                counts[key] = counts.get(key, 0) + int(stations.size)
                if family in ("iid", "ood_dipping") and width == 500 and run_id is not None:
                    hold = centers.setdefault(family, {})
                    for station in stations:
                        dist = abs(float(local_x[b, station]) - 0.5 * width)
                        current = hold.get(run_id)
                        if current is None or dist < current[0]:
                            hold[run_id] = (dist, np.exp(log_pred[b, station]).astype(np.float32))
                seen += int(stations.size)
            if step % 20 == 0:
                print(f"  windows {step} stations {seen}", flush=True)
    rows = []
    ensemble = json.loads((HERE.parent / "dump" / "ensemble_eval_test.json").read_text())["bins"]
    for key, slot in sorted(bins.items()):
        n = counts[key]
        row = {"bin": key, "n_stations": n, "model": NAME}
        for band in ("low", "mid", "high"):
            log_l1 = slot[f"log_l1_{band}"] / n
            column = slot[f"column_{band}"] / n
            row[f"skill_{band}"] = 1.0 - log_l1 / max(column, 1e-6)
            row[f"log_l1_{band}"] = log_l1
        other = ensemble.get(key, {})
        row["ensemble_skill_mid"] = other.get("skill_mid")
        row["ensemble_log_l1_mid"] = other.get("log_l1_mid")
        rows.append(row)
    spectra = {family: {run: spec for run, (_dist, spec) in held.items()} for family, held in centers.items()}
    return pd.DataFrame(rows), spectra


def _treasure(model, device) -> pd.DataFrame:
    import h5py
    import hdf5plugin  # noqa: F401

    from applied_damping import applied_zeta
    from encoding import encode, query_columns
    from fig_treasure import FULL, WIDTH, X_LEFT

    blob = np.load(HERE / "05_treasure_island_full" / "fields.npz")
    freq = np.asarray(blob["freq"])
    truth = np.asarray(blob["truth"])
    column = np.asarray(blob["column"])
    x = np.asarray(blob["x"])
    with h5py.File(FULL, "r") as handle:
        vs = np.asarray(handle["Vs_realization_2D"][:], dtype=np.float64)
        zeta = applied_zeta(handle)
        attrs = handle["params"].attrs
        f1 = float(attrs["damping_freq_first"])
        f2 = float(attrs["damping_freq_second"])
    columns = np.arange(0, vs.shape[1], 20)
    g = np.empty((columns.size, freq.size), dtype=np.float64)
    half = WIDTH // 2
    with torch.inference_mode():
        for start in range(0, columns.size, 8):
            stop = min(start + 8, columns.size)
            encs, queries = [], []
            for column_index in columns[start:stop]:
                src = np.clip(np.arange(column_index - half, column_index + half), 0, vs.shape[1] - 1)
                encs.append(encode("metric", vs[:, src], zeta[:, src], vs.shape[0], f1_hz=f1, f2_hz=f2))
                queries.append(query_columns("metric", np.array([WIDTH / 2.0]), WIDTH))
            pred = model(
                torch.from_numpy(np.stack(encs)).to(device),
                torch.from_numpy(np.stack(queries).astype(np.float32)).to(device),
            ).float().cpu().numpy()[:, 0, :]
            g[start:stop] = pred
            print(f"  treasure {stop}/{columns.size}", flush=True)
    pred_tf = column * np.exp(g)
    mask = (freq >= 0.4) & (freq <= 2.5)
    log_p = np.log(np.clip(pred_tf[:, mask], 1e-6, None))
    log_e = np.log(np.clip(np.asarray(blob["pred"])[:, mask], 1e-6, None))
    log_t = np.log(np.clip(truth[:, mask], 1e-6, None))
    log_c = np.log(np.clip(column[:, mask], 1e-6, None))
    one = np.mean(np.abs(log_p - log_t), axis=1)
    ens = np.mean(np.abs(log_e - log_t), axis=1)
    base = np.mean(np.abs(log_c - log_t), axis=1)
    table = pd.DataFrame({
        "x_m": x,
        "one_model_log_l1_0p4_2p5": one,
        "ensemble_log_l1_0p4_2p5": ens,
        "column_log_l1_0p4_2p5": base,
        "one_model_skill_0p4_2p5": 1.0 - one / np.maximum(base, 1e-6),
        "ensemble_skill_0p4_2p5": 1.0 - ens / np.maximum(base, 1e-6),
    })
    np.savez_compressed(OUT / "treasure_g.npz", x=x, g=g.astype(np.float32))
    return table


def _figure(skill: pd.DataFrame, treasure: pd.DataFrame) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.4))
    for family, color in (("iid", "#0072B2"), ("ood_dipping", "#D55E00"), ("ood_three_layer", "#009E73")):
        part = skill[skill["bin"].str.startswith(family + "_w")].copy()
        part["width"] = part["bin"].str.extract(r"_w(\d+)$").astype(int)
        part = part.sort_values("width")
        axes[0].plot(part["width"], part["ensemble_skill_mid"], color=color, lw=1.2, label=f"{LABEL[family]}, four models")
        axes[0].plot(part["width"], part["skill_mid"], color=color, lw=1.1, ls="--", label=f"{LABEL[family]}, one model")
    axes[0].set_xticks([500, 1000, 1500])
    axes[0].set_xlabel("Interior width (m)")
    axes[0].set_ylabel("Mid-band skill, 0.4–2.5 Hz")
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(0.28, 0.42))
    panel_letter(axes[0], "a")
    axes[1].plot(treasure["x_m"], treasure["column_log_l1_0p4_2p5"], color="#0072B2", lw=1.0, label="1D column")
    axes[1].plot(treasure["x_m"], treasure["one_model_log_l1_0p4_2p5"], color="#E69F00", lw=1.1, label="One model")
    axes[1].plot(treasure["x_m"], treasure["ensemble_log_l1_0p4_2p5"], color="#D55E00", lw=1.2, label="Four models")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("log-L1 of |TF|, 0.4–2.5 Hz")
    axes[1].legend(frameon=False, fontsize=6)
    panel_letter(axes[1], "b")
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    model, stamp = load_fp32(ROOT / NAME, device)
    print(f"loaded {NAME} epoch {stamp['epoch']}", flush=True)
    cache = Cache(CACHE)
    skill, spectra = _score(model, cache, device)
    skill.to_csv(OUT / "skill_by_bin.csv", index=False)
    for family, held in spectra.items():
        runs = np.array(sorted(held), dtype=np.int64)
        np.savez(OUT / f"center_{family}.npz", run=runs, tf=np.stack([held[int(r)] for r in runs]))
        print(f"center {family} {runs.size}", flush=True)
    treasure = _treasure(model, device)
    treasure.to_csv(OUT / "treasure_island.csv", index=False)
    summary = {
        "model": NAME,
        "epoch": int(stamp["epoch"]),
        "why": "Highest validation mid-band skill of the four checkpoints on horizontal, dipping, and realistic sections.",
        "median_treasure_skill": float(np.median(treasure["one_model_skill_0p4_2p5"])),
        "median_ensemble_treasure_skill": float(np.median(treasure["ensemble_skill_0p4_2p5"])),
        "median_treasure_log_l1": float(np.median(treasure["one_model_log_l1_0p4_2p5"])),
        "median_ensemble_treasure_log_l1": float(np.median(treasure["ensemble_log_l1_0p4_2p5"])),
        "median_column_treasure_log_l1": float(np.median(treasure["column_log_l1_0p4_2p5"])),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    _figure(skill, treasure)
    print(skill[["bin", "skill_mid", "ensemble_skill_mid", "log_l1_mid"]].to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
