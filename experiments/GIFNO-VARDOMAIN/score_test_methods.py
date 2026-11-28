#!/usr/bin/env python3
"""Test-split potential of one checkpoint against Pretell, Toro, and Dmult.

The reference is the fixed-window center-station |TF|. Records are every run
in the model's test split. A group summary collapses the 30 seeds of one
Sobol sample so the count is the number of held-out sites.

``--checkpoint`` is required. The forward is the one ``load_model`` binds
(bfloat16 autocast on CUDA when the checkpoint was trained that way). Every
array, the CSV, and ``test_vs_classical.json`` record the checkpoint path and
the weight hash. An array whose sidecar does not match is recomputed.

    python score_test_methods.py --checkpoint ../../checkpoints/vardomain/early_width/stage2_rise50
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER = REPO / "experiments" / "DeepONet-Residual" / "paper"
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(HERE))

from data import KIND_RUN, Cache, make_window  # noqa: E402
from evaluate import load_model  # noqa: E402
from provenance import cache_is_current, full_stamp, write_sidecar  # noqa: E402
from style import METHOD_STYLE, OPENSEES, figsize, panel_letter, plot_curve, use_style  # noqa: E402
from classical_worker import compute_arms, init_worker  # noqa: E402

FIG_DIR = HERE / "figures"
ARMS_DIR = HERE / "figures"
CACHE = REPO / "data" / "vardomain" / "cache"
H5 = {
    "iid": REPO / "data" / "gifno_screen" / "h5",
    "ood_dipping": REPO / "data" / "gifno_screen" / "ood_dipping" / "h5",
}
DOMAINS = (("Horizontal", "iid"), ("Dipping", "ood_dipping"))
MODEL, PRETELL, TORO, DMULT = "Model", "Pretell", "Toro", "Dmult"
ORDER = (MODEL, PRETELL, TORO, DMULT)
STYLE = {
    MODEL: {"color": "#C2410C", "ls": "-", "lw": 1.45, "marker": None, "ms": 3.0, "zorder": 5},
    PRETELL: METHOD_STYLE["Pretell median"],
    TORO: METHOD_STYLE["Toro"],
    DMULT: METHOD_STYLE["Dmult"],
}
WORKERS = 6


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def _tasks(family: str, runs: np.ndarray) -> list[dict]:
    root = H5[family]
    tasks = []
    for run in runs:
        path = root / f"run_{int(run)}.h5"
        with h5py.File(path, "r") as handle:
            params = handle["params"].attrs
            thickness = float(params["H_discretized"] if "H_discretized" in params else params["H"])
            soil = int(params["soil_layer_count"]) if "soil_layer_count" in params else int(round(thickness))
            tasks.append(
                {
                    "family": family,
                    "run": int(run),
                    "vs1": float(params["Vs1"]),
                    "H": thickness,
                    "cov": float(params["CoV"]),
                    "vs2": float(params["Vs2"]),
                    "soil_nz": soil,
                    "f0": float(params["f0_effective"]),
                }
            )
    return tasks


def _arms(family: str, tasks: list[dict]) -> dict[str, np.ndarray]:
    dest = ARMS_DIR / f"test_arms_{family}.npz"
    runs = np.array([t["run"] for t in tasks], dtype=np.int64)
    if dest.is_file():
        blob = np.load(dest)
        if np.array_equal(blob["run"], runs):
            print(f"loaded arms {family} n={runs.size}", flush=True)
            return {key: blob[key] for key in ("pretell", "toro", "dmult")}
    print(f"classical {family} n={runs.size}", flush=True)
    pretell = np.empty((runs.size, 1000), dtype=np.float32)
    toro = np.empty_like(pretell)
    dmult = np.empty_like(pretell)
    done = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx, initializer=init_worker) as pool:
        for row in pool.map(compute_arms, tasks, chunksize=1):
            i = int(np.where(runs == row["run"])[0][0])
            pretell[i] = row["pretell"]
            toro[i] = row["toro"]
            dmult[i] = row["dmult"]
            done += 1
            if done % 25 == 0 or done == runs.size:
                print(f"  {family} {done}/{runs.size}", flush=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, run=runs, pretell=pretell, toro=toro, dmult=dmult)
    return {"pretell": pretell, "toro": toro, "dmult": dmult}


def _predict(model, device, cache: Cache, family: str, runs: np.ndarray, stamp: dict) -> np.ndarray:
    dest = FIG_DIR / f"test_model_{family}.npy"
    if cache_is_current(dest, stamp, runs.size):
        print(f"loaded model {family}", flush=True)
        return np.load(dest)
    camp = next(c for c in cache.campaigns if c.name == family)
    base = next(i for i, (ci, _j) in enumerate(cache.run_of) if cache.campaigns[ci] is camp)
    out = np.empty((runs.size, cache.freq.size), dtype=np.float32)
    for start in range(0, runs.size, 4):
        chunk = runs[start : start + 4]
        xs, qs, priors, mids = [], [], [], []
        for run in chunk:
            x, query, _tf, prior, valid, _info, _xa = make_window(
                cache, "metric", KIND_RUN, int(base + int(run)), 500, 500, 0, False
            )
            v = np.flatnonzero(valid.numpy())
            mid = int(v[len(v) // 2])
            xs.append(x)
            qs.append(query)
            priors.append(prior.numpy()[mid])
            mids.append(mid)
        with torch.no_grad():
            g = model(torch.stack(xs).to(device), torch.stack(qs).to(device)).float().cpu().numpy()
        for j, mid in enumerate(mids):
            out[start + j] = np.exp(priors[j] + g[j, mid]).astype(np.float32)
        if start % 40 == 0:
            print(f"  model {family} {start}/{runs.size}", flush=True)
    np.save(dest, out)
    write_sidecar(dest, {**stamp, "family": family, "n": int(runs.size), "rows": "test-split order"})
    return out


def _scores(freq: np.ndarray, ref: np.ndarray, cand: np.ndarray, f0: np.ndarray) -> dict[str, np.ndarray]:
    n = ref.shape[0]
    pear = np.empty(n)
    gof = np.empty(n)
    for i in range(n):
        r = np.clip(ref[i], 1e-12, None)
        c = np.clip(cand[i], 1e-12, None)
        pear[i] = float(np.corrcoef(c, r)[0, 1])
        w = np.exp(-0.5 * ((freq - float(f0[i])) / 1.5) ** 2)
        w /= max(float(w.sum()), 1e-12)
        gof[i] = float(np.sum(w * np.abs(np.log(c) - np.log(r))))
    return {"pearson": pear, "gof_af": gof}


def _summarize(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model = frame[frame["method"] == MODEL]
    for method, g in frame.groupby("method"):
        other = g.sort_values("run")
        paired = model.sort_values("run")
        rows.append(
            {
                "domain": name,
                "level": "record",
                "method": method,
                "n": int(len(other)),
                "pearson_median": float(other["pearson"].median()),
                "gof_median": float(other["gof_af"].median()),
                "beat_pearson": float("nan") if method == MODEL else float(np.mean(paired["pearson"].to_numpy() > other["pearson"].to_numpy())),
                "beat_anderson": float("nan") if method == MODEL else float(np.mean(paired["gof_af"].to_numpy() < other["gof_af"].to_numpy())),
            }
        )
    group_rows = []
    for method, g in frame.groupby("method"):
        site_p = g.groupby("group")["pearson"].median()
        site_g = g.groupby("group")["gof_af"].median()
        model_p = model.groupby("group")["pearson"].median()
        model_g = model.groupby("group")["gof_af"].median()
        group_rows.append(
            {
                "domain": name,
                "level": "site",
                "method": method,
                "n": int(site_p.size),
                "pearson_median": float(site_p.median()),
                "gof_median": float(site_g.median()),
                "beat_pearson": float("nan") if method == MODEL else float(np.mean(model_p.to_numpy() > site_p.loc[model_p.index].to_numpy())),
                "beat_anderson": float("nan") if method == MODEL else float(np.mean(model_g.to_numpy() < site_g.loc[model_g.index].to_numpy())),
            }
        )
    return pd.DataFrame(rows + group_rows)


def _gallery(name: str, freq, ref, curves, vs1, thick, picked) -> None:
    use_style()
    n = int(len(picked))
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize("double", height_mm=62 * nrows), sharex=True, sharey=True, squeeze=False
    )
    series = (
        (ref, METHOD_STYLE[OPENSEES], "OpenSees"),
        (curves[MODEL], STYLE[MODEL], "Model"),
        (curves[PRETELL], STYLE[PRETELL], "Pretell"),
        (curves[TORO], STYLE[TORO], "Toro"),
    )
    for ax, i, letter in zip(axes.ravel(), picked, "abcdef"):
        for curve, style, label in series:
            plot_curve(ax, freq, np.maximum(curve[int(i)], 1e-6), style, label=label)
        ax.set_title(f"{name}  $V_{{s1}}$={vs1[int(i)]:.0f}, $H$={thick[int(i)]:.0f} m", fontsize=7, pad=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(float(freq[0]), float(freq[-1]))
        panel_letter(ax, letter)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(r"$f$ (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|$TF$|$")
    axes[0, 0].legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_tf_test_" + name.lower())


def _pick(vs1: np.ndarray, thick: np.ndarray, k: int = 6) -> np.ndarray:
    key = np.round(vs1).astype(np.int64) * 1000 + np.round(thick).astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    first = order[np.concatenate([[True], key[order][1:] != key[order][:-1]])]
    first = first[np.argsort(key[first])]
    if first.size > k:
        first = first[np.linspace(0, first.size - 1, k).round().astype(int)]
    return first


def _boxes(frames: dict[str, pd.DataFrame]) -> None:
    use_style()
    labels, pear, gof, colors = [], [], [], []
    for name, _family in DOMAINS:
        df = frames[name]
        short = "Horiz." if name == "Horizontal" else "Dip."
        for method in ORDER:
            g = df.loc[df["method"] == method]
            labels.append(f"{short}\n{method}")
            pear.append(g["pearson"].to_numpy())
            gof.append(g["gof_af"].to_numpy())
            colors.append(STYLE[method]["color"])
    fig, axes = plt.subplots(1, 2, figsize=figsize("double", height_mm=96))
    for ax, data, ylabel, letter in (
        (axes[0], pear, r"Pearson of $|$TF$|$", "a"),
        (axes[1], gof, "Anderson misfit", "b"),
    ):
        bp = ax.boxplot(
            data, widths=0.62, patch_artist=True, showfliers=False,
            medianprops={"color": "black", "lw": 0.8},
            whiskerprops={"lw": 0.6}, capprops={"lw": 0.6}, boxprops={"lw": 0.6},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("#333333")
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=5.5)
        ax.set_ylabel(ylabel)
        panel_letter(ax, letter, x=-0.18)
    axes[0].set_ylim(0.45, 1.02)
    fig.tight_layout()
    _save(fig, "fig_box_test_methods")


def main() -> None:
    global FIG_DIR
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True, help="directory containing best_model.pt")
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args()
    FIG_DIR = args.fig_dir
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    prepared = []
    for name, family in DOMAINS:
        camp_split = np.load(CACHE / family / "split.npy")
        runs = np.flatnonzero(camp_split == "test").astype(int)
        tasks = _tasks(family, runs)
        arms = _arms(family, tasks)
        prepared.append((name, family, runs, tasks, arms))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)
    model.eval()
    stamp = full_stamp(args.checkpoint, device, bool(cfg.get("amp", True)))
    cache = Cache(CACHE)
    freq = cache.freq
    frames: dict[str, pd.DataFrame] = {}
    tables = []
    for name, family, runs, tasks, arms in prepared:
        camp = next(c for c in cache.campaigns if c.name == family)
        groups = np.asarray(camp.group[runs])
        pred = _predict(model, device, cache, family, runs, stamp)
        ref = np.exp(np.asarray(camp.log_tf[runs, 10], dtype=np.float64))
        prior = np.exp(np.asarray(camp.log_tf1d[runs, 10], dtype=np.float64))
        f0 = np.array([t["f0"] for t in tasks], dtype=float)
        vs1 = np.array([t["vs1"] for t in tasks], dtype=float)
        thick = np.array([t["H"] for t in tasks], dtype=float)
        curves = {
            MODEL: pred,
            PRETELL: arms["pretell"],
            TORO: arms["toro"],
            DMULT: arms["dmult"],
            "1D": prior,
        }
        parts = []
        for method, curve in curves.items():
            scored = _scores(freq, ref, np.asarray(curve, dtype=np.float64), f0)
            parts.append(pd.DataFrame({
                "method": method, "run": runs, "group": groups, "vs1": vs1, "H": thick, **scored,
            }))
        frame = pd.concat(parts, ignore_index=True)
        frames[name] = frame
        tables.append(_summarize(name, frame))
        _gallery(name, freq, ref, curves, vs1, thick, _pick(vs1, thick))
        print(name, "records", runs.size, "sites", np.unique(groups).size, flush=True)
    table = pd.concat(tables, ignore_index=True)
    table["checkpoint"] = stamp["checkpoint"]
    table["weight_sha256"] = stamp["weight_sha256"]
    table["inference_dtype"] = stamp["inference_dtype"]
    _boxes(frames)
    table.to_csv(FIG_DIR / "test_vs_classical.csv", index=False)
    manifest = {
        **stamp,
        "classical_arms": "Pretell, Toro, and Dmult do not use the checkpoint. Their npz files are keyed by run id.",
        "model_arrays": [f"test_model_{family}.npy" for _name, family in DOMAINS],
    }
    (FIG_DIR / "test_vs_classical.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(table.to_string(index=False), flush=True)
    print(f"wrote {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
