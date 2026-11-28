#!/usr/bin/env python3
"""Stage-0 residual model against Pretell, Toro, and the damping multiplier.

The curves and boxes use the atlas OpenSees center station, the same reference
as the Pretell / Toro / Dmult figures. A second table scores every arm on the
fixed-window label this model was trained to match. Boxes keep records that
are in this model's test split.

    python plot_model_classical.py --checkpoint ../../checkpoints/vardomain/early_width/stage2_rise50
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER = REPO / "experiments" / "DeepONet-Residual" / "paper"
ATLAS = REPO / "experiments" / "DeepONet-Residual" / "results" / "response_variability" / "tf_atlas"
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(HERE))

from data import KIND_RUN, Cache, make_window  # noqa: E402
from evaluate import load_model  # noqa: E402
from provenance import cache_is_current, full_stamp, write_sidecar  # noqa: E402
from style import METHOD_STYLE, OPENSEES, figsize, panel_letter, plot_curve, use_style  # noqa: E402

FIG_DIR = HERE / "figures"
CACHE = REPO / "data" / "vardomain" / "cache"
PRETELL, TORO, DMULT, MODEL = "Pretell", "Toro", "Dmult", "Model"
ORDER = (MODEL, PRETELL, TORO, DMULT)
STYLE = {
    MODEL: {"color": "#C2410C", "ls": "-", "lw": 1.45, "marker": None, "ms": 3.0, "zorder": 5},
    PRETELL: METHOD_STYLE["Pretell median"],
    TORO: METHOD_STYLE["Toro"],
    DMULT: METHOD_STYLE["Dmult"],
}
DOMAINS = {
    "Horizontal": ("iid", ATLAS / "iid_heldout_pack.npz", ATLAS / "iid_heldout_summary.csv"),
    "Dipping": ("ood_dipping", ATLAS / "dipping_heldout_pack.npz", ATLAS / "dipping_heldout_summary.csv"),
}
ARM_KEY = {PRETELL: "tf_pretell", TORO: "tf_toro", DMULT: "tf_dmult"}


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def _stations(tf: np.ndarray) -> np.ndarray:
    """Center recorder when the array is (record, station, frequency)."""
    a = np.asarray(tf, dtype=np.float64)
    if a.ndim == 3:
        return a[:, a.shape[1] // 2]
    return a


def _scores(freq: np.ndarray, ref: np.ndarray, cand: np.ndarray, f0: np.ndarray) -> dict[str, np.ndarray]:
    n = ref.shape[0]
    pear = np.empty(n)
    gof = np.empty(n)
    rel = np.empty(n)
    for i in range(n):
        r = np.clip(ref[i], 1e-12, None)
        c = np.clip(cand[i], 1e-12, None)
        pear[i] = float(np.corrcoef(c, r)[0, 1])
        rel[i] = float(np.linalg.norm(c - r) / max(np.linalg.norm(r), 1e-12))
        w = np.exp(-0.5 * ((freq - float(f0[i])) / 1.5) ** 2)
        w = w / max(float(w.sum()), 1e-12)
        gof[i] = float(np.sum(w * np.abs(np.log(c) - np.log(r))))
    return {"pearson": pear, "gof_af": gof, "rel_l2": rel}


def _predict(model, device, cache: Cache, flats: np.ndarray, batch: int = 4) -> np.ndarray:
    out = np.empty((flats.size, cache.freq.size), dtype=np.float64)
    for start in range(0, flats.size, batch):
        chunk = flats[start : start + batch]
        xs, qs, priors, mids = [], [], [], []
        for flat in chunk:
            x, query, _tf, prior, valid, _info, _xa = make_window(
                cache, "metric", KIND_RUN, int(flat), 500, 500, 0, False
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
            out[start + j] = np.exp(priors[j] + g[j, mid])
    return out


def _flat_index(cache: Cache, family: str, run: np.ndarray) -> np.ndarray:
    camp = next(c for c in cache.campaigns if c.name == family)
    base = next(i for i, (ci, _j) in enumerate(cache.run_of) if cache.campaigns[ci] is camp)
    return base + run.astype(np.int64)


def _load_domain(name: str, cache: Cache) -> dict:
    family, pack_path, summary_path = DOMAINS[name]
    pack = np.load(pack_path, allow_pickle=True)
    summary = pd.read_csv(summary_path)
    f0 = (
        summary.loc[summary["method"] == "Pretell median", ["sample", "f0"]]
        .drop_duplicates("sample")
        .sort_values("sample")["f0"]
        .to_numpy()
    )
    run = np.asarray(pack["sample_idx"], dtype=int)
    camp = next(c for c in cache.campaigns if c.name == family)
    ref = np.exp(np.asarray(camp.log_tf[run, 10], dtype=np.float64))
    prior = np.exp(np.asarray(camp.log_tf1d[run, 10], dtype=np.float64))
    split = np.asarray(camp.split[run])
    freq = np.asarray(pack["freq"], dtype=float)
    curves = {PRETELL: _stations(pack["tf_pretell"]), TORO: _stations(pack["tf_toro"]), DMULT: _stations(pack["tf_dmult"])}
    return {
        "name": name,
        "freq": freq,
        "run": run,
        "f0": f0,
        "ref": ref,
        "atlas": _stations(pack["tf_opensees"]),
        "prior": prior,
        "split": split,
        "vs1": np.asarray(pack["vs1"], dtype=float),
        "H": np.asarray(pack["H"], dtype=float),
        "curves": curves,
    }


def _frame(domain: dict, model_tf: np.ndarray, reference: str) -> pd.DataFrame:
    rows = []
    arms = {MODEL: model_tf, "1D": domain["prior"], **domain["curves"]}
    ref = domain["atlas"] if reference == "atlas" else domain["ref"]
    for method, curve in arms.items():
        scored = _scores(domain["freq"], ref, curve, domain["f0"])
        part = pd.DataFrame(
            {"reference": reference, "method": method, "sample": np.arange(domain["run"].size), "split": domain["split"], **scored}
        )
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _median_table(frames: dict[str, pd.DataFrame], mask_name: str) -> pd.DataFrame:
    rows = []
    for domain, df in frames.items():
        use = df if mask_name == "all" else df[df["split"] == "test"]
        for (reference, method), g in use.groupby(["reference", "method"]):
            rows.append(
                {
                    "set": mask_name,
                    "reference": reference,
                    "domain": domain,
                    "method": method,
                    "n": int(g["sample"].nunique()),
                    "pearson_median": float(g["pearson"].median()),
                    "gof_median": float(g["gof_af"].median()),
                    "rel_l2_median": float(g["rel_l2"].median()),
                }
            )
    return pd.DataFrame(rows)


def _sample_rows(domain: dict, k: int = 6) -> np.ndarray:
    test = np.flatnonzero(domain["split"] == "test")
    vs1 = np.round(domain["vs1"][test]).astype(int)
    h = np.round(domain["H"][test]).astype(int)
    key = vs1.astype(np.int64) * 1000 + h.astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    picked = order[np.concatenate([[True], key[order][1:] != key[order][:-1]])]
    picked = picked[np.argsort(key[picked])]
    if picked.size > k:
        picked = picked[np.linspace(0, picked.size - 1, k).round().astype(int)]
    return test[picked]


def _gallery(domain: dict, model_tf: np.ndarray) -> None:
    use_style()
    freq = domain["freq"]
    rows = _sample_rows(domain)
    n = int(rows.size)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize("double", height_mm=62 * nrows), sharex=True, sharey=True, squeeze=False
    )
    series = (
        (domain["atlas"], METHOD_STYLE[OPENSEES], "OpenSees"),
        (model_tf, STYLE[MODEL], "Model"),
        (domain["curves"][PRETELL], STYLE[PRETELL], "Pretell"),
        (domain["curves"][TORO], STYLE[TORO], "Toro"),
    )
    for ax, i, letter in zip(axes.ravel(), rows, "abcdef"):
        for curve, style, label in series:
            plot_curve(ax, freq, np.maximum(curve[int(i)], 1e-6), style, label=label)
        ax.set_title(
            f"{domain['name']}  $V_{{s1}}$={domain['vs1'][int(i)]:.0f}, $H$={domain['H'][int(i)]:.0f} m",
            fontsize=7,
            pad=2,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(float(freq[0]), float(freq[-1]))
        panel_letter(ax, letter)
    for ax in axes[-1]:
        ax.set_xlabel(r"$f$ (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|$TF$|$")
    axes[0, 0].legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_tf_model_" + domain["name"].lower())


def _boxes(frames: dict[str, pd.DataFrame]) -> None:
    use_style()
    labels, pear, gof, colors = [], [], [], []
    for domain in ("Horizontal", "Dipping"):
        df = frames[domain]
        df = df[(df["split"] == "test") & (df["reference"] == "atlas") & (df["method"] != "1D")]
        short = "Horiz." if domain == "Horizontal" else "Dip."
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
            data,
            widths=0.62,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "lw": 0.8},
            whiskerprops={"lw": 0.6},
            capprops={"lw": 0.6},
            boxprops={"lw": 0.6},
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
    _save(fig, "fig_box_model_classical")


def main() -> None:
    global FIG_DIR
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True, help="directory containing best_model.pt")
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args()
    FIG_DIR = args.fig_dir
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)
    model.eval()
    stamp = full_stamp(args.checkpoint, device, bool(cfg.get("amp", True)))
    cache = Cache(CACHE)
    frames = {}
    loaded = {}
    for name in DOMAINS:
        domain = _load_domain(name, cache)
        family = DOMAINS[name][0]
        flats = _flat_index(cache, family, domain["run"])
        pred_path = FIG_DIR / f"model_center_{name.lower()}.npy"
        if cache_is_current(pred_path, stamp, flats.size):
            pred = np.load(pred_path)
            print(f"loaded {name} n={flats.size}", flush=True)
        else:
            print(f"predict {name} n={flats.size} test={(domain['split']=='test').sum()}", flush=True)
            pred = _predict(model, device, cache, flats)
            np.save(pred_path, pred)
            write_sidecar(pred_path, {**stamp, "family": family, "n": int(flats.size)})
        loaded[name] = (domain, pred)
        frames[name] = pd.concat(
            [_frame(domain, pred, "atlas"), _frame(domain, pred, "fixed")],
            ignore_index=True,
        )
        _gallery(domain, pred)
    table = pd.concat([_median_table(frames, "test"), _median_table(frames, "all")], ignore_index=True)
    _boxes(frames)
    table["checkpoint"] = stamp["checkpoint"]
    table["weight_sha256"] = stamp["weight_sha256"]
    table["inference_dtype"] = stamp["inference_dtype"]
    table.to_csv(FIG_DIR / "model_vs_classical.csv", index=False)
    print(table.to_string(index=False), flush=True)
    print(f"wrote {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
