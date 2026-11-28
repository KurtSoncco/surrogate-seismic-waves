#!/usr/bin/env python3
"""Paper figures for one checkpoint, and the Case 7 width comparison.

Case 7 uses the 500 m crops and the wider OpenSees crops (1000 m and 1500 m).
A 2000 m window is cut from the 4000 m Q–Vs section about the same sensor.
Records are 15 s, so the comparison band is 0.4–2.5 Hz.

    python plot_results.py --model-dir ../../checkpoints/vardomain/early_width/stage2_rise50
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
REPO = HERE.parents[1]
CASE7A = HERE.parent / "case7a"
PAPER = REPO / "experiments" / "DeepONet-Residual" / "paper"
for path in (PAPER, CASE7A):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
sys.path.insert(0, str(HERE))

from applied_damping import applied_zeta  # noqa: E402
from case7_crops import CROPS, span_columns  # noqa: E402
from case7a import X_LEFT_M  # noqa: E402
from data import Cache, make_window  # noqa: E402
from dense_labels import _ttf  # noqa: E402
from encoding import encode, query_columns  # noqa: E402
from evaluate import load_model  # noqa: E402
from provenance import full_stamp  # noqa: E402
from labels import seiskit_input_acceleration, tf_over_input  # noqa: E402
from section_1d import RHO, rayleigh_xi_target, surface_over_outcrop  # noqa: E402
from style import METHOD_STYLE, figsize, panel_letter, plot_curve, use_style  # noqa: E402
from wave_surrogate.plot_utils.publication import DPI, canvas_bbox  # noqa: E402

SCALE = 1.9776999950408936
WINDOW_S = 1310.72
STRIDE_M = 20
BAND = (0.4, 2.5)
FIG_DIR = HERE / "figures"
CROP_500 = REPO / "data" / "case7a" / "opensees_crops" / "h5"
CROP_WIDE = REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5"
FULL_QVS = CROP_WIDE / "full_qvs.h5"

SITE_LABEL = {
    "flat_west": "Flat west",
    "thinning": "Thinning",
    "center": "Center",
    "ramp": "Ramp",
    "flat_east": "Flat east",
}
SITE_COLOR = {
    "flat_west": "#0072B2",
    "thinning": "#E69F00",
    "center": "#009E73",
    "ramp": "#D55E00",
    "flat_east": "#CC79A7",
}
WIDTH_LS = {500: "-", 1000: "--", 1500: ":", 2000: "-."}


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    kw = {"dpi": DPI, "facecolor": "white", "bbox_inches": canvas_bbox(fig)}
    fig.savefig(FIG_DIR / f"{stem}.pdf", **kw)
    fig.savefig(FIG_DIR / f"{stem}.png", **kw)
    plt.close(fig)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def _band(freq: np.ndarray) -> np.ndarray:
    return (freq >= BAND[0]) & (freq <= BAND[1])


def _crop_path(sample_id: str, width: int) -> Path | None:
    if width == 500:
        path = CROP_500 / f"{sample_id}.h5"
    else:
        path = CROP_WIDE / f"{sample_id}_w{width}.h5"
    return path if path.is_file() else None


def _read_mesh(path: Path) -> dict:
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(path, "r") as f:
        params = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["params"].attrs.items()}
        grid = dict(f["grid"].attrs)
        surf = f["recorders/accel/surface"]
        return {
            "vs": np.asarray(f["Vs_realization_2D"][:], dtype=np.float64),
            "zeta": applied_zeta(f),
            "acc": np.asarray(surf[:], dtype=np.float64),
            "time": np.asarray(f["recorders/accel/time"][:], dtype=np.float64),
            "f1": float(params["damping_freq_first"]),
            "f2": float(params["damping_freq_second"]),
            "motion": float(grid["motion_freq"]),
            "ricker": str(params.get("ricker_is", "acceleration")),
        }


def _window_from_full(center_x: float, width: int) -> dict:
    mesh = _read_mesh(FULL_QVS)
    c0, n, _shift = span_columns(center_x, width)
    sl = slice(c0, c0 + n)
    mesh["vs"] = mesh["vs"][:, sl]
    mesh["zeta"] = mesh["zeta"][:, sl]
    mesh["acc"] = mesh["acc"][:, sl]
    mesh["sensor"] = int(round(center_x - X_LEFT_M)) - c0
    return mesh


def _score_mesh(model, device, mesh: dict, sensor: int) -> dict:
    vs, zeta = mesh["vs"], mesh["zeta"]
    width = vs.shape[1]
    cols = np.arange(0, width, STRIDE_M)
    if sensor not in cols:
        cols = np.sort(np.append(cols, sensor))
    cols = cols[(cols >= 0) & (cols < width)]
    time, dt = mesh["time"], float(mesh["time"][1] - mesh["time"][0])
    freq, raw = tf_over_input(
        mesh["acc"][:, cols],
        seiskit_input_acceleration(mesh["motion"], 0.5, time, mesh["ricker"]),
        dt,
        _ttf(),
        window_s=WINDOW_S,
    )
    truth = raw / SCALE
    tf1d = np.empty_like(truth)
    for j, col in enumerate(cols):
        xi = np.stack([rayleigh_xi_target(freq, z, mesh["f1"], mesh["f2"]) for z in zeta[:, col]])
        tf1d[j] = surface_over_outcrop(freq, vs[:, col], dz=1.0, xi=xi, rho=RHO)
    enc = torch.from_numpy(encode("metric", vs, zeta, vs.shape[0], f1_hz=mesh["f1"], f2_hz=mesh["f2"])).unsqueeze(0)
    query = torch.tensor(query_columns("metric", cols.astype(np.float64), width), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        raw_g = model(enc.to(device), query.to(device)).float().cpu().numpy()[0]
    pred = tf1d * np.exp(raw_g)
    band = _band(freq)
    log_p = np.log(np.maximum(pred[:, band], 1e-6))
    log_t = np.log(np.maximum(truth[:, band], 1e-6))
    log_d = np.log(np.maximum(tf1d[:, band], 1e-6))
    station_l1 = np.abs(log_p - log_t).mean(1)
    station_pear = np.array([_pearson(log_p[j], log_t[j]) for j in range(log_p.shape[0])])
    log_m = float(station_l1.mean())
    log_0 = float(np.abs(log_d - log_t).mean())
    i_s = int(np.where(cols == sensor)[0][0])
    return {
        "freq": freq,
        "band": band,
        "truth": truth[i_s],
        "prior": tf1d[i_s],
        "pred": pred[i_s],
        "pearson": _pearson(np.log(np.maximum(pred[:, band], 1e-6)).ravel(), np.log(np.maximum(truth[:, band], 1e-6)).ravel()),
        "pearson_1d": _pearson(np.log(np.maximum(tf1d[:, band], 1e-6)).ravel(), np.log(np.maximum(truth[:, band], 1e-6)).ravel()),
        "log_l1": float(log_m),
        "skill": float(1.0 - log_m / max(log_0, 1e-6)),
        "n_stations": int(cols.size),
        "record_s": float(time[-1] - time[0]),
        "station_log_l1": station_l1.astype(float).tolist(),
        "station_pearson": station_pear.astype(float).tolist(),
    }


def case7(model, device) -> dict:
    out: dict = {}
    for crop in CROPS:
        name = crop["name"]
        out[name] = {}
        for width in (500, 1000, 1500, 2000):
            if width == 2000:
                mesh = _window_from_full(float(crop["center_x_m"]), width)
                sensor = mesh.pop("sensor")
            else:
                path = _crop_path(crop["sample_id"], width)
                if path is None:
                    continue
                mesh = _read_mesh(path)
                sensor = mesh["vs"].shape[1] // 2
            scored = _score_mesh(model, device, mesh, sensor)
            curves = {k: scored.pop(k) for k in ("freq", "band", "truth", "prior", "pred")}
            out[name][width] = {**scored, "curves": curves}
            print(f"{name:12} w{width} pearson {scored['pearson']:.3f} skill {scored['skill']:.3f}", flush=True)
    return out


def _tf_panel(ax, freq, curves: list[tuple[np.ndarray, dict, str]]) -> None:
    for y, style, label in curves:
        plot_curve(ax, freq, np.maximum(y, 1e-3), style, markers=True, n_marks=6, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 10.0)
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$|\mathrm{TF}|$")


def fig_case7(scored: dict) -> None:
    use_style()
    names = [c["name"] for c in CROPS]
    fig, axes = plt.subplots(1, 5, figsize=figsize("double", height_mm=78), sharey=True)
    for ax, name, letter in zip(axes, names, "abcde"):
        panel = scored[name]
        w500 = panel[500]["curves"]
        freq = w500["freq"]
        series = [(w500["truth"], {**METHOD_STYLE["OpenSees 2-D"], "ls": "-"}, "OpenSees, 500 m")]
        series.append((w500["prior"], {**METHOD_STYLE["1D base case"], "ls": "-."}, "1D"))
        for width, color in ((500, "#C2410C"), (1500, "#0057B8"), (2000, "#009E73")):
            if width not in panel:
                continue
            series.append((
                panel[width]["curves"]["pred"],
                {"color": color, "ls": WIDTH_LS[width], "lw": 1.2, "marker": None, "ms": 2.4, "zorder": 3},
                f"Model, {width} m",
            ))
        _tf_panel(ax, freq, series)
        ax.set_title(SITE_LABEL[name], fontsize=7, pad=2)
        panel_letter(ax, letter)
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_case7_tf")

    fig, axes = plt.subplots(1, 2, figsize=figsize("double", height_mm=70))
    widths = [500, 1000, 1500, 2000]
    for name in names:
        color = SITE_COLOR[name]
        xs, skill, pear, pear0 = [], [], [], []
        for w in widths:
            if w not in scored[name]:
                continue
            row = scored[name][w]
            xs.append(w)
            skill.append(row["skill"])
            pear.append(row["pearson"])
            pear0.append(row["pearson_1d"])
        axes[0].plot(xs, skill, color=color, ls="-", lw=1.2, marker="o", ms=3.2, label=SITE_LABEL[name])
        axes[1].plot(xs, pear, color=color, ls="-", lw=1.2, marker="o", ms=3.2, label=SITE_LABEL[name])
        axes[1].plot(xs, pear0, color=color, ls="--", lw=0.9, marker="x", ms=3.0)
    axes[0].axhline(0.0, color="0.6", lw=0.4)
    axes[0].set_ylabel("skill vs 1D, 0.4–2.5 Hz")
    axes[1].set_ylabel("Pearson of log |TF|")
    for ax, letter in zip(axes, "ab"):
        ax.set_xlabel("window (m)")
        ax.set_xticks(widths)
        panel_letter(ax, letter)
    axes[1].plot([], [], color="0.3", ls="--", marker="x", ms=3, label="1D")
    axes[0].legend(frameon=False, fontsize=6, ncol=1)
    axes[1].legend(frameon=False, fontsize=6, ncol=1)
    fig.tight_layout()
    _save(fig, "fig_case7_width")


def fig_holdout(eval_path: Path) -> None:
    use_style()
    data = json.loads(eval_path.read_text())
    families = (("iid", "IID"), ("ood_dipping", "Dipping"), ("ood_three_layer", "Three-layer"))
    colors = {"iid": "#0072B2", "ood_dipping": "#D55E00", "ood_three_layer": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=figsize("double", height_mm=68))
    for key, label in families:
        xs, skill, pear = [], [], []
        for w in (500, 1000, 1500):
            bin_ = data["bins"].get(f"{key}_w{w}")
            if not bin_:
                continue
            xs.append(w)
            skill.append(bin_["skill_mid"])
            pear.append(bin_["pearson_tr"])
        axes[0].plot(xs, skill, color=colors[key], ls="-", lw=1.2, marker="o", ms=3.2, label=label)
        axes[1].plot(xs, pear, color=colors[key], ls="-", lw=1.2, marker="o", ms=3.2, label=label)
    axes[0].set_ylabel("skill vs g = 0, 0.4–2.5 Hz")
    axes[1].set_ylabel("Pearson, 0.1–2.5 Hz")
    for ax, letter in zip(axes, "ab"):
        ax.set_xlabel("window (m)")
        ax.set_xticks([500, 1000, 1500])
        ax.legend(frameon=False, fontsize=6)
        panel_letter(ax, letter)
    fig.tight_layout()
    _save(fig, "fig_holdout_width")


def fig_examples(model, device, cache: Cache) -> None:
    use_style()
    picks = []
    for family, label in (("iid", "IID"), ("ood_dipping", "Dipping"), ("ood_three_layer", "Three-layer")):
        runs = cache.runs("test", (family,))
        if runs.size == 0:
            continue
        x, query, log_tf, prior, valid, _info, _xa = make_window(cache, "metric", 0, int(runs[0]), 500, 500, 0, False)
        with torch.no_grad():
            g = model(x.unsqueeze(0).to(device), query.unsqueeze(0).to(device)).float().cpu().numpy()[0]
        k = int(np.flatnonzero(valid.numpy())[len(np.flatnonzero(valid.numpy())) // 2])
        pred = np.exp(prior.numpy()[k] + g[k])
        truth = np.exp(log_tf.numpy()[k])
        base = np.exp(prior.numpy()[k])
        picks.append((label, cache.freq, truth, base, pred))
    if not picks:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=figsize("double", height_mm=68), sharey=True)
    if len(picks) == 1:
        axes = [axes]
    for ax, (label, freq, truth, base, pred), letter in zip(axes, picks, "abc"):
        _tf_panel(ax, freq, [
            (truth, METHOD_STYLE["OpenSees 2-D"], "OpenSees"),
            (base, METHOD_STYLE["1D base case"], "1D"),
            (pred, {"color": "#C2410C", "ls": "--", "lw": 1.25, "marker": None, "ms": 2.6, "zorder": 3}, "Model"),
        ])
        ax.set_title(label, fontsize=7, pad=2)
        panel_letter(ax, letter)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    _save(fig, "fig_holdout_tf")


def main() -> None:
    global FIG_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True, help="directory containing best_model.pt")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "vardomain" / "cache")
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args()
    FIG_DIR = args.fig_dir
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.model_dir, device)
    model.eval()
    stamp = full_stamp(args.model_dir, device, bool(cfg.get("amp", True)))
    fig_holdout(args.model_dir / "eval_test.json")
    fig_examples(model, device, Cache(args.cache))
    scored = case7(model, device)
    fig_case7(scored)
    summary = {
        name: {
            str(w): {k: v for k, v in row.items() if k not in ("curves", "station_log_l1", "station_pearson")}
            for w, row in widths.items()
        }
        for name, widths in scored.items()
    }
    (FIG_DIR / "case7_width.json").write_text(json.dumps(summary, indent=2))
    (FIG_DIR / "case7_width.provenance.json").write_text(json.dumps(stamp, indent=2) + "\n")
    print(f"wrote {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
