#!/usr/bin/env python3
"""SVLAB diagnosis for the Case 7 generalization heads.

Scores residual and outcrop+prior on the 5 m surface nodes: zone Pearson
including 0.1–0.4 Hz, the log-correction against the Case 7 1D, the overlap
arm, the three 1D damping conventions, and the off-center spatial gate.

    python diagnose_svlab.py --out-dir ../../checkpoints/case7_generalization/svlab_diagnosis
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from case7a import (
    DX_M,
    REPO,
    ROCK_VS,
    ROCK_ZETA_NORM,
    ZETA,
    ZONES,
    paint_section_1m,
    zone_of,
)
from labels import band_mask
from run_case7a import outcrop_reference_transfer_function
from score_section import dense_center_x, load_models, predict_windows
from score_svlab import BANDS, EPS, LOW_BAND_HZ, zone_scores
from section_1d import element_column, section_tf_1d

SHIFT_M = (15.0, -15.0, 30.0, -30.0)
PROBE_X_M = {"flat_west": -1500.0, "flat_east": 1605.0}
PEARSON_GATE = 0.02
# A flat-zone correction this large is the head moving ground the 1D already fits.
LARGE_DELTA = 0.05
# Convention gap explains the east error when it is at least half the head's |δ|.
CONVENTION_FRACTION = 0.5

HEADS = (
    ("residual", REPO / "checkpoints" / "case7_generalization" / "residual" / "pretrain_s0"),
    ("outcrop_prior12", REPO / "checkpoints" / "case7_generalization" / "outcrop_prior12" / "pretrain_s0"),
)
_DROP = (
    "config",
    "model",
    "data_loader",
    "pod_readout",
    "spectral_layers",
    "xt_readout",
    "losses",
    "curriculum",
    "train",
)


def soil_rock_zeta(vs_1m: np.ndarray) -> np.ndarray:
    """Case 7 zeta with rock at 0.1 of the soil value, the training soil/rock rule."""
    zeta = np.full(vs_1m.shape, ZETA, dtype=np.float64)
    zeta[np.asarray(vs_1m) > ROCK_VS] = ZETA * ROCK_ZETA_NORM
    return zeta


def spatial_gate_zone(center_pearson: float, off_pearson: float, abs_log: float, spread: float) -> dict:
    """Pass when the off-center read stays with the center read and the overlap spread."""
    gap = abs(float(off_pearson) - float(center_pearson))
    agree = float(abs_log) <= float(spread)
    passed = bool(gap <= PEARSON_GATE and agree and np.isfinite([center_pearson, off_pearson, abs_log, spread]).all())
    return {
        "center_pearson": float(center_pearson),
        "off_pearson": float(off_pearson),
        "pearson_gap": float(gap),
        "abs_log_median": float(abs_log),
        "overlap_spread": float(spread),
        "passed": passed,
    }


def convention_tracks(delta_log_rms: float, gap_log_rms: float) -> bool:
    """True when a 1D-convention difference is large enough to be the east error."""
    return bool(gap_log_rms >= CONVENTION_FRACTION * delta_log_rms and gap_log_rms >= LARGE_DELTA)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 0 else float("nan")


def _log_rms(a: np.ndarray, b: np.ndarray) -> float:
    d = np.log(np.maximum(a, EPS)) - np.log(np.maximum(b, EPS))
    return float(np.sqrt(np.mean(d**2)))


def _drop_model_modules() -> None:
    for name in list(sys.modules):
        if name in _DROP or name.startswith("config."):
            sys.modules.pop(name, None)
    for key in list(os.environ):
        if key.startswith("GIFNO_") or key == "TF_TARGET":
            del os.environ[key]


def _overlap_fast(log_tf: np.ndarray, centers: np.ndarray, trusted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log-mean TF at each center from every window that covers it."""
    n_win, nx, _n_freq = log_tf.shape
    origin = 0.5 * (nx - 1)
    step = float(centers[1] - centers[0]) if len(centers) > 1 else 1.0
    half = int(round((nx / 2) / step))
    overlap = np.full((n_win, log_tf.shape[-1]), np.nan)
    spread = np.full(n_win, np.nan)
    for i in range(n_win):
        j0 = max(0, i - half)
        j1 = min(n_win, i + half + 1)
        cols = np.rint(origin + centers[i] - centers[j0:j1]).astype(np.int64)
        ok = (cols >= 0) & (cols < nx)
        if not np.any(ok):
            continue
        stack = log_tf[np.arange(j0, j1)[ok], cols[ok]]
        overlap[i] = stack.mean(axis=0)
        spread[i] = float(stack[:, trusted].std(axis=0).mean())
    return np.exp(overlap), spread


def _read_shift(log_tf: np.ndarray, centers: np.ndarray, shift_m: float) -> np.ndarray:
    """Log TF at each center, read from the window centered ``shift_m`` away."""
    nx = log_tf.shape[1]
    origin = 0.5 * (nx - 1)
    col = int(round(origin - shift_m / DX_M))
    out = np.full((len(centers), log_tf.shape[-1]), np.nan)
    if not 0 <= col < nx:
        return out
    index = {round(float(c), 6): i for i, c in enumerate(centers)}
    for i, x in enumerate(centers):
        j = index.get(round(float(x) + shift_m, 6))
        if j is not None:
            out[i] = log_tf[j, col]
    return out


def _median_abs(a: np.ndarray, b: np.ndarray, rows: np.ndarray, mask: np.ndarray) -> float:
    d = np.abs(a[rows][:, mask] - b[rows][:, mask])
    return float(np.nanmedian(d))


def _zone_abs_log(x: np.ndarray, pred: np.ndarray, base: np.ndarray, freq: np.ndarray) -> dict:
    zones = zone_of(x)
    m = band_mask(freq, BANDS["trusted"])
    out = {}
    for zone in ZONES:
        rows = zones == zone
        out[zone] = float(np.median(np.abs(np.log(np.maximum(pred[rows][:, m], EPS)) - np.log(np.maximum(base[rows][:, m], EPS)))))
    return out


def _three_1d(freq: np.ndarray, vs_elem: np.ndarray, x_m: float) -> dict:
    """Case 7 Rayleigh, training Rayleigh (zeta 0.0075, f2=10), and constant-zeta prior."""
    from absolute_inputs import column_groups, column_tf1d_outcrop, column_tf1d_rayleigh

    col = element_column(vs_elem, x_m)
    f0, _ = column_groups(col[:, None], 5.0)
    case7 = section_tf_1d(freq, vs_elem, np.array([x_m]), "outcrop")[0]
    zeta = np.full_like(col, ZETA)
    train = column_tf1d_rayleigh(col[:, None], zeta[:, None], 5.0, freq)[:, 0]
    const = column_tf1d_outcrop(col[:, None], zeta[:, None], 5.0, freq)[:, 0]
    return {"f0_hz": float(f0[0]), "case7_rayleigh": case7, "train_rayleigh": train, "constant_zeta": const}


def _plot_conventions(freq: np.ndarray, curves: dict, pred: np.ndarray, gt: np.ndarray, title: str, path: Path) -> None:
    m = band_mask(freq, (0.1, 2.5))
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    ax.semilogx(freq[m], gt[m], color="k", lw=1.6, label="SVLAB ground truth")
    ax.semilogx(freq[m], curves["case7_rayleigh"][m], color="C1", lw=1.2, label="1D Case 7 Rayleigh")
    ax.semilogx(freq[m], curves["train_rayleigh"][m], color="C2", lw=1.1, label="1D training Rayleigh")
    ax.semilogx(freq[m], curves["constant_zeta"][m], color="C4", lw=1.1, label="1D constant zeta")
    ax.semilogx(freq[m], pred[m], color="C0", lw=1.3, label="model")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|TF|")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _score_head(name: str, model_dir: Path, pack: dict, vs_1m: np.ndarray, centers: np.ndarray, freq, tf_gt, tf1d, *, zeta=None, prior_mode: str = "constant", tf1d_mult=None) -> dict:
    _drop_model_modules()
    models, device, config, target = load_models([model_dir.resolve()])
    if not bool(getattr(models[0].head, "shared", False)):
        raise RuntimeError(f"{name} head is not the shared spatial readout")
    print(f"  forward {len(centers)} windows on {device}", flush=True)
    log_w = predict_windows(
        models, device, config, vs_1m, centers, batch=16, zeta=zeta, prior_mode=prior_mode
    )
    print("  aggregating overlap and off-center reads", flush=True)
    del models
    log_field = log_w[0]
    nx = log_field.shape[1]
    center_i = nx // 2
    recon = tf1d if tf1d_mult is None else tf1d_mult
    log_base = np.log(np.maximum(recon, EPS))
    if target == "residual":
        # The head emits log(TF / TF_1D) at every column. TF_1D depends only on
        # the physical x being read, so it is added at the query, not on the
        # whole 500 m field.
        log_query = log_field[:, center_i] + log_base
    else:
        log_query = log_field[:, center_i]
    m_tr = band_mask(freq, BANDS["trusted"])
    overlap_lin, spread = _overlap_fast(log_field, centers, m_tr)
    if target == "residual":
        overlap = overlap_lin * recon
    else:
        overlap = overlap_lin
    tf_center = np.exp(log_query)
    logs = [_read_shift(log_field, centers, s) for s in SHIFT_M]
    stacked = np.stack(logs)
    present = np.isfinite(stacked).all(axis=-1)
    off_log = np.full_like(log_field[:, center_i], np.nan)
    for i in range(len(centers)):
        ok = present[:, i]
        if ok.any():
            off_log[i] = np.mean(stacked[ok, i], axis=0)
    if target == "residual":
        off_log = off_log + log_base
    tf_off = np.exp(off_log)
    finite = np.isfinite(off_log[:, 0])
    zones = zone_of(centers)
    gate = {}
    for zone in ZONES:
        rows = np.flatnonzero(zones == zone)
        usable = rows[np.isfinite(tf_off[rows, 0])]
        if usable.size == 0:
            gate[zone] = {"passed": False, "n": 0}
            continue
        c_p = float(np.median([_pearson(tf_center[i, m_tr], tf_gt[i, m_tr]) for i in usable]))
        o_p = float(np.median([_pearson(tf_off[i, m_tr], tf_gt[i, m_tr]) for i in usable]))
        abs_log = _median_abs(np.log(np.maximum(tf_center, EPS)), np.log(np.maximum(tf_off, EPS)), usable, m_tr)
        gate[zone] = spatial_gate_zone(c_p, o_p, abs_log, float(np.nanmedian(spread[usable])))
        gate[zone]["n"] = int(usable.size)
    return {
        "model": name,
        "dir": str(model_dir),
        "target": target,
        "prior_mode": prior_mode,
        "shared": True,
        "center": zone_scores(centers, tf_center, tf_gt, freq),
        "overlap": zone_scores(centers, overlap, tf_gt, freq),
        "off_center": zone_scores(centers[finite], tf_off[finite], tf_gt[finite], freq),
        "correction_abs_log_trusted": _zone_abs_log(centers, tf_center, tf1d, freq),
        "window_spread_log_trusted": {z: float(np.nanmedian(spread[zones == z])) for z in ZONES},
        "spatial_gate": gate,
        "spatial_gate_passed": all(gate[z].get("passed", False) for z in ZONES),
        "_arrays": {"tf_center": tf_center, "tf_off": tf_off, "spread": spread},
    }


def _plot_with_gt(freq, pack, centers, tf_gt, tf_center, out_dir: Path, head: str) -> dict:
    """Rewrite convention plots with SVLAB on them and return the report."""
    report = {}
    for zone, x_m in PROBE_X_M.items():
        i = int(np.argmin(np.abs(centers - x_m)))
        curves = _three_1d(freq, pack["vs"], float(centers[i]))
        pred = tf_center[i]
        trusted = band_mask(freq, BANDS["trusted"])
        low = band_mask(freq, LOW_BAND_HZ)
        row = {
            "x_m": float(centers[i]),
            "f0_hz": curves["f0_hz"],
            "delta_log_rms_trusted": _log_rms(pred[trusted], curves["case7_rayleigh"][trusted]),
            "train_vs_case7_log_rms_trusted": _log_rms(curves["train_rayleigh"][trusted], curves["case7_rayleigh"][trusted]),
            "constant_vs_case7_log_rms_trusted": _log_rms(curves["constant_zeta"][trusted], curves["case7_rayleigh"][trusted]),
            "train_vs_case7_log_rms_low": _log_rms(curves["train_rayleigh"][low], curves["case7_rayleigh"][low]),
        }
        row["tracks_train_rayleigh"] = convention_tracks(row["delta_log_rms_trusted"], row["train_vs_case7_log_rms_trusted"])
        row["tracks_constant_zeta"] = convention_tracks(row["delta_log_rms_trusted"], row["constant_vs_case7_log_rms_trusted"])
        report[zone] = row
        _plot_conventions(freq, curves, pred, tf_gt[i], f"{head}, x = {centers[i]:.0f} m ({zone})", out_dir / f"conventions_{zone}.png")
    return report


def _training_tf1d(vs_1m: np.ndarray, zeta: np.ndarray, centers: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Training-convention Rayleigh |TF| at each window center. Shape (n, F)."""
    from case7a import N_FINE_Z, X_LEFT_M

    gifno = str(REPO / "experiments" / "GIFNO")
    if gifno not in sys.path:
        sys.path.append(gifno)
    from absolute_inputs import column_tf1d_rayleigh

    idx = np.rint(np.asarray(centers) - X_LEFT_M).astype(np.int64)
    cols = np.asarray(vs_1m[:N_FINE_Z, idx], dtype=np.float64)
    z = np.asarray(zeta[:N_FINE_Z, idx], dtype=np.float64)
    return column_tf1d_rayleigh(cols, z, DX_M, freq).T


def run_aligned(out_dir: Path) -> dict:
    """Rescore with soil/rock zeta and the training Rayleigh prior and multiply-back."""
    from case7a import ensure_cache

    out_dir.mkdir(parents=True, exist_ok=True)
    pack = ensure_cache()
    vs_1m = paint_section_1m(pack["vs"])
    zeta = soil_rock_zeta(vs_1m)
    centers = dense_center_x()
    nodes = np.array([__import__("case7a", fromlist=["window_node_index"]).window_node_index(float(x)) for x in centers])
    freq, tf_gt, _scale = outcrop_reference_transfer_function(pack, nodes)
    tf1d = section_tf_1d(freq, pack["vs"], centers, "outcrop")
    tf_train = _training_tf1d(vs_1m, zeta, centers, freq)
    summary = {"prior_mode": "rayleigh", "zeta": "soil_rock", "heads": {}}
    for name, model_dir in HEADS:
        print(f"=== aligned {name} ===", flush=True)
        scored = _score_head(
            name,
            model_dir,
            pack,
            vs_1m,
            centers,
            freq,
            tf_gt,
            tf1d,
            zeta=zeta,
            prior_mode="rayleigh",
            tf1d_mult=tf_train,
        )
        scored.pop("_arrays", None)
        head_dir = out_dir / name
        head_dir.mkdir(parents=True, exist_ok=True)
        (head_dir / "metrics.json").write_text(json.dumps(scored, indent=2) + "\n")
        summary["heads"][name] = {
            "correction_abs_log_trusted": scored["correction_abs_log_trusted"],
            "center_trusted": {z: scored["center"][z]["trusted"]["pearson_median"] for z in ZONES},
            "center_low": {z: scored["center"][z]["low"]["pearson_median"] for z in ZONES},
            "spatial_gate_passed": scored["spatial_gate_passed"],
        }
        _print_head(name, scored)
    east = summary["heads"]["residual"]["correction_abs_log_trusted"]["flat_east"]
    east_b = summary["heads"]["outcrop_prior12"]["correction_abs_log_trusted"]["flat_east"]
    summary["flat_east_delta_still_large"] = bool(min(east, east_b) > LARGE_DELTA)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run(out_dir: Path) -> dict:
    from case7a import ensure_cache

    out_dir.mkdir(parents=True, exist_ok=True)
    pack = ensure_cache()
    vs_1m = paint_section_1m(pack["vs"])
    centers = dense_center_x()
    nodes = np.array([__import__("case7a", fromlist=["window_node_index"]).window_node_index(float(x)) for x in centers])
    freq, tf_gt, scale = outcrop_reference_transfer_function(pack, nodes)
    tf1d = section_tf_1d(freq, pack["vs"], centers, "outcrop")
    summary = {
        "input_scale": float(scale),
        "n_centers": int(len(centers)),
        "tf1d_prior": zone_scores(centers, tf1d, tf_gt, freq),
        "heads": {},
        "west_context": {
            "implemented": False,
            "reason": (
                "The 0.1–0.4 Hz west hole is outside a 500 m window. "
                "No full-section context channel was added, and it is not part of any fine-tune."
            ),
        },
    }
    arrays = {}
    for name, model_dir in HEADS:
        print(f"=== {name} ===", flush=True)
        scored = _score_head(name, model_dir, pack, vs_1m, centers, freq, tf_gt, tf1d)
        arrays[name] = scored.pop("_arrays")
        head_dir = out_dir / name
        head_dir.mkdir(parents=True, exist_ok=True)
        scored["conventions"] = _plot_with_gt(freq, pack, centers, tf_gt, arrays[name]["tf_center"], head_dir, name)
        (head_dir / "metrics.json").write_text(json.dumps(scored, indent=2) + "\n")
        summary["heads"][name] = {
            "target": scored["target"],
            "correction_abs_log_trusted": scored["correction_abs_log_trusted"],
            "spatial_gate_passed": scored["spatial_gate_passed"],
            "spatial_gate": scored["spatial_gate"],
            "center_trusted": {z: scored["center"][z]["trusted"]["pearson_median"] for z in ZONES},
            "center_low": {z: scored["center"][z]["low"]["pearson_median"] for z in ZONES},
            "overlap_trusted": {z: scored["overlap"][z]["trusted"]["pearson_median"] for z in ZONES},
            "conventions": scored["conventions"],
        }
        _print_head(name, scored)
    east = summary["heads"]["residual"]["conventions"]["flat_east"]
    summary["align_inference"] = bool(east["tracks_train_rayleigh"] or east["tracks_constant_zeta"])
    summary["align_reason"] = (
        "flat-east head correction tracks a 1D convention gap"
        if summary["align_inference"]
        else "flat-east 1D convention gap is smaller than the head correction"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez(out_dir / "centers.npz", x=centers, freq=freq, tf_gt=tf_gt, tf1d=tf1d)
    return summary


def _print_head(name: str, scored: dict) -> None:
    print(f"{name} spatial_gate_passed={scored['spatial_gate_passed']}")
    for zone, corr in scored["correction_abs_log_trusted"].items():
        t = scored["center"][zone]["trusted"]["pearson_median"]
        low = scored["center"][zone]["low"]["pearson_median"]
        ov = scored["overlap"][zone]["trusted"]["pearson_median"]
        print(f"  {zone:12s} center P {t:.3f} low P {low:.3f} overlap P {ov:.3f} |δ| {corr:.3f}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "checkpoints" / "case7_generalization" / "svlab_diagnosis",
    )
    ap.add_argument("--aligned", action="store_true")
    args = ap.parse_args()
    if args.aligned:
        run_aligned(args.out_dir)
    else:
        run(args.out_dir)


if __name__ == "__main__":
    main()
