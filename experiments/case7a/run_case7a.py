#!/usr/bin/env python3
"""Score LOGLO-POD on Treasure Island Case 7a against the provided accelerations.

Reference |TF| is the training label: Konno-Ohmachi smoothed FAS of horizontal
surface acceleration divided by the FAS of the center downhole. The checkpoint
stores amplitude only, so waveforms are not compared.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from case7a import (
    DT_RESULTS_S,
    ELEM_M,
    GEOLOGIC_X_MAX,
    GEOLOGIC_X_MIN,
    LX_M,
    DEPTH_M,
    LZ_M,
    N_ROW,
    NX,
    NZ_MAX,
    REPO,
    STRIP_HALF_M,
    WINDOW_LIMIT_M,
    X_LEFT_M,
    X_RIGHT_M,
    ZETA,
    build_model_input,
    build_window_input,
    center_station_index,
    ensure_cache,
    node_x_m,
    recorder_column_indices,
    resample_vs_full_depth,
    station_node_indices,
    station_x_m,
    window_center_x,
    window_node_index,
)

SEISKIT_ROOT = Path("/home/kurt-/seiskit/seiskit")
CHECKPOINT = REPO / "checkpoints" / "tier2_pod64_full7680" / "best_model.pt"
GIFNO_SCREEN = REPO / "data" / "gifno_screen"
LOGLO_DIR = REPO / "experiments" / "GIFNO-FDO-XT-LOGLO-POD"
OUT_DIR = Path(__file__).resolve().parent / "results"

N_FREQ = 1000
SMOOTH_COEFF = 500
FREQ_LO = 0.1
FREQ_HI = 10.0
BANDS = {
    "full": (0.1, 10.0),
    "trusted_0.1_2.5": (0.1, 2.5),
    "core_0.4_2.5": (0.4, 2.5),
    "low_0.1_0.5": (0.1, 0.5),
    "mid_0.5_2": (0.5, 2.0),
    "high_2_10": (2.0, 10.0),
}
_EPS = 1e-12

NOTES = [
    "Reference |TF| is Konno-Ohmachi smoothed FAS(surface horizontal) / "
    "FAS(downhole), 1000 log bins from 0.1 to 10 Hz, coefficient 500.",
    "The downhole is the only rock acceleration, at x = 0 m and 120 m depth. "
    "It is the denominator at all 21 stations. The center station is the "
    "colocated comparison.",
    "Model input is the center 500 m painted from 5 m cells onto a 1 m grid, "
    "top 128 m of the 150 m column. Zeta is set to 0.0075 and divided by its "
    "maximum, so the zeta channel is identically 1. Any uniform damping ratio "
    "normalizes the same way.",
    "Training damping is a zeta field, usually a few percent. This case is "
    "Rayleigh damping at 0.75% at 0.75 Hz and 2.25 Hz.",
    "The Vs profile is gradational. Residual GINO is not scored.",
    "Stations span x = -150 m to 150 m. The rest of the 4000 m section is "
    "outside the operator.",
    "Results are sampled at dt = 0.001 s. The instruction sheet lists an "
    "analysis step of 0.01 s. The saved series are the reference.",
]


class _BlockNumba:
    """Numba's cache locator rejects this synthetic package import."""

    def find_spec(self, name, path, target=None):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("numba disabled while importing seiskit TTF")
        return None


def load_ttf_batch_fast():
    """Import seiskit TTF without executing seiskit/__init__.py (OpenSees)."""
    if "seiskit.ttf.TTF" in sys.modules:
        return sys.modules["seiskit.ttf.TTF"].TTF_batch_fast
    if not SEISKIT_ROOT.is_dir():
        raise FileNotFoundError(f"seiskit package not found at {SEISKIT_ROOT}")
    pkg = types.ModuleType("seiskit")
    pkg.__path__ = [str(SEISKIT_ROOT)]
    pkg.__package__ = "seiskit"
    sys.modules["seiskit"] = pkg
    ttf = types.ModuleType("seiskit.ttf")
    ttf.__path__ = [str(SEISKIT_ROOT / "ttf")]
    ttf.__package__ = "seiskit.ttf"
    sys.modules["seiskit.ttf"] = ttf
    blocked = [key for key in sys.modules if key == "numba" or key.startswith("numba.")]
    saved = {key: sys.modules.pop(key) for key in blocked}
    blocker = _BlockNumba()
    sys.meta_path.insert(0, blocker)
    try:
        from seiskit.ttf.TTF import TTF_batch_fast
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
    return TTF_batch_fast


def reference_transfer_function(
    surface_h: np.ndarray,
    downhole: np.ndarray,
    nodes: np.ndarray,
    dt: float = DT_RESULTS_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Surface/downhole |TF| at ``nodes``. Both arrays are (n_time, n_chan)."""
    ttf = load_ttf_batch_fast()
    surf = np.ascontiguousarray(surface_h[:, nodes].T, dtype=np.float64)
    base_1d = np.asarray(downhole, dtype=np.float64)
    base = np.repeat(base_1d[None, :], surf.shape[0], axis=0)
    freq, mag = ttf(
        base,
        surf,
        dt=float(dt),
        n_points=N_FREQ,
        smooth_coeff=SMOOTH_COEFF,
        dz=1.0,
    )
    return freq.astype(np.float64), mag.astype(np.float32)


def outcrop_reference_transfer_function(pack: dict, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Surface / (c * input) |TF| at ``nodes``; ``c`` fitted against 1D in the flat east zone.

    Unlike surface/downhole, this label is defined the same way at every x and
    matches the relabeled training sets (``relabel_outcrop.py``).
    """
    from labels import CALIBRATION_BAND_HZ, calibrate_input_scale, case7_input_acceleration, tf_over_input
    from section_1d import section_tf_1d

    from case7a import ZONES, zone_of

    ttf = load_ttf_batch_fast()
    dt = float(pack["dt"])
    surface = np.asarray(pack["surface_h"], dtype=np.float64)
    input_acc = case7_input_acceleration(pack["ricker"], np.arange(surface.shape[0]) * dt)
    freq, tf = tf_over_input(surface[:, nodes], input_acc, dt, ttf)

    centers = window_center_x()
    east = centers[zone_of(centers) == "flat_east"]
    if east.size == 0:
        raise RuntimeError(f"no window centers in flat_east {ZONES['flat_east']}")
    east_nodes = np.array([window_node_index(float(x)) for x in east], dtype=np.int64)
    _, tf_east = tf_over_input(surface[:, east_nodes], input_acc, dt, ttf)
    tf1d_east = section_tf_1d(freq, pack["vs"], east, "outcrop")
    scale = calibrate_input_scale(tf_east, tf1d_east, freq, CALIBRATION_BAND_HZ)
    return freq, (tf / scale).astype(np.float32), scale


def case7_reference(pack: dict, nodes: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    if label == "downhole":
        return reference_transfer_function(pack["surface_h"], pack["downhole"], nodes, dt=float(pack["dt"]))
    if label == "outcrop":
        freq, tf, _ = outcrop_reference_transfer_function(pack, nodes)
        return freq, tf
    raise ValueError(f"label must be 'downhole' or 'outcrop', got {label!r}")


def _import_loglo():
    import os

    os.environ["GIFNO_DATA_ROOT"] = str(GIFNO_SCREEN)
    os.environ["GIFNO_POD_NUM_MODES"] = "64"
    if str(LOGLO_DIR) not in sys.path:
        sys.path.insert(0, str(LOGLO_DIR))
    gifno = REPO / "experiments" / "GIFNO"
    if str(gifno) not in sys.path:
        sys.path.insert(1, str(gifno))
    import config
    from model import create_model

    return config, create_model


def predict_transfer_function(
    model_input: np.ndarray,
    checkpoint: Path = CHECKPOINT,
    *,
    pod_modes_path: Path | None = None,
    pod_mean_path: Path | None = None,
    pod_space: str = "linear",
) -> np.ndarray:
    """LOGLO-POD |TF| at the 21 training recorders. Shape (21, n_freq)."""
    model, device = _load_loglo(
        checkpoint,
        pod_modes_path=pod_modes_path,
        pod_mean_path=pod_mean_path,
        pod_space=pod_space,
    )
    grid = _forward(model, device, model_input[None])
    rec = recorder_column_indices()
    return grid[0, rec, :].astype(np.float32)


def _load_loglo(
    checkpoint: Path,
    *,
    pod_modes_path: Path | None = None,
    pod_mean_path: Path | None = None,
    pod_space: str = "linear",
):
    import torch

    _config, create_model = _import_loglo()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        pod_modes_path=pod_modes_path,
        pod_mean_path=pod_mean_path,
        pod_space=pod_space,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device


def _forward(model, device, batch: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        grid = model(torch.from_numpy(np.ascontiguousarray(batch)).to(device))
    return grid.detach().cpu().numpy()


def predict_center_batch(
    model_inputs: np.ndarray,
    checkpoint: Path = CHECKPOINT,
    *,
    model=None,
    device=None,
    batch_size: int = 8,
    pod_modes_path: Path | None = None,
    pod_mean_path: Path | None = None,
    pod_space: str = "linear",
) -> np.ndarray:
    """Center-station |TF| for a stack of windows. Shape (n_windows, n_freq)."""
    if model_inputs.ndim != 4 or model_inputs.shape[1:] != (4, NZ_MAX, NX):
        raise ValueError(f"Expected (N, 4, {NZ_MAX}, {NX}), got {model_inputs.shape}")
    if model is None or device is None:
        model, device = _load_loglo(
            checkpoint,
            pod_modes_path=pod_modes_path,
            pod_mean_path=pod_mean_path,
            pod_space=pod_space,
        )
    center_col = int(recorder_column_indices()[len(recorder_column_indices()) // 2])
    parts: list[np.ndarray] = []
    for start in range(0, model_inputs.shape[0], batch_size):
        grid = _forward(model, device, model_inputs[start : start + batch_size])
        parts.append(grid[:, center_col, :].astype(np.float32))
    return np.concatenate(parts, axis=0)


def predict_recorder_batch(
    model_inputs: np.ndarray,
    checkpoint: Path = CHECKPOINT,
    *,
    model=None,
    device=None,
    batch_size: int = 8,
    pod_modes_path: Path | None = None,
    pod_mean_path: Path | None = None,
    pod_space: str = "linear",
) -> np.ndarray:
    """All 21 recorder |TF|s. Shape (n_windows, 21, n_freq)."""
    if model_inputs.ndim != 4 or model_inputs.shape[1:] != (4, NZ_MAX, NX):
        raise ValueError(f"Expected (N, 4, {NZ_MAX}, {NX}), got {model_inputs.shape}")
    if model is None or device is None:
        model, device = _load_loglo(
            checkpoint,
            pod_modes_path=pod_modes_path,
            pod_mean_path=pod_mean_path,
            pod_space=pod_space,
        )
    rec = recorder_column_indices()
    parts: list[np.ndarray] = []
    for start in range(0, model_inputs.shape[0], batch_size):
        grid = _forward(model, device, model_inputs[start : start + batch_size])
        parts.append(grid[:, rec, :].astype(np.float32))
    return np.concatenate(parts, axis=0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    a64 = a64 - a64.mean()
    b64 = b64 - b64.mean()
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    if denom < _EPS:
        return float("nan")
    return float(np.dot(a64, b64) / denom)


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(true, dtype=np.float64).ravel()
    return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + _EPS))


def _pair_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {"rel_l2": _rel_l2(pred, true), "pearson": _pearson(pred, true)}


def score_transfer_functions(
    pred: np.ndarray,
    true: np.ndarray,
    freq: np.ndarray,
) -> dict:
    center = center_station_index()
    out: dict = {"center": {}, "stations_21": {}, "per_station": []}
    for name, (lo, hi) in BANDS.items():
        mask = (freq >= lo) & (freq <= hi)
        out["center"][name] = _pair_metrics(pred[center, mask], true[center, mask])
        out["stations_21"][name] = _pair_metrics(pred[:, mask], true[:, mask])
    x = station_x_m()
    for i, xv in enumerate(x):
        row: dict = {"x_m": float(xv), "is_center": bool(i == center)}
        for name, (lo, hi) in BANDS.items():
            mask = (freq >= lo) & (freq <= hi)
            metrics = _pair_metrics(pred[i, mask], true[i, mask])
            row[f"pearson_{name}"] = metrics["pearson"]
            row[f"rel_l2_{name}"] = metrics["rel_l2"]
        out["per_station"].append(row)
    return out


_BAND_ORDER = ("full", "low_0.1_0.5", "mid_0.5_2", "high_2_10")
_BAND_LABELS = {
    "full": "0.1–10 Hz",
    "low_0.1_0.5": "0.1–0.5 Hz",
    "mid_0.5_2": "0.5–2 Hz",
    "high_2_10": "2–10 Hz",
}
_BAND_COLORS = {
    "full": "k",
    "low_0.1_0.5": "C0",
    "mid_0.5_2": "C1",
    "high_2_10": "C2",
}


def _vs_depth_edges() -> np.ndarray:
    return np.arange(N_ROW + 1, dtype=np.float64) * ELEM_M


def plot_vs(vs: np.ndarray, out_path: Path) -> None:
    x_edge = X_LEFT_M + ELEM_M * np.arange(vs.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(11, 3.2), constrained_layout=True)
    mesh = ax.pcolormesh(
        x_edge,
        _vs_depth_edges(),
        vs,
        shading="flat",
        cmap="viridis",
    )
    ax.invert_yaxis()
    ax.add_patch(
        Rectangle(
            (-STRIP_HALF_M, 0.0),
            2 * STRIP_HALF_M,
            DEPTH_M,
            fill=False,
            edgecolor="#ff2d2d",
            linewidth=1.2,
        )
    )
    ax.axvline(GEOLOGIC_X_MIN, color="white", lw=0.9, ls="--")
    ax.axvline(GEOLOGIC_X_MAX, color="white", lw=0.9, ls="--")
    ax.plot(0.0, 120.0, marker="o", color="#ff2d2d", markersize=5)
    ax.annotate(
        "downhole 120 m",
        (0.0, 120.0),
        textcoords="offset points",
        xytext=(8, -12),
        color="#ff2d2d",
        fontsize=8,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth (m)")
    ax.set_title(
        "Case 7a Vs. Red box is the center 500 m window. "
        "White dashes mark the geologic interval."
    )
    fig.colorbar(mesh, ax=ax, label="Vs (m/s)", pad=0.01)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pga(surface_h: np.ndarray, surface_v: np.ndarray, out_path: Path) -> None:
    x = node_x_m()
    pga_h = np.max(np.abs(surface_h), axis=0)
    pga_v = np.max(np.abs(surface_v), axis=0)
    fig, ax = plt.subplots(figsize=(11, 3.2), constrained_layout=True)
    ax.axvspan(
        GEOLOGIC_X_MIN, GEOLOGIC_X_MAX, color="0.85", label="geologic interval"
    )
    ax.plot(x, pga_h, color="C0", label="horizontal PGA")
    ax.plot(x, pga_v, color="C1", label="vertical PGA")
    ax.axvline(0.0, color="0.3", linewidth=0.8, linestyle="--")
    ax.set_xlim(X_LEFT_M, X_RIGHT_M)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("peak |acceleration|")
    ax.set_title("Surface PGA. Vertical motion is shown and not scored.")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _freq_edges(freq: np.ndarray) -> np.ndarray:
    log_f = np.log10(freq)
    f_edge = np.empty(freq.size + 1, dtype=np.float64)
    f_edge[1:-1] = 0.5 * (log_f[:-1] + log_f[1:])
    f_edge[0] = log_f[0] - (f_edge[1] - log_f[0])
    f_edge[-1] = log_f[-1] + (log_f[-1] - f_edge[-2])
    return 10.0**f_edge


def _station_x_edges() -> np.ndarray:
    x = station_x_m()
    return np.concatenate([x - 7.5, [x[-1] + 7.5]])


def plot_center_tf(
    freq: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    out_path: Path,
    extras: list[tuple[np.ndarray, str, str]] | None = None,
) -> None:
    center = center_station_index()
    true_1d = true[center] if np.ndim(true) == 2 else true
    pred_1d = pred[center] if np.ndim(pred) == 2 else pred
    curves = [
        (
            pred_1d,
            "0.45" if extras else "C0",
            "top 128 m, flat zeta" if extras else "LOGLO-POD",
        )
    ]
    for series, color, label in extras or []:
        series_1d = series[center] if np.ndim(series) == 2 else series
        curves.append((series_1d, color, label))
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    ax.semilogx(freq, true_1d, color="k", label="reference")
    for series_1d, color, label in curves:
        ax.semilogx(freq, series_1d, color=color, label=label)
    ax.axhline(0.0, color="0.75", lw=0.5)
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|TF|")
    ax.set_title("Center station, x = 0 m, surface / downhole")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_station_tfs(
    freq: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    out_path: Path,
    extras: list[tuple[np.ndarray, str, str]] | None = None,
) -> None:
    x_edge = _station_x_edges()
    f_edge = _freq_edges(freq)
    fields = [("reference |TF|", true)]
    fields.append(
        (
            (
                "top 128 m, flat zeta"
                if extras
                else "LOGLO-POD |TF|"
            )
            + " (below 0 is black)",
            pred,
        )
    )
    for series, _color, label in extras or []:
        fields.append((f"{label} (below 0 is black)", series))
    vmax = float(
        np.percentile(np.concatenate([field.ravel() for _, field in fields]), 99)
    )
    vmax = max(vmax, 1e-3)
    fig, axes = plt.subplots(
        len(fields),
        1,
        figsize=(8.5, 2.4 * len(fields)),
        sharex=True,
        constrained_layout=True,
    )
    if len(fields) == 1:
        axes = [axes]
    for ax, (title, field) in zip(axes, fields, strict=True):
        mesh = ax.pcolormesh(
            x_edge,
            f_edge,
            field.T,
            shading="flat",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_yscale("log")
        ax.set_ylim(FREQ_LO, FREQ_HI)
        ax.set_ylabel("frequency (Hz)")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, label="|TF|", pad=0.01)
    axes[-1].set_xlabel("x (m)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tf_grid(
    freq: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    out_path: Path,
    extras: list[tuple[np.ndarray, str, str]] | None = None,
) -> None:
    """One panel per station, shared axes so amplitudes stay comparable."""
    x = station_x_m()
    center = center_station_index()
    curves = [
        (
            pred,
            "0.45" if extras else "C0",
            "top 128 m" if extras else "LOGLO-POD",
        )
    ]
    for series, color, label in extras or []:
        short = label.removeprefix("full 150 m, ").removesuffix(" zeta")
        curves.append((series, color, short))
    fig, axes = plt.subplots(
        3, 7, figsize=(14, 6.4), sharex=True, sharey=True, constrained_layout=True
    )
    stacked = np.concatenate([true, *[series for series, _, _ in curves]], axis=0)
    y_lo = min(float(stacked.min()), 0.0) - 2.0
    y_hi = float(stacked.max()) * 1.05
    for i, ax in enumerate(axes.ravel()):
        ax.semilogx(freq, true[i], color="k", lw=0.9, label="reference")
        for series, color, label in curves:
            ax.semilogx(freq, series[i], color=color, lw=0.8, label=label)
        ax.set_xlim(FREQ_LO, FREQ_HI)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"x = {x[i]:.0f} m", fontsize=8)
        ax.axhline(0.0, color="0.75", lw=0.5)
        if i == center:
            for spine in ax.spines.values():
                spine.set_color("C3")
                spine.set_linewidth(1.6)
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.supxlabel("frequency (Hz)")
    fig.supylabel("|TF|")
    fig.suptitle("Surface / downhole |TF|. Red frame is the center station.")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tf_median(
    freq: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    out_path: Path,
    extras: list[tuple[np.ndarray, str, str]] | None = None,
) -> None:
    """Median across the 21 stations. Reference band is the 16–84% range."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    med = np.median(true, axis=0)
    lo = np.percentile(true, 16, axis=0)
    hi = np.percentile(true, 84, axis=0)
    ax.fill_between(freq, lo, hi, color="k", alpha=0.12, linewidth=0)
    ax.semilogx(freq, med, color="k", lw=1.4, label="reference")
    curves = [
        (
            pred,
            "0.45" if extras else "C0",
            "top 128 m, flat zeta" if extras else "LOGLO-POD",
        )
    ]
    curves.extend(extras or [])
    for field, color, label in curves:
        ax.semilogx(freq, np.median(field, axis=0), color=color, lw=1.3, label=label)
    ax.axhline(0.0, color="0.75", lw=0.5)
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|TF|")
    ax.set_title("Median |TF| across the center-window stations")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pearson_box(
    per_station: list[dict],
    out_path: Path,
    extra_rows: list[tuple[list[dict], str, str]] | None = None,
) -> None:
    groups = [
        (
            "top 128 m" if extra_rows else "station",
            per_station,
            "0.35" if extra_rows else "C0",
        )
    ]
    for rows, color, label in extra_rows or []:
        groups.append((label, rows, color))
    n_groups = len(groups)
    width = 0.8 / n_groups
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    rng = np.random.default_rng(0)
    for g, (label, rows, color) in enumerate(groups):
        data = [[row[f"pearson_{name}"] for row in rows] for name in _BAND_ORDER]
        offset = (g - (n_groups - 1) / 2) * width
        positions = np.arange(1, len(_BAND_ORDER) + 1, dtype=float) + offset
        ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            showfliers=False,
            manage_ticks=False,
            medianprops={"color": "0.15"},
            boxprops={"color": color},
            whiskerprops={"color": color},
            capprops={"color": color},
        )
        for i, vals in enumerate(data):
            jitter = rng.uniform(-width * 0.15, width * 0.15, size=len(vals))
            ax.scatter(
                np.full(len(vals), positions[i]) + jitter,
                vals,
                s=12,
                color=color,
                zorder=3,
                label=label if i == 0 else None,
            )
        if any(row.get("is_center") for row in rows):
            center = [
                next(row[f"pearson_{name}"] for row in rows if row.get("is_center"))
                for name in _BAND_ORDER
            ]
            ax.scatter(
                positions,
                center,
                s=36,
                marker="D",
                facecolors="none",
                edgecolors="k",
                zorder=4,
                label="center, x = 0" if g == 0 else None,
            )
    ax.set_xticks(np.arange(1, len(_BAND_ORDER) + 1))
    ax.set_xticklabels([_BAND_LABELS[name] for name in _BAND_ORDER])
    ax.axhline(0.0, color="0.55", lw=0.7)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Pearson r")
    ax.set_title("Per-station correlation of |TF|")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_vs_x(
    per_station: list[dict],
    out_path: Path,
    extra_rows: list[tuple[list[dict], str, str]] | None = None,
) -> None:
    x = np.array([row["x_m"] for row in per_station])
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 5.4), sharex=True, constrained_layout=True
    )
    if extra_rows:
        groups = [(per_station, "0.45", "top 128 m, flat zeta")]
        groups.extend(extra_rows)
        for rows, color, label in groups:
            axes[0].plot(
                x,
                [row["pearson_full"] for row in rows],
                marker="o",
                ms=3.5,
                color=color,
                label=label,
            )
            axes[1].plot(
                x,
                [row["rel_l2_full"] for row in rows],
                marker="o",
                ms=3.5,
                color=color,
                label=label,
            )
        axes[0].set_title("Center-window stations, 0.1–10 Hz")
        axes[0].legend(frameon=False, fontsize=8)
    else:
        for name in _BAND_ORDER:
            color = _BAND_COLORS[name]
            label = _BAND_LABELS[name]
            axes[0].plot(
                x,
                [row[f"pearson_{name}"] for row in per_station],
                marker="o",
                ms=3.5,
                color=color,
                label=label,
            )
            axes[1].plot(
                x,
                [row[f"rel_l2_{name}"] for row in per_station],
                marker="o",
                ms=3.5,
                color=color,
                label=label,
            )
        axes[0].set_title("Station scores along the center strip")
        axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[0].axhline(0.0, color="0.55", lw=0.7)
    axes[0].set_ylabel("Pearson r")
    axes[0].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("relative L2")
    axes[1].set_xlabel("x (m)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tf_scatter(
    freq: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    out_path: Path,
    extras: list[tuple[np.ndarray, str, str]] | None = None,
) -> None:
    shown = ("low_0.1_0.5", "mid_0.5_2", "high_2_10")
    curves = [
        (
            pred,
            "0.45" if extras else "C0",
            "top 128 m" if extras else "LOGLO-POD",
        )
    ]
    curves.extend(extras or [])
    fig, axes = plt.subplots(
        1, 3, figsize=(10.6, 3.8), sharex=True, sharey=True, constrained_layout=True
    )
    stacked_true = true
    stacked_pred = np.concatenate([series for series, _, _ in curves], axis=0)
    x_hi = float(np.max(stacked_true)) * 1.05
    y_lo = min(float(np.min(stacked_pred)), 0.0) - 1.0
    y_hi = max(float(np.max(stacked_pred)), x_hi)
    line = [min(0.0, y_lo), max(x_hi, y_hi)]
    for ax, name in zip(axes, shown, strict=True):
        lo, hi = BANDS[name]
        mask = (freq >= lo) & (freq <= hi)
        for series, color, label in curves:
            ax.scatter(
                true[:, mask].ravel(),
                series[:, mask].ravel(),
                s=6,
                alpha=0.15,
                color=color,
                linewidths=0,
                label=label,
            )
        ax.plot(line, line, color="k", lw=0.8)
        ax.set_xlim(0.0, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(_BAND_LABELS[name])
        ax.set_xlabel("reference |TF|")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("predicted |TF|")
    axes[0].legend(frameon=False, fontsize=7, markerscale=2)
    fig.suptitle("Predicted versus reference |TF|. Line is equality.")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tf_error(
    freq: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    out_path: Path,
    extras: list[tuple[np.ndarray, str, str]] | None = None,
) -> None:
    x_edge = _station_x_edges()
    f_edge = _freq_edges(freq)
    panels = [
        (
            "top 128 m, flat zeta" if extras else "LOGLO-POD minus reference",
            pred - true,
        )
    ]
    for series, _color, label in extras or []:
        panels.append((label, series - true))
    lim = float(
        np.percentile(
            np.concatenate([np.abs(err).ravel() for _, err in panels]), 98
        )
    )
    lim = max(lim, 1e-3)
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(8.5, 2.6 * len(panels)),
        sharex=True,
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, err) in zip(axes, panels, strict=True):
        mesh = ax.pcolormesh(
            x_edge,
            f_edge,
            err.T,
            shading="flat",
            cmap="coolwarm",
            vmin=-lim,
            vmax=lim,
        )
        ax.set_yscale("log")
        ax.set_ylim(FREQ_LO, FREQ_HI)
        ax.set_ylabel("frequency (Hz)")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, label="|TF| error", pad=0.01)
    axes[-1].set_xlabel("x (m)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pearson_vs_frequency(
    freq: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    out_path: Path,
    extras: list[tuple[np.ndarray, str, str]] | None = None,
) -> None:
    """Pearson across the 21 stations at each frequency."""

    def _smooth(field: np.ndarray) -> np.ndarray:
        corr = np.array(
            [_pearson(field[:, j], true[:, j]) for j in range(freq.size)]
        )
        half = 10
        smooth = np.empty_like(corr)
        for j in range(freq.size):
            lo = max(0, j - half)
            hi = min(freq.size, j + half + 1)
            smooth[j] = float(np.median(corr[lo:hi]))
        return smooth

    curves = [
        (
            pred,
            "0.45" if extras else "C0",
            "top 128 m, flat zeta" if extras else "median of 21 bins",
        )
    ]
    curves.extend(extras or [])
    fig, ax = plt.subplots(figsize=(7.6, 3.8), constrained_layout=True)
    for field, color, label in curves:
        ax.semilogx(freq, _smooth(field), color=color, lw=1.5, label=label)
    ax.axhline(0.0, color="0.55", lw=0.7)
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("Pearson r across stations")
    ax.set_title("Spatial-pattern correlation, median of 21 frequency bins")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_comparison_plots(
    out_dir: Path,
    freq: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    per_station: list[dict],
    vs: np.ndarray | None = None,
    surface_h: np.ndarray | None = None,
    surface_v: np.ndarray | None = None,
    extras: list[tuple[np.ndarray, str, str]] | None = None,
    extra_rows: list[tuple[list[dict], str, str]] | None = None,
) -> None:
    if vs is not None:
        plot_vs(vs, out_dir / "vs_section.png")
    if surface_h is not None and surface_v is not None:
        plot_pga(surface_h, surface_v, out_dir / "pga.png")
    plot_center_tf(freq, pred, true, out_dir / "tf_center.png", extras)
    plot_center_tf(freq, pred, true, out_dir / "tf_center_full_depth.png", extras)
    plot_station_tfs(freq, pred, true, out_dir / "tf_stations.png", extras)
    plot_tf_grid(freq, pred, true, out_dir / "tf_grid.png", extras)
    plot_tf_median(freq, pred, true, out_dir / "tf_median.png", extras)
    plot_pearson_box(per_station, out_dir / "pearson_box.png", extra_rows)
    plot_metrics_vs_x(per_station, out_dir / "metrics_vs_x.png", extra_rows)
    plot_tf_scatter(freq, pred, true, out_dir / "tf_scatter.png", extras)
    plot_tf_error(freq, pred, true, out_dir / "tf_error.png", extras)
    plot_pearson_vs_frequency(
        freq, pred, true, out_dir / "pearson_vs_frequency.png", extras
    )


def build_metrics(
    scores: dict,
    *,
    checkpoint: Path,
) -> dict:
    return {
        "case": "Treasure Island Case 7a",
        "checkpoint": str(checkpoint),
        "window_m": {
            "x": [-STRIP_HALF_M, STRIP_HALF_M],
            "depth_kept_m": LZ_M,
            "lx_m": LX_M,
        },
        "stations_x_m": station_x_m().tolist(),
        "center_station_index": center_station_index(),
        "zeta": ZETA,
        "scores": scores,
        "notes": NOTES,
    }


def _score_curve(pred: np.ndarray, true: np.ndarray, freq: np.ndarray) -> dict[str, dict]:
    scored: dict[str, dict] = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freq >= lo) & (freq <= hi)
        scored[name] = _pair_metrics(pred[mask], true[mask])
    return scored


def _per_station_scores(
    x: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    freq: np.ndarray,
) -> list[dict]:
    rows = []
    for i, xv in enumerate(x):
        row: dict = {"x_m": float(xv)}
        scored = _score_curve(pred[i], true[i], freq)
        for name, metrics in scored.items():
            row[f"pearson_{name}"] = metrics["pearson"]
            row[f"rel_l2_{name}"] = metrics["rel_l2"]
        rows.append(row)
    return rows


FULL_SECTION_NOTES = [
    "Full-depth inputs average the whole 150 m column into 128 bins. Rows near "
    "the bottom of that 128-vector are zero padding in training.",
    "15 m stations whose 500 m window lies inside the mesh run from -1740 m to "
    "1740 m. The geometric limits are -1750 m and 1750 m, which are not on that "
    "15 m lattice.",
    "One downhole record is the denominator at every x.",
    "soil_rock zeta is 1 where the binned Vs is at most 760 m/s and 0.1 where "
    "Vs is above 760 m/s.",
]


def plot_center_full_depth(
    freq: np.ndarray,
    true: np.ndarray,
    top128: np.ndarray,
    flat: np.ndarray,
    soil_rock: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    ax.semilogx(freq, true, color="k", label="reference")
    ax.semilogx(freq, top128, color="0.45", label="top 128 m, flat zeta")
    ax.semilogx(freq, flat, color="C0", label="full 150 m, flat zeta")
    ax.semilogx(freq, soil_rock, color="C3", label="full 150 m, soil/rock zeta")
    ax.axhline(0.0, color="0.75", lw=0.5)
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|TF|")
    ax.set_title("Center station, x = 0 m")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_full_section_scores(
    x: np.ndarray,
    flat_rows: list[dict],
    rock_rows: list[dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(8.2, 5.4), sharex=True, constrained_layout=True
    )
    for ax in axes:
        ax.axvspan(
            GEOLOGIC_X_MIN,
            GEOLOGIC_X_MAX,
            color="0.85",
            label="geologic interval",
        )
    axes[0].plot(
        x,
        [row["pearson_full"] for row in flat_rows],
        color="C0",
        lw=1.2,
        label="flat zeta",
    )
    axes[0].plot(
        x,
        [row["pearson_full"] for row in rock_rows],
        color="C3",
        lw=1.2,
        label="soil/rock zeta",
    )
    axes[1].plot(
        x,
        [row["rel_l2_full"] for row in flat_rows],
        color="C0",
        lw=1.2,
        label="flat zeta",
    )
    axes[1].plot(
        x,
        [row["rel_l2_full"] for row in rock_rows],
        color="C3",
        lw=1.2,
        label="soil/rock zeta",
    )
    axes[0].axhline(0.0, color="0.55", lw=0.7)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("Pearson r")
    axes[1].set_ylabel("relative L2")
    axes[1].set_xlabel("x (m)")
    axes[0].set_title("Full-depth windows, 0.1–10 Hz")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_full_section(args: argparse.Namespace, pack: dict | None = None) -> dict:
    """Slide 500 m windows across the extended mesh at full depth."""
    if pack is None:
        pack = ensure_cache(
            args.cache,
            args.instructions_zip,
            args.results_zip,
            force=args.force_cache,
        )
    centers = window_center_x()
    nodes = np.array([window_node_index(float(x)) for x in centers], dtype=np.int64)
    freq, tf_true = case7_reference(pack, nodes, args.label)
    vs_binned = resample_vs_full_depth(pack["vs"])
    flat_in = np.stack(
        [build_window_input(vs_binned, float(x), "flat") for x in centers]
    )
    rock_in = np.stack(
        [build_window_input(vs_binned, float(x), "soil_rock") for x in centers]
    )
    model, device = _load_loglo(args.checkpoint)
    tf_flat = predict_center_batch(flat_in, model=model, device=device)
    tf_rock = predict_center_batch(rock_in, model=model, device=device)
    flat_rows = _per_station_scores(centers, tf_flat, tf_true, freq)
    rock_rows = _per_station_scores(centers, tf_rock, tf_true, freq)

    center_i = int(np.flatnonzero(np.isclose(centers, 0.0))[0])
    top_path = args.out_dir / "tf_pred.npy"
    if top_path.is_file():
        top_center = np.load(top_path)[center_station_index()]
    else:
        top_center = predict_center_batch(
            build_model_input(pack["vs"], zeta=ZETA)[None],
            model=model,
            device=device,
        )[0]
    center_scores = {
        "top128_flat": _score_curve(top_center, tf_true[center_i], freq),
        "full_flat": _score_curve(tf_flat[center_i], tf_true[center_i], freq),
        "full_soil_rock": _score_curve(tf_rock[center_i], tf_true[center_i], freq),
    }
    section = {
        "x_m": centers.tolist(),
        "geologic_x_m": [GEOLOGIC_X_MIN, GEOLOGIC_X_MAX],
        "window_x_m": [float(centers[0]), float(centers[-1])],
        "geometric_limit_m": [-WINDOW_LIMIT_M, WINDOW_LIMIT_M],
        "center_x0": center_scores,
        "flat": flat_rows,
        "soil_rock": rock_rows,
        "notes": FULL_SECTION_NOTES,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "full_x.npy", centers.astype(np.float32))
    np.save(args.out_dir / "full_freq.npy", freq.astype(np.float32))
    np.save(args.out_dir / "full_tf_true.npy", tf_true)
    np.save(args.out_dir / "full_tf_flat.npy", tf_flat)
    np.save(args.out_dir / "full_tf_soil_rock.npy", tf_rock)
    plot_center_full_depth(
        freq,
        tf_true[center_i],
        top_center,
        tf_flat[center_i],
        tf_rock[center_i],
        args.out_dir / "tf_center_full_depth.png",
    )
    plot_full_section_scores(
        centers,
        flat_rows,
        rock_rows,
        args.out_dir / "full_section_scores.png",
    )
    metrics_path = args.out_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
    else:
        metrics = build_metrics(
            {"center": {}, "stations_21": {}}, checkpoint=args.checkpoint
        )
    metrics["full_section"] = section
    notes = list(metrics.get("notes", []))
    for note in FULL_SECTION_NOTES:
        if note not in notes:
            notes.append(note)
    metrics["notes"] = notes
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return section


def _center_window_arms(
    args: argparse.Namespace, pack: dict
) -> tuple[np.ndarray, np.ndarray]:
    """21-station |TF| of the x = 0 window on the full 150 m column."""
    flat_path = args.out_dir / "tf_pred_full_flat_21.npy"
    rock_path = args.out_dir / "tf_pred_full_soil_rock_21.npy"
    if flat_path.is_file() and rock_path.is_file():
        return np.load(flat_path), np.load(rock_path)
    vs_binned = resample_vs_full_depth(pack["vs"])
    inputs = np.stack(
        [
            build_window_input(vs_binned, 0.0, "flat"),
            build_window_input(vs_binned, 0.0, "soil_rock"),
        ]
    )
    pred = predict_recorder_batch(inputs, args.checkpoint)
    np.save(flat_path, pred[0])
    np.save(rock_path, pred[1])
    return pred[0], pred[1]


def resolve_logpod(args: argparse.Namespace) -> tuple[Path, Path] | None:
    """Checkpoint and model dir for the log-|TF| arm, when both exist."""
    ckpt = getattr(args, "logpod_checkpoint", None)
    model_dir = getattr(args, "logpod_model_dir", None)
    if ckpt is None or model_dir is None:
        return None
    ckpt = Path(ckpt)
    model_dir = Path(model_dir)
    modes = model_dir / "pod_modes.npy"
    mean = model_dir / "pod_mean.npy"
    if not ckpt.is_file() or not modes.is_file() or not mean.is_file():
        return None
    space_path = model_dir / "pod_space.txt"
    if space_path.is_file() and space_path.read_text().strip() != "log":
        raise ValueError(f"{space_path} is not log; refusing to exp a linear basis")
    return ckpt, model_dir


def _logpod_center_prediction(
    args: argparse.Namespace, pack: dict
) -> np.ndarray | None:
    """21-station |TF| of the top-128 m center window from a log-POD checkpoint."""
    found = resolve_logpod(args)
    if found is None:
        return None
    ckpt, model_dir = found
    cache = args.out_dir / "tf_pred_logpod_21.npy"
    if cache.is_file():
        return np.load(cache)
    pred = predict_transfer_function(
        build_model_input(pack["vs"], zeta=ZETA),
        ckpt,
        pod_modes_path=model_dir / "pod_modes.npy",
        pod_mean_path=model_dir / "pod_mean.npy",
        pod_space="log",
    )
    np.save(cache, pred)
    return pred


def _record_logpod_scores(
    args: argparse.Namespace,
    pred: np.ndarray,
    true: np.ndarray,
    freq: np.ndarray,
) -> None:
    """Add the log-POD center scores without dropping the full-section block."""
    found = resolve_logpod(args)
    if found is None:
        return
    scored = score_transfer_functions(pred, true, freq)
    scored.pop("per_station", None)
    path = args.out_dir / "metrics.json"
    metrics = json.loads(path.read_text()) if path.is_file() else {}
    metrics["log_pod"] = {
        "checkpoint": str(found[0]),
        "model_dir": str(found[1]),
        "scores": scored,
    }
    path.write_text(json.dumps(metrics, indent=2))


def redraw_all(args: argparse.Namespace) -> None:
    """Rewrite every figure from saved arrays, including the full-depth arms."""
    pack = ensure_cache(args.cache, args.instructions_zip, args.results_zip)
    freq = np.load(args.out_dir / "freq.npy")
    true = np.load(args.out_dir / "tf_true.npy")
    top = np.load(args.out_dir / "tf_pred.npy")
    flat21, rock21 = _center_window_arms(args, pack)
    extras = [
        (flat21, "C0", "full 150 m, flat zeta"),
        (rock21, "C3", "full 150 m, soil/rock zeta"),
    ]
    top_rows = score_transfer_functions(top, true, freq)["per_station"]
    extra_rows = [
        (
            score_transfer_functions(flat21, true, freq)["per_station"],
            "C0",
            "full 150 m, flat",
        ),
        (
            score_transfer_functions(rock21, true, freq)["per_station"],
            "C3",
            "full 150 m, soil/rock",
        ),
    ]
    log_pred = _logpod_center_prediction(args, pack)
    if log_pred is not None:
        extras.append((log_pred, "C2", "log POD"))
        extra_rows.append(
            (
                score_transfer_functions(log_pred, true, freq)["per_station"],
                "C2",
                "log POD",
            )
        )
        _record_logpod_scores(args, log_pred, true, freq)
    write_comparison_plots(
        args.out_dir,
        freq,
        top,
        true,
        top_rows,
        vs=pack["vs"],
        surface_h=pack["surface_h"],
        surface_v=pack["surface_v"],
        extras=extras,
        extra_rows=extra_rows,
    )
    centers = np.load(args.out_dir / "full_x.npy")
    full_freq = np.load(args.out_dir / "full_freq.npy")
    full_true = np.load(args.out_dir / "full_tf_true.npy")
    plot_full_section_scores(
        centers,
        _per_station_scores(
            centers, np.load(args.out_dir / "full_tf_flat.npy"), full_true, full_freq
        ),
        _per_station_scores(
            centers,
            np.load(args.out_dir / "full_tf_soil_rock.npy"),
            full_true,
            full_freq,
        ),
        args.out_dir / "full_section_scores.png",
    )


def run(args: argparse.Namespace) -> dict:
    pack = ensure_cache(
        args.cache,
        args.instructions_zip,
        args.results_zip,
        force=args.force_cache,
    )
    nodes = station_node_indices()
    freq, tf_true = case7_reference(pack, nodes, args.label)
    model_input = build_model_input(pack["vs"], zeta=ZETA)
    tf_pred = predict_transfer_function(model_input, args.checkpoint)
    if tf_pred.shape != tf_true.shape:
        raise RuntimeError(f"pred {tf_pred.shape} != reference {tf_true.shape}")
    scores = score_transfer_functions(tf_pred, tf_true, freq)
    metrics = build_metrics(scores, checkpoint=args.checkpoint)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(args.out_dir / "freq.npy", freq.astype(np.float32))
    np.save(args.out_dir / "tf_true.npy", tf_true)
    np.save(args.out_dir / "tf_pred.npy", tf_pred)
    run_full_section(args, pack)
    redraw_all(args)
    return metrics


def plots_only(args: argparse.Namespace) -> dict:
    redraw_all(args)
    return json.loads((args.out_dir / "metrics.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instructions-zip", type=Path, default=None)
    parser.add_argument("--results-zip", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--logpod-checkpoint",
        type=Path,
        default=None,
        help="Optional log-|TF| POD weights. Drawn only when this file exists.",
    )
    parser.add_argument(
        "--logpod-model-dir",
        type=Path,
        default=None,
        help="Directory with the log-space pod_modes.npy and pod_mean.npy",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--label",
        choices=("downhole", "outcrop"),
        default="downhole",
        help="Reference TF: surface/downhole(x=0), or surface/(c*input) as in relabeled training data",
    )
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Redraw every figure from saved arrays, including the full-depth arms",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Score sliding full-depth windows and skip the center-strip redraw",
    )
    args = parser.parse_args()
    from case7a import DEFAULT_CACHE, DEFAULT_INSTRUCTIONS_ZIP, DEFAULT_RESULTS_ZIP

    if args.instructions_zip is None:
        args.instructions_zip = DEFAULT_INSTRUCTIONS_ZIP
    if args.results_zip is None:
        args.results_zip = DEFAULT_RESULTS_ZIP
    if args.cache is None:
        args.cache = DEFAULT_CACHE
    if args.full_only:
        section = run_full_section(args)
        center = section["center_x0"]
        flat = center["full_flat"]["full"]
        rock = center["full_soil_rock"]["full"]
        print(
            f"x=0 full flat rel L2 {flat['rel_l2']:.3f}  pearson {flat['pearson']:.3f}"
        )
        print(
            "x=0 soil/rock rel L2 "
            f"{rock['rel_l2']:.3f}  pearson {rock['pearson']:.3f}"
        )
        print(f"wrote {args.out_dir}")
        return
    metrics = plots_only(args) if args.plots_only else run(args)
    center = metrics["scores"]["center"]["full"]
    stations = metrics["scores"]["stations_21"]["full"]
    print(
        f"center rel L2 {center['rel_l2']:.3f}  pearson {center['pearson']:.3f}"
    )
    print(
        "21 stations rel L2 "
        f"{stations['rel_l2']:.3f}  pearson {stations['pearson']:.3f}"
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
