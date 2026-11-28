#!/usr/bin/env python3
"""Two figures of the 4 m lateral sawtooth.

The traces are the 1500 m dense OpenSees section, where stage 2 puts the
largest share of its lateral power at 4 m. The spectrum uses all three dense
sections. OpenSees is log|TF| at 2.5 Hz. The models are the log correction g
at the same frequency. The trace keeps periods of 16 m and shorter so the
tooth is visible; the spectrum is the full interior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
VARDOMAIN = HERE.parent
REPO = VARDOMAIN.parents[1]
sys.path.insert(0, str(VARDOMAIN))
sys.path.insert(0, str(REPO / "experiments" / "case7a"))
sys.path.insert(0, str(REPO))

from data import Cache  # noqa: E402
from dense_labels import _ttf  # noqa: E402
from encoding import encode, query_columns  # noqa: E402
from labels import seiskit_input_acceleration, tf_over_input  # noqa: E402
from paper.p128n128_case7 import load_fp32  # noqa: E402
from wave_surrogate.plot_utils.publication import (  # noqa: E402
    apply_publication_style,
    figsize,
    panel_letter,
    savefig,
)

DENSE = Path("/tmp/dense_recorders")
CKPT = REPO / "checkpoints" / "vardomain"
MODELS = (
    ("Stage 2", CKPT / "early_width" / "stage2_rise50", "#E69F00"),
    ("P128", CKPT / "v2_ablations" / "P128", "#009E73"),
    ("P128n128", CKPT / "v2_ablations" / "P128n128", "#D55E00"),
)
OPENSEES_COLOR = "#0072B2"
OUT = HERE / "p128n128"
TRACE_SECTION = "time_w1500"
ZOOM_M = 64
SHORT_M = 16.0


def _share_spectrum(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=np.float64)
    series = series - series.mean()
    power = np.abs(np.fft.rfft(series)) ** 2
    freq = np.fft.rfftfreq(series.size, d=1.0)
    keep = freq > 0
    total = power[keep].sum()
    period = 1.0 / freq[keep]
    return period, power[keep] / max(total, 1e-30)


def _short(series: np.ndarray, max_period: float = SHORT_M) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    spec = np.fft.rfft(series - series.mean())
    freq = np.fft.rfftfreq(series.size, d=1.0)
    spec[freq < 1.0 / max_period] = 0.0
    return np.fft.irfft(spec, n=series.size)


def _section(path: Path, k_model: int, models: dict) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        params = handle["params"].attrs
        pad = int(params["pad_m"])
        interior = int(params["lx_interior_m"])
        vs = np.asarray(handle["Vs_realization_2D"][:, pad:pad + interior], dtype=np.float64)
        zeta = np.asarray(handle["Damping_zeta"][:, pad:pad + interior], dtype=np.float64)
        rock = np.asarray(handle["rock_mask"][:, pad:pad + interior])
        acc = np.asarray(handle["recorders/accel/surface"][:, pad:pad + interior], dtype=np.float64)
        time = np.asarray(handle["recorders/accel/time"][:], dtype=np.float64)
        f1 = float(params["rayleigh_f1_hz"])
        f2 = float(params["rayleigh_f2_hz"])
        motion = float(params["motion_freq_hz"])
        ricker = str(params["ricker_is"])
    dt = float(time[1] - time[0])
    freq, mag = tf_over_input(
        acc, seiskit_input_acceleration(motion, 0.5, time, ricker), dt, _ttf(), window_s=1310.72,
    )
    k_osees = int(np.argmin(np.abs(freq - 2.5)))
    hits = [
        int(np.flatnonzero(rock[:, j])[0]) if np.any(rock[:, j]) else rock.shape[0]
        for j in range(interior)
    ]
    x = torch.from_numpy(encode("metric", vs, zeta, int(max(hits)), f1_hz=f1, f2_hz=f2))
    cols = torch.tensor(
        query_columns("metric", np.arange(interior, dtype=np.float64), interior), dtype=torch.float32,
    )
    out = {"OpenSees": np.log(np.maximum(mag[:, k_osees], 1e-8))}
    device = next(next(iter(models.values())).parameters()).device
    x = x.unsqueeze(0).to(device)
    cols = cols.unsqueeze(0).to(device)
    for name, model in models.items():
        with torch.no_grad():
            out[name] = model(x, cols).float()[0, :, k_model].cpu().numpy()
    out["x"] = np.arange(interior, dtype=np.float64)
    return out


def _zoom(residual: np.ndarray, width: int) -> int:
    kernel = np.ones(width) / width
    power = np.convolve(residual ** 2, kernel, mode="valid")
    return int(np.argmax(power))


def main() -> None:
    apply_publication_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Cache(REPO / "data" / "vardomain" / "cache")
    k_model = int(np.argmin(np.abs(np.asarray(cache.freq) - 2.5)))
    loaded = {name: load_fp32(path, device) for name, path, _color in MODELS}
    sections = {
        path.stem: _section(path, k_model, loaded)
        for path in sorted(DENSE.glob("time_w*.h5"))
    }
    trace = sections[TRACE_SECTION]
    start = _zoom(_short(trace["Stage 2"]), ZOOM_M)
    stop = start + ZOOM_M
    x = trace["x"][start:stop]

    stage = _short(trace["Stage 2"])[start:stop]
    fig, ax = plt.subplots(figsize=figsize("double", height_mm=78))
    ax.plot(x, _short(trace["OpenSees"])[start:stop], color=OPENSEES_COLOR, lw=1.1, label="OpenSees, 1 m")
    for name, _path, color in MODELS:
        ax.plot(x, _short(trace[name])[start:stop], color=color, lw=1.1, label=name)
    peaks = np.flatnonzero((stage[1:-1] > stage[:-2]) & (stage[1:-1] > stage[2:]) & (stage[1:-1] > 0.03)) + 1
    pair = next((i, j) for i, j in zip(peaks, peaks[1:]) if 3 <= j - i <= 5)
    y_mark = float(np.max(np.abs(stage))) * 1.22
    ax.annotate(
        "",
        xy=(x[pair[1]], y_mark),
        xytext=(x[pair[0]], y_mark),
        arrowprops={"arrowstyle": "<->", "color": "0.25", "lw": 0.6},
        annotation_clip=False,
    )
    ax.text((x[pair[0]] + x[pair[1]]) / 2, y_mark, "4 m", ha="center", va="bottom", color="0.25")
    ax.set_xlabel("Position along the 1500 m interior (m)")
    ax.set_ylabel("Log amplitude, periods ≤ 16 m")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-y_mark * 0.85, y_mark * 1.35)
    fig.subplots_adjust(top=0.82)
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98), frameon=False)
    savefig(fig, OUT / "fig_sawtooth_trace.pdf", png=True, kind="line")

    order = ("time_w500", "time_w1000", "time_w1500")
    letters = ("a", "b", "c")
    titles = ("500 m", "1000 m", "1500 m")
    fig, axes = plt.subplots(1, 3, figsize=figsize("double", height_mm=78), sharey=True)
    for ax, key, letter, title in zip(axes, order, letters, titles):
        series = sections[key]
        period, share = _share_spectrum(series["OpenSees"])
        ax.plot(period, share, color=OPENSEES_COLOR, lw=1.0, label="OpenSees, 1 m")
        for name, _path, color in MODELS:
            period, share = _share_spectrum(series[name])
            ax.plot(period, share, color=color, lw=1.0, label=name)
        ax.axvline(4.0, color="0.45", lw=0.5, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(2.0, series["x"][-1])
        ax.set_xlabel("Wavelength (m)")
        ax.set_title(title)
        panel_letter(ax, letter)
    axes[0].set_ylabel("Share of lateral power")
    fig.subplots_adjust(top=0.78, wspace=0.18)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98), frameon=False,
    )
    savefig(fig, OUT / "fig_sawtooth_spectrum.pdf", png=True, kind="line")
    print(f"zoom {start:.0f}–{stop:.0f} m on {TRACE_SECTION}")
    print(f"wrote {OUT / 'fig_sawtooth_trace.pdf'}")
    print(f"wrote {OUT / 'fig_sawtooth_spectrum.pdf'}")


if __name__ == "__main__":
    main()
