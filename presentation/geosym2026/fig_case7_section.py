"""Case 7 geologic section, Pearson along x, and log-log |TF|.

Three rows: the Vs section, Pearson of log|TF| on 0.1–2.5 Hz, then one |TF|
panel per marked station. The red squares are lettered (a)–(e) to match the
panels. West of −650 m and east of 1050 m the instruction-sheet model repeats
one column. Those extensions are left off.

full_qvs.h5 is the 15 s Q–Vs solve of the 4000 m interior. Each station is the
center of a 500 m window, the same cut as score_svlab_stage1.py. Pearson is of
log|TF| on 0.1–2.5 Hz. On that band, x = −1660 m is where the section departs
most from its own column. The outcrop scale is score_record_length.SCALE.

    python fig_case7_section.py
"""

from __future__ import annotations

import sys

import h5py
import hdf5plugin  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

import figstyle as fs

sys.path.insert(0, str(fs.REPO / "experiments" / "case7a"))
sys.path.insert(0, str(fs.REPO / "experiments" / "GIFNO-VARDOMAIN"))

from applied_damping import applied_zeta  # noqa: E402
from case7a import ELEM_M, GEOLOGIC_X_MAX, GEOLOGIC_X_MIN, X_LEFT_M, ensure_cache  # noqa: E402
from case7_crops import span_columns  # noqa: E402
from encoding import encode, query_columns  # noqa: E402
from evaluate import load_model  # noqa: E402
from labels import LABEL_WINDOW_S, seiskit_input_acceleration, tf_over_input  # noqa: E402
from run_case7a import load_ttf_batch_fast  # noqa: E402
from score_record_length import SCALE  # noqa: E402
from score_svlab_stage1 import CACHE_FREQ, _column_tf  # noqa: E402

FULL = fs.REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5" / "full_qvs.h5"
CACHE = fs.DATA / "case7_section_ai.npz"
THIRDS_CACHE = fs.DATA / "case7_section_ai_thirds.npz"
WIDTH = 500
STRIDE_M = 20
X_STATION = -1660.0
# −2000 + 4000/3 and −2000 + 8000/3, on the nearest metre.
THIRDS_M = (-667.0, 667.0)
# Inside the geologic interval. One-third and two-thirds of −600 to 1000 m,
# on the 20 m stations. −60 and 460 are 7 m off −66.7 and 466.7.
SHOW = (
    (-600.0, "−600 m", ""),
    (-60.0, "−60 m", ""),
    (0.0, "0", ""),
    (460.0, "460 m", ""),
    (1000.0, "1000 m", ""),
)
# Bedrock incline inside the geologic model, from the section figure.
RAMP_M = (600.0, 1000.0)
BAND = (0.1, 2.5)
# case7a.ZONES, named as on the full-section slide.
ZONES = (
    ("Flat west", -np.inf, -650.0, "#D7DCE3"),
    ("Basin", -650.0, 1050.0, "#D9E6F2"),
    ("Ramp", 1050.0, 1300.0, "#F6E4C4"),
    ("Flat east", 1300.0, np.inf, "#D9EADF"),
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 0 else float("nan")


def _stations() -> np.ndarray:
    x = np.arange(-1740.0, 1740.0 + STRIDE_M, STRIDE_M)
    if not np.any(np.isclose(x, X_STATION)):
        raise RuntimeError(f"{X_STATION:.0f} m is not on the {STRIDE_M:.0f} m stations")
    return x


def _predict(x: np.ndarray | None = None) -> dict[str, np.ndarray]:
    x = _stations() if x is None else np.asarray(x, dtype=np.float64)
    freq_m = np.load(CACHE_FREQ).astype(np.float64)
    with h5py.File(FULL, "r") as handle:
        vs = np.asarray(handle["Vs_realization_2D"][:], dtype=np.float64)
        zeta = applied_zeta(handle)
        time = np.asarray(handle["recorders/accel/time"][:], dtype=np.float64)
        acc = np.asarray(handle["recorders/accel/surface"][:], dtype=np.float64)
        x_all = np.asarray(handle["recorders/accel/surface"].attrs["x_m"], dtype=np.float64)
        params = handle["params"].attrs
        f1 = float(params["damping_freq_first"])
        f2 = float(params["damping_freq_second"])
        motion = float(handle["grid"].attrs["motion_freq"])
        ricker = params.get("ricker_is", "acceleration")
        if isinstance(ricker, bytes):
            ricker = ricker.decode()
    cols = np.array([int(np.argmin(np.abs(x_all - v))) for v in x], dtype=np.int64)
    if np.max(np.abs(x_all[cols] - x)) > 1e-6:
        raise RuntimeError("a station is not a mesh column")
    dt = float(time[1] - time[0])
    freq, raw = tf_over_input(
        acc[:, cols],
        seiskit_input_acceleration(motion, 0.5, time, str(ricker)),
        dt,
        load_ttf_batch_fast(),
        window_s=LABEL_WINDOW_S,
    )
    truth = raw / SCALE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _cfg = load_model(fs.CHECKPOINT, device)
    model.eval()
    q = float(query_columns("metric", np.array([WIDTH / 2.0]), WIDTH)[0])
    query = torch.tensor([[q]], dtype=torch.float32, device=device)
    pred = np.empty((x.size, freq.size), dtype=np.float64)
    with torch.no_grad():
        for i, center in enumerate(x):
            c0, n, shift = span_columns(float(center), WIDTH)
            if shift != 0.0 or n != WIDTH:
                raise RuntimeError(f"window at x={center:.0f} is shifted by {shift:.0f} m")
            vs_w = vs[:, c0 : c0 + n]
            zeta_w = zeta[:, c0 : c0 + n]
            mid = n // 2
            tf1d = _column_tf(freq_m, vs_w[:, mid], zeta_w[:, mid], f1, f2)
            enc = torch.from_numpy(
                encode("metric", vs_w, zeta_w, vs_w.shape[0], f1_hz=f1, f2_hz=f2)
            ).unsqueeze(0)
            g = model(enc.to(device), query).float().cpu().numpy()[0, 0]
            pred[i] = np.interp(freq, freq_m, tf1d * np.exp(g))
            if i % 25 == 0:
                print(f"station {i + 1}/{x.size}  x={center:.0f} m  g {float(g.mean()):+.3f}", flush=True)
    band = (freq >= BAND[0]) & (freq <= BAND[1])
    pear = np.array([
        _pearson(np.log(np.maximum(pred[i, band], 1e-6)), np.log(np.maximum(truth[i, band], 1e-6)))
        for i in range(x.size)
    ])
    return {"x": x, "freq": freq, "truth": truth, "pred": pred, "pearson": pear}


def _cross_section(ax):
    """Case 7 Vs on the geologic interval. Depth increases downward."""
    vs = np.asarray(ensure_cache()["vs"], dtype=np.float64)
    x_edge = X_LEFT_M + ELEM_M * np.arange(vs.shape[1] + 1)
    j0 = int(np.searchsorted(x_edge, GEOLOGIC_X_MIN))
    j1 = int(np.searchsorted(x_edge, GEOLOGIC_X_MAX))
    vs, x_edge = vs[:, j0:j1], x_edge[j0 : j1 + 1]
    z_edge = ELEM_M * np.arange(vs.shape[0] + 1)
    mesh = ax.pcolormesh(
        x_edge, z_edge, vs, cmap=fs.CMAP, norm=LogNorm(150, 2400),
        shading="flat", rasterized=True,
    )
    ax.set_ylim(z_edge[-1], 0)
    ax.set_yticks([0, 50, 100, 150])
    ax.set_ylabel("Depth (m)")
    ax.axhline(0, color=fs.TRUTH, lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
    return mesh


def _shade(ax, x: np.ndarray) -> None:
    lo, hi = float(x.min()), float(x.max())
    for _name, a, b, color in ZONES:
        ax.axvspan(max(a, lo), min(b, hi), color=color, lw=0, zorder=0)


def _zone_labels(ax, x: np.ndarray) -> None:
    lo, hi = float(x.min()), float(x.max())
    trans = ax.get_xaxis_transform()
    for name, a, b, _color in ZONES:
        left, right = max(a, lo), min(b, hi)
        ax.text((left + right) / 2, 1.02, name, transform=trans, ha="center", va="bottom",
                fontsize=13, color=fs.MUTED, clip_on=False)


def _load(path, x: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if path.is_file():
        print(f"using {path}")
        return dict(np.load(path))
    pack = _predict(x)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pack)
    print(f"wrote {path}")
    return pack


def _at(pack: dict[str, np.ndarray], xq: float) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(pack["x"])
    i = int(np.argmin(np.abs(x - xq)))
    if abs(float(x[i]) - xq) > 0.51:
        raise RuntimeError(f"x = {xq:.0f} m is not in this set (nearest {x[i]:.0f})")
    return pack["truth"][i], pack["pred"][i], float(pack["pearson"][i])


def _freq_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10.0)
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Frequency (Hz)")


def main() -> None:
    pack = _load(CACHE)
    x, pear = np.asarray(pack["x"]), np.asarray(pack["pearson"])
    freq = pack["freq"]
    inside = (x >= -650.0) & (x <= 1050.0)
    curves = {xq: _at(pack, xq) for xq, _title, _note in SHOW}
    for xq, _title, _note in SHOW:
        print(f"x = {xq:.0f} m  Pearson {curves[xq][2]:.3f}")
    print(f"geologic {pear[inside].min():.3f}–{pear[inside].max():.3f}")

    fs.apply()
    fig = plt.figure(figsize=(fs.FULL_W, 9.6))
    gs = fig.add_gridspec(3, len(SHOW), height_ratios=[1.25, 1.0, 1.2], left=0.09, right=0.90,
                          bottom=0.055, top=0.95, hspace=0.55, wspace=0.22)
    ax_sec = fig.add_subplot(gs[0, :])
    ax = fig.add_subplot(gs[1, :], sharex=ax_sec)
    mesh = _cross_section(ax_sec)
    ax_sec.set_xlim(-650, 1050)
    ax_sec.tick_params(axis="x", labelbottom=False)
    trans = ax_sec.get_xaxis_transform()
    ax_sec.text((-650 + RAMP_M[0]) / 2, 1.03, "Basin", transform=trans, ha="center", va="bottom",
                fontsize=13, color=fs.MUTED, clip_on=False)
    ax_sec.text(sum(RAMP_M) / 2, 1.03, "Bedrock Incline", transform=trans, ha="center", va="bottom",
                fontsize=13, color=fs.MUTED, clip_on=False)
    pos = ax_sec.get_position()
    gap = pos.y0 - ax.get_position().y1
    drop = gap - 0.006
    ax_sec.set_position([pos.x0, pos.y0 - drop, pos.width, pos.height])
    pos = ax_sec.get_position()
    cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.014, pos.height])
    cb = fig.colorbar(mesh, cax=cax, ticks=[150, 300, 600, 1200, 2400])
    cb.ax.set_yticklabels(["150", "300", "600", "1200", "2400"])
    cb.ax.yaxis.set_minor_locator(plt.NullLocator())
    cb.set_label("$V_s$ (m/s)", fontsize=13)
    cb.ax.tick_params(labelsize=11)

    ax.plot(x[inside], pear[inside], color=fs.AI, lw=2.8, zorder=3, label="This work")
    mark = "#C62828"
    mark_x = [xq for xq, _t, _n in SHOW]
    mark_y = [curves[xq][2] for xq in mark_x]
    ax.scatter(mark_x, mark_y, s=56, marker="s", color=mark, edgecolor="white", linewidths=0.8, zorder=4)
    for i, (xq, yv) in enumerate(zip(mark_x, mark_y)):
        ax.annotate(f"({chr(ord('a') + i)})", (xq, yv), textcoords="offset points", xytext=(0, 8),
                    ha="center", va="bottom", fontsize=13, color=mark, fontweight="semibold",
                    zorder=5, clip_on=False)
    ax.set_xticks([-600, -300, 0, 300, 600, 1000])
    ax.set_ylim(0.955, 1.008)
    ax.set_yticks([0.96, 0.98, 1.00], ["0.96", "0.98", "1.00"])
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Pearson of log$|TF|$\n0.1–2.5 Hz")
    ax.legend(loc="lower right", fontsize=14)

    span = [np.concatenate([t, p]) for t, p, _ in curves.values()]
    ymax = max(float(np.max(v)) for v in span)
    b_axes = []
    for i, (xq, title, note) in enumerate(SHOW):
        bx = fig.add_subplot(gs[2, i], sharey=b_axes[0] if b_axes else None)
        b_axes.append(bx)
        truth_i, pred_i, p_i = curves[xq]
        bx.loglog(freq, np.maximum(pred_i, 1e-3), color=fs.AI, lw=2.4, label="This work", zorder=3)
        bx.loglog(freq, np.maximum(truth_i, 1e-3), color=fs.TRUTH, lw=1.7, label="2D simulation", zorder=4)
        _freq_axis(bx)
        head = f"({chr(ord('a') + i)})  {title}"
        if note:
            head += f"\n{note}"
        bx.set_title(head, loc="center", fontsize=13, color=mark, fontweight="semibold", pad=4)
        bx.tick_params(labelsize=12)
        bx.xaxis.label.set_size(13)
        bx.text(0.50, 0.04, rf"$\rho = {p_i:.2f}$", transform=bx.transAxes, ha="center", va="bottom",
                fontsize=12, color=fs.MUTED)
        if i == 0:
            bx.set_ylabel("$|TF|$")
        else:
            bx.tick_params(labelleft=False)
    b_axes[0].set_ylim(0.5, ymax * 1.45)
    b_axes[0].set_yticks([0.5, 1, 10], ["0.5", "1", "10"])
    b_axes[0].yaxis.set_minor_locator(plt.NullLocator())
    handles, labels = b_axes[0].get_legend_handles_labels()
    tf_top = b_axes[0].get_position().y1
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.915, tf_top - 0.02),
               fontsize=12, frameon=False, borderaxespad=0)
    fs.save(fig, "fig_case7_section")


if __name__ == "__main__":
    main()
