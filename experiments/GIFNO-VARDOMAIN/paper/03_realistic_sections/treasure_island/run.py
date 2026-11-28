#!/usr/bin/env python3
"""Treasure Island zones: leftover, distance to the manifold, and the section figure.

Leftover in a zone is log_l1 / (1 - skill) from the saved zone table, which is
the column error on 0.4–2.5 Hz. Scalar distance uses log impedance, log
thickness, log f0, and log f2, standardized on the manifold training rows.

    python run.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "scripts" / "_entry.py").is_file():
        sys.path.insert(0, str(_parent / "scripts"))
        break
from _entry import enter

enter(_HERE)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from bootstrap_ci import block_bootstrap_median
from case7_crops import CROPS, soil_quarter_wave_hz
from paths import DUMP, FIG, MANIFEST, REPO
from plotting import apply_publication_style, panel_letter, save_figure

OUT = _HERE.parent
ROCK_VS = 760.0
SHOW = ("flat_west", "thinning", "ramp")
SVLAB = REPO / "presentation" / "geosym2026" / "data" / "svlab_stage2_tf.npz"


def _manifold_cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with MANIFEST.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("stratum") != "blind" and row.get("split") != "blind"]
    cloud = np.vstack([_csv_scalars(row) for row in rows])
    mu = cloud.mean(axis=0)
    sigma = np.maximum(cloud.std(axis=0), 1e-6)
    return cloud, mu, sigma


def _csv_scalars(row: dict) -> np.ndarray:
    return np.array(
        [
            np.log(float(row["vs_contact_mps"]) / float(row["vs_surface_mps"])),
            np.log(float(row["soil_thickness_m"])),
            np.log(max(float(row["f0_flat_hz"]), 1e-6)),
            np.log(max(float(row["rayleigh_f2_hz"]), 1e-6)),
        ],
        dtype=np.float64,
    )


def _zone_scalars(vs: np.ndarray, x_edge: np.ndarray, center: float) -> np.ndarray:
    x = 0.5 * (x_edge[:-1] + x_edge[1:])
    col = vs[:, int(np.argmin(np.abs(x - center)))]
    rock = np.flatnonzero(col >= ROCK_VS)
    contact = int(rock[0]) if rock.size else col.size - 1
    thickness = max(contact, 1)
    f0 = soil_quarter_wave_hz(col, ROCK_VS)
    return np.log(
        np.array(
            [col[min(contact, col.size - 1)] / max(col[0], 1.0), thickness, max(f0, 1e-6), max(3.0 * f0, 1e-6)],
            dtype=np.float64,
        )
    )


def _distance(point: np.ndarray, cloud: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    z = (cloud - mu) / sigma
    p = (point - mu) / sigma
    return float(np.min(np.linalg.norm(z - p, axis=1)))


def _leftover(table: dict) -> list[dict]:
    rows = []
    for name, widths in table.items():
        for width, cell in widths.items():
            skill = float(cell["skill"])
            log_l1 = float(cell["log_l1"])
            column = log_l1 / max(1.0 - skill, 1e-6)
            rows.append(
                {
                    "zone": name,
                    "width_m": int(width),
                    "model_log_l1": log_l1,
                    "skill": skill,
                    "column_log_l1": column,
                    "pearson": float(cell["pearson"]),
                    "n_stations": int(cell["n_stations"]),
                }
            )
    return rows


def _section():
    from case7a import ELEM_M, X_LEFT_M, ensure_cache

    pack = ensure_cache()
    vs = np.asarray(pack["vs"], dtype=np.float64)
    x_edge = X_LEFT_M + ELEM_M * np.arange(vs.shape[1] + 1)
    return vs, x_edge


def main() -> None:
    case7_path = DUMP / "ensemble_case7_width.json"
    if not case7_path.is_file():
        case7_path = FIG / "stage2" / "case7_width.json"
    table = json.loads(case7_path.read_text())
    leftover = _leftover(table)
    cloud, mu, sigma = _manifold_cloud()
    distances = []
    vs = x_edge = None
    try:
        vs, x_edge = _section()
    except Exception as exc:
        print(f"section cache unavailable: {exc}", flush=True)
    if vs is not None:
        for crop in CROPS:
            point = _zone_scalars(vs, x_edge, float(crop["center_x_m"]))
            distances.append(
                {
                    "zone": crop["name"],
                    "center_x_m": float(crop["center_x_m"]),
                    "distance": _distance(point, cloud, mu, sigma),
                    "log_impedance": float(point[0]),
                    "log_thickness": float(point[1]),
                    "log_f0": float(point[2]),
                    "log_f2": float(point[3]),
                }
            )
    block = None
    metrics_path = DUMP / "ensemble_svlab_metrics.json"
    if not metrics_path.is_file():
        metrics_path = FIG / "stage2" / "svlab_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    along_block = metrics.get("pearson_vs_x", {})
    stage = along_block.get("Ensemble", along_block.get("Stage 2", {}))
    along = next((values for key, values in stage.items() if "0.4" in key and "2.5" in key), None)
    x_m = metrics.get("x_m")
    if along is not None and x_m is not None:
        block = block_bootstrap_median(np.asarray(along, dtype=np.float64), np.asarray(x_m, dtype=np.float64), rng=np.random.default_rng(0))

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"leftover": leftover, "scalar_distance": distances, "block_bootstrap_pearson_core": block}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))

    curves = None
    svlab_curves = DUMP / "ensemble_svlab.npz"
    if not svlab_curves.is_file():
        svlab_curves = SVLAB
    if svlab_curves.is_file():
        blob = np.load(svlab_curves)
        curves = {key: np.asarray(blob[key]) for key in ("x", "freq", "tf_gt", "pred", "prior")}

    apply_publication_style()
    fig = plt.figure(figsize=(7.48, 5.6))
    ax0 = fig.add_subplot(2, 1, 1)
    if vs is not None:
        x = 0.5 * (x_edge[:-1] + x_edge[1:])
        z = np.arange(vs.shape[0] + 1, dtype=np.float64)
        show = (x >= -1800) & (x <= 1800)
        j0, j1 = int(np.flatnonzero(show)[0]), int(np.flatnonzero(show)[-1])
        image = ax0.pcolormesh(x_edge[j0 : j1 + 2], z, vs[:, j0 : j1 + 1], cmap="cividis", norm=LogNorm(vmin=100, vmax=2000))
        fig.colorbar(image, ax=ax0, fraction=0.03, pad=0.02, label=r"$V_s$ (m/s)")
        for crop, color in zip(CROPS, ("#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#009E73")):
            center = float(crop["center_x_m"])
            ax0.plot([center - 250, center + 250], [8, 8], color=color, lw=1.4)
            ax0.text(center, 18, crop["name"].replace("_", " "), ha="center", va="bottom", fontsize=6, color=color)
        ax0.set_ylim(vs.shape[0], 0)
        ax0.set_ylabel("Depth (m)")
    else:
        ax0.set_axis_off()
        ax0.text(0.5, 0.5, "Vs section cache is not on this machine", ha="center", va="center")
    ax0.set_xlabel("x (m)")
    panel_letter(ax0, "a")

    axes = [fig.add_subplot(2, 3, i) for i in (4, 5, 6)]
    for ax, name, letter in zip(axes, SHOW, "bcd"):
        if curves is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "TF curves not saved", ha="center", va="center", transform=ax.transAxes)
        else:
            center = float(next(crop["center_x_m"] for crop in CROPS if crop["name"] == name))
            i = int(np.argmin(np.abs(curves["x"] - center)))
            freq = curves["freq"]
            column = np.clip(curves["prior"][i], 1e-12, None)
            g_true = np.log(np.clip(curves["tf_gt"][i], 1e-12, None)) - np.log(column)
            g_pred = np.log(np.clip(curves["pred"][i], 1e-12, None)) - np.log(column)
            ax.axhline(0.0, color="#0072B2", ls="--", lw=1.0, label="Column")
            ax.plot(freq, g_true, color="#000000", lw=1.1, label="OpenSees")
            ax.plot(freq, g_pred, color="#D55E00", lw=1.1, label="Model")
            ax.set_xscale("log")
            ax.set_xlim(0.4, 10.0)
            ax.set_xlabel(r"$f$ (Hz)")
        ax.set_title(name.replace("_", " "))
        panel_letter(ax, letter)
    axes[0].set_ylabel(r"$g$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.48, 0.48), ncol=3, frameon=False, fontsize=7)
    fig.suptitle("OpenSees model of the Treasure Island geometry", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(hspace=0.62)
    save_figure(fig, OUT / "central.pdf")
    print(json.dumps({"zones": len(leftover), "distances": len(distances), "curves": curves is not None}, indent=2))


if __name__ == "__main__":
    main()
