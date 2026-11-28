#!/usr/bin/env python3
"""Wall-clock per section at 500, 1000, 1500, and 2000 m.

The column is the 1D engine at every interior station. The surrogate is one
batched forward of four copies of that window. OpenSees is the same 15 s
solve on Savio (``hpc/savio_timing15.sh``), not this machine. ``--opensees``
runs that solve locally. The 0.5 s pilot is not used.

    python run.py
    python run.py --opensees
"""

from __future__ import annotations

import json
import os
import sys
import time
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
import torch

from encoding import encode, query_columns
from generate_deep import sample_deep
from paths import CACHE, STAGE2
from plotting import apply_publication_style, save_figure
from section_1d import rayleigh_xi_target, surface_over_outcrop

OUT = _HERE.parent
WIDTHS = (500, 1000, 1500, 2000)
DURATION_S = 15.0
BATCH = 4


def _row(width: int) -> dict:
    row = sample_deep(1, seed=1000 + width, prefix=f"time{width}")[0]
    row["lx_m"] = float(width)
    row["duration_s"] = DURATION_S
    row["sample_id"] = f"time_w{width}"
    row["damping_family"] = "qvs"
    return row


def _mesh(row: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    import section_generator as sg

    vs, zeta, _rock = sg.section_field(row)
    return np.asarray(vs, dtype=np.float64), np.asarray(zeta, dtype=np.float64), int(vs.shape[0]), int(vs.shape[1])


def _time_column(vs: np.ndarray, zeta: np.ndarray, freq: np.ndarray, f1: float, f2: float) -> float:
    started = time.perf_counter()
    # a0 and a1 scale with zeta, so xi(z, f) = zeta(z) * xi(zeta=1).
    unit = rayleigh_xi_target(freq, 1.0, f1, f2)
    for j in range(vs.shape[1]):
        xi = np.asarray(zeta[:, j], dtype=np.float64)[:, None] * unit
        surface_over_outcrop(freq, vs[:, j], xi=xi)
        if j and j % 250 == 0:
            print(f"  column {j}/{vs.shape[1]}", flush=True)
    return time.perf_counter() - started


def _time_surrogate(vs: np.ndarray, zeta: np.ndarray, model, device, f1: float, f2: float) -> float:
    width = int(vs.shape[1])
    x = torch.from_numpy(encode("metric", vs, zeta, vs.shape[0], f1_hz=f1, f2_hz=f2)).unsqueeze(0)
    stations = np.linspace(0, width - 1, num=min(width, 512), dtype=np.float64)
    query = torch.tensor(query_columns("metric", stations, width), dtype=torch.float32).view(1, -1)
    batch_x = x.repeat(BATCH, 1, 1, 1)
    batch_q = query.repeat(BATCH, 1)
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model(batch_x.to(device), batch_q.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        model(batch_x.to(device), batch_q.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - started) / BATCH


def _time_opensees(row: dict) -> dict:
    from run_sections import simulate

    work = OUT / "opensees"
    dest = simulate(row, work, duration_s=DURATION_S)
    import h5py

    with h5py.File(dest, "r") as handle:
        wall = float(handle.attrs["wall_s"])
        nx = int(handle["Vs_realization_2D"].shape[1])
        nz = int(handle["Vs_realization_2D"].shape[0])
    dest.unlink(missing_ok=True)
    return {"wall_s": wall, "opensees_nx": nx, "opensees_nz": nz, "status": "solved"}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opensees", action="store_true", help="run the 15 s solves on this machine")
    ap.add_argument("--skip-opensees", action="store_true", help="ignored; local OpenSees is off unless --opensees")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from evaluate import load_model

    model, _cfg = load_model(STAGE2, device)
    rows = []
    prepared = []
    for width in WIDTHS:
        row = _row(width)
        vs, zeta, nz, nx = _mesh(row)
        f1 = float(row["rayleigh_f1_hz"])
        f2 = float(row["rayleigh_f2_hz"])
        print(f"timing column and surrogate width {width} nx {nx} nz {nz}", flush=True)
        column_s = _time_column(vs, zeta, freq, f1, f2)
        surrogate_s = _time_surrogate(vs, zeta, model, device, f1, f2)
        item = {
            "width_m": width,
            "depth_m": nz,
            "interior_nx": nx,
            "n_freq": int(freq.size),
            "duration_s": DURATION_S,
            "column_s": column_s,
            "surrogate_s": surrogate_s,
            "surrogate_batch": BATCH,
            "device": str(device),
            "cores": os.cpu_count(),
        }
        rows.append(item)
        prepared.append(row)
        print(json.dumps(item), flush=True)
        _write(rows)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.opensees and not args.skip_opensees:
        for item, row in zip(rows, prepared):
            print(f"opensees width {item['width_m']}", flush=True)
            try:
                item.update(_time_opensees(row))
            except Exception as exc:
                item["opensees_status"] = f"unavailable: {exc}"
            print(json.dumps(item), flush=True)
            _write(rows)


def _write(rows: list[dict]) -> None:
    (OUT / "results.json").write_text(json.dumps({"rows": rows}, indent=2))
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.48, 3.3))
    x = np.arange(len(rows), dtype=float)
    names = ("column_s", "surrogate_s", "wall_s")
    labels = ("Column", "Surrogate", "OpenSees")
    colors = ("#0072B2", "#D55E00", "#000000")
    bar = 0.24
    for offset, key, label, color in zip((-bar, 0.0, bar), names, labels, colors):
        vals = [float(row.get(key, np.nan)) for row in rows]
        ax.bar(x + offset, vals, width=bar, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(row["width_m"]) for row in rows])
    ax.set_xlabel("Interior width (m)")
    ax.set_ylabel("Seconds per section")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")


if __name__ == "__main__":
    main()
