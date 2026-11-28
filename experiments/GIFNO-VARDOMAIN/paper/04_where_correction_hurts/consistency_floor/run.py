#!/usr/bin/env python3
"""g_true on the flat-contact CoV = 0 manifold rows.

The floor is the median, over those center stations, of mean |g_true| on
0.4–2.5 Hz. The spectrum is the median of signed g_true(f) with a 5–95%
envelope. The empirical dispersion f_max is the first frequency above 2.5 Hz
where the envelope half-width exceeds twice its mid-band median.

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

from metrics import band_mask
from paths import CACHE, LABELS, MANIFEST
from plotting import apply_publication_style, panel_letter, save_figure

OUT = _HERE.parent


def _flat_cov0() -> list[dict]:
    with MANIFEST.open() as handle:
        rows = list(csv.DictReader(handle))
    kept = []
    for row in rows:
        extra = row.get("extra_features") or ""
        if row.get("feature_name") == "flat" and extra in ("", "[]") and float(row["vs_cov"]) == 0.0:
            kept.append(row)
    return kept


def _assigned_split(sample_id: str, csv_split: str) -> str:
    """Cache split. The CSV marks only the blind rows; val is assigned later."""
    path = CACHE / "sections" / f"manifold__{sample_id}.npz"
    if not path.is_file():
        return csv_split
    return str(np.load(path, allow_pickle=True)["split"])


def _center_g(row: dict) -> tuple[np.ndarray, np.ndarray, float]:
    path = LABELS / f"{row['sample_id']}.npz"
    blob = np.load(path)
    freq = np.asarray(blob["freq"], dtype=np.float64)
    x = np.asarray(blob["x_m"], dtype=np.float64)
    mid = int(np.argmin(np.abs(x - 0.5 * float(row["lx_m"]))))
    tf = np.clip(np.asarray(blob["tf"][mid], dtype=np.float64), 1e-6, None)
    tf1d = np.clip(np.asarray(blob["tf1d"][mid], dtype=np.float64), 1e-6, None)
    return freq, np.log(tf) - np.log(tf1d), float(x[mid])


def main() -> None:
    rows = _flat_cov0()
    missing = [row["sample_id"] for row in rows if not (LABELS / f"{row['sample_id']}.npz").is_file()]
    if missing:
        raise SystemExit(f"flat CoV=0 labels missing: {missing}")
    curves = []
    table = []
    freq = None
    for row in rows:
        freq, g, x_mid = _center_g(row)
        curves.append(g)
        table.append(
            {
                "sample_id": row["sample_id"],
                "stratum": row["stratum"],
                "split": _assigned_split(row["sample_id"], row["split"]),
                "lx_m": float(row["lx_m"]),
                "damping_family": row["damping_family"],
                "x_mid_m": x_mid,
            }
        )
    stack = np.vstack(curves)
    freq = np.asarray(freq)
    median = np.median(stack, axis=0)
    q05 = np.quantile(stack, 0.05, axis=0)
    q95 = np.quantile(stack, 0.95, axis=0)
    half = 0.5 * (q95 - q05)
    mid = band_mask(freq, "mid")
    abs_mid = np.mean(np.abs(stack[:, mid]), axis=1)
    floor = float(np.median(abs_mid))
    mid_half = float(np.median(half[mid]))
    high = np.flatnonzero(freq > 2.5)
    risen = high[half[high] > 2.0 * mid_half]
    f_max = float(freq[int(risen[0])]) if risen.size else None
    counts: dict[str, int] = {}
    for row in table:
        counts[row["split"]] = counts.get(row["split"], 0) + 1

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    np.savez(OUT / "g_true.npz", freq=freq, g=stack, median=median, q05=q05, q95=q95)
    payload = {
        "n": len(table),
        "split_counts": counts,
        "floor_midband_median_abs_g": floor,
        "envelope_halfwidth_midband_median": mid_half,
        "empirical_f_max_hz": f_max,
        "rule": (
            "Center station of each flat-contact vs_cov=0 row. "
            "Floor is the median of mean |g_true| on 0.4-2.5 Hz. "
            "Empirical f_max is the first frequency above 2.5 Hz where the "
            "5-95% half-width exceeds twice its mid-band median."
        ),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.48, 3.4))
    ax.fill_between(freq, q05, q95, color="#56B4E9", alpha=0.35, linewidth=0, label="5–95%")
    ax.plot(freq, median, color="#0072B2", lw=1.2, label="Median")
    ax.axhline(0.0, color="#000000", lw=0.6)
    ax.axvspan(0.4, 2.5, color="#F0E442", alpha=0.25, lw=0)
    if f_max is not None:
        ax.axvline(f_max, color="#D55E00", lw=1.0, ls="--", label=rf"Empirical $f_{{\max}}$ = {f_max:.2f} Hz")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10.0)
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$g_{\mathrm{true}}(f)$")
    ax.set_title(rf"Floor {floor:.3f} on 0.4–2.5 Hz  (n = {len(table)})")
    ax.legend(frameon=False, fontsize=7)
    panel_letter(ax, "a")
    fig.tight_layout()
    save_figure(fig, OUT / "central.pdf")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
