#!/usr/bin/env python3
"""Center-station column on a rigid base, scored like the practice table.

The stored column is the elastic half-space outcrop of the profile under the
center recorder, with each layer's zeta as Rayleigh damping at the record f1
and at 10 Hz. The rigid column uses that same profile and that same damping.
It stops at the soil-rock contact (first row with Vs >= 760 m/s) and returns
surface motion over the motion of that base. OpenSees stays the half-space label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE.parents[2].parent / "experiments" / "case7a"))

from bootstrap_ci import bootstrap_cluster_statistic  # noqa: E402
from center_curves import FAMILIES  # noqa: E402
from metrics import anderson, pearson  # noqa: E402
from paths import CACHE  # noqa: E402
from section_1d import RHO, _EPS, rayleigh_xi_target, surface_over_outcrop  # noqa: E402

OUT = HERE / "03_state_of_practice"
CENTER = 10
ROCK_VS = 760.0
F2_HZ = 10.0


def _rigid(freq: np.ndarray, vs: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """|u(surface) / u(base)|. ``xi`` is (n_layer, n_freq)."""
    omega = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)
    u = np.ones(freq.shape, dtype=np.complex128)
    tau = np.zeros_like(u)
    for v, x in zip(vs, xi):
        vs_c = v * np.sqrt(1.0 + 2.0j * x)
        kh = omega / vs_c
        c, s = np.cos(kh), np.sin(kh)
        gk = 1j * omega * RHO * vs_c
        u, tau = u * c + tau * (s / gk), -u * gk * s + tau * c
    return 1.0 / np.maximum(np.abs(u), _EPS)


def _family(family: str) -> dict[str, np.ndarray]:
    root = CACHE / family
    split = np.load(root / "split.npy").astype(str)
    rows = np.flatnonzero(split == "test")
    rec_x = np.load(root / "rec_x.npy")
    col = int(rec_x[CENTER])
    vs = np.asarray(np.load(root / "vs.npy", mmap_mode="r")[rows, :, col], dtype=np.float64)
    zeta = np.asarray(np.load(root / "zeta.npy", mmap_mode="r")[rows, :, col], dtype=np.float64)
    n_active = np.asarray(np.load(root / "n_active.npy")[rows], dtype=np.int32)
    f1 = np.asarray(np.load(root / "rayleigh_f1.npy")[rows], dtype=np.float64)
    freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
    truth = np.exp(np.asarray(np.load(root / "log_tf.npy", mmap_mode="r")[rows, CENTER], dtype=np.float64))
    stored = np.exp(np.asarray(np.load(root / "log_tf1d.npy", mmap_mode="r")[rows, CENTER], dtype=np.float64))
    group = np.asarray(np.load(root / "group.npy")[rows])
    runs = np.asarray(np.load(root / "run_index.npy")[rows])
    outcrop = np.empty((rows.size, freq.size), dtype=np.float64)
    rigid = np.empty_like(outcrop)
    for i in range(rows.size):
        n = int(n_active[i])
        column_vs = vs[i, :n]
        column_zeta = zeta[i, :n]
        xi = np.stack([rayleigh_xi_target(freq, float(z), float(f1[i]), F2_HZ) for z in column_zeta])
        outcrop[i] = surface_over_outcrop(freq, column_vs, dz=1.0, xi=xi, rho=RHO)
        rock = np.flatnonzero(column_vs >= ROCK_VS)
        soil_n = int(rock[0]) if rock.size else n
        if soil_n < 1:
            soil_n = n
        rigid[i] = _rigid(freq, column_vs[:soil_n], xi[:soil_n])
        if (i + 1) % 100 == 0 or i + 1 == rows.size:
            print(f"  {family} {i + 1}/{rows.size}", flush=True)
    return {
        "run": runs, "group": group, "freq": freq, "f1": f1,
        "opensees": truth, "stored": stored, "outcrop": outcrop, "rigid": rigid,
    }


def _save(family: str, frame: dict[str, np.ndarray]) -> None:
    dest = OUT / f"rigid_{family}.npz"
    np.savez_compressed(dest, run=frame["run"], tf=frame["rigid"].astype(np.float32))
    print(f"  wrote {dest}", flush=True)


def _score(freq, truth, curve, f0) -> dict[str, np.ndarray]:
    band = (freq >= 0.4) & (freq <= 2.5)
    whole = (freq >= 0.1) & (freq <= 10.0)
    log_c = np.log(np.clip(curve, 1e-12, None))
    log_t = np.log(np.clip(truth, 1e-12, None))
    return {
        "pearson": np.array([pearson(curve[i], truth[i]) for i in range(truth.shape[0])]),
        "anderson": np.array([anderson(freq, truth[i], curve[i], f0[i]) for i in range(truth.shape[0])]),
        "log_l1_mid": np.mean(np.abs(log_c[:, band] - log_t[:, band]), axis=1),
        "log_l1_full": np.mean(np.abs(log_c[:, whole] - log_t[:, whole]), axis=1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, family in FAMILIES.items():
        print(name, flush=True)
        frame = _family(family)
        gap = np.median(np.abs(np.log(np.clip(frame["outcrop"], 1e-12, None)) - np.log(np.clip(frame["stored"], 1e-12, None))))
        print(f"  median |log outcrop − stored column| {gap:.4e}", flush=True)
        _save(family, frame)
        for method, curve in (("Column, elastic outcrop", frame["outcrop"]), ("Column, rigid base", frame["rigid"])):
            scored = _score(frame["freq"], frame["opensees"], curve, frame["f1"])
            stat_p = bootstrap_cluster_statistic(scored["pearson"], frame["group"], np.median, rng=np.random.default_rng(0))
            stat_a = bootstrap_cluster_statistic(scored["anderson"], frame["group"], np.median, rng=np.random.default_rng(0))
            site = pd.DataFrame({"value": scored["pearson"], "group": frame["group"]}).groupby("group")["value"].median()
            rows.append({
                "domain": name,
                "method": method,
                "n_records": int(frame["run"].size),
                "n_sites": stat_p["n_clusters"],
                "pearson_record": stat_p["point"],
                "pearson_low": stat_p["low"],
                "pearson_high": stat_p["high"],
                "pearson_site": float(site.median()),
                "anderson_record": stat_a["point"],
                "anderson_low": stat_a["low"],
                "anderson_high": stat_a["high"],
                "log_l1_0p4_2p5": float(np.median(scored["log_l1_mid"])),
                "log_l1_0p1_10": float(np.median(scored["log_l1_full"])),
                "outcrop_matches_stored_log": float(gap),
            })
    table = pd.DataFrame(rows)
    dest = OUT / "table_rigid_column.csv"
    table.to_csv(dest, index=False)
    print(table.to_string(index=False))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
