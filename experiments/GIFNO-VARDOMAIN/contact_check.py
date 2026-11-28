#!/usr/bin/env python3
"""Checks for the contact channels before a training arm is started."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from data import KIND_SECTION, Cache, make_window
from model import CONTACT_DEPTH_M, contact_raw, fit_contact_whitening


def _step_profile() -> None:
    vs = np.concatenate([
        np.full(40, 200.0),
        np.full(40, 500.0),
        np.full(40, 1200.0),
    ])[:, None]
    log_vs = torch.from_numpy(np.log(vs / 400.0).astype(np.float32)).unsqueeze(0)
    observed = torch.ones_like(log_vs, dtype=torch.bool)
    raw = contact_raw(log_vs, observed)[0, :, 0, 0].numpy()
    depth_m = raw[:6] * CONTACT_DEPTH_M
    if abs(depth_m[0] - 80.0) > 0.1 or abs(depth_m[1] - 40.0) > 0.1:
        raise SystemExit(f"threshold depths {depth_m[:2]}")
    if not (raw[6] == 1.0 and raw[7] == 1.0):
        raise SystemExit(f"flags {raw[6:8]}")
    if not (depth_m[2] < depth_m[4]):
        raise SystemExit(f"jumps not ordered by depth {depth_m[2:6]}")
    hot = np.full((16, 1), 1000.0)
    log_hot = torch.from_numpy(np.log(hot / 400.0).astype(np.float32)).unsqueeze(0)
    filled = contact_raw(log_hot, torch.ones_like(log_hot, dtype=torch.bool))[0, :, 0, 0]
    if float(filled[0]) != 1.0 or float(filled[6]) != 0.0:
        raise SystemExit(f"surface-above-760 should fill, got {filled.tolist()}")
    print("synthetic profiles ok", flush=True)


def _pair_fraction(depth: np.ndarray, flag: np.ndarray) -> float:
    both = (flag[:-1] > 0.5) & (flag[1:] > 0.5)
    if not np.any(both):
        return float("nan")
    jump = np.abs(np.diff(depth)) * CONTACT_DEPTH_M > 5.0
    return float(np.mean(jump[both]))


def main() -> None:
    _step_profile()
    cache = Cache(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/vardomain/cache"))
    mean, std = fit_contact_whitening(cache, column_stride=10)
    print("train mean", np.array2string(mean, precision=4), flush=True)
    print("train std", np.array2string(std, precision=4), flush=True)

    def columns(vs: np.ndarray) -> np.ndarray:
        field = np.asarray(vs, dtype=np.float64)
        log_vs = np.log(np.maximum(field, 1e-6) / 400.0).astype(np.float32)
        raw = contact_raw(
            torch.from_numpy(log_vs)[None],
            torch.ones((1, log_vs.shape[0], log_vs.shape[1]), dtype=torch.bool),
        )
        return raw[0, :, 0, :].numpy()

    names = ("iid", "sections")
    stores = {name: [] for name in names}
    for flat in cache.runs("val", ("iid",)):
        camp, j = cache.campaign(int(flat))
        n_active = int(camp.n_active[j])
        stores["iid"].append(columns(np.asarray(camp.vs[j, :n_active])))
    for j in cache.section_indices("val"):
        stores["sections"].append(columns(np.asarray(cache.sections[int(j)].vs)))
    for name, parts in stores.items():
        raw = np.concatenate(parts, axis=1)
        whitened = (raw[:6] - mean[:, None]) / std[:, None]
        print(f"{name} columns {raw.shape[1]}", flush=True)
        print(f"  fill760 {1.0 - float(raw[6].mean()):.3f} fill360 {1.0 - float(raw[7].mean()):.3f}", flush=True)
        print(f"  flicker760 {_pair_fraction(raw[0], raw[6]):.3f} flicker360 {_pair_fraction(raw[1], raw[7]):.3f}", flush=True)
        print("  whitened std", np.array2string(whitened.std(axis=1), precision=3), flush=True)

    from encoding import CH_LATERAL

    sec_j = int(cache.section_indices("val")[0])
    sec = cache.sections[sec_j]
    width = min(int(sec.lx), 1000)
    differed = []
    for offset in (0.0, 10.0):
        window = make_window(cache, "metric", KIND_SECTION, sec_j, width, 0, 20, False, offset)
        x = window[0].numpy()
        raw = contact_raw(
            torch.from_numpy(x[0:1]),
            torch.from_numpy(x[2:3] > 0.5),
        )[0, 0, 0].numpy()
        interior = x[CH_LATERAL, 0] > 0.5
        differed.append(raw[interior])
    change = float(np.mean(np.abs(differed[0] - differed[1])) * CONTACT_DEPTH_M)
    print(f"jitter depth change on val section {sec_j} {change:.3f} m", flush=True)
    if change < 0.05:
        raise SystemExit("jitter did not move the contact depth")
    print("checks done", flush=True)


if __name__ == "__main__":
    main()
