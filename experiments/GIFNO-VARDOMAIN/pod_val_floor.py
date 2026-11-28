#!/usr/bin/env python3
"""Project true validation residuals onto a saved 128-mode POD.

The fit-set floor is the error on curves the basis was built from. This scores
the curves the loss is judged on. Projection uses the stored sqrt(f) modes.
Band shares are that weight's fraction of the 0.1–10 Hz sum, so they add to one.

The stored grid is log-spaced, so equal weight per bin is already equal weight
per decade. ``per_hz`` weights each bin by its width in Hz. That is the linear
grid the flat column is not.

    python pod_val_floor.py --cache data/vardomain/cache --pod checkpoints/vardomain/v2_ablations/A
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data import Cache
from model import spectral_weight


BANDS = (("low", 0.1, 0.4), ("mid", 0.4, 2.5), ("high", 2.5, 10.0))


def _curves_runs(cache: Cache, family: str) -> np.ndarray:
    rows = []
    for flat in cache.runs("val", (family,)):
        camp, j = cache.campaign(int(flat))
        rows.append(np.asarray(camp.log_tf[j], dtype=np.float64) - np.asarray(camp.log_tf1d[j], dtype=np.float64))
    if not rows:
        return np.zeros((0, cache.freq.size))
    return np.concatenate(rows, axis=0)


def _curves_sections(cache: Cache, split: str) -> np.ndarray:
    rows = []
    for j in cache.section_indices(split):
        sec = cache.sections[int(j)]
        x = sec.x_m.astype(np.float64)
        keep = (x >= 5.0) & (x <= sec.lx - 5.0) & np.isclose(np.mod(x, 20.0), 0.0, atol=0.2)
        if not np.any(keep):
            continue
        g = np.asarray(sec.log_tf, dtype=np.float64) - np.asarray(sec.log_tf1d, dtype=np.float64)
        rows.append(g[keep])
    if not rows:
        return np.zeros((0, cache.freq.size))
    return np.concatenate(rows, axis=0)


def _remainder(curves: np.ndarray, modes: np.ndarray, mean: np.ndarray, weight: np.ndarray) -> np.ndarray:
    err = np.empty_like(curves)
    for i in range(0, curves.shape[0], 2048):
        block = curves[i : i + 2048]
        coeff = ((block - mean) * weight) @ modes.T
        recon = mean + coeff @ modes / weight
        err[i : i + 2048] = np.abs(block - recon)
    return err


def _edge_widths(coord: np.ndarray) -> np.ndarray:
    """Trapezoid width of each sample along a monotone coordinate."""
    width = np.empty_like(coord)
    width[0] = coord[1] - coord[0]
    width[-1] = coord[-1] - coord[-2]
    width[1:-1] = 0.5 * (coord[2:] - coord[:-2])
    return width


def _score(abs_err: np.ndarray, freq: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    profile = abs_err.mean(axis=0)
    total = float((weight * profile).sum())
    out = {"total": total}
    for name, lo, hi in BANDS:
        mask = (freq >= lo) & (freq <= hi)
        part = float((weight[mask] * profile[mask]).sum())
        out[name] = part
        out[name + "_share"] = part / total if total else float("nan")
        out[name + "_mean"] = float(profile[mask].mean())
    return out


def report(cache: Cache, pod: Path) -> None:
    freq = np.asarray(cache.freq, dtype=np.float64)
    modes = np.load(pod / "pod_modes.npy").astype(np.float64)
    mean = np.load(pod / "pod_mean.npy").astype(np.float64)
    fit_w = np.load(pod / "pod_weight.npy").astype(np.float64)
    dlog = _edge_widths(np.log10(freq))
    df = _edge_widths(freq)
    spacing = np.diff(np.log10(freq))
    print(
        f"log10 spacing mean {spacing.mean():.6f} std {spacing.std():.3g} "
        f"(equal per bin is per decade when this std is ~0)"
    )
    for name, lo, hi in BANDS:
        n = int(np.sum((freq >= lo) & (freq <= hi)))
        print(f"  {name:4s} {n:4d} bins  {np.log10(hi / lo):.3f} decades  {hi - lo:.2f} Hz")
    sqrt_w = spectral_weight(freq)
    weights = {
        "per_bin": np.ones_like(freq),
        "per_decade": dlog,
        "per_hz": df,
        "sqrt_f": sqrt_w,
        "sqrt_f_x2_low": sqrt_w * np.where((freq >= 0.1) & (freq <= 0.4), 2.0, 1.0),
    }
    groups = {
        "iid_val": _curves_runs(cache, "iid"),
        "manifold_val": _curves_sections(cache, "val"),
        "dipping_val": _curves_runs(cache, "ood_dipping"),
        "three_layer_val": _curves_runs(cache, "ood_three_layer"),
        "manifold_blind": _curves_sections(cache, "blind"),
    }
    print(f"freq {freq.size} from {freq[0]:.4g} to {freq[-1]:.4g} Hz, modes {modes.shape[0]}")
    for name, curves in groups.items():
        print(f"\n{name} n={curves.shape[0]}")
        true = np.abs(curves)
        left = _remainder(curves, modes, mean, fit_w)
        for wname, weight in weights.items():
            got = _score(true, freq, weight)
            floor = _score(left, freq, weight)
            kept = {
                band: 1.0 - floor[band + "_mean"] / got[band + "_mean"]
                for band, _, _ in BANDS
            }
            print(
                f"  {wname:16s} true {got['total']:.4f} shares "
                f"{got['low_share']:.3f}/{got['mid_share']:.3f}/{got['high_share']:.3f} "
                f"mean {got['low_mean']:.4f}/{got['mid_mean']:.4f}/{got['high_mean']:.4f}"
            )
            print(
                f"  {'floor':16s} left {floor['total']:.4f} shares "
                f"{floor['low_share']:.3f}/{floor['mid_share']:.3f}/{floor['high_share']:.3f} "
                f"mean {floor['low_mean']:.4f}/{floor['mid_mean']:.4f}/{floor['high_mean']:.4f} "
                f"kept {kept['low']:.3f}/{kept['mid']:.3f}/{kept['high']:.3f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--pod", type=Path, required=True)
    args = ap.parse_args()
    report(Cache(args.cache), args.pod)


if __name__ == "__main__":
    main()
