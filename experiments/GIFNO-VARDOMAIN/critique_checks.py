#!/usr/bin/env python3
"""Three checks the POD floor does not answer.

1. Skill of the current checkpoint against the stored 1D prior, per band and
   family. Skill is 1 - (model log-L1) / mean|g|. The average is the mean of
   per-width means, which is how the trainer prints low/mid/high.
2. Whether 5-20 m lateral Vs survives the frozen 16-component column PCA, and
   what wavelength the global and local spectral paths can represent.
3. Spread of high-band g across random-field seeds of one geology. That spread
   is a floor only for a model that does not see the seed. This model is given
   the realization, so the spread is the realization-specific part of g.

    python critique_checks.py --cache data/vardomain/cache --pod checkpoints/vardomain/v2_ablations/A
    python critique_checks.py --cache ... --pod ... --skill
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from data import FAMILY_NAME, Cache
from encoding import NX_METRIC


BANDS = (("low", 0.1, 0.4), ("mid", 0.4, 2.5), ("high", 2.5, 10.0))
VS_REF = 400.0
ZETA_REF = 0.02
SOIL_VS = 1500.0
GLOBAL_MODES = 96
LOCAL_PATCH_M = 20
LOCAL_MIN_M = 10.0


def _band_mask(freq: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freq >= lo) & (freq <= hi)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def seed_spread(cache: Cache) -> None:
    """RMS of g minus the seed-mean, at matched stations, for groups with two or more seeds."""
    freq = np.asarray(cache.freq, dtype=np.float64)
    print("\nseed spread of g = log|TF| - log|TF_1D|")
    buckets: dict[tuple, list[np.ndarray]] = defaultdict(list)
    for flat in cache.runs("val", ("iid", "ood_dipping", "ood_three_layer")):
        camp, j = cache.campaign(int(flat))
        g = np.asarray(camp.log_tf[j], dtype=np.float64) - np.asarray(camp.log_tf1d[j], dtype=np.float64)
        buckets[(camp.name, int(camp.group[j]))].append(g)
    _report_groups(freq, buckets)

    sections: dict[tuple, list[np.ndarray]] = defaultdict(list)
    for j in list(cache.section_indices("val")) + list(cache.section_indices("train")):
        sec = cache.sections[int(j)]
        g = np.asarray(sec.log_tf, dtype=np.float64) - np.asarray(sec.log_tf1d, dtype=np.float64)
        x = np.asarray(sec.x_m, dtype=np.float64)
        keep = (x >= 5.0) & (x <= sec.lx - 5.0) & np.isclose(np.mod(x, 20.0), 0.0, atol=0.2)
        if int(np.sum(keep)) < 2:
            continue
        sections[("manifold", int(sec.group), int(sec.lx))].append(g[keep])
    _report_groups(freq, sections)


def _report_groups(freq: np.ndarray, buckets: dict) -> None:
    by_family: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for key, curves in buckets.items():
        family = key[0]
        shapes = {c.shape for c in curves}
        if len(curves) < 2 or len(shapes) != 1:
            continue
        stack = np.stack(curves, axis=0)
        dev = stack - stack.mean(axis=0, keepdims=True)
        for name, lo, hi in BANDS:
            mask = _band_mask(freq, lo, hi)
            by_family[f"{family}:{name}"].append((
                _rms(dev[..., mask]),
                float(np.mean(np.abs(stack[..., mask]))),
                int(stack.shape[0]),
            ))
    families = []
    for label in by_family:
        fam = label.split(":")[0]
        if fam not in families:
            families.append(fam)
    for fam in families:
        n_groups = len(by_family[f"{fam}:high"])
        seeds = [row[2] for row in by_family[f"{fam}:high"]]
        print(f"  {fam:16s} groups {n_groups:4d}  seeds/group {min(seeds)}-{max(seeds)}")
        for name, _, _ in BANDS:
            rows = by_family[f"{fam}:{name}"]
            spread = float(np.median([row[0] for row in rows]))
            mag = float(np.median([row[1] for row in rows]))
            print(f"    {name:4s}  median seed-RMS {spread:.4f}  median mean|g| {mag:.4f}  spread/|g| {spread / mag:.3f}")


def _lateral_power(field: np.ndarray) -> dict[str, float]:
    """Power of a (depth, x) field in wavelength bands. dx is 1 m. The mean of each row is removed."""
    centered = field - field.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(centered, axis=1)
    power = np.abs(spec) ** 2
    power[:, 0] = 0.0
    n = field.shape[1]
    k = np.arange(power.shape[1])
    wavelength = np.full(k.shape, np.inf)
    wavelength[1:] = n / k[1:]
    bands = {
        "lt10": wavelength < 10.0,
        "m10_20": (wavelength >= 10.0) & (wavelength < 20.0),
        "m20_42": (wavelength >= 20.0) & (wavelength < NX_METRIC / GLOBAL_MODES),
        "gt42": wavelength >= NX_METRIC / GLOBAL_MODES,
    }
    return {name: float(power[:, mask].sum()) for name, mask in bands.items()}


def _pca_reconstruct(vs: np.ndarray, zeta: np.ndarray, mean: np.ndarray, components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column PCA of interleaved log Vs and log zeta. Returns soil log Vs and its reconstruction."""
    height = int(components.shape[1] // 2)
    rows = min(int(vs.shape[0]), height)
    soil = np.median(vs[:rows], axis=1) < SOIL_VS
    if not np.any(soil):
        soil = np.ones(rows, dtype=bool)
    feat = np.broadcast_to(mean, (vs.shape[1], mean.shape[0])).copy()
    log_vs = np.log(np.maximum(vs[:rows], 1e-6) / VS_REF)
    log_zeta = np.log(np.maximum(zeta[:rows], 1e-8) / ZETA_REF)
    feat[:, 0 : 2 * rows : 2] = log_vs.T
    feat[:, 1 : 2 * rows : 2] = log_zeta.T
    code = (feat - mean) @ components.T
    recon = mean + code @ components
    hat = recon[:, 0 : 2 * rows : 2].T
    return log_vs[soil], hat[soil]


def input_resolution(cache: Cache, pod: Path) -> None:
    mean = np.load(pod / "pca_mean.npy").astype(np.float64)
    components = np.load(pod / "pca_components.npy").astype(np.float64)
    shortest = NX_METRIC / GLOBAL_MODES
    print("\ninput resolution")
    print(
        f"  canvas {NX_METRIC} columns at 1 m. Global rfft keeps {GLOBAL_MODES} modes, "
        f"shortest wavelength {shortest:.2f} m."
    )
    print(
        f"  Local patches are {LOCAL_PATCH_M} m. The training cutoff keeps wavelengths "
        f">= {LOCAL_MIN_M:.0f} m inside a patch, so 5 m is dropped there and 10-20 m is kept."
    )
    print("  The high-pass residual (4 m pool) is a third path and is not cut at that wavelength.")
    print(f"  PCA is {components.shape[0]} components of a depth column, not a lateral filter.")
    totals = {key: 0.0 for key in ("lt10", "m10_20", "m20_42", "gt42")}
    kept = dict(totals)
    err = dict(totals)
    sq = 0.0
    n_cell = 0
    n_sec = 0
    for j in cache.section_indices("val"):
        sec = cache.sections[int(j)]
        true, hat = _pca_reconstruct(
            np.asarray(sec.vs, dtype=np.float64),
            np.asarray(sec.zeta, dtype=np.float64),
            mean,
            components,
        )
        for key, value in _lateral_power(true).items():
            totals[key] += value
        for key, value in _lateral_power(hat).items():
            kept[key] += value
        for key, value in _lateral_power(true - hat).items():
            err[key] += value
        sq += float(np.sum((true - hat) ** 2))
        n_cell += int(true.size)
        n_sec += 1
    order = ("lt10", "m10_20", "m20_42", "gt42")
    labels = {"lt10": "<10 m", "m10_20": "10-20 m", "m20_42": "20-42 m", "gt42": ">42 m"}
    grand = sum(totals.values())
    print(f"  manifold val sections {n_sec}, soil-cell log-Vs RMSE {np.sqrt(sq / max(n_cell, 1)):.4f}")
    print(f"  {'band':8s} {'true share':>10s} {'kept share':>10s} {'error/true':>10s}")
    for key in order:
        base = totals[key]
        print(
            f"  {labels[key]:8s} {base / grand:10.3f} {kept[key] / grand:10.3f} "
            f"{err[key] / base:10.3f}"
        )


def _width_mean(bucket: dict[int, list[float]]) -> float:
    if not bucket:
        return float("nan")
    return float(np.mean([np.mean(rows) for rows in bucket.values()]))


def skill(cache: Cache, pod: Path) -> None:
    import inspect

    import torch
    from torch.utils.data import DataLoader

    from data import WindowSet, eval_records
    from evaluate import predict
    from model import load_vardomain_state, model_from_config

    ckpt_path = pod / "last.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    modes = np.load(pod / "pod_modes.npy")
    mean = np.load(pod / "pod_mean.npy")
    pca_mean = np.load(pod / "pca_mean.npy")
    pca_components = np.load(pod / "pca_components.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_config(cfg, modes, mean, pca_mean, pca_components)
    load_vardomain_state(model, ckpt["state_dict"])
    model.to(device)
    model.eval()
    wanted = dict(
        iid_limit=None, positions=1, families=("iid", "ood_dipping", "ood_three_layer"),
        section_stride=5, widths=(500,), extra_section_splits=(),
    )
    accepted = inspect.signature(eval_records).parameters
    records = eval_records(cache, "val", **{key: value for key, value in wanted.items() if key in accepted})
    loader = DataLoader(
        WindowSet(cache, "metric", records, 3, seed=0, mirror=False),
        batch_size=4, shuffle=False, num_workers=0,
    )
    freq = np.asarray(cache.freq, dtype=np.float64)
    err: dict[tuple[str, str], dict[int, list]] = defaultdict(lambda: defaultdict(list))
    base: dict[tuple[str, str], dict[int, list]] = defaultdict(lambda: defaultdict(list))
    n_batch = 0
    print(f"\nskill from {ckpt_path.name} epoch {ckpt.get('epoch')} on {device}, {len(records)} windows", flush=True)
    for log_pred, log_true, log_prior, info, _xa, _xl in predict(model, loader, device, False, "residual"):
        pred = np.asarray(log_pred, dtype=np.float64)
        true = np.asarray(log_true, dtype=np.float64)
        prior = np.asarray(log_prior, dtype=np.float64)
        for i in range(pred.shape[0]):
            fam = FAMILY_NAME.get(int(info[i, 4]), "section")
            width = int(info[i, 2])
            for name, lo, hi in BANDS:
                mask = _band_mask(freq, lo, hi)
                err[(fam, name)][width].append(float(np.mean(np.abs(pred[i, mask] - true[i, mask]))))
                base[(fam, name)][width].append(float(np.mean(np.abs(prior[i, mask] - true[i, mask]))))
        n_batch += 1
        if n_batch % 100 == 0:
            print(f"  batches {n_batch}", flush=True)
    print(f"  {'family':16s} {'band':4s} {'model':>8s} {'mean|g|':>8s} {'skill':>8s}")
    families = ("iid", "manifold", "ood_dipping", "ood_three_layer")
    for fam in families:
        for name, _, _ in BANDS:
            model_l1 = _width_mean(err[(fam, name)])
            prior_l1 = _width_mean(base[(fam, name)])
            skill_value = 1.0 - model_l1 / prior_l1
            print(f"  {fam:16s} {name:4s} {model_l1:8.4f} {prior_l1:8.4f} {skill_value:8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--pod", type=Path, required=True)
    ap.add_argument("--skill", action="store_true")
    args = ap.parse_args()
    cache = Cache(args.cache)
    input_resolution(cache, args.pod)
    seed_spread(cache)
    if args.skill:
        skill(cache, args.pod)


if __name__ == "__main__":
    main()
