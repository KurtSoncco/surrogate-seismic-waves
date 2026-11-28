#!/usr/bin/env python3
"""Phase 1 data audit from caches already on disk.

No OpenSees launch and no frozen-model forward. The finite-element floor in
plan section 1.3 is not measured here. The noise-floor placeholder in
``gates.py`` is left unchanged.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = Path(__file__).resolve().parents[1]
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from score_harness import band_mask  # noqa: E402

REPO = ANALYSIS.parents[2]
CACHE = REPO / "data" / "vardomain" / "cache"
POD = REPO / "checkpoints" / "vardomain" / "v2_ablations" / "P128n128"
OUT = Path(__file__).resolve().parent
BANDS = ("low", "mid", "high", "overall")
W_TRAIN_M = 500.0
DX_M = 1.0
F_EDGES = (0.1, 0.4, 2.5, 10.0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "median": np.nan, "p10": np.nan, "p90": np.nan}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _band_l1(curves: np.ndarray, freq: np.ndarray, band: str) -> np.ndarray:
    """Mean absolute value on ``band``. ``curves`` is (..., freq)."""
    mask = band_mask(freq, band)
    return np.abs(curves[..., mask]).mean(axis=-1)


def census(freq: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        folder = CACHE / family
        split = np.load(folder / "split.npy")
        group = np.load(folder / "group.npy")
        labels, counts = np.unique(split, return_counts=True)
        count_of = {str(k): int(v) for k, v in zip(labels, counts)}
        crossed = 0
        for gid in np.unique(group):
            if len(set(split[group == gid].tolist())) > 1:
                crossed += 1
        vs = np.load(folder / "vs.npy", mmap_mode="r")
        rows.append(
            {
                "item": "1.1",
                "family": family,
                "n_records": int(split.size),
                "n_groups": int(np.unique(group).size),
                "n_train": count_of.get("train", 0),
                "n_val": count_of.get("val", 0),
                "n_test": count_of.get("test", 0),
                "groups_crossing_splits": crossed,
                "vs_shape": "x".join(str(x) for x in vs.shape),
                "n_freq": int(freq.size),
                "f_min_hz": float(freq[0]),
                "f_max_hz": float(freq[-1]),
                "width_m": W_TRAIN_M,
                "note": "500 m strip stored in the cache. IID depth axis is NZ_IID = 110.",
            }
        )
    manifest = Path(
        "/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data/vardomain/manifold150/manifest.csv"
    )
    man_rows = list(csv.DictReader(manifest.open()))
    complete = [r for r in man_rows if r.get("status") in (None, "", "complete")]
    npz_files = sorted((CACHE / "sections").glob("manifold__man_*.npz"))
    width_counts: dict[int, int] = {}
    damp_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for path in npz_files:
        with np.load(path) as blob:
            width = int(blob["vs"].shape[1])
            damp = str(blob["damping"])
            split_name = str(blob["split"])
        width_counts[width] = width_counts.get(width, 0) + 1
        damp_counts[damp] = damp_counts.get(damp, 0) + 1
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
    rows.append(
        {
            "item": "1.1",
            "family": "manifold",
            "n_records": len(npz_files),
            "n_manifest_rows": len(man_rows),
            "n_manifest_complete": len(complete),
            "n_train": split_counts.get("train", 0),
            "n_val": split_counts.get("val", 0),
            "n_blind": split_counts.get("blind", 0),
            "widths": str(dict(sorted(width_counts.items()))),
            "damping": str(damp_counts),
            "soil_fill": "gradient",
            "note": "H5s are on Box at manifold150/h5. Local cache is the npz sections. Round-0 2000 m meshes are a different tree and are not counted here.",
        }
    )
    round0 = REPO / "data" / "vardomain" / "case7_sections_round0_h5"
    n_round0 = len(list(round0.glob("*.h5"))) if round0.is_dir() else 0
    rows.append(
        {
            "item": "1.1",
            "family": "round0_sections",
            "n_records": n_round0,
            "note": "Flagged. 2000 x 150 m, no free-field pad. Labels in section_labels/round0 carry CSV zeta. Not a clean 2000 m domain.",
        }
    )
    ti = REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5"
    rows.append(
        {
            "item": "1.1",
            "family": "treasure_island",
            "n_records": 1,
            "files": "full_qvs.h5, full_svlab.h5",
            "present": str(ti.is_dir()),
            "note": "full_svlab.h5 is not a second solve. Record length 15 s. Not on the Box mount.",
        }
    )
    return rows


def coverage(freq: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    scatter = {}
    for family, width in (("iid", 500.0), ("ood_dipping", 500.0), ("ood_three_layer", 500.0)):
        folder = CACHE / family
        vs = np.load(folder / "vs.npy", mmap_mode="r")
        n_active = np.load(folder / "n_active.npy")
        split = np.load(folder / "split.npy")
        rec_x = np.load(folder / "rec_x.npy")
        center = int(rec_x[rec_x.size // 2])
        n = vs.shape[0]
        h_m = n_active.astype(np.float64)
        vs_surface = np.empty(n, dtype=np.float64)
        vs_mean = np.empty(n, dtype=np.float64)
        cov = np.empty(n, dtype=np.float64)
        for start in range(0, n, 256):
            stop = min(n, start + 256)
            block = np.asarray(vs[start:stop], dtype=np.float32)
            surface = block[:, 0, :]
            vs_surface[start:stop] = surface.mean(axis=1)
            cov[start:stop] = surface.std(axis=1) / np.maximum(surface.mean(axis=1), 1e-6)
            for j, row in enumerate(block):
                depth = int(n_active[start + j])
                column = row[: max(depth, 1), center]
                vs_mean[start + j] = float(column.mean())
        f0 = vs_mean / (4.0 * np.maximum(h_m, 1.0))
        for band_f in F_EDGES:
            lam_over_w = vs_surface / (band_f * width)
            stats = _quantiles(lam_over_w)
            rows.append(
                {
                    "item": "1.2",
                    "family": family,
                    "width_m": width,
                    "quantity": f"lambda_over_W_at_{band_f:g}Hz",
                    **stats,
                }
            )
        rows.append({"item": "1.2", "family": family, "quantity": "H_m", **_quantiles(h_m)})
        rows.append({"item": "1.2", "family": family, "quantity": "Vs_surface_mps", **_quantiles(vs_surface)})
        rows.append({"item": "1.2", "family": family, "quantity": "f0_quarter_wave_hz", **_quantiles(f0)})
        rows.append({"item": "1.2", "family": family, "quantity": "surface_CoV", **_quantiles(cov)})
        scatter[family] = {
            "split": split,
            "lam_0p1": vs_surface / (0.1 * width),
            "h": h_m,
        }
    # Manifold widths from the cached sections, surface Vs at the center column.
    lam = []
    widths = []
    for path in sorted((CACHE / "sections").glob("manifold__man_*.npz")):
        with np.load(path) as blob:
            field = np.asarray(blob["vs"], dtype=np.float32)
        width = float(field.shape[1])
        surface = field[0]
        lam.append(float(surface.mean() / (0.1 * width)))
        widths.append(width)
    rows.append(
        {
            "item": "1.2",
            "family": "manifold",
            "quantity": "lambda_over_W_at_0.1Hz",
            **_quantiles(np.asarray(lam)),
        }
    )
    return rows, scatter, np.asarray(lam), np.asarray(widths)


def points_per_wavelength() -> list[dict]:
    rows = []
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        vs = np.load(CACHE / family / "vs.npy", mmap_mode="r")
        # Surface row is the slowest soil the 10 Hz wave sees in this crop.
        surface = np.asarray(vs[:, 0, :], dtype=np.float32)
        vs_min = float(surface.min())
        ppw = vs_min / (10.0 * DX_M)
        rows.append(
            {
                "item": "1.3",
                "family": family,
                "dx_m": DX_M,
                "vs_min_surface_mps": vs_min,
                "points_per_wavelength_10hz": ppw,
                "fe_floor_measured": "no",
                "note": "Mesh-refinement and repeated-solve reruns were not launched.",
            }
        )
    rows.append(
        {
            "item": "1.3",
            "family": "treasure_island",
            "record_s": 15.0,
            "fe_floor_measured": "no",
            "note": "15 s is the short record. Full-versus-truncated TF leakage was not recomputed.",
        }
    )
    return rows


def signal(freq: np.ndarray) -> tuple[list[dict], dict]:
    rows = []
    curves = {}
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        folder = CACHE / family
        log_tf = np.load(folder / "log_tf.npy", mmap_mode="r")
        log_1d = np.load(folder / "log_tf1d.npy", mmap_mode="r")
        split = np.load(folder / "split.npy")
        per_band = {band: [] for band in BANDS}
        per_split = {band: {name: [] for name in ("train", "val", "test")} for band in BANDS}
        for start in range(0, log_tf.shape[0], 256):
            stop = min(log_tf.shape[0], start + 256)
            g = np.asarray(log_tf[start:stop], dtype=np.float32) - np.asarray(log_1d[start:stop], dtype=np.float32)
            # Mean over stations, then the band mean. One number per record.
            g_rec = g.mean(axis=1)
            for band in BANDS:
                values = _band_l1(g_rec, freq, band)
                per_band[band].append(values)
                for name in ("train", "val", "test"):
                    mask = split[start:stop] == name
                    if np.any(mask):
                        per_split[band][name].append(values[mask])
        packed = {band: np.concatenate(parts) for band, parts in per_band.items()}
        curves[family] = packed
        for band in BANDS:
            stats = _quantiles(packed[band])
            row = {"item": "1.4", "family": family, "split": "all", "band": band, "width_m": W_TRAIN_M, **stats}
            rows.append(row)
            for name in ("train", "val", "test"):
                parts = per_split[band][name]
                values = np.concatenate(parts) if parts else np.array([])
                rows.append(
                    {
                        "item": "1.4",
                        "family": family,
                        "split": name,
                        "band": band,
                        "width_m": W_TRAIN_M,
                        **_quantiles(values),
                    }
                )
    return rows, curves


def realization(freq: np.ndarray) -> list[dict]:
    rows = []
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        folder = CACHE / family
        group = np.load(folder / "group.npy")
        log_tf = np.load(folder / "log_tf.npy", mmap_mode="r")
        log_1d = np.load(folder / "log_tf1d.npy", mmap_mode="r")
        rec_x = np.load(folder / "rec_x.npy")
        station = int(rec_x.size // 2)
        order = np.argsort(group, kind="mergesort")
        group = group[order]
        breaks = np.flatnonzero(np.diff(group)) + 1
        spans = np.split(order, breaks)
        for band in BANDS:
            pair_medians = []
            for index in spans:
                if index.size < 2:
                    continue
                g = np.asarray(log_tf[index, station], dtype=np.float32) - np.asarray(
                    log_1d[index, station], dtype=np.float32
                )
                mask = band_mask(freq, band)
                # Mean |g_i - g_j| on the band, upper triangle only.
                diff = np.abs(g[:, None, :] - g[None, :, :])[..., mask].mean(axis=-1)
                iu = np.triu_indices(index.size, k=1)
                pair_medians.append(float(np.median(diff[iu])))
            rows.append(
                {
                    "item": "1.5",
                    "family": family,
                    "band": band,
                    "station": "center",
                    "n_groups": len(pair_medians),
                    **{k if k != "n" else "n_pair_groups": v for k, v in _quantiles(np.asarray(pair_medians)).items()},
                    "note": "Median within-group pairwise |g| , then the quantile across groups. Same (H, CoV, f0) or Sobol id.",
                }
            )
    return rows


def pod_floor(freq: np.ndarray) -> list[dict]:
    modes = np.load(POD / "pod_modes.npy").astype(np.float64)
    mean = np.load(POD / "pod_mean.npy").astype(np.float64)
    weight = np.load(POD / "pod_weight.npy").astype(np.float64)
    rows = []
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        folder = CACHE / family
        split = np.load(folder / "split.npy")
        log_tf = np.load(folder / "log_tf.npy", mmap_mode="r")
        log_1d = np.load(folder / "log_tf1d.npy", mmap_mode="r")
        station = log_tf.shape[1] // 2
        for split_name in ("val", "test"):
            index = np.flatnonzero(split == split_name)
            g = np.asarray(log_tf[index, station], dtype=np.float64) - np.asarray(
                log_1d[index, station], dtype=np.float64
            )
            coeff = ((g - mean) * weight) @ modes.T
            recon = mean + coeff @ modes / weight
            for band in BANDS:
                err = _band_l1(g - recon, freq, band)
                rows.append(
                    {
                        "item": "1.6",
                        "family": family,
                        "split": split_name,
                        "band": band,
                        "station": "center",
                        **_quantiles(err),
                        "note": "Reconstruction log-L1 of g_true in the stored 128 modes.",
                    }
                )
    man_err = {band: [] for band in BANDS}
    for path in sorted((CACHE / "sections").glob("manifold__man_*.npz")):
        with np.load(path) as blob:
            g = np.asarray(blob["log_tf"], dtype=np.float64) - np.asarray(blob["log_tf1d"], dtype=np.float64)
        center = g.shape[0] // 2
        curve = g[center]
        coeff = ((curve - mean) * weight) @ modes.T
        recon = mean + coeff @ modes / weight
        for band in BANDS:
            man_err[band].append(float(_band_l1(curve - recon, freq, band)))
    for band in BANDS:
        rows.append(
            {
                "item": "1.6",
                "family": "manifold",
                "split": "cached_sections",
                "band": band,
                "station": "center",
                **_quantiles(np.asarray(man_err[band])),
                "note": "Input PCA was not applied. physical_features is 4-wide and pca_components.npy is (128, 512).",
            }
        )
    return rows


def splits() -> list[dict]:
    rows = []
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        split = np.load(CACHE / family / "split.npy")
        group = np.load(CACHE / family / "group.npy")
        sample = np.load(CACHE / family / "sample_id.npy")
        crossed_g = 0
        crossed_s = 0
        for gid in np.unique(group):
            if len(set(split[group == gid].tolist())) > 1:
                crossed_g += 1
        for sid in np.unique(sample):
            if len(set(split[sample == sid].tolist())) > 1:
                crossed_s += 1
        rows.append(
            {
                "item": "1.7",
                "family": family,
                "groups_crossing_splits": crossed_g,
                "sample_ids_crossing_splits": crossed_s,
                "n_groups": int(np.unique(group).size),
                "note": "split_by_group, seed 42 in build_cache. sample_id in these caches is the group code.",
            }
        )
    # Manifold group_id against split.
    by_group: dict[str, set[str]] = {}
    for path in sorted((CACHE / "sections").glob("manifold__man_*.npz")):
        with np.load(path) as blob:
            gid = str(blob["group_id"])
            by_group.setdefault(gid, set()).add(str(blob["split"]))
    crossed = sum(1 for labels in by_group.values() if len(labels) > 1)
    rows.append(
        {
            "item": "1.7",
            "family": "manifold",
            "groups_crossing_splits": crossed,
            "n_groups": len(by_group),
            "note": "Epoch 46 selects on manifold validation log-L1. AGREE_FLOOR = 0.0136. Blind is not a selection input. Adversarial validation and nearest-neighbor versus model error were not run; they need the frozen forward.",
        }
    )
    return rows


def plot_census(rows: list[dict], path: Path) -> None:
    families = [r["family"] for r in rows if r["family"] in ("iid", "ood_dipping", "ood_three_layer", "manifold")]
    counts = [r["n_records"] for r in rows if r["family"] in ("iid", "ood_dipping", "ood_three_layer", "manifold")]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(families, counts, color="#3c6e71")
    ax.set_ylabel("records in cache")
    ax.set_title("Census of cached campaigns")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_coverage(scatter: dict, man_lam: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"iid": "#1d3557", "ood_dipping": "#e09f3e", "ood_three_layer": "#9b2226"}
    for family, pack in scatter.items():
        test = pack["split"] == "test"
        ax.scatter(pack["h"][test], pack["lam_0p1"][test], s=8, alpha=0.45, c=colors[family], label=f"{family} test")
    if man_lam.size:
        ax.axhline(float(np.median(man_lam)), color="#2a9d8f", lw=1.2, label="manifold median λ/W at 0.1 Hz")
    ax.set_xlabel("stored depth rows (m at 1 m cells)")
    ax.set_ylabel("λ/W at 0.1 Hz, surface Vs")
    ax.set_title("500 m training sits at large λ/W below 0.4 Hz")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_signal(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    families = list(curves)
    x = np.arange(len(BANDS))
    width = 0.25
    for i, family in enumerate(families):
        medians = [float(np.median(curves[family][band])) for band in BANDS]
        ax.bar(x + (i - 1) * width, medians, width, label=family)
    ax.axhline(0.0136, color="black", ls="--", lw=0.8, label="placeholder floor 0.0136")
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_ylabel("median L1(g_true)")
    ax.set_title("Column residual by band. Floor is still the placeholder.")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_bars(rows: list[dict], family_key: str, value_key: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    families = []
    for row in rows:
        if row.get("band") == "mid" and row["family"] not in families:
            families.append(row["family"])
    x = np.arange(len(families))
    vals = []
    for family in families:
        match = [
        r
        for r in rows
        if r["family"] == family and r.get("band") == "mid" and r.get("split", "test") in ("test", "cached_sections", None)
    ] or [r for r in rows if r["family"] == family and r.get("band") == "mid"]
        vals.append(float(match[0][value_key]) if match else np.nan)
    ax.bar(x, vals, color="#457b9d")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_ppw(rows: list[dict], path: Path) -> None:
    keep = [r for r in rows if "points_per_wavelength_10hz" in r]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar([r["family"] for r in keep], [r["points_per_wavelength_10hz"] for r in keep], color="#6d597a")
    ax.set_ylabel("points per wavelength at 10 Hz")
    ax.set_title("Finest surface Vs on a 1 m cell. FE floor not measured.")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_splits(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar(
        [r["family"] for r in rows],
        [r["groups_crossing_splits"] for r in rows],
        color="#2a9d8f",
    )
    ax.set_ylabel("groups in more than one split")
    ax.set_title("Split leakage. Zero means the group stays in one split.")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_summary(census_rows, cov_rows, ppw_rows, sig_rows, real_rows, pod_rows, split_rows) -> None:
    def grab(rows, **want):
        for row in rows:
            if all(row.get(k) == v for k, v in want.items()):
                return row
        return {}

    def num(value) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.4g}"

    iid = grab(census_rows, family="iid")
    dip = grab(census_rows, family="ood_dipping")
    tri = grab(census_rows, family="ood_three_layer")
    man = grab(census_rows, family="manifold")
    sig_low = grab(sig_rows, family="ood_three_layer", split="test", band="low")
    sig_mid = grab(sig_rows, family="ood_three_layer", split="test", band="mid")
    real_mid = grab(real_rows, family="iid", band="mid")
    pod_mid = grab(pod_rows, family="iid", split="test", band="mid")
    text = f"""# Phase 1 audit

G1 is not closed. The finite-element floor was not measured, because the section 1.3 reruns were not launched. The frozen-model error drivers in 1.8 were not run. `gates.noise_floor_log_l1` stays the placeholder 0.0136. `metrics.FLOOR_LOG_L1` was not edited.

## 1.1 Census

IID cache is {iid.get('n_records')} records, {iid.get('n_groups')} groups, splits {iid.get('n_train')}/{iid.get('n_val')}/{iid.get('n_test')}. The Vs array is {iid.get('vs_shape')}: a 500 m strip padded to 110 depth rows. Dipping is {dip.get('n_records')} records ({dip.get('n_train')}/{dip.get('n_val')}/{dip.get('n_test')}). Three-layer matches that count. Manifold npz files: {man.get('n_records')}, splits train/val/blind {man.get('n_train')}/{man.get('n_val')}/{man.get('n_blind')}, widths {man.get('widths')}, damping {man.get('damping')}. The Box manifest has {man.get('n_manifest_rows')} rows and {man.get('n_manifest_complete')} marked complete. Every cached manifold section is `soil_fill=gradient`. Round-0 is 64 local H5s of unpadded 2000 m sections and stays flagged. Treasure Island `full_qvs.h5` is local; `full_svlab.h5` is the same solve.

## 1.2 Coverage

Coordinates use the surface Vs and a 1 m cell. On the 500 m families, λ/W at 0.1 Hz has median {num(grab(cov_rows, family='iid', quantity='lambda_over_W_at_0.1Hz').get('median'))} (IID) and the low band is several wavelengths longer than the training window. That is the empty cell for a 0.1–0.4 Hz lateral question trained at 500 m. Quarter-wave f0·H/Vs is 1/4 when f0 is Vs_column/(4H); the figure shows λ/W instead, which is the quantity that moves.

## 1.3 Label floors

Points per wavelength at 10 Hz use the slowest surface Vs on the 1 m grid. IID minimum is {num(grab(ppw_rows, family='iid').get('points_per_wavelength_10hz'))}. No mesh refinement and no repeated solve were launched, so there is no FE log-L1 floor. The 15 s Treasure Island record was not compared with a truncated spectrum.

## 1.4 Signal

L1(g_true) is the mean absolute column residual, averaged over the 21 stations. Three-layer test median is {num(sig_low.get('median'))} on 0.1–0.4 Hz and {num(sig_mid.get('median'))} on 0.4–2.5 Hz. The published three-layer skill of about −5.689 is a mean of per-station skills on a small residual, which G0 already separated from pooled log-L1. Skill stays suppressed at the placeholder until an FE floor exists. Bands are reported as absolute error.

## 1.5 Realization floor

Within each IID group of 30 field draws that share (H, CoV, f0), the median pairwise |g| at the center station, then the median across groups, is {num(real_mid.get('median'))} on 0.4–2.5 Hz. That is an OpenSees–OpenSees spread for a fixed θ. It is a ceiling only for a model that does not see the realization. P128n128 sees the field, so beating this number is a later claim, not a result of this audit.

## 1.6 Representation floors

Output: g_true at the center station projected with `pod_modes.npy`, `pod_mean.npy`, and `pod_weight.npy`, reconstructed as mean + coeff @ modes / weight. IID test median reconstruction log-L1 on 0.4–2.5 Hz is {num(pod_mid.get('median'))}. Input: `physical_features` is four scalars per column and `pca_components.npy` is (128, 512). No mapping between them was invented, so the depth-PCA floor is not a number in this pass. Treasure Island was not projected; its transfer function is not in the manifold npz cache.

## 1.7 Split integrity

Groups crossing a split: IID {grab(split_rows, family='iid').get('groups_crossing_splits')}, dipping {grab(split_rows, family='ood_dipping').get('groups_crossing_splits')}, three-layer {grab(split_rows, family='ood_three_layer').get('groups_crossing_splits')}, manifold {grab(split_rows, family='manifold').get('groups_crossing_splits')}. The epoch-46 rule in `train.py` selects on manifold validation log-L1 and requires bf16 agreement under `AGREE_FLOOR = 0.0136`. Blind is logged and is not the selection score. Adversarial validation and nearest-neighbor distance versus model error wait on the frozen forward.

## 1.8 Error drivers

Not run. No P128n128 forward was started, and the full test spectra are not already on disk as a harness table. Spearman ranks and the twenty worst records are absent. Paper 2 claims that need those ranks, or a band below a measured noise floor, stay open.
"""
    (OUT / "audit_summary.md").write_text(text)


def main() -> None:
    raise SystemExit(
        "run_audit.py is a superseded draft and would overwrite the Phase 1 tables. "
        "Run audit_census.py through audit_errors.py. The gate is audit_summary.md."
    )
    return
    OUT.mkdir(parents=True, exist_ok=True)
    freq = np.load(CACHE / "freq.npy").astype(np.float64)
    census_rows = census(freq)
    cov_rows, scatter, man_lam, _widths = coverage(freq)
    ppw_rows = points_per_wavelength()
    sig_rows, curves = signal(freq)
    real_rows = realization(freq)
    pod_rows = pod_floor(freq)
    split_rows = splits()
    _write_csv(OUT / "census.csv", census_rows)
    _write_csv(OUT / "coverage.csv", cov_rows)
    _write_csv(OUT / "label_floors.csv", ppw_rows)
    _write_csv(OUT / "signal.csv", sig_rows)
    _write_csv(OUT / "realization_floor.csv", real_rows)
    _write_csv(OUT / "pod_floor.csv", pod_rows)
    _write_csv(OUT / "splits.csv", split_rows)
    _write_csv(
        OUT / "error_drivers.csv",
        [
            {
                "item": "1.8",
                "status": "not_run",
                "note": "Frozen P128n128 forward was not started. No Spearman table.",
            }
        ],
    )
    plot_census(census_rows, OUT / "census.png")
    plot_coverage(scatter, man_lam, OUT / "coverage.png")
    plot_ppw(ppw_rows, OUT / "label_floors.png")
    plot_signal(curves, OUT / "signal.png")
    plot_bars(real_rows, "family", "median", "Within-group OpenSees spread of g, mid band", "median pairwise |g|", OUT / "realization_floor.png")
    plot_bars(pod_rows, "family", "median", "POD reconstruction of g_true, mid band, center", "median log-L1", OUT / "pod_floor.png")
    plot_splits(split_rows, OUT / "splits.png")
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    ax.axis("off")
    ax.set_title("1.8 frozen-model error drivers")
    ax.text(0.0, 0.5, "Not run. No forward of P128n128 in this pass.", transform=ax.transAxes, va="center")
    fig.tight_layout()
    fig.savefig(OUT / "error_drivers.png", dpi=120)
    plt.close(fig)
    write_summary(census_rows, cov_rows, ppw_rows, sig_rows, real_rows, pod_rows, split_rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
