#!/usr/bin/env python3
"""Phase 1.6 POD and depth-PCA floors.

Output projection follows ``pod_val_floor`` and ``model.reconstruct_pod``.
The stored depth basis is 128 by 512. ``depth_probe.physical_features`` returns
four scalars, so it is reported beside the column reconstruction and is not
multiplied by that basis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import (
    AUDIT, BAND_NAMES, CACHE, CKPT, PCA_HEIGHT, REPO,
    band_l1, column_codes, provenance, save_fig, use_agg,
)
from depth_probe import physical_features
from model import reconstruct_pod

POD = CKPT


def _load_pod():
    modes = np.load(POD / "pod_modes.npy").astype(np.float64)
    mean = np.load(POD / "pod_mean.npy").astype(np.float64)
    weight = np.load(POD / "pod_weight.npy").astype(np.float64)
    return modes, mean, weight


def _reconstruct(g: np.ndarray, modes, mean, weight) -> np.ndarray:
    coeff = ((g - mean) * weight) @ modes.T
    recon = reconstruct_pod(
        torch.as_tensor(mean),
        torch.as_tensor(coeff)[:, None, :],
        torch.as_tensor(modes),
        torch.as_tensor(weight),
    )
    return recon[:, 0, :].detach().cpu().numpy()


def _accumulate(g: np.ndarray, recon: np.ndarray, freq: np.ndarray, sink: dict, key: str) -> None:
    signal = band_l1(g, freq)
    floor = band_l1(g - recon, freq)
    slot = sink.setdefault(key, {band: {"signal": 0.0, "floor": 0.0, "n": 0} for band in BAND_NAMES})
    for band in BAND_NAMES:
        slot[band]["signal"] += float(np.sum(signal[band]))
        slot[band]["floor"] += float(np.sum(floor[band]))
        slot[band]["n"] += int(np.asarray(signal[band]).size)


def _campaign_pod(family: str, freq, modes, mean, weight, sink: dict) -> None:
    root = CACHE / family
    split = np.load(root / "split.npy").astype(str)
    valid = np.load(root / "valid.npy").astype(bool)
    log_tf = np.load(root / "log_tf.npy", mmap_mode="r")
    log_prior = np.load(root / "log_tf1d.npy", mmap_mode="r")
    step = 64
    for start in range(0, split.size, step):
        stop = min(start + step, split.size)
        keep = np.flatnonzero(valid[start:stop])
        if keep.size == 0:
            continue
        g = np.asarray(log_tf[start:stop][keep], dtype=np.float64) - np.asarray(log_prior[start:stop][keep], dtype=np.float64)
        flat = g.reshape(-1, g.shape[-1])
        recon = _reconstruct(flat, modes, mean, weight).reshape(g.shape)
        labels = split[start:stop][keep]
        for lab in np.unique(labels):
            mask = labels == lab
            _accumulate(g[mask], recon[mask], freq, sink, f"{family}:{lab}")
            if lab in ("val", "test"):
                _accumulate(g[mask], recon[mask], freq, sink, lab)


def _section_pod(freq, modes, mean, weight, sink: dict) -> None:
    for path in sorted((CACHE / "sections").glob("*.npz")):
        blob = np.load(path)
        g = np.asarray(blob["log_tf"], dtype=np.float64) - np.asarray(blob["log_tf1d"], dtype=np.float64)
        x = np.asarray(blob["x_m"], dtype=np.float64)
        lx = float(np.asarray(blob["vs"]).shape[1])
        keep = (x >= 5.0) & (x <= lx - 5.0)
        if x.size == g.shape[0] and np.any(keep):
            g = g[keep]
        if g.size == 0:
            continue
        recon = _reconstruct(g, modes, mean, weight)
        lab = str(blob["split"])
        _accumulate(g, recon, freq, sink, f"manifold:{lab}")
        _accumulate(g, recon, freq, sink, "manifold")


def _treasure_pod(freq_cache, modes, mean, weight, sink: dict) -> None:
    blob = np.load(REPO / "experiments" / "GIFNO-VARDOMAIN" / "paper" / "geotech_v2" / "05_treasure_island_full" / "fields.npz")
    freq = np.asarray(blob["freq"], dtype=np.float64)
    if freq.shape != freq_cache.shape or np.max(np.abs(freq - freq_cache)) > 1e-5:
        sink["treasure_island"] = "freq_mismatch"
        return
    g = np.asarray(blob["g_true"], dtype=np.float64)
    recon = _reconstruct(g, modes, mean, weight)
    _accumulate(g, recon, freq, sink, "treasure_island")


def _pca_record_rows() -> list[dict]:
    mean = np.load(POD / "pca_mean.npy").astype(np.float64)
    components = np.load(POD / "pca_components.npy").astype(np.float64)
    if components.shape != (128, 512) or mean.shape != (512,):
        raise ValueError(f"expected PCA (128, 512) and mean (512,), got {components.shape} {mean.shape}")
    rows = []
    codes = []
    ids = []

    def _add(sample_id, family, split, group_id, vs, zeta, n_active, columns):
        code, err = column_codes(vs, zeta, n_active, columns, mean, components)
        center = vs.shape[1] // 2
        column = np.asarray(vs[:n_active, center], dtype=np.float64)
        damp = np.asarray(zeta[:n_active, center], dtype=np.float64)
        phys = physical_features(column[:, None], damp[:, None]).ravel()
        rows.append({
            "sample_id": sample_id,
            "family": family,
            "split": split,
            "group_id": group_id,
            "pca_recon_l1": float(np.mean(err)),
            "n_columns": int(columns.size),
            "phys_log_f0": float(phys[0]),
            "phys_log_thickness": float(phys[1]),
            "phys_log_vs10": float(phys[2]),
            "phys_log_zeta": float(phys[3]),
            "note": (
                "pca_recon_l1 is the mean absolute error of the interleaved log(Vs/400), log(zeta/0.02) "
                f"columns through pca_components {tuple(components.shape)}. "
                "depth_probe.physical_features is the four scalars stored here; it is not a 512-vector and is not multiplied by that basis."
            ),
        })
        codes.append(code.mean(axis=0).astype(np.float32))
        ids.append(sample_id)

    for family in ("iid", "ood_dipping", "ood_three_layer"):
        root = CACHE / family
        vs = np.load(root / "vs.npy", mmap_mode="r")
        zeta = np.load(root / "zeta.npy", mmap_mode="r")
        n_active = np.load(root / "n_active.npy")
        split = np.load(root / "split.npy").astype(str)
        group = np.load(root / "group.npy")
        columns = np.arange(0, vs.shape[-1], 50)
        for i in range(int(split.size)):
            if i % 500 == 0:
                print(f"pca {family} {i}", flush=True)
            _add(f"{family}:{i}", family, split[i], int(group[i]), vs[i], zeta[i], int(n_active[i]), columns)
    for path in sorted((CACHE / "sections").glob("*.npz")):
        blob = np.load(path)
        sid = path.stem.split("__")[-1]
        vs = np.asarray(blob["vs"])
        columns = np.arange(0, vs.shape[1], 40)
        _add(sid, "manifold", str(blob["split"]), str(blob["group_id"]) if "group_id" in blob.files else "",
             vs, np.asarray(blob["zeta"]), int(vs.shape[0]), columns)
    import h5py
    import hdf5plugin  # noqa: F401
    ti = REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5" / "full_qvs.h5"
    with h5py.File(ti, "r") as handle:
        vs = np.asarray(handle["Vs_realization_2D"][:], dtype=np.float64)
        zeta = np.asarray(handle["Damping_zeta"][:], dtype=np.float64)
    _add("treasure_island", "treasure_island", "blind", "treasure_island", vs, zeta, int(vs.shape[0]),
         np.arange(0, vs.shape[1], 80))
    stacked = np.stack(codes).astype(np.float32)
    np.save(AUDIT / "pca_codes.npy", stacked)
    pd.DataFrame({"sample_id": ids}).to_csv(AUDIT / "pca_codes_index.csv", index=False)
    return rows


def _rows_from_sink(sink: dict) -> list[dict]:
    rows = []
    for key, payload in sink.items():
        if not isinstance(payload, dict):
            rows.append({"pool": key, "band": "all", "status": "failed", "note": str(payload)})
            continue
        for band in BAND_NAMES:
            item = payload[band]
            n = max(int(item["n"]), 1)
            signal = item["signal"] / n
            floor = item["floor"] / n
            rows.append({
                "table": "pod_floor",
                "pool": key,
                "band": band,
                "n_curves": int(item["n"]),
                "signal_l1": signal,
                "pod_floor_l1": floor,
                "fraction_left": floor / signal if signal else np.nan,
                "fraction_kept": 1.0 - floor / signal if signal else np.nan,
                "status": "measured",
                "note": "Station-mean |g − P128 g| after the stored 128-mode weighted projection. reconstruct_pod inverts the weight.",
            })
    return rows


def plot(frame: pd.DataFrame, path) -> None:
    use_agg()
    import matplotlib.pyplot as plt

    pools = ["val", "test", "manifold", "treasure_island"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(BAND_NAMES))
    for pool in pools:
        vals = []
        for band in BAND_NAMES:
            hit = frame[(frame["pool"] == pool) & (frame["band"] == band)]
            vals.append(float(hit["pod_floor_l1"].iloc[0]) if len(hit) else np.nan)
        ax.plot(x, vals, marker="o", label=pool)
    ax.set_xticks(x, BAND_NAMES)
    ax.set_ylabel("POD floor, mean |g − g_hat|")
    ax.set_title("128-mode floor of g_true")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, path)
    plt.close(fig)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
    modes, mean, weight = _load_pod()
    toy = np.zeros((1, freq.size), dtype=np.float64)
    left = _reconstruct(toy, modes, mean, weight)
    coeff = ((toy - mean) * weight) @ modes.T
    numpy_recon = mean + coeff @ modes / weight
    if np.max(np.abs(left - numpy_recon)) > 1e-5:
        raise AssertionError("reconstruct_pod does not match the pod_val_floor product")
    sink: dict = {}
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        print(f"pod {family}", flush=True)
        _campaign_pod(family, freq, modes, mean, weight, sink)
    print("pod manifold", flush=True)
    _section_pod(freq, modes, mean, weight, sink)
    print("pod treasure", flush=True)
    _treasure_pod(freq, modes, mean, weight, sink)
    pod_rows = _rows_from_sink(sink)
    print("depth pca", flush=True)
    pca_rows = _pca_record_rows()
    frame = pd.DataFrame(pod_rows)
    pca = pd.DataFrame(pca_rows)
    # Headline depth error by family and split, next to the POD rows.
    depth_rows = []
    for (family, split), part in pca.groupby(["family", "split"], sort=False):
        depth_rows.append({
            "table": "depth_pca",
            "pool": f"{family}:{split}",
            "family": family,
            "split": split,
            "band": "column",
            "n_records": int(len(part)),
            "pca_recon_l1_median": float(part["pca_recon_l1"].median()),
            "pca_recon_l1_mean": float(part["pca_recon_l1"].mean()),
            "status": "measured",
            "note": pca["note"].iloc[0],
        })
    out = pd.concat([frame, pd.DataFrame(depth_rows)], ignore_index=True)
    out.to_csv(AUDIT / "representation.csv", index=False)
    pca.to_csv(AUDIT / "pca_records.csv", index=False)
    plot(frame, AUDIT / "representation.png")
    provenance(
        AUDIT / "representation.provenance.json",
        campaign="iid, ood_dipping, ood_three_layer, manifold sections, treasure island fields.npz g_true",
        engine="stored log_tf1d; Treasure Island g_true is the paper fields.npz curve",
        outcrop="stored prior subtracted before the projection",
        damping="as stored on each campaign",
        fft=1310.72,
        checkpoint=str(CKPT),
        epoch=46,
        dtype="float64 projection of stored curves; the seismic model was not run",
        unit="station",
        sample_ids=list(sink),
        difference=(
            "POD coefficients are ((g − mean) * weight) @ modes.T, the projection in pod_val_floor.py. "
            "Reconstruction calls model.reconstruct_pod. "
            f"Depth PCA components are {tuple(np.load(POD / 'pca_components.npy').shape)} on interleaved log Vs and log zeta. "
            "physical_features is stored as four scalars because that function does not return a 512-vector. "
            f"PCA height used by the basis is {PCA_HEIGHT}."
        ),
    )
    print(f"representation pools {frame['pool'].nunique() if len(frame) else 0}", flush=True)


if __name__ == "__main__":
    main()
