#!/usr/bin/env python3
"""Phase 1.2 dimensionless coverage of the training density and the held-out sets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import AUDIT, BOX, CACHE, REPO, corr_length_m, provenance, save_fig, use_agg
from depth_probe import physical_features, quarter_wavelength_hz

EDGES = (0.1, 0.4, 2.5, 10.0)


def _manifest_by_run(path, run_key: str) -> dict[int, dict]:
    if not Path_ok(path):
        return {}
    frame = pd.read_csv(path)
    key = run_key if run_key in frame.columns else frame.columns[0]
    out = {}
    for row in frame.to_dict(orient="records"):
        out[int(row[key])] = row
    return out


def Path_ok(path) -> bool:
    from pathlib import Path
    return Path(path).is_file()


def _soil_rock(vs: np.ndarray, n_active: int, soil_m: float) -> tuple[float, float, float, float]:
    """Travel-time Vs, arithmetic Vs, rock Vs, and thickness actually used."""

    column = np.asarray(vs[: int(n_active)], dtype=np.float64)
    if column.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    soil_n = int(np.clip(round(float(soil_m)), 1, column.size - 1 if column.size > 1 else 1))
    soil = column[:soil_n]
    rock = column[soil_n:] if soil_n < column.size else column[-1:]
    travel = float(np.sum(1.0 / np.maximum(soil, 1e-6)))
    vs_harm = float(soil_n / travel)
    vs_arith = float(np.mean(soil))
    vs_rock = float(np.mean(rock))
    return vs_harm, vs_arith, vs_rock, float(soil_n)


def _lambda_fields(vs: float, width: float) -> dict:
    out = {}
    for freq in EDGES:
        lam = float(vs) / freq if np.isfinite(vs) and vs > 0 else np.nan
        tag = str(freq).replace(".", "p")
        out[f"lambda_m_{tag}"] = lam
        out[f"lambda_over_w_{tag}"] = lam / width if width else np.nan
        out[f"depth_over_lambda_{tag}"] = np.nan  # filled by caller with H
    return out


def _campaign_rows(name: str, manifest: dict[int, dict], width_m: float) -> list[dict]:
    root = CACHE / name
    vs = np.load(root / "vs.npy", mmap_mode="r")
    zeta = np.load(root / "zeta.npy", mmap_mode="r")
    n_active = np.load(root / "n_active.npy")
    split = np.load(root / "split.npy").astype(str)
    group = np.load(root / "group.npy")
    run_index = np.load(root / "run_index.npy")
    valid = np.load(root / "valid.npy").astype(bool)
    rec_x = np.load(root / "rec_x.npy")
    center = int(rec_x[len(rec_x) // 2])
    rows = []
    for i in range(int(split.size)):
        meta = manifest.get(int(run_index[i]), {})
        soil_m = float(meta.get("H_discretized") or meta.get("H1_discretized") or n_active[i])
        if "H2_discretized" in meta and meta["H2_discretized"] not in ("", None) and not (isinstance(meta["H2_discretized"], float) and np.isnan(meta["H2_discretized"])):
            soil_m = float(meta["H_discretized"]) if "H_discretized" in meta else float(meta["H1_discretized"]) + float(meta["H2_discretized"])
        vs_h, vs_a, vs_r, used = _soil_rock(vs[i, :, center], int(n_active[i]), soil_m)
        column = np.asarray(vs[i, : int(n_active[i]), center], dtype=np.float64)
        zeta_c = np.asarray(zeta[i, : int(n_active[i]), center], dtype=np.float64)
        phys = physical_features(column[:, None], zeta_c[:, None]).ravel()
        f0_q = float(quarter_wavelength_hz(column[:, None])[0]) if column.size else np.nan
        f0 = float(meta["f0_effective"]) if meta.get("f0_effective") not in (None, "") else f0_q
        cov = float(meta["CoV"]) if meta.get("CoV") not in (None, "") else np.nan
        if name == "ood_three_layer" and meta.get("CoV1") not in (None, ""):
            cov = float(meta["CoV1"])
        r_manifest = meta.get("rH", meta.get("rH1", ""))
        r_manifest = float(r_manifest) if r_manifest not in (None, "") else np.nan
        depth_row = min(int(used) // 2, vs.shape[1] - 1) if np.isfinite(used) else 0
        r_hat = corr_length_m(np.log(np.maximum(np.asarray(vs[i, depth_row], dtype=np.float64), 1.0)))
        r_h = r_manifest if np.isfinite(r_manifest) else r_hat
        lam = _lambda_fields(vs_h, width_m)
        for freq, key in ((0.1, "0p1"), (0.4, "0p4"), (2.5, "2p5"), (10.0, "10p0")):
            lam[f"depth_over_lambda_{key}"] = (used * freq / vs_h) if vs_h and np.isfinite(vs_h) else np.nan
        rows.append({
            "sample_id": f"{name}:{i}",
            "family": name,
            "split": split[i],
            "role": split[i],
            "valid": bool(valid[i]),
            "group_id": int(group[i]),
            "run_index": int(run_index[i]),
            "width_m": width_m,
            "mesh_m": 1500,
            "H_m": used,
            "vs_harm": vs_h,
            "vs_arith": vs_a,
            "vs_rock": vs_r,
            "impedance": vs_r / vs_a if vs_a else np.nan,
            "f0_hz": f0,
            "f0_quarter_hz": f0_q,
            "f0_H_over_vs": (f0 * used / vs_a) if vs_a else np.nan,
            "cov": cov,
            "r_h_m": r_h,
            "r_h_manifest_m": r_manifest,
            "r_h_estimated_m": r_hat,
            "r_h_over_w": r_h / width_m if width_m else np.nan,
            "r_h_over_lambda_0p4": (r_h * 0.4 / vs_h) if vs_h else np.nan,
            "zeta_mean": float(np.mean(zeta_c)) if zeta_c.size else np.nan,
            "phys_log_f0": float(phys[0]) if phys.size else np.nan,
            "phys_log_thickness": float(phys[1]) if phys.size > 1 else np.nan,
            "phys_log_vs10": float(phys[2]) if phys.size > 2 else np.nan,
            "phys_log_zeta": float(phys[3]) if phys.size > 3 else np.nan,
            **lam,
        })
    return rows


def _manifold_rows() -> list[dict]:
    manifest = BOX / "vardomain" / "manifold150" / "manifest.csv"
    meta = {}
    if manifest.is_file():
        frame = pd.read_csv(manifest)
        meta = {str(r["sample_id"]): r for r in frame.to_dict(orient="records")}
    rows = []
    sec = CACHE / "sections"
    for path in sorted(sec.glob("*.npz")):
        sid = path.stem.split("__")[-1]
        blob = np.load(path, allow_pickle=True)
        info = meta.get(sid, {})
        vs = np.asarray(blob["vs"], dtype=np.float64)
        zeta = np.asarray(blob["zeta"], dtype=np.float64)
        width = float(vs.shape[1])
        soil_m = float(info.get("soil_thickness_m") or min(80.0, vs.shape[0] - 1))
        center = vs.shape[1] // 2
        vs_h, vs_a, vs_r, used = _soil_rock(vs[:, center], vs.shape[0], soil_m)
        column = vs[: int(max(used, 1)), center]
        zeta_c = zeta[: column.size, center]
        phys = physical_features(column[:, None], zeta_c[:, None]).ravel()
        f0 = float(info["f0_flat_hz"]) if info.get("f0_flat_hz") not in (None, "") else float(quarter_wavelength_hz(column[:, None])[0])
        cov = float(info["vs_cov"]) if info.get("vs_cov") not in (None, "") else np.nan
        r_manifest = float(info["vs_corr_h_m"]) if info.get("vs_corr_h_m") not in (None, "") else np.nan
        depth_row = min(int(used) // 2, vs.shape[0] - 1)
        r_hat = corr_length_m(np.log(np.maximum(vs[depth_row], 1.0)))
        r_h = r_manifest if np.isfinite(r_manifest) else r_hat
        lam = _lambda_fields(vs_h, width)
        for freq, key in ((0.1, "0p1"), (0.4, "0p4"), (2.5, "2p5"), (10.0, "10p0")):
            lam[f"depth_over_lambda_{key}"] = (used * freq / vs_h) if vs_h else np.nan
        rows.append({
            "sample_id": sid,
            "family": "manifold",
            "split": str(blob["split"]),
            "role": str(blob["split"]),
            "valid": True,
            "group_id": str(blob["group_id"]) if "group_id" in blob.files else str(info.get("group_id", "")),
            "run_index": sid,
            "width_m": width,
            "mesh_m": width,
            "H_m": used,
            "vs_harm": vs_h,
            "vs_arith": vs_a,
            "vs_rock": vs_r,
            "impedance": vs_r / vs_a if vs_a else np.nan,
            "f0_hz": f0,
            "f0_quarter_hz": float(quarter_wavelength_hz(column[:, None])[0]),
            "f0_H_over_vs": (f0 * used / vs_a) if vs_a else np.nan,
            "cov": cov,
            "r_h_m": r_h,
            "r_h_manifest_m": r_manifest,
            "r_h_estimated_m": r_hat,
            "r_h_over_w": r_h / width if width else np.nan,
            "r_h_over_lambda_0p4": (r_h * 0.4 / vs_h) if vs_h else np.nan,
            "zeta_mean": float(np.mean(zeta_c)) if zeta_c.size else np.nan,
            "phys_log_f0": float(phys[0]),
            "phys_log_thickness": float(phys[1]),
            "phys_log_vs10": float(phys[2]),
            "phys_log_zeta": float(phys[3]),
            "damping_family": str(blob["damping"]) if "damping" in blob.files else str(info.get("damping_family", "")),
            "soil_fill": str(info.get("soil_fill", "")),
            **lam,
        })
    return rows


def _treasure_row() -> dict:
    import h5py
    import hdf5plugin  # noqa: F401

    path = REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5" / "full_qvs.h5"
    with h5py.File(path, "r") as handle:
        vs = np.asarray(handle["Vs_realization_2D"][:], dtype=np.float64)
        zeta_field = np.asarray(handle["Damping_zeta"][:], dtype=np.float64)
        f0 = float(handle["params"].attrs.get("f0_hz", np.nan))
    # Soil is the profile above the first rock-speed row at the section center.
    center = vs.shape[1] // 2
    column = vs[:, center]
    rock = np.flatnonzero(column > 700.0)
    soil_m = float(rock[0]) if rock.size else float(column.size // 2)
    vs_h, vs_a, vs_r, used = _soil_rock(column, column.size, soil_m)
    width = 2000.0
    depth_row = min(int(used) // 2, vs.shape[0] - 1)
    r_hat = corr_length_m(np.log(np.maximum(vs[depth_row], 1.0)))
    mid = vs[depth_row]
    soil_mask = mid < 700.0
    cov = float(np.std(mid[soil_mask]) / np.mean(mid[soil_mask])) if np.any(soil_mask) else np.nan
    lam = _lambda_fields(vs_h, width)
    for freq, key in ((0.1, "0p1"), (0.4, "0p4"), (2.5, "2p5"), (10.0, "10p0")):
        lam[f"depth_over_lambda_{key}"] = (used * freq / vs_h) if vs_h else np.nan
    zeta = zeta_field[: int(used), center]
    phys = physical_features(column[: int(used), None], np.maximum(zeta, 1e-8)[:, None]).ravel()
    return {
        "sample_id": "treasure_island",
        "family": "treasure_island",
        "split": "blind",
        "role": "treasure",
        "valid": True,
        "group_id": "treasure_island",
        "run_index": "full_qvs",
        "width_m": width,
        "mesh_m": 4000.0,
        "H_m": used,
        "vs_harm": vs_h,
        "vs_arith": vs_a,
        "vs_rock": vs_r,
        "impedance": vs_r / vs_a if vs_a else np.nan,
        "f0_hz": f0,
        "f0_quarter_hz": float(quarter_wavelength_hz(column[: int(used), None])[0]),
        "f0_H_over_vs": (f0 * used / vs_a) if vs_a else np.nan,
        "cov": cov,
        "r_h_m": r_hat,
        "r_h_manifest_m": np.nan,
        "r_h_estimated_m": r_hat,
        "r_h_over_w": r_hat / width,
        "r_h_over_lambda_0p4": (r_hat * 0.4 / vs_h) if vs_h else np.nan,
        "zeta_mean": float(np.mean(zeta)),
        "phys_log_f0": float(phys[0]),
        "phys_log_thickness": float(phys[1]),
        "phys_log_vs10": float(phys[2]),
        "phys_log_zeta": float(phys[3]),
        "note": "Scored window is 2000 m. The OpenSees section is 4000 m. CoV is the lateral coefficient of variation at mid-soil, not a random-field input.",
        **lam,
    }


def _empty_cell_note(frame: pd.DataFrame) -> str:
    train = frame[(frame["family"] == "iid") & (frame["split"] == "train") & frame["valid"]]
    ti = frame[frame["family"] == "treasure_island"]
    if train.empty or ti.empty:
        return "Treasure Island or the IID training split was missing, so the empty-cell check was not made."
    # Training width is 500 m, so lambda/W at 0.1 Hz lives in one vertical strip.
    train_lo = float(train["lambda_over_w_0p1"].quantile(0.05))
    train_hi = float(train["lambda_over_w_0p1"].quantile(0.95))
    ti_val = float(ti["lambda_over_w_0p1"].iloc[0])
    wide = frame[(frame["family"] == "manifold") & (frame["width_m"] >= 1500)]
    wide_med = float(wide["lambda_over_w_0p1"].median()) if len(wide) else float("nan")
    inside = train_lo <= ti_val <= train_hi
    return (
        f"IID train lambda/W at 0.1 Hz, 5th–95th percentile, is {train_lo:.2f}–{train_hi:.2f} "
        f"because every training window is 500 m. Treasure Island at the 2000 m window is {ti_val:.2f} "
        f"({'inside' if inside else 'outside'} that strip). "
        f"Manifold sections at least 1500 m have median lambda/W at 0.1 Hz of {wide_med:.2f}. "
        "The low-frequency lateral question on a 500 m training window sits where the training density has no other width."
    )


def plot(frame: pd.DataFrame, path) -> None:
    use_agg()
    import matplotlib.pyplot as plt

    train = frame[(frame["family"] == "iid") & (frame["split"] == "train")]
    panels = (
        ("width_m", "lambda_over_w_0p1", "width (m)", "λ/W at 0.1 Hz"),
        ("width_m", "lambda_over_w_0p4", "width (m)", "λ/W at 0.4 Hz"),
        ("cov", "impedance", "CoV", "impedance Vs_rock/Vs_soil"),
        ("depth_over_lambda_0p1", "depth_over_lambda_2p5", "H/λ at 0.1 Hz", "H/λ at 2.5 Hz"),
    )
    overlays = (
        ("val", (frame["split"] == "val") & frame["family"].isin(["iid", "ood_dipping", "ood_three_layer"]), "C1"),
        ("test", (frame["split"] == "test") & frame["family"].isin(["iid", "ood_dipping", "ood_three_layer"]), "C3"),
        ("manifold", frame["family"] == "manifold", "C2"),
        ("Treasure Island", frame["family"] == "treasure_island", "C4"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4))
    for ax, (xkey, ykey, xlabel, ylabel) in zip(axes.ravel(), panels):
        ax.hexbin(train[xkey], train[ykey], gridsize=30, cmap="Blues", mincnt=1, linewidths=0)
        for label, mask, color in overlays:
            part = frame[mask]
            ax.scatter(part[xkey], part[ykey], s=18 if label != "Treasure Island" else 60, c=color,
                       label=label, alpha=0.8, zorder=3, marker="*" if label == "Treasure Island" else "o")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Training density and held-out coverage", y=1.02)
    fig.tight_layout()
    save_fig(fig, path)
    plt.close(fig)


def main() -> None:
    from pathlib import Path

    AUDIT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(_campaign_rows("iid", _manifest_by_run(BOX / "transfer_function" / "manifest.csv", "run_index"), 500.0))
    rows.extend(_campaign_rows("ood_dipping", _manifest_by_run(BOX / "ood_dipping" / "manifest.csv", "index"), 500.0))
    rows.extend(_campaign_rows("ood_three_layer", _manifest_by_run(BOX / "ood_three_layer" / "manifest.csv", "index"), 500.0))
    try:
        rows.extend(_manifold_rows())
    except Exception as exc:
        rows.append({"sample_id": "manifold", "family": "manifold", "read_ok": False, "note": f"{type(exc).__name__}: {exc}"})
    try:
        rows.append(_treasure_row())
    except Exception as exc:
        rows.append({"sample_id": "treasure_island", "family": "treasure_island", "read_ok": False,
                     "note": f"{type(exc).__name__}: {exc}"})
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT / "coverage.csv", index=False)
    note = _empty_cell_note(frame) if "lambda_over_w_0p1" in frame.columns else "coverage coordinates were not computed"
    (AUDIT / "coverage_empty_cells.txt").write_text(note + "\n")
    if "lambda_over_w_0p1" in frame.columns and frame["lambda_over_w_0p1"].notna().any():
        plot(frame, AUDIT / "coverage.png")
    provenance(
        AUDIT / "coverage.provenance.json",
        campaign="iid, ood_dipping, ood_three_layer, manifold cache, treasure island full_qvs",
        engine="not a transfer-function score; coordinates from Vs, zeta, and manifests",
        outcrop="not used",
        damping="stored zeta; IID and OOD are the Q(Vs) field, manifold as labeled",
        fft=None,
        checkpoint=None,
        epoch=None,
        dtype="not a model score",
        unit="record",
        sample_ids=frame["sample_id"].astype(str).tolist(),
        difference=(
            "One row per record. W is 500 m for the IID and OOD strips and the interior length for a manifold section. "
            "Treasure Island is placed at the 2000 m scored window. "
            "f0·H/Vs uses the manifest f0 and the arithmetic soil Vs, so it is not forced to 1/4. "
            "λ uses the travel-time soil velocity. "
            + note
        ),
    )
    print(f"coverage rows {len(frame)}", flush=True)
    print(note, flush=True)


if __name__ == "__main__":
    main()
