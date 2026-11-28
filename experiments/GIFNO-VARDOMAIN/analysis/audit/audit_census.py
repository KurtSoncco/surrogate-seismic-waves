#!/usr/bin/env python3
"""Phase 1.1 census. Counts and flags. No H5 is copied into the repo."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from common import AUDIT, BOX, CACHE, REPO, provenance, save_fig, use_agg


def _n_h5(path: Path) -> int | None:
    if not path.is_dir():
        return None
    return sum(1 for p in path.iterdir() if p.suffix == ".h5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _h5_probe(path: Path) -> dict:
    import h5py
    import hdf5plugin  # noqa: F401

    out = {"read_ok": True, "note": ""}
    try:
        with h5py.File(path, "r") as handle:
            grid = dict(handle["grid"].attrs) if "grid" in handle else {}
            params = dict(handle["params"].attrs) if "params" in handle else {}
            out["dx_m"] = float(grid.get("dx", np.nan))
            out["dz_m"] = float(grid.get("dz", np.nan))
            out["lx_m"] = float(grid.get("Lx", np.nan))
            out["lz_m"] = float(grid.get("Lz", np.nan))
            out["duration_s"] = float(params["duration"]) if "duration" in params else np.nan
            if "recorders/accel/time" in handle:
                time = np.asarray(handle["recorders/accel/time"][:], dtype=np.float64)
                out["record_s"] = float(time[-1] - time[0]) if time.size > 1 else np.nan
                out["n_time"] = int(time.size)
            else:
                out["record_s"] = np.nan
                out["n_time"] = 0
            out["damping_applied"] = "damping_applied" in handle or "damping_applied" in params
            if "Vs_realization_2D" in handle:
                out["vs_shape"] = "x".join(str(s) for s in handle["Vs_realization_2D"].shape)
            else:
                out["vs_shape"] = ""
            damp = params.get("damping", "")
            if isinstance(damp, bytes):
                damp = damp.decode()
            out["damping_attr"] = str(damp)
            out["pad_m"] = float(params["pad_m"]) if "pad_m" in params else np.nan
    except Exception as exc:
        out = {
            "read_ok": False, "note": f"{type(exc).__name__}: {exc}",
            "dx_m": np.nan, "dz_m": np.nan, "lx_m": np.nan, "lz_m": np.nan,
            "duration_s": np.nan, "record_s": np.nan, "n_time": 0,
            "damping_applied": False, "vs_shape": "", "damping_attr": "", "pad_m": np.nan,
        }
    return out


def _cache_row(name: str) -> dict:
    root = CACHE / name
    row = {"campaign": name, "location": "local cache", "read_ok": root.is_dir()}
    if not root.is_dir():
        row["note"] = "cache directory missing"
        return row
    split = np.load(root / "split.npy").astype(str)
    group = np.load(root / "group.npy")
    valid = np.load(root / "valid.npy")
    n_active = np.load(root / "n_active.npy")
    counts = {k: int(v) for k, v in zip(*np.unique(split, return_counts=True))}
    row.update({
        "n_records": int(split.size),
        "n_train": counts.get("train", 0),
        "n_val": counts.get("val", 0),
        "n_test": counts.get("test", 0),
        "n_blind": counts.get("blind", 0),
        "n_invalid": int((~valid.astype(bool)).sum()),
        "n_groups": int(np.unique(group).size),
        "depth_m_min": int(n_active.min()),
        "depth_m_max": int(n_active.max()),
        "depth_m_median": float(np.median(n_active)),
        "width_m": 500,
        "mesh_m": 1500,
        "clean_domain": True,
        "note": "500 m strip of a 1500 m mesh; rows past n_active repeat the bottom row. NZ_IID cap is 110.",
    })
    return row


def _manifold_rows(manifest: Path) -> list[dict]:
    if not manifest.is_file():
        return [{"campaign": "manifold150", "location": str(manifest), "read_ok": False,
                 "note": "manifest missing"}]
    frame = pd.read_csv(manifest)
    rows = []
    complete = frame[frame["status"] == "complete"] if "status" in frame.columns else frame
    def _add(part: pd.DataFrame, label: str) -> None:
        if part.empty:
            return
        split_counts = part["split"].astype(str).value_counts().to_dict() if "split" in part.columns else {}
        width_counts = part["lx_m"].value_counts().to_dict() if "lx_m" in part.columns else {}
        damp = part["damping_family"].astype(str).value_counts().to_dict() if "damping_family" in part.columns else {}
        fill = part["soil_fill"].astype(str).value_counts().to_dict() if "soil_fill" in part.columns else {}
        rows.append({
            "campaign": f"manifold150 {label}",
            "location": str(manifest),
            "read_ok": True,
            "n_records": int(len(part)),
            "n_train": int(split_counts.get("train", 0)),
            "n_val": int(split_counts.get("val", 0)),
            "n_test": int(split_counts.get("test", 0)),
            "n_blind": int(split_counts.get("blind", 0)),
            "n_groups": int(part["group_id"].nunique()) if "group_id" in part.columns else np.nan,
            "width_counts": str({str(k): int(v) for k, v in width_counts.items()}),
            "damping_counts": str({str(k): int(v) for k, v in damp.items()}),
            "soil_fill_counts": str({str(k): int(v) for k, v in fill.items()}),
            "depth_m_min": float(part["depth_m"].min()) if "depth_m" in part.columns else np.nan,
            "depth_m_max": float(part["depth_m"].max()) if "depth_m" in part.columns else np.nan,
            "duration_s": float(part["duration_s"].median()) if "duration_s" in part.columns else np.nan,
            "clean_domain": True,
            "note": "Manifest split column is the design split. Training uses manifold_holdout. H5s stay on Box.",
        })
    _add(frame, "design rows")
    _add(complete, "status complete")
    return rows


def _section_cache_row() -> dict:
    sec = CACHE / "sections"
    files = sorted(sec.glob("*.npz")) if sec.is_dir() else []
    splits: dict[str, int] = {}
    widths: dict[str, int] = {}
    damps: dict[str, int] = {}
    fills_note = ""
    failed = 0
    for path in files:
        try:
            blob = np.load(path, allow_pickle=True)
            lab = str(blob["split"])
            splits[lab] = splits.get(lab, 0) + 1
            widths[str(int(blob["vs"].shape[1]))] = widths.get(str(int(blob["vs"].shape[1])), 0) + 1
            if "damping" in blob.files:
                damps[str(blob["damping"])] = damps.get(str(blob["damping"]), 0) + 1
        except Exception:
            failed += 1
    return {
        "campaign": "manifold cache sections",
        "location": str(sec),
        "read_ok": failed == 0 and bool(files),
        "n_records": len(files),
        "n_train": splits.get("train", 0),
        "n_val": splits.get("val", 0),
        "n_test": splits.get("test", 0),
        "n_blind": splits.get("blind", 0),
        "width_counts": str(widths),
        "damping_counts": str(damps),
        "n_read_failures": failed,
        "clean_domain": True,
        "note": "Interiors stored after the pad crop. Troughs are dropped when Cache loads, not necessarily here.",
    }


def _round0_row() -> dict:
    h5_dir = REPO / "data" / "vardomain" / "case7_sections_round0_h5"
    files = sorted(h5_dir.glob("*.h5")) if h5_dir.is_dir() else []
    durations = []
    applied = 0
    pads = 0
    shapes = []
    failures = 0
    for path in files:
        probe = _h5_probe(path)
        if not probe["read_ok"]:
            failures += 1
            continue
        durations.append(probe["duration_s"])
        applied += int(bool(probe["damping_applied"]))
        pads += int(np.isfinite(probe["pad_m"]) and probe["pad_m"] > 0)
        shapes.append(probe["vs_shape"])
    labels = REPO / "data" / "vardomain" / "section_labels"
    label_counts = {p.name: len(list(p.glob("*.npz"))) for p in labels.iterdir()} if labels.is_dir() else {}
    return {
        "campaign": "round0 sections",
        "location": str(h5_dir),
        "read_ok": failures == 0 and bool(files),
        "n_files": len(files),
        "n_records": len(files),
        "duration_s": float(np.median(durations)) if durations else np.nan,
        "duration_min_s": float(np.min(durations)) if durations else np.nan,
        "n_damping_applied": applied,
        "n_with_pad": pads,
        "vs_shapes": str(sorted(set(shapes))),
        "label_npz_counts": str(label_counts),
        "clean_domain": False,
        "note": (
            "Flagged, not a clean 2000 m domain. No free-field pad, so a 2000 m window includes the absorbing boundary. "
            "No damping_applied dataset. section_labels/round0 carries CSV zeta in the 1D prior. "
            "round1_fixed is the corrected 1D engine on the same unpadded mesh. "
            f"Read failures: {failures}."
        ),
    }


def _treasure_rows() -> list[dict]:
    h5 = REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5"
    qvs = h5 / "full_qvs.h5"
    sv = h5 / "full_svlab.h5"
    rows = []
    probe = _h5_probe(qvs) if qvs.is_file() else {"read_ok": False, "note": "full_qvs.h5 missing"}
    digest_q = _sha256(qvs) if qvs.is_file() else ""
    digest_s = _sha256(sv) if sv.is_file() else ""
    same = bool(digest_q) and digest_q == digest_s
    rows.append({
        "campaign": "treasure island full_qvs",
        "location": str(qvs),
        "read_ok": bool(probe.get("read_ok")),
        "n_records": 1,
        "duration_s": probe.get("duration_s", np.nan),
        "record_s": probe.get("record_s", np.nan),
        "n_time": probe.get("n_time", 0),
        "dx_m": probe.get("dx_m", np.nan),
        "lx_m": probe.get("lx_m", np.nan),
        "lz_m": probe.get("lz_m", np.nan),
        "vs_shape": probe.get("vs_shape", ""),
        "damping_applied": probe.get("damping_applied", False),
        "pad_m": probe.get("pad_m", np.nan),
        "sha256": digest_q,
        "clean_domain": False,
        "note": (
            "One 15 s Q(Vs) section. The scored window is 2000 m on this 4000 m mesh, with edge columns repeated past the ends. "
            + (probe.get("note") or "")
        ),
    })
    motion_same = False
    vs_same = False
    zeta_max = float("nan")
    if qvs.is_file() and sv.is_file():
        import h5py
        import hdf5plugin  # noqa: F401

        with h5py.File(qvs, "r") as left, h5py.File(sv, "r") as right:
            vs_same = bool(np.array_equal(np.asarray(left["Vs_realization_2D"][:]), np.asarray(right["Vs_realization_2D"][:])))
            motion_same = bool(np.array_equal(
                np.asarray(left["recorders/accel/surface"][:]),
                np.asarray(right["recorders/accel/surface"][:]),
            ))
            zeta_max = float(np.max(np.abs(
                np.asarray(left["Damping_zeta"][:], dtype=np.float64)
                - np.asarray(right["Damping_zeta"][:], dtype=np.float64)
            )))
    rows.append({
        "campaign": "full_svlab",
        "location": str(sv),
        "read_ok": sv.is_file(),
        "n_records": 1 if sv.is_file() else 0,
        "sha256": digest_s,
        "identical_to_full_qvs": same,
        "vs_equal": vs_same,
        "surface_accel_equal": motion_same,
        "damping_zeta_max_abs": zeta_max,
        "clean_domain": False,
        "note": (
            "Not a second OpenSees solve. "
            + (
                "Byte-identical to full_qvs.h5."
                if same
                else "SHA-256 differs because Damping_zeta and the damping attributes differ, "
                f"while Vs equal={vs_same} and surface acceleration equal={motion_same}. "
                "The surface motion was not scored as a separate solve."
            )
        ),
    })
    return rows


def build() -> pd.DataFrame:
    rows: list[dict] = []
    for name in ("iid", "ood_dipping", "ood_three_layer"):
        rows.append(_cache_row(name))
    rows.append(_section_cache_row())
    rows.extend(_manifold_rows(BOX / "vardomain" / "manifold150" / "manifest.csv"))
    design = BOX / "vardomain" / "manifold150" / "design.csv"
    rows.append({
        "campaign": "manifold150 design.csv",
        "location": str(design),
        "read_ok": design.is_file(),
        "n_records": int(sum(1 for _ in design.open()) - 1) if design.is_file() else 0,
        "note": "Design table on Box. Not copied into the repo.",
    })
    probes = {
        "iid h5 (box)": BOX / "h5",
        "ood_dipping h5 (box)": BOX / "ood_dipping" / "h5",
        "ood_three_layer h5 (box)": BOX / "ood_three_layer" / "h5",
        "corner_is h5 (box)": BOX / "corner_is" / "h5",
        "case7 round0 h5 (box)": BOX / "case7_sections" / "round0" / "h5",
        "manifold150 h5 (box)": BOX / "vardomain" / "manifold150" / "h5",
    }
    samples = {
        "iid h5 (box)": BOX / "h5" / "run_0.h5",
        "ood_dipping h5 (box)": None,
        "ood_three_layer h5 (box)": None,
        "corner_is h5 (box)": None,
        "case7 round0 h5 (box)": None,
        "manifold150 h5 (box)": None,
    }
    for label, directory in probes.items():
        count = _n_h5(directory) if directory.exists() else None
        sample = samples[label]
        if sample is None and directory.is_dir():
            found = next((p for p in directory.iterdir() if p.suffix == ".h5"), None)
            sample = found
        probe = _h5_probe(sample) if sample is not None and Path(sample).is_file() else {
            "read_ok": directory.is_dir(), "note": "no sample h5 opened" if directory.is_dir() else "directory missing",
        }
        rows.append({
            "campaign": label,
            "location": str(directory),
            "read_ok": bool(probe.get("read_ok")) and count is not None,
            "n_files": count if count is not None else 0,
            "sample_h5": "" if sample is None else str(sample),
            "dx_m": probe.get("dx_m", np.nan),
            "dz_m": probe.get("dz_m", np.nan),
            "lx_m": probe.get("lx_m", np.nan),
            "lz_m": probe.get("lz_m", np.nan),
            "duration_s": probe.get("duration_s", np.nan),
            "record_s": probe.get("record_s", np.nan),
            "damping_applied": probe.get("damping_applied", False),
            "damping_attr": probe.get("damping_attr", ""),
            "pad_m": probe.get("pad_m", np.nan),
            "vs_shape": probe.get("vs_shape", ""),
            "clean_domain": label.startswith("iid") or label.startswith("ood") or label.startswith("manifold"),
            "note": probe.get("note") or "Counted on Box. Arrays were not copied into the repo. corner_is is inventoried and is not a cache family.",
        })
    rows.append(_round0_row())
    rows.extend(_treasure_rows())
    fixed = REPO / "data" / "vardomain" / "outcrop_fixed_window"
    rows.append({
        "campaign": "outcrop_fixed_window",
        "location": str(fixed),
        "read_ok": fixed.is_dir(),
        "note": "Local fixed-window labels. cache/meta.json fixed is empty, so this directory is not named as the cache source by that file alone.",
        "campaigns_present": str(sorted(p.name for p in fixed.iterdir())) if fixed.is_dir() else "",
    })
    frame = pd.DataFrame(rows)
    return frame


def plot(frame: pd.DataFrame, path: Path) -> None:
    use_agg()
    import matplotlib.pyplot as plt

    keep = frame[frame["campaign"].isin(["iid", "ood_dipping", "ood_three_layer", "manifold cache sections"])]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    labels = keep["campaign"].tolist()
    x = np.arange(len(labels))
    train = keep["n_train"].fillna(0).to_numpy()
    val = keep["n_val"].fillna(0).to_numpy()
    test = keep["n_test"].fillna(0).to_numpy()
    blind = keep["n_blind"].fillna(0).to_numpy()
    ax.bar(x, train, label="train")
    ax.bar(x, val, bottom=train, label="val")
    ax.bar(x, test, bottom=train + val, label="test")
    ax.bar(x, blind, bottom=train + val + test, label="blind")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("records")
    ax.set_title("Cache census")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, path)
    plt.close(fig)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    frame = build()
    csv_path = AUDIT / "census.csv"
    frame.to_csv(csv_path, index=False)
    plot(frame, AUDIT / "census.png")
    ids = frame["campaign"].astype(str).tolist()
    provenance(
        AUDIT / "census.provenance.json",
        campaign="iid, ood_dipping, ood_three_layer, manifold150, round0, treasure island, corner_is",
        engine="not scored; census of stored campaigns",
        outcrop="not scored",
        damping="qvs for IID and OOD caches; manifold mixed; round0 has no damping_applied",
        fft=1310.72,
        checkpoint=None,
        epoch=None,
        dtype="not a model score",
        unit="record",
        sample_ids=ids,
        difference=(
            "Counts and flags only. Round-0 2000 m sections are flagged because the mesh has no free-field pad. "
            "full_svlab.h5 is compared by SHA-256 to full_qvs.h5 and is not treated as a second solve. "
            "Box H5 arrays were not copied into the repo."
        ),
    )
    print(f"census rows {len(frame)} -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
