#!/usr/bin/env python3
"""Interpolation, width consistency, skill against g=0, the blind 16, and Case 7 slices.

    python score_holdout.py --model-dir ../../checkpoints/vardomain/stage2_residual_s0 \
        --cache ../../data/vardomain/cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import Cache, WindowSet, eval_records
from evaluate import load_model, predict
from torch.utils.data import DataLoader

CASE7_FULL = Path(__file__).resolve().parents[2] / "data" / "case7a" / "opensees_crops" / "context" / "h5"
ROUND1_LAB = Path(__file__).resolve().parents[2] / "data" / "vardomain" / "section_labels" / "round1_fixed"
ROUND1_H5 = Path(__file__).resolve().parents[2] / "data" / "case7a" / "stampede_round0" / "h5"


def _lerp(x_src: np.ndarray, y_src: np.ndarray, x_q: np.ndarray) -> np.ndarray:
    order = np.argsort(x_src)
    src, ys = x_src[order], y_src[order]
    return np.stack([np.interp(x_q, src, ys[:, f]) for f in range(ys.shape[1])], axis=1).astype(np.float32)


def interpolation(model, cache: Cache, device, target: str) -> dict:
    """Hold out the 5 m stations that are not on the 20 m training grid."""
    records = [r for r in eval_records(cache, "val", families=(), positions=1, section_stride=5) if r["kind"] == "section"]
    if not records:
        return {"n_sections": 0}
    loader = DataLoader(WindowSet(cache, "metric", records, 2, seed=0, mirror=False), batch_size=1, shuffle=False)
    mid = (cache.freq >= 0.4) & (cache.freq <= 2.5)
    model_err, label_err, self_err = [], [], []
    contact_model, contact_label, flat_model, flat_label = [], [], [], []
    for log_pred, log_true, _prior, info, x_abs, _xl in predict(model, loader, device, False, target):
        x = x_abs.astype(np.float64)
        train = np.isclose(np.mod(x, 20.0), 0.0, atol=0.2)
        hold = ~train
        if train.sum() < 2 or hold.sum() == 0:
            continue
        from_labels = _lerp(x[train], log_true[train], x[hold])
        from_model = _lerp(x[train], log_pred[train], x[hold])
        err_m = np.abs(log_pred[hold][:, mid] - log_true[hold][:, mid]).mean(1)
        err_l = np.abs(from_labels[:, mid] - log_true[hold][:, mid]).mean(1)
        err_s = np.abs(from_model[:, mid] - log_true[hold][:, mid]).mean(1)
        model_err.extend(err_m.tolist())
        label_err.extend(err_l.tolist())
        self_err.extend(err_s.tolist())
        sec = cache.sections[int(info[0, 1])]
        grad = np.abs(np.gradient(np.asarray(sec.vs[0], dtype=np.float64)))
        g = np.interp(x[hold], np.arange(sec.lx), grad)
        hi = g >= np.median(grad)
        if hi.any():
            contact_model.extend(err_m[hi].tolist())
            contact_label.extend(err_l[hi].tolist())
        if (~hi).any():
            flat_model.extend(err_m[~hi].tolist())
            flat_label.extend(err_l[~hi].tolist())
    def mean(v):
        return float(np.mean(v)) if v else None
    return {
        "n_stations": len(model_err),
        "model_log_l1_mid": mean(model_err),
        "label_interp_log_l1_mid": mean(label_err),
        "model_interp_log_l1_mid": mean(self_err),
        "contact_model": mean(contact_model),
        "contact_label_interp": mean(contact_label),
        "flat_model": mean(flat_model),
        "flat_label_interp": mean(flat_label),
    }


def width_consistency(model, cache: Cache, device, target: str) -> dict:
    """Centered 1000 m and 1500 m crops of each 2000 m validation section, inner overlap."""
    sections = [i for i, s in enumerate(cache.sections) if s.split == "val" and s.lx >= 2000]
    if not sections:
        return {"n_sections": 0}
    spreads = []
    for j in sections:
        crops = []
        for width, start in ((2000, 0), (1500, 250), (1000, 500)):
            rec = [{
                "kind": "section", "index": j, "draw": 0, "fixed": True, "allow_mirror": False,
                "width": width, "start": start, "stride": 20, "family": "manifold",
            }]
            loader = DataLoader(WindowSet(cache, "metric", rec, 2, seed=0, mirror=False), batch_size=1)
            log_pred, _true, _prior, _info, x_abs, _xl = next(predict(model, loader, device, False, target))
            crops.append((x_abs.astype(np.float64), log_pred))
        # overlap of the three inner regions: 700 m to 1300 m on the 2000 m section
        ref_x, ref_y = crops[0]
        keep = (ref_x >= 700.0) & (ref_x <= 1300.0)
        if keep.sum() == 0:
            continue
        acc = []
        for x, y in crops[1:]:
            acc.append(np.abs(_lerp(x, y, ref_x[keep]) - ref_y[keep]).mean())
        spreads.append(float(np.mean(acc)))
    return {"n_sections": len(spreads), "mean_abs_log_spread": float(np.mean(spreads)) if spreads else None}


def case7_slices(model, device, target: str) -> dict:
    """Centered windows of the 4000 m Q–Vs section. The full section does not fit the canvas."""
    path = CASE7_FULL / "full_qvs.h5"
    if not path.is_file():
        return {"skipped": "full_qvs.h5 is not on this machine"}
    import sys

    import h5py
    import hdf5plugin  # noqa: F401

    case7a = Path(__file__).resolve().parents[1] / "case7a"
    if str(case7a) not in sys.path:
        sys.path.insert(0, str(case7a))
    from applied_damping import applied_zeta
    from dense_labels import _ttf
    from encoding import encode, query_columns
    from labels import seiskit_input_acceleration, tf_over_input
    from section_1d import RHO, rayleigh_xi_target, surface_over_outcrop

    out = {}
    with h5py.File(path, "r") as f:
        params = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["params"].attrs.items()}
        vs_full = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
        zeta_full = applied_zeta(f)
        surf = f["recorders/accel/surface"]
        x_nodes = np.asarray(surf.attrs["x_m"], dtype=np.float64)
        time = np.asarray(f["recorders/accel/time"][:], dtype=np.float64)
        grid = dict(f["grid"].attrs)
        acc_all = None  # loaded per window
        dt = float(time[1] - time[0])
        out["record_s"] = float(time[-1] - time[0])
        ricker = str(params.get("ricker_is", "acceleration"))
        scale = 1.982620120048523 if ricker == "velocity" else 1.9776999950408936
        f1 = float(params.get("damping_freq_first", params.get("f0_effective", 1.0)))
        f2 = float(params.get("damping_freq_second", 10.0))
        lx = int(round(float(grid["Lx"])))
        center = lx // 2
        for width in (500, 1000, 1500, 2000):
            lo = center - width // 2
            step = 20
            cols = np.arange(lo, lo + width, step)
            cols = cols[cols < vs_full.shape[1]]
            acc = np.asarray(surf[:, cols], dtype=np.float64)
            freq, tf_raw = tf_over_input(
                acc, seiskit_input_acceleration(float(grid["motion_freq"]), 0.5, time, ricker),
                dt, _ttf(), window_s=1310.72,
            )
            tf1d = np.empty_like(tf_raw)
            for j, col in enumerate(cols):
                xi = np.stack([rayleigh_xi_target(freq, z, f1, f2) for z in zeta_full[:, col]])
                tf1d[j] = surface_over_outcrop(freq, vs_full[:, col], dz=float(grid["dz"]), xi=xi, rho=RHO)
            vs = vs_full[:, lo : lo + width]
            zeta = zeta_full[:, lo : lo + width]
            x_local = (cols - lo).astype(np.float64)
            enc = torch.from_numpy(encode("metric", vs, zeta, vs.shape[0], f1_hz=f1, f2_hz=f2)).unsqueeze(0)
            query = torch.tensor(query_columns("metric", x_local, width), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                raw = model(enc.to(device), query.to(device)).float().cpu().numpy()[0]
            prior = np.log(np.maximum(tf1d, 1e-6))
            truth = np.log(np.maximum(tf_raw / scale, 1e-6))
            pred = prior + raw if target == "residual" else raw
            band = (freq >= 0.4) & (freq <= 2.5)
            err = np.abs(pred[:, band] - truth[:, band]).mean()
            err0 = np.abs(prior[:, band] - truth[:, band]).mean()
            out[f"w{width}"] = {"log_l1_mid": float(err), "g0_log_l1_mid": float(err0), "skill_mid": float(1.0 - err / max(err0, 1e-6))}
    out["note"] = "centered on the 4000 m section; the whole section is not one input"
    return out


def dense_metres(model, cache: Cache, device, target: str, label_dir: Path) -> dict:
    """Metres the 5 m loss never saw, on the 24 sections labeled at 1 m."""
    if not label_dir.is_dir():
        return {"skipped": f"{label_dir} is not written yet"}
    files = sorted(label_dir.glob("*.npz"))
    if not files:
        return {"n_sections": 0}
    by_name = {s.name: i for i, s in enumerate(cache.sections)}
    from encoding import encode, query_columns

    mid = (cache.freq >= 0.4) & (cache.freq <= 2.5)
    model_err, label_err = [], []
    for path in files:
        key = f"manifold__{path.stem}"
        if key not in by_name:
            continue
        sec = cache.sections[by_name[key]]
        lab = np.load(path)
        x = np.asarray(lab["x_m"], dtype=np.float64)
        truth = np.log(np.maximum(np.asarray(lab["tf"], dtype=np.float64), 1e-6))
        hold = ~np.isclose(np.mod(x, 5.0), 0.0, atol=0.2)
        src = ~hold
        if hold.sum() == 0 or src.sum() < 2:
            continue
        enc = torch.from_numpy(encode("metric", sec.vs, sec.zeta, sec.vs.shape[0], f1_hz=sec.f1_hz, f2_hz=sec.f2_hz)).unsqueeze(0)
        query = torch.tensor(query_columns("metric", x, sec.lx), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw = model(enc.to(device), query.to(device)).float().cpu().numpy()[0]
        # column prior lives on the 5 m stations; interpolate it onto the 1 m query
        prior_log = _lerp(sec.x_m.astype(np.float64), np.asarray(sec.log_tf1d, dtype=np.float64), x)
        pred = prior_log + raw if target == "residual" else raw
        from_labels = _lerp(x[src], truth[src], x[hold])
        model_err.append(float(np.abs(pred[hold][:, mid] - truth[hold][:, mid]).mean()))
        label_err.append(float(np.abs(from_labels[:, mid] - truth[hold][:, mid]).mean()))
    def mean(v):
        return float(np.mean(v)) if v else None
    return {"n_sections": len(model_err), "model_log_l1_mid": mean(model_err), "label_interp_log_l1_mid": mean(label_err)}


def round1_boundary(model, device, target: str) -> dict:
    """Unpadded 2000 m sections, stations at least 500 m from each absorbing end."""
    files = sorted(ROUND1_LAB.glob("sec_*.npz"))
    if not files or not ROUND1_H5.is_dir():
        return {"skipped": "round1_fixed labels or H5 are not on this machine"}
    import sys

    import h5py
    import hdf5plugin  # noqa: F401

    case7a = Path(__file__).resolve().parents[1] / "case7a"
    if str(case7a) not in sys.path:
        sys.path.insert(0, str(case7a))
    from applied_damping import applied_zeta
    from encoding import encode, query_columns

    err, err0 = [], []
    for path in files:
        h5 = ROUND1_H5 / f"{path.stem}.h5"
        if not h5.is_file():
            continue
        lab = np.load(path)
        x = np.asarray(lab["x_m"], dtype=np.float64)
        keep = (x >= 500.0) & (x <= x[-1] - 500.0)
        if not keep.any():
            continue
        with h5py.File(h5, "r") as f:
            vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
            zeta = applied_zeta(f)
            params = f["params"].attrs
            f1 = float(params.get("damping_freq_first", 1.0))
            f2 = float(params.get("damping_freq_second", 10.0))
        enc = torch.from_numpy(encode("metric", vs, zeta, vs.shape[0], f1_hz=f1, f2_hz=f2)).unsqueeze(0)
        query = torch.tensor(query_columns("metric", x[keep], vs.shape[1]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw = model(enc.to(device), query.to(device)).float().cpu().numpy()[0]
        truth = np.log(np.maximum(np.asarray(lab["tf"], dtype=np.float64)[keep], 1e-6))
        prior = np.log(np.maximum(np.asarray(lab["tf1d"], dtype=np.float64)[keep], 1e-6))
        pred = prior + raw if target == "residual" else raw
        freq = np.asarray(lab["freq"], dtype=np.float64)
        band = (freq >= 0.4) & (freq <= 2.5)
        err.append(float(np.abs(pred[:, band] - truth[:, band]).mean()))
        err0.append(float(np.abs(prior[:, band] - truth[:, band]).mean()))
    if not err:
        return {"n_sections": 0}
    e, e0 = float(np.mean(err)), float(np.mean(err0))
    return {"n_sections": len(err), "log_l1_mid": e, "g0_log_l1_mid": e0, "skill_mid": float(1.0 - e / max(e0, 1e-6))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--skip-case7", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.model_dir, device)
    model.eval()
    cache = Cache(args.cache)
    target = cfg.get("target", "residual")
    from evaluate import evaluate

    report = {
        "interpolation": interpolation(model, cache, device, target),
        "dense_metres": dense_metres(model, cache, device, target, args.cache.parent / "manifold_labels" / "stride1"),
        "width_consistency": width_consistency(model, cache, device, target),
        "blind": evaluate(model, cache, "metric", "blind", device, batch=2, workers=0, iid_limit=0, positions=1, target=target, families=()),
        "per_width": evaluate(model, cache, "metric", "val", device, batch=2, workers=0, iid_limit=8, positions=1, target=target).get("bins", {}),
    }
    if not args.skip_case7:
        report["case7"] = case7_slices(model, device, target)
        report["round1_boundary"] = round1_boundary(model, device, target)
    dest = args.model_dir / "holdout.json"
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_width"}, indent=2))


if __name__ == "__main__":
    main()
