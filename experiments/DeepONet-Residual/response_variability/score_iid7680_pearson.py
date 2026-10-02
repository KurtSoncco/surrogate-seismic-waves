#!/usr/bin/env python3
"""Score residual DeepONet ln|TF| Pearson on all 7680 IID runs.

Same center-recorder definition as Box ``hallal_vs_2d/pearson_center.csv``.
Reuses the n3000 signed cache and fills the other runs from H5 + nominal Haskell.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from data import ResidualDeepONetDataset, dataset_kwargs_from_blob  # noqa: E402
from eval_ood import _load_residual_model  # noqa: E402
from haskell_baseline import haskell_nominal_af_within  # noqa: E402
from model import apply_query_freq  # noqa: E402
from residual_signed import (  # noqa: E402
    load_manifest,
    resolve_h5_path,
    stack_field_columns,
)
from train import _device, _forward, apply_norms  # noqa: E402
from unified_metrics import tf_from_residual  # noqa: E402

import h5py  # noqa: E402

N = 7680
N_REC = config.N_LATERAL
N_FREQ = config.N_FREQ
CENTER = N_REC // 2
CACHE = config.CACHE_DIR / "iid7680_full"
SRC = config.CACHE_DIR / "n3000_seed42"
CKPT = _EXP / "checkpoints" / "M7680_gino_rebal_ft.pt"
OUT = config.RESULTS_DIR / "response_variability" / "gno_vs_classical"
BOX = Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data/hallal_vs_2d")


def _open(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        mm = np.load(path, mmap_mode="r+")
        if mm.shape != shape or mm.dtype != dtype:
            raise RuntimeError(f"{path} has {mm.shape} {mm.dtype}, expected {shape} {dtype}")
        return mm
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def build_cache() -> None:
    if (CACHE / "READY").is_file():
        print(f"[cache] reuse {CACHE}", flush=True)
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    src_idx = np.load(SRC / "sample_indices.npy")
    loc = {int(g): i for i, g in enumerate(src_idx)}
    src_fields = np.load(SRC / "fields.npy", mmap_mode="r")
    src_vs = np.load(SRC / "vs_col.npy", mmap_mode="r")
    src_tf = np.load(SRC / "tf1d_nom.npy", mmap_mode="r")
    src_meta = dict(np.load(SRC / "meta.npz", allow_pickle=True))

    fields = _open(CACHE / "fields.npy", (N, 3, config.NZ_MAX, N_REC), np.float32)
    vs_col = _open(CACHE / "vs_col.npy", (N, N_REC), np.float32)
    tf1d = _open(CACHE / "tf1d_nom.npy", (N, N_REC, N_FREQ), np.float32)
    _open(CACHE / "r_nom_signed.npy", (N, N_REC, N_FREQ), np.float32)

    meta = {k: np.empty(N, dtype=src_meta[k].dtype) for k in src_meta}
    filled = np.zeros(N, dtype=bool)
    for g, i in loc.items():
        fields[g] = src_fields[i]
        vs_col[g] = src_vs[i]
        tf1d[g] = src_tf[i]
        for k in meta:
            meta[k][g] = src_meta[k][i]
        filled[g] = True
    print(f"[cache] copied {int(filled.sum())} from n3000", flush=True)

    freq = np.load(config.TF_FREQ_PATH)
    rec = np.load(config.RECORDER_X_IDX_PATH)
    np.save(CACHE / "freq.npy", freq)
    np.save(CACHE / "recorder_x.npy", rec)
    np.save(CACHE / "sample_indices.npy", np.arange(N, dtype=np.int64))

    manifest = load_manifest()
    nom_cache: dict[tuple[float, float, float], np.ndarray] = {}
    missing = np.flatnonzero(~filled)
    for g in tqdm(missing, desc="fill missing IID"):
        g = int(g)
        row = manifest[g]
        h5_path = resolve_h5_path(row["h5_path"])
        with h5py.File(h5_path, "r") as f:
            params = {k: f["params"].attrs[k] for k in f["params"].attrs}
            vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
            zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
        vs = vs[:, config.X_SLICE_START : config.X_SLICE_END]
        zeta = zeta[:, config.X_SLICE_START : config.X_SLICE_END]
        vs1 = float(params["Vs1"])
        H = float(params.get("H_discretized", params.get("H")))
        vs2 = float(params["Vs2"])
        key = (round(vs1, 6), round(H, 6), round(vs2, 6))
        if key not in nom_cache:
            nom_cache[key] = haskell_nominal_af_within(
                freq,
                vs1=vs1,
                H=H,
                vs2=vs2,
                xi=float(config.DEFAULT_XI_TREND),
                rho=config.RHO,
            ).astype(np.float32)
        tf1d[g] = nom_cache[key][None, :]
        soil_nz = int(params.get("soil_layer_count", params.get("H_discretized", vs.shape[0])))
        fld, vc = stack_field_columns(vs, zeta, rec, soil_nz=soil_nz, nz=int(vs.shape[0]))
        fields[g] = fld
        vs_col[g] = vc
        meta["sample_idx"][g] = g
        meta["run_index"][g] = int(row.get("run_index", g))
        meta["h5_path"][g] = f"run_{g}.h5"
        meta["rf_seed"][g] = int(params["rf_seed"])
        meta["rH"][g] = float(params["rH"])
        meta["aHV"][g] = float(params["aHV"])
        meta["CoV"][g] = float(params["CoV"])
        meta["Vs1"][g] = vs1
        meta["Vs2"][g] = vs2
        meta["H"][g] = H
        meta["soil_nz"][g] = soil_nz
        meta["nz"][g] = int(vs.shape[0])
        meta["xi_damp"][g] = float(config.DEFAULT_XI_TREND)
        filled[g] = True
    fields.flush()
    vs_col.flush()
    tf1d.flush()
    np.savez(CACHE / "meta.npz", **meta)
    (CACHE / "READY").write_text("ok\n")
    print(f"[cache] ready {CACHE}  nominal keys={len(nom_cache)}", flush=True)


def predict_center(*, batch_size: int = 8, chunk: int = 64) -> np.ndarray:
    import torch

    out_path = OUT / "tf_deeponet_center.npy"
    prog_path = OUT / "tf_deeponet_center_done.npy"
    OUT.mkdir(parents=True, exist_ok=True)
    pred = _open(out_path, (N, N_FREQ), np.float32)
    done = int(np.load(prog_path)) if prog_path.is_file() else 0
    if done >= N:
        print(f"[predict] reuse {out_path}", flush=True)
        return pred

    device = _device()
    model, blob, stats, trunk_set = _load_residual_model(CKPT, device)
    serial = bool(blob.get("serial_tf1d", True))
    log_residual = bool(blob.get("log_residual", False))
    ds_kw = dataset_kwargs_from_blob(blob)
    mode = blob.get("branch_mode", "single")
    t_mean = stats["target_mean"].to(device)
    t_std = stats["target_std"].to(device)
    print(f"[predict] device={device} resume={done}", flush=True)

    with torch.no_grad():
        for start in range(done, N, chunk):
            stop = min(N, start + chunk)
            idx = np.arange(start, stop)
            ds = ResidualDeepONetDataset(
                CACHE,
                idx,
                target="R_nom",
                trunk_set=trunk_set,
                n_freq=config.N_FREQ_EVAL,
                serial_tf1d=serial,
                **ds_kw,
            )
            apply_norms(ds, stats)
            apply_query_freq(model, getattr(ds, "freq_s", None))
            hats = []
            for b0 in range(0, len(ds), batch_size):
                batch = ds[b0] if batch_size == 1 else _collate(ds, b0, min(len(ds), b0 + batch_size))
                pred_n = _forward(
                    model,
                    batch["fields"].to(device),
                    batch["stoch"].to(device),
                    batch["trunk_y"].to(device),
                    mode,
                    geom_flags=batch.get("geom_flags"),
                    rH=batch.get("rH"),
                )
                raw = (pred_n * t_std + t_mean).cpu().numpy()
                tf1d = batch["tf1d"].numpy()
                hats.append(tf_from_residual(tf1d, raw, log_residual=log_residual))
            stacked = np.concatenate(hats, axis=0).reshape(len(ds), ds.n_rec, -1)
            pred[start:stop] = stacked[:, CENTER, :].astype(np.float32)
            pred.flush()
            np.save(prog_path, np.array(stop, dtype=np.int64))
            print(f"[predict] {stop}/{N}", flush=True)
    return pred


def _collate(ds, start: int, stop: int) -> dict:
    import torch

    items = [ds[i] for i in range(start, stop)]
    keys = items[0].keys()
    return {k: torch.stack([it[k] for it in items], dim=0) for k in keys}


def score(pred: np.ndarray) -> None:
    import pandas as pd

    csv = pd.read_csv(BOX / "pearson_center.csv")
    with h5py.File(BOX / "tf_2d_center.h5") as f:
        ref = np.asarray(f["tf_center"][:], dtype=np.float64)
    if ref.shape != pred.shape:
        raise RuntimeError(f"shape {pred.shape} vs reference {ref.shape}")
    eps = 1e-12
    a = np.log(np.maximum(np.abs(np.asarray(pred, dtype=np.float64)), eps))
    b = np.log(np.maximum(np.abs(ref), eps))
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    r = np.full(N, np.nan)
    ok = den > eps
    r[ok] = (a[ok] * b[ok]).sum(axis=1) / den[ok]
    csv = csv.sort_values("index")
    csv["r_deeponet"] = r
    csv.to_csv(OUT / "pearson_center_iid7680.csv", index=False)

    def summ(x: np.ndarray) -> dict[str, float]:
        qs = np.percentile(x, [10, 25, 50, 75, 90])
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "p10": float(qs[0]),
            "p25": float(qs[1]),
            "p50": float(qs[2]),
            "p75": float(qs[3]),
            "p90": float(qs[4]),
        }

    summary = {
        "checkpoint": str(CKPT),
        "metric": "Pearson r of ln|TF| at center recorder vs OpenSees 2D center TF",
        "n": N,
        "DeepONet": summ(r),
        "Toro": summ(csv["r_toro"].to_numpy()),
        "Passeri": summ(csv["r_passeri"].to_numpy()),
        "Dmult": summ(csv["r_dmult"].to_numpy()),
        "paired": {
            "deeponet_gt_toro": float(np.mean(r > csv["r_toro"])),
            "deeponet_gt_passeri": float(np.mean(r > csv["r_passeri"])),
            "deeponet_gt_dmult": float(np.mean(r > csv["r_dmult"])),
            "deeponet_gt_best_classical": float(
                np.mean(r > np.maximum(csv["r_toro"], csv["r_passeri"]))
            ),
            "median_delta_toro": float(np.median(r - csv["r_toro"])),
            "median_delta_passeri": float(np.median(r - csv["r_passeri"])),
            "median_delta_dmult": float(np.median(r - csv["r_dmult"])),
        },
    }
    (OUT / "summary_iid7680.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    build_cache()
    pred = predict_center()
    score(pred)


if __name__ == "__main__":
    main()
