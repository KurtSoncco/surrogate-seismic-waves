#!/usr/bin/env python3
"""Autoencoders against the frozen bases, including a 0.01–50 Hz query.

Labels exist only on the cache grid, 0.1–10 Hz. Mean absolute error is reported
there. The coordinate decoder is also queried from 0.01 to 50 Hz. Outside the
label grid those values are predictions, not errors. The window autoencoder has
no frequency coordinate, so it is scored only on 0.1–10 Hz.

Column autoencoders are the saved code-2 and code-32 checkpoints. Each code is
read out with the same ridge as the masked-PCA probe.

    python span_compare.py --cache ... --weights .../ae_pretrain --out .../span
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import Cache
from depth_bound import CHUNK, _codes, _receiver_batch, _runs, accumulate
from depth_probe import (
    PCA_RANK,
    RIDGE,
    NormalEquations,
    curve_scores,
    pod_coefficients,
    reconstruct,
)
from model import ColumnDepthAutoencoder, CurveEncoder, SparseWindowAutoencoder, SpectralDecoder, spectral_weight
from pod_rank import frequency_weights, weighted_band_errors
from pretrain_ae import DEPTH_REF_M, VS_REF, ZETA_REF, _refuse_shipped, select_curves, split_rows
from screen_hifreq import BANDS, WINDOW, reconstruct_curves
from train import _curves

SPAN_MIN_HZ = 0.01
SPAN_MAX_HZ = 50.0
SPAN_N = 256
DEPTH_BANDS = ((0, 10), (10, 30), (30, 60), (60, 110), (100, 110))
FOCUS = (64, 128)


def _band_row(freq: np.ndarray, per_bin: np.ndarray) -> dict[str, float]:
    out = {"mae": float(per_bin.mean())}
    for lo, hi in BANDS:
        chosen = (freq >= lo) & (freq <= hi)
        out[f"{lo:g}-{hi:g}"] = float(per_bin[chosen].mean())
    return out


def _log_interp(curves: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Linear interpolation in log frequency. The caller must mask off-grid bins."""
    x = np.log(np.asarray(src, dtype=np.float64))
    y = np.log(np.asarray(dst, dtype=np.float64))
    index = np.clip(np.searchsorted(x, y, side="right") - 1, 0, x.size - 2)
    step = (y - x[index]) / (x[index + 1] - x[index])
    return curves[:, index] * (1.0 - step) + curves[:, index + 1] * step


def _load_coordinate(path: Path, n_freq: int) -> tuple[CurveEncoder, SpectralDecoder, int]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    rank = int(blob["k"])
    encoder = CurveEncoder(n_freq, rank, 256)
    decoder = SpectralDecoder(rank, 256)
    encoder.load_state_dict(blob["encoder"])
    decoder.load_state_dict(blob["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, rank


def _decode(encoder: CurveEncoder, decoder: SpectralDecoder, curves: np.ndarray, freq: np.ndarray, batch: int) -> np.ndarray:
    freq_t = torch.as_tensor(freq, dtype=torch.float32)
    pieces = []
    with torch.no_grad():
        for start in range(0, curves.shape[0], batch):
            chunk = torch.as_tensor(curves[start : start + batch], dtype=torch.float32)
            pred = decoder(encoder(chunk)[:, None, :], freq_t).squeeze(1)
            pieces.append(pred.numpy())
    return np.concatenate(pieces, axis=0)


def _span_record(pred: np.ndarray, truth: np.ndarray, wide: np.ndarray, labeled: np.ndarray) -> dict:
    """Error on label frequencies. Off-grid bins keep the prediction only."""
    mae = np.full(wide.shape[0], np.nan)
    if labeled.any():
        mae[labeled] = np.abs(pred[:, labeled] - truth[:, labeled]).mean(axis=0)
    wings = {}
    for lo, hi in ((SPAN_MIN_HZ, 0.1), (10.0, SPAN_MAX_HZ)):
        chosen = (wide >= lo) & (wide <= hi) & ~labeled
        sample = pred[:, chosen]
        wings[f"{lo:g}-{hi:g}"] = {
            "n_freq": int(chosen.sum()),
            "pred_mean": float(sample.mean()) if sample.size else None,
            "pred_abs_mean": float(np.abs(sample).mean()) if sample.size else None,
            "pred_std": float(sample.std()) if sample.size else None,
            "error": None,
        }
    inside = (wide >= 0.1) & (wide <= 10.0) & labeled
    return {
        "per_freq_mae": [None if not np.isfinite(value) else float(value) for value in mae],
        "in_band": _band_row(wide[inside], mae[inside]) if inside.any() else {},
        "wings": wings,
    }


def spectral_comparison(cache: Cache, weights: Path, batch: int) -> dict:
    freq = np.asarray(cache.freq, dtype=np.float64)
    curves, _groups = _curves(cache, "residual")
    curves = select_curves(curves, 0, 0)
    train_idx, hold_idx = split_rows(curves.shape[0], 0)
    train, hold = curves[train_idx], curves[hold_idx]
    print(f"spectral train {train.shape[0]} hold {hold.shape[0]}", flush=True)
    weight = frequency_weights(freq, train)["sqrt_f"]
    native = {
        "uniform": weighted_band_errors(train, hold, freq, np.ones_like(freq), FOCUS),
        "sqrt_f": weighted_band_errors(train, hold, freq, weight, FOCUS),
    }
    wide = np.geomspace(SPAN_MIN_HZ, SPAN_MAX_HZ, SPAN_N)
    labeled = (wide >= freq[0]) & (wide <= freq[-1])
    truth = _log_interp(hold, freq, wide)
    decoded = {}
    for name, path in (("coordinate_k32", weights / "full/spectral_k32.pt"), ("coordinate_k64", weights / "full/spectral_k64.pt")):
        encoder, decoder, rank = _load_coordinate(path, freq.size)
        on_grid = _decode(encoder, decoder, hold, freq, batch)
        on_wide = _decode(encoder, decoder, hold, wide, batch)
        decoded[name] = {
            "rank": rank,
            "native": _band_row(freq, np.abs(on_grid - hold).mean(axis=0)),
            "span_hz": _span_record(on_wide, truth, wide, labeled),
        }
        row = decoded[name]["native"]
        print(
            f"{name} native mae {row['mae']:.4f} 0.1-0.4 {row['0.1-0.4']:.4f} 5-10 {row['5-10']:.4f}",
            flush=True,
        )
    window = SparseWindowAutoencoder(WINDOW, 3, knn=5, block=4, stride=1)
    window.load_state_dict(torch.load(weights / "hifreq/sparse_window_ae.pt", map_location="cpu", weights_only=True))
    window.eval()
    recon = reconstruct_curves(window, hold, torch.device("cpu"), batch)
    decoded["window"] = {"native": _band_row(freq, np.abs(recon - hold).mean(axis=0)), "span_hz": None}
    row = decoded["window"]["native"]
    print(f"window native mae {row['mae']:.4f} 5-10 {row['5-10']:.4f}", flush=True)
    defined = spectral_weight(freq)
    return {
        "target": "log|TF| - log|TF_1D|",
        "split": "select_curves seed 0, split seed 0, the holdout the autoencoders were trained against",
        "label_hz": [float(freq[0]), float(freq[-1])],
        "query_hz": [SPAN_MIN_HZ, SPAN_MAX_HZ, SPAN_N],
        "query_note": "Mean absolute error is only computed where the query sits inside the label grid. 0.01–0.1 Hz and 10–50 Hz are predictions.",
        "sqrt_weight_matches_training_definition": bool(np.allclose(weight, defined)),
        "pod_native": native,
        "autoencoder": decoded,
        "wide_hz": wide.tolist(),
    }


def _load_column(path: Path) -> ColumnDepthAutoencoder:
    state = torch.load(path, map_location="cpu", weights_only=True)
    code = int(state["lift.weight"].shape[0]) if "lift.weight" in state else 2
    model = ColumnDepthAutoencoder(2, code_dim=code)
    model.load_state_dict(state)
    model.eval()
    return model


def _pack(vs: np.ndarray, zeta: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count, height, width = vs.shape
    flat = count * width
    values = np.empty((1, 2, height, flat), dtype=np.float32)
    # (N, H, W) -> (H, N*W), column n*W+w matching the receiver-curve order.
    values[0, 0] = np.log(np.maximum(vs, 1e-6) / VS_REF).astype(np.float32).transpose(1, 0, 2).reshape(height, flat)
    values[0, 1] = np.log(np.maximum(zeta, 1e-8) / ZETA_REF).astype(np.float32).transpose(1, 0, 2).reshape(height, flat)
    mask = np.ones((1, height, flat), dtype=np.float32)
    depth = np.empty((1, 1, height, flat), dtype=np.float32)
    depth[0, 0] = (np.arange(height, dtype=np.float32) / DEPTH_REF_M)[:, None]
    return torch.from_numpy(values), torch.from_numpy(mask), torch.from_numpy(depth)


@torch.no_grad()
def _column_forward(model: ColumnDepthAutoencoder, vs: np.ndarray, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, mask, depth = _pack(vs, zeta)
    surface, recon = model(values, mask, depth)
    code = surface[0, :, 0, :].T.numpy().astype(np.float64)
    err = (recon - values).abs().sum(dim=(0, 1, 3)).numpy()
    count = mask.sum(dim=(0, 2)).numpy() * values.shape[1]
    return code, err, count


def column_comparison(cache: Cache, weights: Path) -> dict:
    pca64, pod, n_train = accumulate(cache)
    taken = min(PCA_RANK, pca64["rank"])
    pca16 = {"mean": pca64["mean"], "components": pca64["components"][:taken], "rank": taken}
    models = {
        "column_code2": _load_column(weights / "full/depth_ae.pt"),
        "column_code32": _load_column(weights / "depth_code32/depth_ae.pt"),
    }
    print(" ".join(f"{name} code {model.code_dim}" for name, model in models.items()), flush=True)
    targets = int(pod["modes"].shape[0])
    eqs = {
        "pca16": NormalEquations(PCA_RANK, targets),
        "pca64": NormalEquations(int(pca64["components"].shape[0]), targets),
        "combined": NormalEquations(PCA_RANK + 4, targets),
    }
    eqs.update({name: NormalEquations(model.code_dim, targets) for name, model in models.items()})
    index, camp = _runs(cache, "train")
    cols = np.asarray(camp.rec_x, dtype=np.int64)
    freq_n = int(np.asarray(cache.freq).size)
    heights = camp.n_active[index]
    for soil in np.unique(heights):
        chosen = index[heights == soil]
        for start in range(0, chosen.size, CHUNK):
            rows = chosen[start : start + CHUNK]
            vs = np.asarray(camp.vs[rows, : int(soil)][:, :, cols], dtype=np.float64)
            zeta = np.asarray(camp.zeta[rows, : int(soil)][:, :, cols], dtype=np.float64)
            paired, physical = _receiver_batch(camp, rows, int(soil), cols)
            curves = np.asarray(camp.log_tf1d[rows], dtype=np.float64).reshape(-1, freq_n)
            coeff = pod_coefficients(curves, pod)
            codes = _codes(paired, physical, pca16, pca64)
            for name in ("pca16", "pca64", "combined"):
                eqs[name].add(codes[name], coeff)
            for name, model in models.items():
                code, _err, _count = _column_forward(model, vs, zeta)
                eqs[name].add(code, coeff)
    solutions = {name: eq.solve(RIDGE) for name, eq in eqs.items()}
    freq = np.asarray(cache.freq, dtype=np.float64)
    val_index, val_camp = _runs(cache, "val")
    val_cols = np.asarray(val_camp.rec_x, dtype=np.int64)
    buckets: dict[str, list[np.ndarray]] = {name: [] for name in ("mean", *solutions)}
    targets_v: list[np.ndarray] = []
    profile = {name: {"sum": np.zeros(int(val_camp.n_active.max())), "count": np.zeros(int(val_camp.n_active.max()))} for name in models}
    heights = val_camp.n_active[val_index]
    for soil in np.unique(heights):
        chosen = val_index[heights == soil]
        for start in range(0, chosen.size, CHUNK):
            rows = chosen[start : start + CHUNK]
            vs = np.asarray(val_camp.vs[rows, : int(soil)][:, :, val_cols], dtype=np.float64)
            zeta = np.asarray(val_camp.zeta[rows, : int(soil)][:, :, val_cols], dtype=np.float64)
            paired, physical = _receiver_batch(val_camp, rows, int(soil), val_cols)
            curves = np.asarray(val_camp.log_tf1d[rows], dtype=np.float64).reshape(-1, freq.size)
            codes = _codes(paired, physical, pca16, pca64)
            buckets["mean"].append(np.repeat(pod["mean"][None, :], curves.shape[0], axis=0))
            for name in ("pca16", "pca64", "combined"):
                buckets[name].append(reconstruct(eqs[name].predict(codes[name], solutions[name]), pod))
            for name, model in models.items():
                code, err, count = _column_forward(model, vs, zeta)
                buckets[name].append(reconstruct(eqs[name].predict(code, solutions[name]), pod))
                profile[name]["sum"][: err.size] += err
                profile[name]["count"][: count.size] += count
            targets_v.append(curves)
    target = np.concatenate(targets_v, axis=0)
    scored = {name: curve_scores(np.concatenate(rows, axis=0), target, freq) for name, rows in buckets.items()}
    for name, accum in profile.items():
        bands = {}
        for lo, hi in DEPTH_BANDS:
            sl = slice(lo, min(hi, accum["sum"].size))
            denom = float(accum["count"][sl].sum())
            bands[f"{lo}-{hi}"] = float(accum["sum"][sl].sum() / denom) if denom else None
        total = float(accum["count"].sum())
        scored[name]["profile_masked_l1"] = float(accum["sum"].sum() / total) if total else None
        scored[name]["profile_bands_m"] = bands
        row = scored[name]
        print(
            f"{name} mae {row['mae']:.4f} profile {row['profile_masked_l1']:.4f} "
            f"0-10 {bands['0-10']:.4f} 60-110 {bands['60-110']:.4f}",
            flush=True,
        )
    for name in ("mean", "pca16", "pca64", "combined"):
        print(f"{name} mae {scored[name]['mae']:.4f}", flush=True)
    return {
        "target": "log|TF_1D| at each validation receiver",
        "label_hz": [float(freq[0]), float(freq[-1])],
        "note": "The column code is scored by the same ridge readout as masked PCA. Profile L1 is the autoencoder's own reconstruction of log Vs and log zeta, which has no frequency.",
        "n_train_curves": n_train,
        "pod_variance_explained": pod["explained"],
        "scores": scored,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()
    _refuse_shipped(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    cache = Cache(args.cache)
    report = {
        "spectral": spectral_comparison(cache, args.weights, args.batch),
        "column": column_comparison(cache, args.weights),
    }
    (args.out / "report.json").write_text(json.dumps(report))
    print(f"wrote {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
