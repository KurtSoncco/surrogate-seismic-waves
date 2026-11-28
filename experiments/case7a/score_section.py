#!/usr/bin/env python3
"""Score absolute-input LOGLO models on the whole Case 7 section.

Reference: surface / (c * input) at every 15 m station (laterally consistent,
same label as the relabeled training sets). Each ``--model-dir`` holds
``best_model.pt``, ``pod_modes.npy``, ``pod_mean.npy`` and ``run_env.json``
(the ``GIFNO_*`` settings it was trained with). Several dirs form a deep
ensemble: log-mean prediction, log-std as epistemic spread.

Arms, all scored by zone and band:

* ``tf1d_prior``: vertical-SH 1D outcrop TF under each station (no learning)
* ``center``:     window centered on x, center column (shared head) or center recorder
* ``overlap``:    log-mean over every window whose field includes x
* ``all21``:      legacy 21-slot arm; shared head also writes ``dense`` at 5 m nodes

``correction_abs_log_trusted`` is the median |log TF - log TF_1D| per zone;
in the flat zones it should be near zero. The overlap spread (std of log TF across those windows) measures how far the
500 m window operator is from a consistent whole-section operator.

    python score_section.py --model-dir ../../checkpoints/loglo_abs_residual_s0 \
        --model-dir ../../checkpoints/loglo_abs_residual_s1 --out-dir results/section_abs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from case7a import (
    DZ_M,
    ELEM_M,
    N_FINE_Z,
    NODES_EACH_SIDE,
    REPO,
    STATION_SPACING_M,
    WINDOW_LIMIT_M,
    ZETA,
    ZONES,
    build_window_input_absolute,
    ensure_cache,
    paint_section_1m,
    window_center_x,
    window_node_index,
    zone_of,
)
from labels import CORE_BAND_HZ, TRUSTED_BAND_HZ, band_mask
from run_case7a import outcrop_reference_transfer_function
from section_1d import section_tf_1d

BANDS = {"full": (0.1, 10.0), "trusted": TRUSTED_BAND_HZ, "core": CORE_BAND_HZ}
LOGLO_DIR = REPO / "experiments" / "GIFNO-FDO-XT-LOGLO-POD"
GIFNO_DIR = REPO / "experiments" / "GIFNO"
OFFSETS = np.arange(-NODES_EACH_SIDE, NODES_EACH_SIDE + 1)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 0 else float("nan")


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))


def zone_scores(x: np.ndarray, pred: np.ndarray, true: np.ndarray, freq: np.ndarray) -> dict:
    zones = zone_of(x)
    out: dict = {}
    for zone in ZONES:
        rows = np.flatnonzero(zones == zone)
        out[zone] = {"n": int(rows.size)}
        for band, rng in BANDS.items():
            m = band_mask(freq, rng)
            p = [_pearson(pred[i, m], true[i, m]) for i in rows]
            r = [_rel_l2(pred[i, m], true[i, m]) for i in rows]
            out[zone][band] = {"pearson_median": float(np.median(p)), "rel_l2_median": float(np.median(r))}
    return out


def _load_env(model_dir: Path) -> dict[str, str]:
    path = model_dir / "run_env.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; write the GIFNO_* settings used for training there")
    return {k: str(v) for k, v in json.loads(path.read_text()).items()}


def load_models(model_dirs: list[Path]):
    """All ensemble members must share the input settings (one config import)."""
    envs = [_load_env(d) for d in model_dirs]
    keys = ("GIFNO_INPUT_ENCODING", "GIFNO_NZ_MAX", "GIFNO_PRIOR_CHANNELS", "GIFNO_DIMNORM_FILM", "GIFNO_POD_SPACE")
    for e in envs[1:]:
        if any(e.get(k) != envs[0].get(k) for k in keys):
            raise ValueError("ensemble members differ in input/POD settings")
    os.environ.update(envs[0])
    os.environ["GIFNO_MODEL_DIR"] = str(model_dirs[0])
    for p in (str(GIFNO_DIR), str(LOGLO_DIR)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    if "config" in sys.modules:
        raise RuntimeError("a 'config' module was imported before the LOGLO config")
    import torch

    import config
    from model import create_model

    if config.INPUT_ENCODING != "absolute":
        raise ValueError("score_section.py is for INPUT_ENCODING=absolute models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for d in model_dirs:
        m = create_model(pod_modes_path=d / "pod_modes.npy", pod_mean_path=d / "pod_mean.npy").to(device)
        m.load_state_dict(torch.load(d / "best_model.pt", map_location=device, weights_only=True), strict=True)
        m.eval()
        models.append(m)
    target = envs[0].get("TF_TARGET", "residual")
    return models, device, config, target


def dense_center_x() -> np.ndarray:
    """5 m SVLAB nodes whose 500 m window lies inside the extended mesh."""
    k_lo = int(np.ceil(-WINDOW_LIMIT_M / ELEM_M))
    k_hi = int(np.floor(WINDOW_LIMIT_M / ELEM_M))
    return ELEM_M * np.arange(k_lo, k_hi + 1, dtype=np.float64)


def predict_windows(
    models,
    device,
    config,
    vs_1m: np.ndarray,
    centers: np.ndarray,
    batch: int = 8,
    zeta: float | np.ndarray | None = None,
    prior_mode: str = "constant",
) -> np.ndarray:
    """log |TF|. Shared head: (M, n_win, nx, F). Legacy: (M, n_win, 21, F)."""
    import torch

    rec = config.recorder_x_indices()
    shared = bool(getattr(models[0].head, "shared", False))
    n_prior = int(config.PRIOR_CHANNELS)
    prior_log = None
    if n_prior > 0:
        gifno = str(REPO / "experiments" / "GIFNO")
        if gifno not in sys.path:
            sys.path.append(gifno)
        from absolute_inputs import column_tf1d_outcrop, column_tf1d_rayleigh, prior_frequencies

        if zeta is None or np.isscalar(zeta):
            zeta_full = np.full(vs_1m.shape, ZETA if zeta is None else float(zeta), dtype=np.float64)
        else:
            zeta_full = np.asarray(zeta, dtype=np.float64)
        propagator = column_tf1d_rayleigh if prior_mode == "rayleigh" else column_tf1d_outcrop
        tf_prior = propagator(
            vs_1m[:N_FINE_Z], zeta_full[:N_FINE_Z], DZ_M, prior_frequencies(n_prior)
        )
        prior_log = np.log(np.maximum(tf_prior, 1e-12)).astype(np.float32)
    out = []
    for m in models:
        rows = []
        for i in range(0, len(centers), batch):
            xb = np.stack(
                [
                    build_window_input_absolute(
                        vs_1m,
                        float(c),
                        nz_max=config.NZ_MAX,
                        n_prior=n_prior,
                        **({} if zeta is None else {"zeta": zeta}),
                        prior_mode=prior_mode,
                        prior_log=prior_log,
                    )
                    for c in centers[i : i + batch]
                ]
            )
            with torch.no_grad():
                grid = m(torch.from_numpy(xb).to(device)).cpu().numpy()
            take = grid if shared else grid[:, rec, :]
            rows.append(np.log(np.maximum(take, 1e-6)))
        out.append(np.concatenate(rows))
    return np.stack(out)


def _overlap_from_dense(log_mean: np.ndarray, centers: np.ndarray, trusted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log-mean TF at each window center from every overlapping dense field."""
    n_win, nx, n_freq = log_mean.shape
    origin = 0.5 * (nx - 1)
    overlap = np.full((n_win, n_freq), np.nan)
    spread = np.full(n_win, np.nan)
    for i, x in enumerate(centers):
        stack = []
        for j, c in enumerate(centers):
            col = int(round(origin + (float(x) - float(c))))
            if 0 <= col < nx:
                stack.append(log_mean[j, col])
        if not stack:
            continue
        stack = np.stack(stack)
        overlap[i] = stack.mean(axis=0)
        spread[i] = float(stack[:, trusted].std(axis=0).mean())
    return np.exp(overlap), spread


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "section_abs")
    args = ap.parse_args()

    pack = ensure_cache()
    if args.model_dir:
        models, device, config, target = load_models([d.resolve() for d in args.model_dir])
        shared = bool(getattr(models[0].head, "shared", False))
    else:
        models, device, config, target, shared = [], None, None, None, False

    centers = dense_center_x() if shared else window_center_x()
    nodes = np.array([window_node_index(float(x)) for x in centers], dtype=np.int64)
    freq, tf_true, scale = outcrop_reference_transfer_function(pack, nodes)
    tf1d = section_tf_1d(freq, pack["vs"], centers, "outcrop")

    results = {"tf1d_prior": zone_scores(centers, tf1d, tf_true, freq)}
    arrays = {"x": centers, "freq": freq, "tf_true": tf_true, "tf1d_prior": tf1d}

    if args.model_dir:
        log_w = predict_windows(models, device, config, paint_section_1m(pack["vs"]), centers)
        m_tr = band_mask(freq, TRUSTED_BAND_HZ)
        if shared:
            nx = log_w.shape[2]
            center_i = nx // 2
            x_grid = centers[:, None] + (np.arange(nx) - center_i)
            tf1d_grid = section_tf_1d(freq, pack["vs"], x_grid.ravel(), "outcrop").reshape(len(centers), nx, -1)
            if target == "residual":
                log_w = log_w + np.log(np.maximum(tf1d_grid, 1e-6))[None]
            log_mean = log_w.mean(axis=0)
            tf_center = np.exp(log_mean[:, center_i])
            ens_std = log_w[:, :, center_i].std(axis=0) if len(models) > 1 else np.zeros_like(tf_center)
            overlap, spread = _overlap_from_dense(log_mean, centers, m_tr)
            results["center"] = zone_scores(centers, tf_center, tf_true, freq)
            results["overlap"] = zone_scores(centers, overlap, tf_true, freq)
            results["dense"] = results["center"]
        else:
            rec_x = centers[:, None] + OFFSETS[None, :] * STATION_SPACING_M
            uniq, inv = np.unique(np.round(rec_x, 6), return_inverse=True)
            inv = inv.reshape(rec_x.shape)
            tf1d_u = section_tf_1d(freq, pack["vs"], uniq, "outcrop")
            if target == "residual":
                log_w = log_w + np.log(np.maximum(tf1d_u[inv], 1e-6))[None]
            log_mean = log_w.mean(axis=0)
            center_i = NODES_EACH_SIDE
            tf_center = np.exp(log_mean[:, center_i])
            ens_std = log_w[:, :, center_i].std(axis=0) if len(models) > 1 else np.zeros_like(tf_center)
            step = {round(float(c), 6): i for i, c in enumerate(centers)}
            overlap = np.full_like(tf_center, np.nan)
            spread = np.full(len(centers), np.nan)
            for i, x in enumerate(centers):
                stack = []
                for k in OFFSETS:
                    j = step.get(round(float(x - k * STATION_SPACING_M), 6))
                    if j is not None:
                        stack.append(log_mean[j, center_i + k])
                stack = np.stack(stack)
                overlap[i] = np.exp(stack.mean(axis=0))
                spread[i] = float(stack[:, m_tr].std(axis=0).mean())
            results["center"] = zone_scores(centers, tf_center, tf_true, freq)
            results["overlap"] = zone_scores(centers, overlap, tf_true, freq)
            uniq_nodes = np.array([window_node_index(float(x)) for x in uniq], dtype=np.int64)
            _, tf_true_u, _ = outcrop_reference_transfer_function(pack, uniq_nodes)
            results["all21"] = zone_scores(
                rec_x.ravel(), np.exp(log_mean).reshape(-1, freq.size), tf_true_u[inv.ravel()], freq
            )

        zones = zone_of(centers)
        correction = np.abs(np.log(np.maximum(tf_center, 1e-6)) - np.log(np.maximum(tf1d, 1e-6)))[:, m_tr].mean(axis=1)
        results["correction_abs_log_trusted"] = {z: float(np.median(correction[zones == z])) for z in ZONES}
        results["window_spread_log_trusted"] = {z: float(np.nanmedian(spread[zones == z])) for z in ZONES}
        results["ensemble_std_log_trusted"] = {z: float(np.median(ens_std[zones == z][:, m_tr])) for z in ZONES}
        results["models"] = [str(d) for d in args.model_dir]
        results["target"] = target
        results["collocation"] = "dense_5m" if shared else "stations_21"
        arrays.update(tf_center=tf_center, tf_overlap=overlap, window_spread=spread, ensemble_std=ens_std)

    results["input_scale"] = scale
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(args.out_dir / f"{k}.npy", np.asarray(v, dtype=np.float32))
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    for arm in ("tf1d_prior", "center", "overlap", "all21", "dense"):
        if arm not in results:
            continue
        print(arm)
        for z, s in results[arm].items():
            print(f"  {z:10s} trusted P {s['trusted']['pearson_median']:.3f} L2 {s['trusted']['rel_l2_median']:.3f}"
                  f" | core P {s['core']['pearson_median']:.3f} L2 {s['core']['rel_l2_median']:.3f}")


if __name__ == "__main__":
    main()
