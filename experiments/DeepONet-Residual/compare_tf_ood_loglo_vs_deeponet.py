#!/usr/bin/env python3
"""OOD TF head-to-head: LOGLO-POD vs ResUNet DeepONet (TF1D_nom + R_hat).

Uses cached OpenSees GT from ood_scores_full7680 when available.
Campaigns: dipping (960) + three_layer (960).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

import config as dn_config
from compare_tf_loglo_vs_deeponet import (
    _pearson_across_freq,
    _r2,
    _rel_l2,
    _summarize,
    fit_norms_from_train,
    load_deeponet,
    load_loglo,
)
from data import build_recorder_fields

_EPS = 1e-12


def _normalize_ood_params(params: dict[str, Any]) -> dict[str, float | int]:
    """Map dipping vs three_layer H5 attrs to DeepONet branch scalars."""
    if "rf_seed" in params:
        rf_seed = int(params["rf_seed"])
        rH = float(params["rH"])
        aHV = float(params["aHV"])
        CoV = float(params["CoV"])
        H = float(params.get("H_discretized", params.get("H", 0)))
        vs2 = float(params["Vs2"])
        soil_nz = int(
            params.get(
                "soil_layer_count",
                params.get("H_discretized", params.get("H", 0)),
            )
        )
    else:
        # three_layer: two GRF seeds — use layer-1 seed for xi replay (mix data convention)
        rf_seed = int(params.get("seed1", params.get("seed2", 0)))
        rH = float(params.get("rH1", params.get("rH2", 0)))
        aHV = float(params.get("aHV1", params.get("aHV2", 0)))
        c1 = float(params.get("CoV1", 0))
        c2 = float(params.get("CoV2", 0))
        CoV = float((c1 + c2) / 2.0)
        h1 = float(params.get("H1", params.get("H1_discretized", 0)))
        h2 = float(params.get("H2", params.get("H2_discretized", 0)))
        H = h1 + h2
        vs2 = float(params.get("Vs_bedrock", params.get("Vs2", 0)))
        l1 = int(params.get("layer1_count", params.get("H1_discretized", h1)))
        l2 = int(params.get("layer2_count", params.get("H2_discretized", h2)))
        soil_nz = l1 + l2
    return {
        "rf_seed": rf_seed,
        "rH": rH,
        "aHV": aHV,
        "CoV": CoV,
        "H": H,
        "Vs1": float(params.get("Vs1", 0)),
        "Vs2": vs2,
        "soil_nz": soil_nz,
    }


def _read_h5_fields(h5_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with h5py.File(h5_path, "r") as f:
        vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
        zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
        params = {k: f["params"].attrs[k] for k in f["params"].attrs}
    return vs, zeta, params


def _ood_roots() -> tuple[Path, Path]:
    ood_root = Path(
        os.environ.get(
            "OOD_H5_ROOT",
            str(Path.home() / "surrogate-seismic-waves/data/ood_h5"),
        )
    )
    gt_root = Path(
        os.environ.get(
            "OOD_GT_ROOT",
            str(
                Path.home()
                / "surrogate-seismic-waves/checkpoints/ood_scores_full7680"
            ),
        )
    )
    return ood_root, gt_root


def _haskell_nom_tf1d(
    h5_path: Path, freq: np.ndarray, recorder_x: np.ndarray
) -> np.ndarray:
    from haskell_baseline import haskell_at_columns, haskell_nominal_af_within

    vs, zeta, params = _read_h5_fields(h5_path)
    norm = _normalize_ood_params(params)
    vs_crop = vs[:, dn_config.X_SLICE_START : dn_config.X_SLICE_END]
    zeta_crop = zeta[:, dn_config.X_SLICE_START : dn_config.X_SLICE_END]
    vs2 = float(norm["Vs2"])
    soil_nz = int(norm["soil_nz"])

    tf1d_col = haskell_at_columns(
        freq,
        vs_crop,
        zeta_crop,
        recorder_x,
        dz=dn_config.DZ,
        vs_rock=vs2,
        soil_nz=soil_nz,
        rho=dn_config.RHO,
    ).astype(np.float32)

    vs1 = float(norm["Vs1"])
    H = float(norm["H"])
    xi = float(dn_config.DEFAULT_XI_TREND)
    tf1d_nom_1d = haskell_nominal_af_within(
        freq, vs1=vs1, H=H, vs2=vs2, xi=xi, rho=dn_config.RHO
    ).astype(np.float32)
    return np.broadcast_to(tf1d_nom_1d[None, :], tf1d_col.shape).copy()


@torch.no_grad()
def predict_deeponet_ood(
    model: torch.nn.Module,
    h5_path: Path,
    freq: np.ndarray,
    recorder_x: np.ndarray,
    stats: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (TF_hat, TF1D_nom) shape (n_rec, n_freq)."""
    from features import fourier_freq_features, spectral_kl_coefficients

    vs, zeta, params = _read_h5_fields(h5_path)
    norm = _normalize_ood_params(params)
    vs = vs[:, dn_config.X_SLICE_START : dn_config.X_SLICE_END]
    zeta = zeta[:, dn_config.X_SLICE_START : dn_config.X_SLICE_END]
    nz = int(vs.shape[0])
    H = float(norm["H"])
    soil_nz = int(norm["soil_nz"])
    rf_seed = int(norm["rf_seed"])
    rH = float(norm["rH"])
    aHV = float(norm["aHV"])
    CoV = float(norm["CoV"])
    xi_damp = float(dn_config.DEFAULT_XI_TREND)

    fields = build_recorder_fields(vs, zeta, nz=nz, recorder_x=recorder_x)
    xi_vals, _ = spectral_kl_coefficients(
        rf_seed=rf_seed,
        rH=rH,
        aHV=aHV,
        nx=dn_config.NX,
        nz=nz,
        dx=dn_config.DX,
        dz=dn_config.DZ,
        k=dn_config.K_XI,
    )
    stoch = np.concatenate(
        [xi_vals, np.array([rH, aHV, CoV, xi_damp], dtype=np.float32)]
    ).astype(np.float32)

    tf1d_nom = _haskell_nom_tf1d(h5_path, freq, recorder_x)

    n_rec = len(recorder_x)
    n = max(1, min(soil_nz, vs.shape[0]))
    cols = recorder_x.astype(int)
    vs_col = vs[:n, cols].mean(axis=0)
    x_m = (cols.astype(np.float64) + 0.5) * dn_config.DX
    sin_f, cos_f = fourier_freq_features(
        freq, f_min=dn_config.FREQ_START_HZ, f_max=dn_config.FREQ_END_HZ
    )

    rows = []
    for ri in range(n_rec):
        vs_c = float(max(vs_col[ri], _EPS))
        for j, f in enumerate(freq):
            f = float(f)
            lam = vs_c / max(f, _EPS)
            rows.append(
                np.array(
                    [
                        f * H / vs_c,
                        float(sin_f[j]),
                        float(cos_f[j]),
                        float(x_m[ri] / max(lam, _EPS)),
                    ],
                    dtype=np.float32,
                )
            )
    trunk_y = np.stack(rows, axis=0)

    stoch_t = (torch.from_numpy(stoch) - stats["stoch_mean"]) / stats["stoch_std"]
    trunk_t = (torch.from_numpy(trunk_y) - stats["trunk_mean"]) / stats["trunk_std"]
    fields_t = torch.from_numpy(fields.copy())

    pred_n = model(
        fields_t.unsqueeze(0).to(device),
        stoch_t.unsqueeze(0).to(device),
        trunk_t.unsqueeze(0).to(device),
    )[0].cpu()
    pred = pred_n * stats["target_std"] + stats["target_mean"]
    R = pred.numpy().reshape(n_rec, len(freq))
    return (tf1d_nom + R).astype(np.float32), tf1d_nom


def _campaign_jobs(
    campaign: str, ood_root: Path, gt_root: Path, limit: int | None
) -> list[tuple[str, Path, Path]]:
    name = f"ood_{campaign}"
    h5_dir = ood_root / name / "h5"
    manifest_path = ood_root / name / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    jobs: list[tuple[str, Path, Path]] = []
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row.get("index", row.get("run_index", -1)))
            if idx < 0:
                continue
            h5_path = h5_dir / f"run_{idx}.h5"
            case_dir = gt_root / campaign / f"run_{idx}"
            if h5_path.is_file() and (case_dir / "tf_true.npy").is_file():
                jobs.append((campaign, h5_path, case_dir))
    if limit is not None:
        jobs = jobs[:limit]
    return jobs


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "loglo": {
            "rel_l2": _summarize([r["loglo_rel_l2"] for r in rows]),
            "pearson_f": _summarize([r["loglo_pearson_f"] for r in rows]),
            "r2": _summarize([r["loglo_r2"] for r in rows]),
        },
        "deeponet": {
            "rel_l2": _summarize([r["deeponet_rel_l2"] for r in rows]),
            "pearson_f": _summarize([r["deeponet_pearson_f"] for r in rows]),
            "r2": _summarize([r["deeponet_r2"] for r in rows]),
        },
        "tf1d_nom_only": {
            "rel_l2": _summarize([r["tf1d_rel_l2"] for r in rows]),
            "pearson_f": _summarize([r["tf1d_pearson_f"] for r in rows]),
            "r2": _summarize([r["tf1d_r2"] for r in rows]),
        },
        "win_rate_rel_l2": {
            "loglo_better": float(
                np.mean([r["loglo_rel_l2"] < r["deeponet_rel_l2"] for r in rows])
            ),
            "deeponet_better": float(
                np.mean([r["deeponet_rel_l2"] < r["loglo_rel_l2"] for r in rows])
            ),
        },
    }


def main() -> None:
    campaigns = os.environ.get("OOD_CAMPAIGNS", "dipping,three_layer").split(",")
    limit = int(os.environ.get("MAX_SAMPLES", "0")) or None
    cache_tag = os.environ.get("CACHE_TAG", "full7680_seed42")
    cache_dir = dn_config.CACHE_DIR / cache_tag
    seed = int(os.environ.get("SEED", dn_config.SEED))
    out_dir = Path(
        os.environ.get(
            "COMPARE_OUT",
            str(dn_config.RESULTS_DIR / "compare_gino_loglo"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ood_root, gt_root = _ood_roots()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ood-compare] device={device} ood={ood_root} gt={gt_root}", flush=True)

    stats = fit_norms_from_train(cache_dir, seed)
    dn_ckpt = Path(
        os.environ.get(
            "DEEPONET_CKPT",
            str(
                dn_config.CHECKPOINT_DIR
                / "M7680_gino_rebal_ft.pt"
            ),
        )
    )
    dn_model, _ = load_deeponet(dn_ckpt, device)
    loglo_model, build_input_from_h5, predict_tf, loglo_ckpt = load_loglo(device)
    recorder_x = np.load(dn_config.RECORDER_X_IDX_PATH)

    all_rows: list[dict[str, Any]] = []
    by_campaign: dict[str, list[dict[str, Any]]] = {}

    for camp in campaigns:
        camp = camp.strip()
        partial_path = out_dir / f"partial_{camp}.json"
        if partial_path.is_file() and os.environ.get("FORCE_RERUN", "0") != "1":
            print(f"[ood-compare] skip {camp} (found {partial_path})", flush=True)
            rows = json.loads(partial_path.read_text())
            by_campaign[camp] = rows
            all_rows.extend(rows)
            continue

        jobs = _campaign_jobs(camp, ood_root, gt_root, limit)
        print(f"[ood-compare] {camp}: {len(jobs)} cases", flush=True)
        rows: list[dict[str, Any]] = []
        for j, (_, h5_path, case_dir) in enumerate(jobs):
            tf_true = np.load(case_dir / "tf_true.npy")
            freq = np.load(case_dir / "freq.npy")
            n_rec, n_freq = tf_true.shape

            x = build_input_from_h5(h5_path)
            tf_loglo = predict_tf(loglo_model, x, device)
            tf_dn, tf1d = predict_deeponet_ood(
                dn_model, h5_path, freq, recorder_x, stats, device
            )

            row = {
                "campaign": camp,
                "case": case_dir.name,
                "h5_path": str(h5_path),
                "loglo_rel_l2": _rel_l2(tf_loglo, tf_true),
                "deeponet_rel_l2": _rel_l2(tf_dn, tf_true),
                "tf1d_rel_l2": _rel_l2(tf1d, tf_true),
                "loglo_pearson_f": _pearson_across_freq(
                    tf_true, tf_loglo, n_rec, n_freq
                ),
                "deeponet_pearson_f": _pearson_across_freq(
                    tf_true, tf_dn, n_rec, n_freq
                ),
                "tf1d_pearson_f": _pearson_across_freq(
                    tf_true, tf1d, n_rec, n_freq
                ),
                "loglo_r2": _r2(tf_true, tf_loglo),
                "deeponet_r2": _r2(tf_true, tf_dn),
                "tf1d_r2": _r2(tf_true, tf1d),
            }
            rows.append(row)
            all_rows.append(row)
            if (j + 1) % 50 == 0 or j == 0:
                print(
                    f"  [{camp}] {j+1}/{len(jobs)}  "
                    f"LOGLO={row['loglo_rel_l2']:.3f}  "
                    f"DN={row['deeponet_rel_l2']:.3f}  "
                    f"TF1D={row['tf1d_rel_l2']:.3f}",
                    flush=True,
                )
        partial_path.write_text(json.dumps(rows, indent=2))
        print(f"[ood-compare] wrote partial {partial_path}", flush=True)
        by_campaign[camp] = rows

    summary = {
        "deeponet_ckpt": str(dn_ckpt),
        "loglo_ckpt": str(loglo_ckpt),
        "note": "OOD dipping + three_layer; GT from ood_scores_full7680 cache",
        "campaigns": {c: _aggregate(by_campaign[c]) for c in by_campaign},
        "combined": _aggregate(all_rows),
    }

    out_json = out_dir / "summary.json"
    out_json.write_text(
        json.dumps({"summary": summary, "per_case": all_rows}, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[ood-compare] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
