#!/usr/bin/env python3
"""Head-to-head TF comparison: LOGLO-POD vs ResUNet DeepONet-Residual (R_nom).

Evaluates both on the DeepONet full7680 *test split* (same sample indices),
full 1000-frequency TFs (not the 50-freq train screen).

Metrics (per sample, then mean/median):
  - rel_L2 on linear |TF|
  - Pearson across frequency (mean over recorders)
  - R² on flattened TF

Usage (Lambda):
  export GIFNO_DATA_ROOT=$HOME/gifno_data
  export GIFNO_H5_DIR=$GIFNO_DATA_ROOT/h5
  export GIFNO_TF_DIR=$GIFNO_DATA_ROOT/transfer_function
  export GIFNO_MODEL_DIR=$HOME/surrogate-seismic-waves/checkpoints/tier2_pod64_full7680
  export GIFNO_POD_NUM_MODES=64 GIFNO_LATENT_CHANNELS=128 GIFNO_NUM_FNO_LAYERS=5
  export GIFNO_DEEPONET_LATENT_DIM=128
  cd experiments/DeepONet-Residual
  python -u compare_tf_loglo_vs_deeponet.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import config as dn_config
from data import (
    ResidualDeepONetDataset,
    build_recorder_fields,
    make_splits,
    stoch_dim,
    trunk_feature_names,
)
from model import build_model
from residual_signed import resolve_h5_path

_EPS = 1e-12


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), _EPS))


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    y = y.ravel().astype(np.float64)
    p = p.ravel().astype(np.float64)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _pearson_across_freq(y: np.ndarray, p: np.ndarray, n_rec: int, n_freq: int) -> float:
    y = y.reshape(n_rec, n_freq).astype(np.float64)
    p = p.reshape(n_rec, n_freq).astype(np.float64)
    cors = []
    for i in range(n_rec):
        a, b = y[i], p[i]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        cors.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(cors)) if cors else 0.0


def _summarize(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def load_deeponet(ckpt: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    trunk_set = blob.get("trunk_set", "full")
    field_encoder = blob.get("field_encoder", "resunet")
    branch_mode = blob.get("branch_mode", "single")
    model = build_model(
        branch_mode,
        field_channels=dn_config.FIELD_CHANNELS,
        stoch_dim=stoch_dim(),
        trunk_dim=len(trunk_feature_names(trunk_set)),
        latent_dim=dn_config.LATENT_DIM,
        field_hidden=dn_config.FIELD_HIDDEN,
        branch_hidden=dn_config.BRANCH_HIDDEN,
        trunk_hidden=dn_config.TRUNK_HIDDEN,
        trunk_layers=dn_config.TRUNK_LAYERS,
        field_encoder=field_encoder,
        n_recorders=dn_config.N_LATERAL,
    )
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    return model, blob


def fit_norms_from_train(cache_dir: Path, seed: int) -> dict[str, torch.Tensor]:
    """Match training: z-score from train split only (same as train.py).

    Uses ResidualDeepONetDataset but only on train indices — same code path as
    training. For full7680 this is heavy once; prefer cached norms if present.
    """
    norms_path = cache_dir / f"train_norms_seed{seed}.pt"
    if norms_path.exists():
        print(f"[compare] loading cached norms {norms_path}", flush=True)
        return torch.load(norms_path, map_location="cpu", weights_only=False)

    meta = np.load(cache_dir / "meta.npz", allow_pickle=True)
    n = len(meta["sample_idx"])
    splits = make_splits(n, seed=seed)
    print(
        f"[compare] building train norms over {len(splits.train)} samples "
        "(one-time preload)...",
        flush=True,
    )
    train_ds = ResidualDeepONetDataset(
        cache_dir, splits.train, target="R_nom", trunk_set="full"
    )
    from train import fit_and_apply_norms

    stats = fit_and_apply_norms(train_ds)
    torch.save(stats, norms_path)
    print(f"[compare] wrote norms → {norms_path}", flush=True)
    return stats


@torch.no_grad()
def predict_deeponet_full_tf(
    model: torch.nn.Module,
    *,
    local_i: int,
    cache_dir: Path,
    meta: dict,
    tf1d_nom: np.ndarray,
    freq: np.ndarray,
    recorder_x: np.ndarray,
    stats: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """Return TF_hat = TF1D_nom + R_hat on all frequencies, shape (n_rec, n_freq)."""
    sys.path.insert(0, str(dn_config.RESIDUAL_DIR))
    from features import fourier_freq_features, spectral_kl_coefficients

    h5_path = Path(str(meta["h5_path"][local_i]))
    # Remap legacy Box paths
    if not h5_path.exists():
        h5_path = resolve_h5_path(str(h5_path))
    nz = int(meta["nz"][local_i])
    H = float(meta["H"][local_i])
    soil_nz = int(meta["soil_nz"][local_i])

    import h5py

    with h5py.File(h5_path, "r") as f:
        vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
        zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
    vs = vs[:, dn_config.X_SLICE_START : dn_config.X_SLICE_END]
    zeta = zeta[:, dn_config.X_SLICE_START : dn_config.X_SLICE_END]

    fields_path = cache_dir / "fields_rec.npy"
    if fields_path.exists():
        fields = np.load(fields_path, mmap_mode="r")[local_i]
        fields = np.asarray(fields, dtype=np.float32)
    else:
        fields = build_recorder_fields(vs, zeta, nz=nz, recorder_x=recorder_x)

    # stochastic
    rf_seed = int(meta["rf_seed"][local_i])
    rH = float(meta["rH"][local_i])
    aHV = float(meta["aHV"][local_i])
    CoV = float(meta["CoV"][local_i])
    xi_damp = float(meta["xi_damp"][local_i]) if "xi_damp" in meta else dn_config.DEFAULT_XI_TREND
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
                        f * H / vs_c,  # f_star
                        float(sin_f[j]),
                        float(cos_f[j]),
                        float(x_m[ri] / max(lam, _EPS)),  # x/lambda
                    ],
                    dtype=np.float32,
                )
            )
    trunk_y = np.stack(rows, axis=0)  # (n_rec*n_freq, 4)

    # apply train norms
    stoch_t = (torch.from_numpy(stoch) - stats["stoch_mean"]) / stats["stoch_std"]
    trunk_t = (torch.from_numpy(trunk_y) - stats["trunk_mean"]) / stats["trunk_std"]
    fields_t = torch.from_numpy(fields)

    pred_n = model(
        fields_t.unsqueeze(0).to(device),
        stoch_t.unsqueeze(0).to(device),
        trunk_t.unsqueeze(0).to(device),
    )[0].cpu()
    pred = pred_n * stats["target_std"] + stats["target_mean"]
    R = pred.numpy().reshape(n_rec, len(freq))
    return (tf1d_nom + R).astype(np.float32)


def load_loglo(device: torch.device):
    import importlib
    import importlib.util

    loglo_dir = Path(__file__).resolve().parents[1] / "GIFNO-FDO-XT-LOGLO-POD"
    gifno_dir = Path(__file__).resolve().parents[1] / "GIFNO"
    dn_dir = Path(__file__).resolve().parent

    # Prefer LOGLO/GIFNO packages over DeepONet-Residual for create_model imports.
    sys.path = [p for p in sys.path if Path(p).resolve() != dn_dir.resolve()]
    for p in (str(gifno_dir), str(loglo_dir)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    # Drop cached modules that collide by name
    for name in list(sys.modules):
        if name in ("config", "model", "capability_check") or name.startswith(
            "capability_check"
        ):
            # keep dn_config reference via dn_config global
            if name == "config":
                continue
            del sys.modules[name]

    cfg_path = loglo_dir / "config.py"
    spec = importlib.util.spec_from_file_location("loglo_pod_config", cfg_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {cfg_path}")
    loglo_config = importlib.util.module_from_spec(spec)
    sys.modules["loglo_pod_config"] = loglo_config
    sys.modules["config"] = loglo_config
    spec.loader.exec_module(loglo_config)
    loglo_config.setup_import_paths()

    # Force LOGLO model.py
    model_path = loglo_dir / "model.py"
    mspec = importlib.util.spec_from_file_location("loglo_pod_model", model_path)
    if mspec is None or mspec.loader is None:
        raise ImportError(f"Cannot load {model_path}")
    loglo_model_mod = importlib.util.module_from_spec(mspec)
    sys.modules["loglo_pod_model"] = loglo_model_mod
    sys.modules["model"] = loglo_model_mod
    mspec.loader.exec_module(loglo_model_mod)

    from capability_check import build_input_from_h5, load_model, predict_tf

    ckpt = Path(
        os.environ.get(
            "LOGLO_CKPT",
            str(
                Path.home()
                / "surrogate-seismic-waves/checkpoints/tier2_pod64_full7680/best_model.pt"
            ),
        )
    )
    model = load_model(ckpt, device)

    # Restore DeepONet import path / config binding for remaining DeepONet code
    if str(dn_dir) not in sys.path:
        sys.path.insert(0, str(dn_dir))
    sys.modules["config"] = dn_config
    # Keep LOGLO helpers bound to their modules; don't rebind model globally
    return model, build_input_from_h5, predict_tf, ckpt
def main() -> None:
    cache_tag = os.environ.get("CACHE_TAG", "full7680_seed42")
    cache_dir = dn_config.CACHE_DIR / cache_tag
    seed = int(os.environ.get("SEED", dn_config.SEED))
    max_samples = int(os.environ.get("MAX_SAMPLES", "0")) or None
    out_dir = Path(
        os.environ.get(
            "COMPARE_OUT",
            str(dn_config.RESULTS_DIR / "compare_tf_loglo_vs_deeponet"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compare] device={device} cache={cache_dir}", flush=True)

    meta = dict(np.load(cache_dir / "meta.npz", allow_pickle=True))
    sample_indices = np.load(cache_dir / "sample_indices.npy")
    tf1d_nom = np.load(cache_dir / "tf1d_nom.npy", mmap_mode="r")
    tf_all = np.load(dn_config.TF_PER_SAMPLE_PATH, mmap_mode="r")
    freq = np.load(dn_config.TF_FREQ_PATH)
    recorder_x = np.load(dn_config.RECORDER_X_IDX_PATH)

    n = len(sample_indices)
    splits = make_splits(n, seed=seed)
    test_local = splits.test
    if max_samples is not None:
        test_local = test_local[:max_samples]
    print(f"[compare] test samples: {len(test_local)} / {n}", flush=True)

    print("[compare] fitting DeepONet train norms (preload train split)...", flush=True)
    stats = fit_norms_from_train(cache_dir, seed)

    dn_ckpt = Path(
        os.environ.get(
            "DEEPONET_CKPT",
            str(
                dn_config.CHECKPOINT_DIR
                / "single_resunet_full_R_nom_full7680_seed42.pt"
            ),
        )
    )
    print(f"[compare] DeepONet ckpt={dn_ckpt}", flush=True)
    dn_model, dn_blob = load_deeponet(dn_ckpt, device)

    print("[compare] loading LOGLO...", flush=True)
    loglo_model, build_input_from_h5, predict_tf, loglo_ckpt = load_loglo(device)
    print(f"[compare] LOGLO ckpt={loglo_ckpt}", flush=True)

    rows = []
    for j, local_i in enumerate(test_local):
        local_i = int(local_i)
        sidx = int(sample_indices[local_i])
        tf_true = np.asarray(tf_all[sidx], dtype=np.float32)
        tf1d = np.asarray(tf1d_nom[local_i], dtype=np.float32)

        # DeepONet
        tf_dn = predict_deeponet_full_tf(
            dn_model,
            local_i=local_i,
            cache_dir=cache_dir,
            meta=meta,
            tf1d_nom=tf1d,
            freq=freq,
            recorder_x=recorder_x,
            stats=stats,
            device=device,
        )

        # LOGLO
        h5_path = Path(str(meta["h5_path"][local_i]))
        if not h5_path.exists():
            h5_path = resolve_h5_path(str(h5_path))
        x = build_input_from_h5(h5_path)
        tf_loglo = predict_tf(loglo_model, x, device)

        n_rec, n_freq = tf_true.shape
        row = {
            "local_i": local_i,
            "sample_idx": sidx,
            "loglo_rel_l2": _rel_l2(tf_loglo, tf_true),
            "deeponet_rel_l2": _rel_l2(tf_dn, tf_true),
            "tf1d_rel_l2": _rel_l2(tf1d, tf_true),
            "loglo_pearson_f": _pearson_across_freq(tf_true, tf_loglo, n_rec, n_freq),
            "deeponet_pearson_f": _pearson_across_freq(tf_true, tf_dn, n_rec, n_freq),
            "tf1d_pearson_f": _pearson_across_freq(tf_true, tf1d, n_rec, n_freq),
            "loglo_r2": _r2(tf_true, tf_loglo),
            "deeponet_r2": _r2(tf_true, tf_dn),
            "tf1d_r2": _r2(tf_true, tf1d),
        }
        rows.append(row)
        if (j + 1) % 25 == 0 or j == 0:
            print(
                f"[{j+1}/{len(test_local)}] sidx={sidx}  "
                f"LOGLO relL2={row['loglo_rel_l2']:.3f}  "
                f"DeepONet relL2={row['deeponet_rel_l2']:.3f}  "
                f"TF1D relL2={row['tf1d_rel_l2']:.3f}",
                flush=True,
            )

    summary = {
        "n_test": len(rows),
        "cache_tag": cache_tag,
        "seed": seed,
        "deeponet_ckpt": str(dn_ckpt),
        "loglo_ckpt": str(loglo_ckpt),
        "note": (
            "Same DeepONet full7680 test split; full 1000 freqs; "
            "DeepONet = TF1D_nom + R_hat (ResUNet); LOGLO = direct TF surrogate"
        ),
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

    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps({"summary": summary, "per_sample": rows}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[compare] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
