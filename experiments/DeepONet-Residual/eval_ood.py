#!/usr/bin/env python3
"""Score Haskell nom/col (± optional R_hat) on Box ood_dipping / ood_three_layer.

Does **not** default to ~/seiskit/neural-operator/experiments/. Roots are
``$GIFNO_DATA_ROOT/ood_*`` or ``GIFNO_OOD_DIPPING`` / ``GIFNO_OOD_THREE_LAYER``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import config
import numpy as np

_RES = config.RESIDUAL_DIR
if str(_RES) not in sys.path:
    sys.path.insert(0, str(_RES))

from haskell_baseline import (  # noqa: E402
    haskell_at_columns,
    haskell_nominal_af_within,
    haskell_nominal_layered_af_within,
)
from ood_io import (  # noqa: E402
    clamp_residual,
    crop_variability,
    default_ood_roots,
    discover_h5_files,
    load_or_compute_tf,
    nominal_layer_params,
    read_h5_sample,
    recorder_x_indices,
    soil_nz_from_params,
)

_EPS = 1e-12


def _metrics(y: np.ndarray, p: np.ndarray, *, n_rec: int, n_freq: int) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    return {
        "r2": _r2(y, p),
        "rel_l2": _rel_l2(y, p),
        "pearson": _pearson(y, p),
        "pearson_freq": _pearson_across_freq(y, p, n_rec=n_rec, n_freq=n_freq),
    }


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _rel_l2(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.linalg.norm(y - p) / max(np.linalg.norm(y), 1e-12))


def _pearson(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.float64).ravel()
    p = p.astype(np.float64).ravel()
    if y.size < 2 or y.std() < 1e-12 or p.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def _pearson_across_freq(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_rec: int,
    n_freq: int,
) -> float:
    y = y.astype(np.float64).ravel()
    p = p.astype(np.float64).ravel()
    q = n_rec * n_freq
    if y.size % q != 0:
        return _pearson(y, p)
    n_s = y.size // q
    Y = y.reshape(n_s, n_rec, n_freq)
    P = p.reshape(n_s, n_rec, n_freq)
    cors: list[float] = []
    for i in range(n_s):
        for r in range(n_rec):
            a, b = Y[i, r], P[i, r]
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            cors.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(cors)) if cors else 0.0


def _aggregate(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out: dict[str, float] = {"n": float(len(rows))}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[f"{prefix}_{k}_mean"] = float(np.mean(vals))
        out[f"{prefix}_{k}_median"] = float(np.median(vals))
    return out


def _ood_stoch(params: dict[str, Any], nz: int) -> tuple[np.ndarray, str]:
    """Map OOD attrs onto the IID stochastic branch (ξ, rH, aHV, CoV, ξ_damp)."""
    from features import spectral_kl_coefficients

    if "rf_seed" in params:
        rf_seed = int(params["rf_seed"])
        rH = float(params["rH"])
        aHV = float(params["aHV"])
        cov = float(params["CoV"])
        note = "iid_like_rf_seed"
    else:
        rf_seed = int(params.get("seed1", params.get("seed", 0)))
        rH = float(params.get("rH1", params.get("rH", 1.0)))
        aHV = float(params.get("aHV1", params.get("aHV", 1.0)))
        cov = float(params.get("CoV1", params.get("CoV", 0.0)))
        note = "three_layer_layer1_standin"
    xi_damp = float(config.DEFAULT_XI_TREND)
    xi_vals, _ = spectral_kl_coefficients(
        rf_seed=rf_seed,
        rH=rH,
        aHV=aHV,
        nx=config.NX,
        nz=nz,
        dx=config.DX,
        dz=config.DZ,
        k=config.K_XI,
    )
    stoch = np.concatenate(
        [xi_vals, np.array([rH, aHV, cov, xi_damp], dtype=np.float32)]
    ).astype(np.float32)
    return stoch, note


def _ood_fields(
    vs: np.ndarray, zeta: np.ndarray, recorder_x: np.ndarray, nz: int, soil_nz: int
) -> tuple[np.ndarray, np.ndarray]:
    from data import normalize_vs_surface, normalize_zeta_max, pad_depth

    vs_c = crop_variability(vs)
    zeta_c = crop_variability(zeta)
    vs_pad = pad_depth(vs_c, config.NZ_MAX)
    zeta_pad = pad_depth(zeta_c, config.NZ_MAX)
    vs_n = normalize_vs_surface(vs_pad)
    zeta_n = normalize_zeta_max(zeta_pad, nz)
    z_imp = (config.RHO * vs_pad).astype(np.float32)
    z_imp = z_imp / max(float(z_imp.max()), _EPS)
    cols = recorder_x.astype(int)
    fields = np.stack(
        [vs_n[:, cols], zeta_n[:, cols], z_imp[:, cols]], axis=0
    ).astype(np.float32)
    n = max(1, min(int(soil_nz), vs_c.shape[0]))
    vs_col = vs_c[:n, cols].mean(axis=0).astype(np.float32)
    return fields, vs_col


def _load_residual_model(ckpt_path: Path, device):
    import torch
    from model import build_model

    from data import stoch_dim, trunk_feature_names

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    trunk_set = blob.get("trunk_set", "full")
    serial = bool(blob.get("serial_tf1d", False))
    trunk_dim = len(trunk_feature_names(trunk_set)) + (1 if serial else 0)
    model = build_model(
        blob.get("branch_mode", "single"),
        field_channels=config.FIELD_CHANNELS,
        stoch_dim=stoch_dim(),
        trunk_dim=trunk_dim,
        latent_dim=config.LATENT_DIM,
        field_hidden=config.FIELD_HIDDEN,
        branch_hidden=config.BRANCH_HIDDEN,
        trunk_hidden=config.TRUNK_HIDDEN,
        trunk_layers=config.TRUNK_LAYERS,
        field_encoder=blob.get("field_encoder", "conv"),
    )
    model.load_state_dict(blob["model"])
    model.to(device)
    model.eval()
    stats = blob.get("stats")
    if stats is None:
        raise RuntimeError(
            f"{ckpt_path} has no training stats; retrain with the updated train.py"
        )
    stats = {k: torch.as_tensor(v) for k, v in stats.items()}
    return model, blob, stats, trunk_set


def _predict_rhat(
    model,
    blob: dict[str, Any],
    stats: dict[str, Any],
    trunk_set: str,
    *,
    fields: np.ndarray,
    stoch: np.ndarray,
    vs_col: np.ndarray,
    H: float,
    freq: np.ndarray,
    recorder_x: np.ndarray,
    device,
    tf1d: np.ndarray | None = None,
) -> np.ndarray:
    import torch
    from features import fourier_freq_features
    from train import _forward

    from data import append_serial_tf1d, build_trunk_queries, trunk_feature_names

    sin_f, cos_f = fourier_freq_features(
        freq, f_min=config.FREQ_START_HZ, f_max=config.FREQ_END_HZ
    )
    trunk = build_trunk_queries(
        vs_col=vs_col,
        H=H,
        recorder_x=recorder_x,
        freq_s=freq,
        sin_f=sin_f,
        cos_f=cos_f,
        trunk_names=trunk_feature_names(trunk_set),
    )
    if blob.get("serial_tf1d") and tf1d is not None:
        trunk = append_serial_tf1d(trunk, tf1d)
    with torch.no_grad():
        stoch_t = (torch.from_numpy(stoch) - stats["stoch_mean"]) / stats["stoch_std"]
        trunk_t = (torch.from_numpy(trunk) - stats["trunk_mean"]) / stats["trunk_std"]
        fields_t = torch.from_numpy(fields).unsqueeze(0).to(device)
        stoch_t = stoch_t.unsqueeze(0).to(device)
        trunk_t = trunk_t.unsqueeze(0).to(device)
        pred_n = _forward(
            model, fields_t, stoch_t, trunk_t, blob.get("branch_mode", "single")
        )
        pred = pred_n * stats["target_std"].to(device) + stats["target_mean"].to(device)
        n_rec = len(recorder_x)
        n_freq = len(freq)
        return pred.squeeze(0).cpu().numpy().reshape(n_rec, n_freq)


def eval_one_h5(
    h5_path: Path,
    *,
    tf_cache_dir: Path,
    recorder_x: np.ndarray,
    freq_ref: np.ndarray | None,
    model_pack: tuple | None,
    clamp_mode: str,
    device: Any,
    force_tf: bool,
) -> dict[str, Any]:
    vs, zeta, params, extra = read_h5_sample(h5_path)
    tf, freq = load_or_compute_tf(h5_path, tf_cache_dir, force=force_tf)
    if freq_ref is not None and len(freq) == len(freq_ref):
        freq = np.asarray(freq_ref, dtype=np.float64)
    vs_c = crop_variability(vs)
    zeta_c = crop_variability(zeta)
    nom = nominal_layer_params(params)
    vs2 = float(nom["vs2"])
    soil_nz = soil_nz_from_params(params, vs_c.shape[0])
    rec = recorder_x
    if rec.max() >= vs_c.shape[1]:
        rec = np.clip(rec, 0, vs_c.shape[1] - 1)

    tf1d_col = haskell_at_columns(
        freq,
        vs_c,
        zeta_c,
        rec,
        dz=config.DZ,
        vs_rock=vs2,
        soil_nz=soil_nz,
        rho=config.RHO,
    ).astype(np.float64)
    tf1d_nom_1d = haskell_nominal_af_within(
        freq,
        vs1=float(nom["vs1"]),
        H=float(nom["H"]),
        vs2=vs2,
        xi=float(config.DEFAULT_XI_TREND),
        rho=config.RHO,
    )
    tf1d_nom = np.broadcast_to(tf1d_nom_1d[None, :], tf1d_col.shape).copy()
    tf = np.asarray(tf, dtype=np.float64)
    n_rec, n_freq = tf.shape

    tf1d_nom3 = None
    true_layers = nom.get("true_layers")
    if true_layers is not None:
        tf1d_nom3_1d = haskell_nominal_layered_af_within(
            freq,
            H=true_layers["H"],
            Vs=true_layers["Vs"],
            vs_rock=float(true_layers["vs_rock"]),
            xi=float(config.DEFAULT_XI_TREND),
            rho=config.RHO,
        )
        tf1d_nom3 = np.broadcast_to(tf1d_nom3_1d[None, :], tf1d_col.shape).copy()

    row: dict[str, Any] = {
        "h5": str(h5_path),
        "nom_source": nom["source"],
        "nom_misspecified": bool(nom["misspecified"]),
        "vs_shape": extra["vs_shape"],
        "soil_nz": soil_nz,
        "H_nom": float(nom["H"]),
        "Vs1_nom": float(nom["vs1"]),
        "Vs2_nom": vs2,
    }
    if true_layers is not None:
        row["nom3_source"] = true_layers["source"]
        row["H1_nom3"] = float(true_layers["H"][0])
        row["H2_nom3"] = float(true_layers["H"][1])
        row["Vs_mid_nom3"] = float(true_layers["Vs"][1])
    m_nom = _metrics(tf, tf1d_nom, n_rec=n_rec, n_freq=n_freq)
    m_col = _metrics(tf, tf1d_col, n_rec=n_rec, n_freq=n_freq)
    row.update({f"haskell_nom_{k}": v for k, v in m_nom.items()})
    row.update({f"haskell_col_{k}": v for k, v in m_col.items()})
    if tf1d_nom3 is not None:
        m_nom3 = _metrics(tf, tf1d_nom3, n_rec=n_rec, n_freq=n_freq)
        row.update({f"haskell_nom3_{k}": v for k, v in m_nom3.items()})
        row["delta_rel_l2_nom3_vs_nom"] = float(m_nom["rel_l2"] - m_nom3["rel_l2"])
        row["delta_rel_l2_nom3_vs_col"] = float(m_col["rel_l2"] - m_nom3["rel_l2"])
    r_nom = tf - tf1d_nom
    r_col = tf - tf1d_col
    row["mean_abs_R_nom"] = float(np.mean(np.abs(r_nom)))
    row["mean_abs_R_col"] = float(np.mean(np.abs(r_col)))

    if model_pack is not None:
        model, blob, stats, trunk_set = model_pack
        fields, vs_col = _ood_fields(vs, zeta, rec, int(vs_c.shape[0]), soil_nz)
        stoch, stoch_note = _ood_stoch(params, int(vs_c.shape[0]))
        row["stoch_note"] = stoch_note
        r_hat = _predict_rhat(
            model,
            blob,
            stats,
            trunk_set,
            fields=fields,
            stoch=stoch,
            vs_col=vs_col,
            H=float(nom["H"]),
            freq=np.asarray(freq, dtype=np.float64),
            recorder_x=rec,
            device=device,
            tf1d=tf1d_nom3 if (blob.get("serial_tf1d") and tf1d_nom3 is not None) else tf1d_nom,
        )
        r_hat = clamp_residual(r_hat, clamp_mode)
        prior = (
            tf1d_nom3
            if (blob.get("serial_tf1d") and tf1d_nom3 is not None)
            else tf1d_nom
        )
        tf_hat = prior + np.asarray(r_hat, dtype=np.float64)
        m_plus = _metrics(tf, tf_hat, n_rec=n_rec, n_freq=n_freq)
        row.update({f"nom_plus_Rhat_{k}": v for k, v in m_plus.items()})
        row["delta_rel_l2_vs_nom"] = float(m_nom["rel_l2"] - m_plus["rel_l2"])
        row["delta_r2_vs_nom"] = float(m_plus["r2"] - m_nom["r2"])
        row["delta_rel_l2_vs_col"] = float(m_col["rel_l2"] - m_plus["rel_l2"])
        if tf1d_nom3 is not None:
            tf_hat3 = tf1d_nom3 + np.asarray(r_hat, dtype=np.float64)
            m_plus3 = _metrics(tf, tf_hat3, n_rec=n_rec, n_freq=n_freq)
            row.update({f"nom3_plus_Rhat_{k}": v for k, v in m_plus3.items()})
            row["delta_rel_l2_nom3R_vs_nom3"] = float(
                m_nom3["rel_l2"] - m_plus3["rel_l2"]
            )
            row["delta_rel_l2_nom3R_vs_col"] = float(
                m_col["rel_l2"] - m_plus3["rel_l2"]
            )
    return row


def eval_corpus(
    name: str,
    root: Path,
    *,
    tf_cache_dir: Path,
    model_pack: tuple | None,
    clamp_mode: str,
    device: Any,
    limit: int | None,
    force_tf: bool,
) -> dict[str, Any]:
    h5s = discover_h5_files(root)
    if limit is not None:
        h5s = h5s[: int(limit)]
    rec = recorder_x_indices(root)
    rows: list[dict[str, Any]] = []
    print(f"[{name}] {len(h5s)} H5 under {root}", flush=True)
    for i, p in enumerate(h5s):
        print(f"[{name}] {i + 1}/{len(h5s)} {p.name}", flush=True)
        rows.append(
            eval_one_h5(
                p,
                tf_cache_dir=tf_cache_dir,
                recorder_x=rec,
                freq_ref=None,
                model_pack=model_pack,
                clamp_mode=clamp_mode,
                device=device,
                force_tf=force_tf,
            )
        )
    nom_rows = [
        {k.replace("haskell_nom_", ""): v for k, v in r.items() if k.startswith("haskell_nom_")}
        for r in rows
    ]
    col_rows = [
        {k.replace("haskell_col_", ""): v for k, v in r.items() if k.startswith("haskell_col_")}
        for r in rows
    ]
    summary: dict[str, Any] = {
        "corpus": name,
        "root": str(root),
        "n_h5": len(h5s),
        "nom_source": rows[0]["nom_source"] if rows else None,
        "nom_misspecified": rows[0]["nom_misspecified"] if rows else None,
        **_aggregate(nom_rows, "haskell_nom"),
        **_aggregate(col_rows, "haskell_col"),
    }
    if rows and "nom_plus_Rhat_rel_l2" in rows[0]:
        plus_rows = [
            {
                k.replace("nom_plus_Rhat_", ""): v
                for k, v in r.items()
                if k.startswith("nom_plus_Rhat_")
            }
            for r in rows
        ]
        summary.update(_aggregate(plus_rows, "nom_plus_Rhat"))
        summary["frac_Rhat_beats_nom_rel_l2"] = float(
            np.mean([r["delta_rel_l2_vs_nom"] > 0 for r in rows])
        )
        summary["frac_Rhat_beats_col_rel_l2"] = float(
            np.mean([r["delta_rel_l2_vs_col"] > 0 for r in rows])
        )
    if rows and "haskell_nom3_rel_l2" in rows[0]:
        nom3_rows = [
            {
                k.replace("haskell_nom3_", ""): v
                for k, v in r.items()
                if k.startswith("haskell_nom3_")
            }
            for r in rows
        ]
        summary.update(_aggregate(nom3_rows, "haskell_nom3"))
        summary["nom3_source"] = rows[0].get("nom3_source")
        summary["frac_nom3_beats_nom_rel_l2"] = float(
            np.mean([r["delta_rel_l2_nom3_vs_nom"] > 0 for r in rows])
        )
        summary["frac_nom3_beats_col_rel_l2"] = float(
            np.mean([r["delta_rel_l2_nom3_vs_col"] > 0 for r in rows])
        )
    if rows and "nom3_plus_Rhat_rel_l2" in rows[0]:
        plus3_rows = [
            {
                k.replace("nom3_plus_Rhat_", ""): v
                for k, v in r.items()
                if k.startswith("nom3_plus_Rhat_")
            }
            for r in rows
        ]
        summary.update(_aggregate(plus3_rows, "nom3_plus_Rhat"))
        summary["frac_nom3R_beats_nom3_rel_l2"] = float(
            np.mean([r["delta_rel_l2_nom3R_vs_nom3"] > 0 for r in rows])
        )
        summary["frac_nom3R_beats_col_rel_l2"] = float(
            np.mean([r["delta_rel_l2_nom3R_vs_col"] > 0 for r in rows])
        )
    summary["mean_abs_R_nom"] = float(np.mean([r["mean_abs_R_nom"] for r in rows])) if rows else None
    summary["mean_abs_R_col"] = float(np.mean([r["mean_abs_R_col"] for r in rows])) if rows else None
    return {"summary": summary, "per_file": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=config.DEFAULT_CHECKPOINT if config.DEFAULT_CHECKPOINT.is_file() else None,
        help="Residual checkpoint (default: shipped serial P3 mix). Pass empty via --haskell-only.",
    )
    p.add_argument(
        "--haskell-only",
        action="store_true",
        help="Skip the residual net; score Haskell nom/col only.",
    )
    p.add_argument("--clamp", choices=["none", "tanh", "zero"], default="none")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force-tf", action="store_true")
    p.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="Restrict to these corpus names (repeatable). Default: both Box ood_*.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=config.RESULTS_DIR / "ood_eval.json",
    )
    args = p.parse_args()
    if args.haskell_only:
        args.checkpoint = None
    device: Any = "cpu"
    model_pack = None
    if args.checkpoint is not None:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_pack = _load_residual_model(args.checkpoint, device)

    report: dict[str, Any] = {
        "GIFNO_DATA_ROOT": str(config.data_root()),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "clamp": args.clamp,
        "corpora": {},
    }
    roots = default_ood_roots()
    if args.corpus:
        wanted = {c.strip() for c in args.corpus}
        missing = wanted - set(roots)
        if missing:
            raise SystemExit(
                f"unknown --corpus {sorted(missing)}; known={sorted(roots)}"
            )
        roots = {k: v for k, v in roots.items() if k in wanted}
    for name, root in roots.items():
        tf_cache = config.CACHE_DIR / f"{name}_tf"
        pack = eval_corpus(
            name,
            root,
            tf_cache_dir=tf_cache,
            model_pack=model_pack,
            clamp_mode=args.clamp,
            device=device,
            limit=args.limit,
            force_tf=args.force_tf,
        )
        report["corpora"][name] = pack
        s = pack["summary"]
        print(f"\n=== {name} n={s['n_h5']} nom={s['nom_source']} ===", flush=True)
        for k in sorted(s):
            if k in ("corpus", "root"):
                continue
            print(f"  {k}: {s[k]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
