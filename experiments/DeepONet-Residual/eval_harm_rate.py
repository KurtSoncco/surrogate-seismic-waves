#!/usr/bin/env python3
"""E0+: harm-rate / fail-soft / peak-amp of residual GINO vs Haskell and LOGLO.

No training. Scores an existing residual checkpoint on mix held-out slices
and optionally LOGLO on the *same* files if a LOGLO ckpt is set.

    cd experiments/DeepONet-Residual
    python eval_harm_rate.py --ckpt checkpoints/M7680_gino_rebal_ft.pt
    python eval_harm_rate.py --synthetic
    python eval_harm_rate.py --ckpt checkpoints/M7680_gino_rebal_ft.pt --full7680
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
from unified_metrics import score_leftover_batch, win_rate

_EXP = Path(__file__).resolve().parent


def _freq_grid() -> np.ndarray:
    return np.logspace(
        np.log10(config.FREQ_START_HZ),
        np.log10(config.FREQ_END_HZ),
        config.N_FREQ,
    ).astype(np.float64)


def _score_loader(
    model,
    loader,
    stats,
    device,
    *,
    n_rec: int,
    n_freq: int,
    freq,
    log_residual: bool,
) -> dict:
    from train import _forward
    from model import apply_query_freq

    apply_query_freq(model, freq)
    model.eval()
    t_mean = stats["target_mean"].to(device)
    t_std = stats["target_std"].to(device)
    p_all, tf1d_all, tf2d_all = [], [], []
    gate_all: list[np.ndarray] = []
    f0_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            fields = batch["fields"].to(device)
            stoch = batch["stoch"].to(device)
            trunk_y = batch["trunk_y"].to(device)
            pred_n = _forward(
                model,
                fields,
                stoch,
                trunk_y,
                "single",
                geom_flags=batch.get("geom_flags"),
            )
            pred = pred_n * t_std + t_mean
            p_all.append(pred.cpu().numpy())
            tf1d_all.append(batch["tf1d"].numpy())
            tf2d_all.append(batch["tf2d"].numpy())
            if "f0_tts" in batch:
                f0_all.append(
                    np.asarray(batch["f0_tts"].numpy(), dtype=np.float64).ravel()
                )
            gate = getattr(model, "last_gate", None)
            if gate is not None:
                gate_all.append(gate.cpu().numpy().reshape(gate.shape[0], -1).mean(-1))
    p = np.concatenate(p_all, axis=0)
    tf1d = np.concatenate(tf1d_all, axis=0)
    tf2d = np.concatenate(tf2d_all, axis=0)
    gate = np.concatenate(gate_all) if gate_all else None
    r_true = tf2d - tf1d
    return score_leftover_batch(
        tf1d=tf1d,
        r_true=r_true,
        r_hat=p,
        tf2d=tf2d,
        freq=freq[:n_freq] if freq is not None else None,
        n_rec=n_rec,
        n_freq=n_freq,
        gate=gate,
        log_residual=log_residual,
        f0_calc=np.concatenate(f0_all) if f0_all else None,
    )


def _load_residual(ckpt: Path, device):
    from model import build_from_checkpoint_blob

    from data import (
        infer_trunk_scales,
        stoch_dim,
        stoch_layout_from_blob,
        trunk_feature_names,
        trunk_in_features_from_state,
    )

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    serial = bool(blob.get("serial_tf1d", False))
    trunk_set = blob.get("trunk_set", "full")
    trunk_scales = infer_trunk_scales(
        trunk_set=trunk_set,
        serial=serial,
        recorded=int(blob.get("trunk_scales", 1) or 1),
        trunk_in_features=trunk_in_features_from_state(blob.get("model") or {}),
    )
    blob["trunk_scales"] = int(trunk_scales)
    layout = stoch_layout_from_blob(blob)
    blob["stoch_layout"] = layout
    sdim = stoch_dim(layout=layout)
    blob["stoch_dim"] = int(sdim)
    trunk_dim = len(trunk_feature_names(trunk_set, trunk_scales)) + (1 if serial else 0)
    model = build_from_checkpoint_blob(
        blob,
        field_channels=config.FIELD_CHANNELS,
        stoch_dim=sdim,
        trunk_dim=trunk_dim,
        latent_dim=config.LATENT_DIM,
        field_hidden=config.FIELD_HIDDEN,
        branch_hidden=config.BRANCH_HIDDEN,
        trunk_hidden=config.TRUNK_HIDDEN,
        trunk_layers=config.TRUNK_LAYERS,
    )
    model.load_state_dict(blob["model"])
    model.to(device)
    model.eval()
    stats = {k: torch.as_tensor(v) for k, v in blob["stats"].items()}
    return model, blob, stats


def _compact(mets: dict[str, Any]) -> dict[str, Any]:
    skip = {"rel_l2_tf_hat_per_sample", "rel_l2_tf_1d_per_sample"}
    return {k: v for k, v in mets.items() if k not in skip}


def score_gino(ckpt: Path, out_dir: Path, *, full7680: bool = False) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    from data import ResidualDeepONetDataset, dataset_kwargs_from_blob
    from mix_ladder import mix_test_parts
    from train import apply_norms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, blob, stats = _load_residual(ckpt, device)
    serial = bool(blob.get("serial_tf1d", True))
    log_residual = bool(blob.get("log_residual", False))
    tests = mix_test_parts()
    if full7680:
        cache = config.CACHE_DIR / "full7680_seed42"
        idx_path = cache / "splits" / f"iid_n7680_seed{config.SEED}.npz"
        from data import make_splits

        if (cache / "r_nom_signed.npy").is_file():
            meta = np.load(cache / "meta.npz", allow_pickle=True)
            splits = make_splits(len(meta["sample_idx"]), seed=config.SEED)
            tests = {**tests, "iid_full7680": (cache, splits.test)}
        elif idx_path.is_file():
            pass
    report: dict[str, Any] = {
        "ckpt": str(ckpt),
        "log_residual": log_residual,
        "note": (
            "harm_rate = P(||TF_hat-TF2D|| > ||TF1D-TF2D||); "
            "E0+ adds logspec, peak ΔlnA, band L2, harm-by-leftover-quantile"
        ),
        "domains": {},
    }
    for name, (cache, idx) in tests.items():
        if not (cache / "r_nom_signed.npy").is_file():
            report["domains"][name] = {
                "skipped": True,
                "reason": f"missing cache {cache}",
            }
            continue
        ds = ResidualDeepONetDataset(
            cache,
            idx,
            target="R_nom",
            trunk_set="full",
            n_freq=config.N_FREQ_EVAL,
            serial_tf1d=serial,
            **dataset_kwargs_from_blob(blob),
        )
        apply_norms(ds, stats)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        mets = _score_loader(
            model,
            loader,
            stats,
            device,
            n_rec=ds.n_rec,
            n_freq=len(ds.f_idx),
            freq=ds.freq_s,
            log_residual=log_residual,
        )
        report["domains"][name] = mets
        print(
            f"[harm] {name} n={mets['n']} harm_rate={mets['harm_rate']:.3f} "
            f"relL2_hat={mets['rel_l2_tf_hat']:.3f} relL2_1d={mets['rel_l2_tf_1d']:.3f} "
            f"logspec={mets['logspec_rel_l2_tf_hat']:.3f} "
            f"dlnA={mets.get('peak_dlnA_vs_2d_mean', float('nan')):.3f} "
            f"harm_small={mets.get('harm_rate_small_leftover', float('nan')):.3f}",
            flush=True,
        )
    report["loglo"] = score_loglo_on_mix(device, tests, n_freq=config.N_FREQ_EVAL)
    _attach_win_rates(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    compact = {
        **{k: v for k, v in report.items() if k != "domains"},
        "domains": {
            n: _compact(m) if isinstance(m, dict) else m
            for n, m in report["domains"].items()
        },
        "loglo": {
            **{k: v for k, v in report["loglo"].items() if k != "domains"},
            "domains": {
                n: _compact(m) if isinstance(m, dict) else m
                for n, m in report["loglo"].get("domains", {}).items()
            },
        }
        if isinstance(report.get("loglo"), dict)
        else report.get("loglo"),
    }
    path = out_dir / "harm_rate.json"
    path.write_text(json.dumps(compact, indent=2, default=str))
    print(f"[harm] wrote {path}", flush=True)
    return report


def _attach_win_rates(report: dict[str, Any]) -> None:
    loglo = report.get("loglo") or {}
    if loglo.get("skipped"):
        return
    wins: dict[str, Any] = {}
    for name, gino in report.get("domains", {}).items():
        lo = (loglo.get("domains") or {}).get(name)
        if not isinstance(gino, dict) or not isinstance(lo, dict):
            continue
        if gino.get("skipped") or lo.get("skipped"):
            continue
        ga = gino.get("rel_l2_tf_hat_per_sample") or []
        la = lo.get("rel_l2_tf_hat_per_sample") or []
        if not ga or not la:
            continue
        wins[name] = {
            "gino_beats_loglo": win_rate(ga, la),
            "loglo_beats_gino": win_rate(la, ga),
            "gino_beats_haskell": win_rate(
                ga, gino.get("rel_l2_tf_1d_per_sample") or []
            ),
            "loglo_beats_haskell": win_rate(
                la, lo.get("rel_l2_tf_1d_per_sample") or []
            ),
        }
    if wins:
        report["win_rate"] = wins


def os_environ_ckpt() -> Path | None:
    env = os.environ.get("GIFNO_MODEL_DIR") or os.environ.get("LOGLO_CKPT")
    if not env:
        default = (
            Path.home()
            / "surrogate-seismic-waves"
            / "checkpoints"
            / "tier2_pod64_full7680"
            / "best_model.pt"
        )
        if default.is_file():
            return default
        return None
    p = Path(env)
    for cand in (p / "best_model.pt", p):
        if cand.is_file():
            return cand
    return None


def score_loglo_on_mix(
    device: torch.device,
    tests: dict[str, tuple[Path, np.ndarray]],
    *,
    n_freq: int,
) -> dict[str, Any]:
    ckpt = os_environ_ckpt()
    if ckpt is None:
        return {
            "skipped": True,
            "reason": "set GIFNO_MODEL_DIR or LOGLO_CKPT to score LOGLO on the same slices",
        }
    try:
        from compare_tf_loglo_vs_deeponet import load_loglo
        from residual_signed import resolve_h5_path
        from data import freq_screen_indices
    except Exception as exc:
        return {"skipped": True, "reason": f"LOGLO import failed: {exc}"}

    try:
        loglo_model, build_input_from_h5, predict_tf, used = load_loglo(device)
    except Exception as exc:
        return {
            "skipped": True,
            "reason": f"LOGLO load failed: {exc}",
            "ckpt": str(ckpt),
        }

    out: dict[str, Any] = {"skipped": False, "ckpt": str(used), "domains": {}}
    for name, (cache, idx) in tests.items():
        if not (cache / "r_nom_signed.npy").is_file():
            out["domains"][name] = {"skipped": True, "reason": f"missing cache {cache}"}
            continue
        meta = dict(np.load(cache / "meta.npz", allow_pickle=True))
        tf1d_all = np.load(cache / "tf1d_nom.npy", mmap_mode="r")
        r_all = np.load(cache / "r_nom_signed.npy", mmap_mode="r")
        freq = (
            np.load(cache / "freq.npy")
            if (cache / "freq.npy").is_file()
            else _freq_grid()
        )
        f_idx = freq_screen_indices(freq, n_freq)
        tf2d_path = cache / "tf2d.npy"
        tf2d_local = np.load(tf2d_path, mmap_mode="r") if tf2d_path.is_file() else None
        hats, t1, t2, rt = [], [], [], []
        n_ok = 0
        n_miss = 0
        for local_i in np.asarray(idx).ravel():
            local_i = int(local_i)
            h5 = Path(str(meta["h5_path"][local_i]))
            if not h5.is_file():
                h5 = resolve_h5_path(str(h5))
            if not h5.is_file():
                n_miss += 1
                continue
            try:
                x = build_input_from_h5(h5)
                pred = np.asarray(predict_tf(loglo_model, x, device), dtype=np.float64)
            except Exception:
                n_miss += 1
                continue
            tf1d = np.asarray(tf1d_all[local_i][:, f_idx], dtype=np.float64)
            if tf2d_local is not None:
                tf2d = np.asarray(tf2d_local[local_i][:, f_idx], dtype=np.float64)
            else:
                tf2d = tf1d + np.asarray(r_all[local_i][:, f_idx], dtype=np.float64)
            if pred.shape[-1] != tf1d.shape[-1]:
                pred = pred[:, f_idx] if pred.shape[-1] >= n_freq else pred
            if pred.shape != tf1d.shape:
                n_miss += 1
                continue
            hats.append(pred.reshape(-1))
            t1.append(tf1d.reshape(-1))
            t2.append(tf2d.reshape(-1))
            rt.append((tf2d - tf1d).reshape(-1))
            n_ok += 1
        if n_ok == 0:
            out["domains"][name] = {
                "skipped": True,
                "reason": f"no H5 files scored (miss={n_miss})",
            }
            continue
        mets = score_leftover_batch(
            tf1d=np.stack(t1),
            r_true=np.stack(rt),
            r_hat=np.stack(hats) - np.stack(t1),
            tf2d=np.stack(t2),
            freq=np.asarray(freq)[f_idx],
            n_rec=t1[0].size // len(f_idx),
            n_freq=len(f_idx),
            tf_hat=np.stack(hats),
        )
        mets["n_h5_missing"] = int(n_miss)
        out["domains"][name] = mets
        print(
            f"[harm] LOGLO {name} n={mets['n']} harm_rate={mets['harm_rate']:.3f} "
            f"relL2={mets['rel_l2_tf_hat']:.3f} logspec={mets['logspec_rel_l2_tf_hat']:.3f}",
            flush=True,
        )
    return out


def synthetic_report() -> dict[str, Any]:
    """Deterministic toy tensors so E0 is testable without GIFNO caches."""
    rng = np.random.default_rng(0)
    n_s, n_rec, n_f = 8, 4, 16
    freq = np.linspace(0.1, 10.0, n_f)
    tf1d = np.abs(rng.normal(1.0, 0.1, size=(n_s, n_rec * n_f))) + 0.2
    r_true = rng.normal(0.0, 0.05, size=tf1d.shape)
    tf2d = tf1d + r_true
    r_hat_good = 0.8 * r_true
    r_hat_bad = -2.0 * r_true
    good = score_leftover_batch(
        tf1d=tf1d,
        r_true=r_true,
        r_hat=r_hat_good,
        tf2d=tf2d,
        freq=freq,
        n_rec=n_rec,
        n_freq=n_f,
    )
    bad = score_leftover_batch(
        tf1d=tf1d,
        r_true=r_true,
        r_hat=r_hat_bad,
        tf2d=tf2d,
        freq=freq,
        n_rec=n_rec,
        n_freq=n_f,
    )
    r_log = np.log(np.maximum(tf2d, 1e-8)) - np.log(np.maximum(tf1d, 1e-8))
    log_good = score_leftover_batch(
        tf1d=tf1d,
        r_true=r_true,
        r_hat=0.8 * r_log,
        tf2d=tf2d,
        freq=freq,
        n_rec=n_rec,
        n_freq=n_f,
        log_residual=True,
    )
    return {
        "good_residual": good,
        "harmful_residual": bad,
        "good_log_residual": log_good,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        type=Path,
        default=config.DEFAULT_CHECKPOINT,
        help="Residual GINO checkpoint (default M7680_gino_rebal_ft.pt).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=config.RESULTS_DIR / "unified_encoder",
    )
    p.add_argument("--synthetic", action="store_true")
    p.add_argument(
        "--full7680",
        action="store_true",
        help="Also score the full7680 IID test split if the cache exists.",
    )
    args = p.parse_args()
    if args.synthetic:
        rep = synthetic_report()
        args.out.mkdir(parents=True, exist_ok=True)
        compact = {k: _compact(v) for k, v in rep.items()}
        (args.out / "harm_rate_synthetic.json").write_text(
            json.dumps(compact, indent=2)
        )
        print(json.dumps(compact, indent=2))
        return
    if not args.ckpt.is_file():
        print(
            f"[harm] no checkpoint at {args.ckpt}; writing synthetic placeholder",
            flush=True,
        )
        rep = synthetic_report()
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "harm_rate.json"
        path.write_text(
            json.dumps(
                {
                    "ckpt": str(args.ckpt),
                    "skipped_live": True,
                    "reason": "checkpoint missing; synthetic diagnostic only",
                    "synthetic": {k: _compact(v) for k, v in rep.items()},
                    "loglo": {
                        "skipped": True,
                        "reason": "no live checkpoint; LOGLO not scored",
                    },
                },
                indent=2,
            )
        )
        print(f"[harm] wrote {path}", flush=True)
        return
    score_gino(args.ckpt, args.out, full7680=args.full7680)


if __name__ == "__main__":
    main()
