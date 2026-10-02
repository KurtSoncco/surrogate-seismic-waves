#!/usr/bin/env python3
"""Station hold-out scoring for kernel-query GNO.

Train may see only even (or interior) recorders. This scores the held-out
stations against:

  * a kernel-GNO checkpoint (if given)
  * linear interpolation of neighboring shipped branch ``p``
  * 1-D Haskell-only (R̂ = 0)

Does not overwrite ``eval_spatial_leftover.py`` (scatter *across* the 21
stations). Nested 21-station ship gates stay on ``score_ship_gates.py``.

Kill rule: if odd-station Pearson is no better than interpolate-p, the kernel
is not learning space — keep the ship checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data import ResidualDeepONetDataset, query_station_indices
from mix_ladder import mix_test_parts
from model import apply_query_freq, gno_core, interp_along_x
from response_variability.metrics import anderson_frequency_domain, pearson
from train import (
    _forward,
    _pearson_across_freq,
    _r2,
    _rel_l2,
    apply_checkpoint_stats,
    evaluate,
    stats_from_checkpoint,
)

OUT_DIR = config.RESULTS_DIR / "arch_train"
HOLD_FROM_TRAIN = {"even": "odd", "interior": "edge", "all": "odd"}
TRAIN_FROM_HOLD = {"odd": "even", "edge": "interior"}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _metrics_from_arrays(
    r_true: np.ndarray,
    r_hat: np.ndarray,
    tf2d: np.ndarray,
    tf1d: np.ndarray,
    freq: np.ndarray,
    *,
    n_rec: int,
    n_freq: int,
) -> dict[str, float]:
    tf_hat = tf1d + r_hat
    anderson = [
        anderson_frequency_domain(freq, tf2d[i, r], tf_hat[i, r])
        for i in range(tf2d.shape[0])
        for r in range(n_rec)
    ]
    return {
        "r2_R": _r2(r_true, r_hat),
        "rel_l2_R": _rel_l2(r_true, r_hat),
        "pearson_R": pearson(r_true, r_hat),
        "pearson_R_freq": _pearson_across_freq(
            r_true.ravel(), r_hat.ravel(), n_rec=n_rec, n_freq=n_freq
        ),
        "rel_l2_TF": _rel_l2(tf2d, tf_hat),
        "pearson_TF": pearson(tf2d, tf_hat),
        "anderson_mean": float(np.nanmean(anderson)) if anderson else float("nan"),
    }


def _per_recorder_rel_l2(
    r_true: np.ndarray, r_hat: np.ndarray, x_m: np.ndarray
) -> list[dict[str, float]]:
    n_rec = r_true.shape[1]
    rows = []
    for i in range(n_rec):
        rows.append(
            {
                "recorder": int(i),
                "x_m": float(x_m[i]),
                "rel_l2_R": _rel_l2(r_true[:, i], r_hat[:, i]),
            }
        )
    return rows


def _load_model(ckpt: Path, device: torch.device):
    from eval_ood import _load_residual_model

    model, blob, stats, trunk_set = _load_residual_model(ckpt, device)
    return model, blob, stats, trunk_set


def _ds(
    cache: Path,
    idx: np.ndarray,
    *,
    query_split: str,
    support_stride: int,
    n_freq: int,
    serial: bool,
    stoch_layout: str,
) -> ResidualDeepONetDataset:
    return ResidualDeepONetDataset(
        cache,
        idx,
        target="R_nom",
        trunk_set="full",
        n_freq=n_freq,
        serial_tf1d=serial,
        stoch_layout=stoch_layout,
        query_split=query_split,
        support_stride=support_stride,
    )


@torch.no_grad()
def interpolate_ship_p(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    train_split: str,
    hold_split: str,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    freq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate shipped branch ``p`` from train stations onto hold-out x."""
    model.eval()
    apply_query_freq(model, freq)
    t_mean = stats["target_mean"].to(device)
    t_std = stats["target_std"].to(device)
    r_true, r_hat, tf1d_a, tf2d_a = [], [], [], []
    core = gno_core(model)
    for batch in loader:
        fields = batch["fields"].to(device)
        stoch = batch["stoch"].to(device)
        trunk_y = batch["trunk_y"].to(device)
        n_all = int(batch["query_x"].shape[-1]) if "query_x" in batch else fields.shape[-1]
        train_idx = query_station_indices(n_all, train_split)
        hold_idx = query_station_indices(n_all, hold_split)
        _forward(
            model,
            fields,
            stoch,
            trunk_y,
            "single",
            geom_flags=batch.get("geom_flags"),
            rH=batch.get("rH"),
            query_x=batch.get("query_x"),
            support_x=batch.get("support_x"),
        )
        p = getattr(core, "last_branch", None)
        bq = getattr(core, "last_trunk", None)
        if p is None or bq is None:
            raise RuntimeError("ship model did not expose last_branch / last_trunk")
        n_freq = bq.shape[1] // n_all
        x = batch["query_x"].to(device=p.device, dtype=p.dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0).expand(p.shape[0], -1)
        p_tr = p[:, train_idx]
        p_h = interp_along_x(p_tr, x[0, train_idx], x[0, hold_idx])
        b_h = bq.view(p.shape[0], n_all, n_freq, -1)[:, hold_idx]
        r_n = (p_h.unsqueeze(2) * b_h).sum(dim=-1)
        bias = getattr(core, "bias", None)
        if bias is not None:
            r_n = r_n + bias.reshape(-1)[0]
        r = (r_n * t_std + t_mean).reshape(p.shape[0], len(hold_idx), n_freq)
        tf1d = batch["tf1d"].cpu().numpy().reshape(p.shape[0], n_all, n_freq)[:, hold_idx]
        tf2d = batch["tf2d"].cpu().numpy().reshape(p.shape[0], n_all, n_freq)[:, hold_idx]
        tgt = batch["target_raw"].cpu().numpy().reshape(p.shape[0], n_all, n_freq)[:, hold_idx]
        r_hat.append(r.detach().cpu().numpy())
        r_true.append(tgt)
        tf1d_a.append(tf1d)
        tf2d_a.append(tf2d)
    return (
        np.concatenate(r_true, 0),
        np.concatenate(r_hat, 0),
        np.concatenate(tf1d_a, 0),
        np.concatenate(tf2d_a, 0),
    )


def _collect_holdout(
    model: torch.nn.Module | None,
    ds: ResidualDeepONetDataset,
    blob_stats: dict[str, torch.Tensor] | None,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray] | None:
    if model is None or blob_stats is None:
        return None
    apply_checkpoint_stats(blob_stats, ds)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    mets = evaluate(
        model,
        loader,
        device,
        "single",
        blob_stats,
        n_rec=ds.n_rec,
        n_freq=len(ds.f_idx),
        freq=ds.freq_s,
    )
    r_true, r_hat, tf1d, tf2d = [], [], [], []
    t_mean = blob_stats["target_mean"].to(device)
    t_std = blob_stats["target_std"].to(device)
    model.eval()
    apply_query_freq(model, ds.freq_s)
    with torch.no_grad():
        for batch in loader:
            pred_n = _forward(
                model,
                batch["fields"].to(device),
                batch["stoch"].to(device),
                batch["trunk_y"].to(device),
                "single",
                geom_flags=batch.get("geom_flags"),
                rH=batch.get("rH"),
                query_x=batch.get("query_x"),
                support_x=batch.get("support_x"),
            )
            pred = (pred_n * t_std + t_mean).cpu().numpy()
            n_rec, n_freq = ds.n_rec, len(ds.f_idx)
            r_hat.append(pred.reshape(-1, n_rec, n_freq))
            r_true.append(batch["target_raw"].numpy().reshape(-1, n_rec, n_freq))
            tf1d.append(batch["tf1d"].numpy().reshape(-1, n_rec, n_freq))
            tf2d.append(batch["tf2d"].numpy().reshape(-1, n_rec, n_freq))
    return {
        "metrics": mets,
        "r_true": np.concatenate(r_true, 0),
        "r_hat": np.concatenate(r_hat, 0),
        "tf1d": np.concatenate(tf1d, 0),
        "tf2d": np.concatenate(tf2d, 0),
    }


def score_domain(
    name: str,
    cache: Path,
    idx: np.ndarray,
    *,
    hold_split: str,
    train_split: str,
    kernel_model,
    kernel_stats,
    ship_model,
    ship_stats,
    serial: bool,
    stoch_layout: str,
    support_stride: int,
    n_freq: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    hold_ds = _ds(
        cache,
        idx,
        query_split=hold_split,
        support_stride=support_stride,
        n_freq=n_freq,
        serial=serial,
        stoch_layout=stoch_layout,
    )
    full_ds = _ds(
        cache,
        idx,
        query_split="all",
        support_stride=0,
        n_freq=n_freq,
        serial=serial,
        stoch_layout=stoch_layout,
    )
    n_rec = hold_ds.n_rec
    n_freq_g = len(hold_ds.f_idx)
    freq = np.asarray(hold_ds.freq_s)
    x_m = np.asarray(hold_ds.query_x, dtype=np.float64)

    out: dict[str, Any] = {"n": len(hold_ds), "n_rec": n_rec, "hold_split": hold_split}

    kern = _collect_holdout(
        kernel_model, hold_ds, kernel_stats, device=device, batch_size=batch_size
    )
    if kern is not None:
        extra = _metrics_from_arrays(
            kern["r_true"],
            kern["r_hat"],
            kern["tf2d"],
            kern["tf1d"],
            freq,
            n_rec=n_rec,
            n_freq=n_freq_g,
        )
        out["kernel"] = {**kern["metrics"], **extra}
        out["kernel_per_recorder"] = _per_recorder_rel_l2(
            kern["r_true"], kern["r_hat"], x_m
        )

    key = "target_raw" if "target_raw" in hold_ds._cache[0] else "target"
    r_true = np.stack([it[key].numpy() for it in hold_ds._cache]).reshape(
        len(hold_ds), n_rec, n_freq_g
    )
    tf1d = np.stack([it["tf1d"].numpy() for it in hold_ds._cache]).reshape(
        len(hold_ds), n_rec, n_freq_g
    )
    tf2d = np.stack([it["tf2d"].numpy() for it in hold_ds._cache]).reshape(
        len(hold_ds), n_rec, n_freq_g
    )
    haskell_r = np.zeros_like(r_true)
    out["haskell"] = _metrics_from_arrays(
        r_true, haskell_r, tf2d, tf1d, freq, n_rec=n_rec, n_freq=n_freq_g
    )

    if ship_model is not None and ship_stats is not None:
        apply_checkpoint_stats(ship_stats, full_ds)
        loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        rt, rh, t1, t2 = interpolate_ship_p(
            ship_model,
            loader,
            train_split=train_split,
            hold_split=hold_split,
            stats=ship_stats,
            device=device,
            freq=freq,
        )
        out["interpolate_p"] = _metrics_from_arrays(
            rt, rh, t2, t1, freq, n_rec=n_rec, n_freq=n_freq_g
        )
        out["interpolate_p_per_recorder"] = _per_recorder_rel_l2(rt, rh, x_m)

    k_p = (out.get("kernel") or {}).get("pearson_R_freq")
    i_p = (out.get("interpolate_p") or {}).get("pearson_R_freq")
    out["beats_interpolate_p"] = bool(
        k_p is not None and i_p is not None and float(k_p) > float(i_p)
    )
    return out


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Spatial query hold-out",
        "",
        f"Hold-out split: `{report.get('hold_out')}` "
        f"(train queries `{report.get('train_split')}`).",
        "",
        "Kill rule: odd-station Pearson must beat interpolating shipped `p`.",
        "",
        "| Domain | kernel Pearson_R_freq | interpolate-p | Haskell | beats interp-p |",
        "|--------|----------------------:|--------------:|--------:|:--------------:|",
    ]
    for dname, rec in (report.get("domains") or {}).items():
        k = (rec.get("kernel") or {}).get("pearson_R_freq")
        i = (rec.get("interpolate_p") or {}).get("pearson_R_freq")
        h = (rec.get("haskell") or {}).get("pearson_R_freq")
        beat = rec.get("beats_interpolate_p")
        def _f(x):
            return "—" if x is None else f"{float(x):.3f}"

        lines.append(
            f"| {dname} | {_f(k)} | {_f(i)} | {_f(h)} | "
            f"{'yes' if beat else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=None, help="Kernel-query checkpoint.")
    p.add_argument(
        "--ship-ckpt",
        type=Path,
        default=config.DEFAULT_CHECKPOINT,
        help="Shipped GINO for interpolate-p.",
    )
    p.add_argument(
        "--hold-out",
        choices=["odd", "edge"],
        default="odd",
        help="Stations to score. odd pairs with even train; edge with interior.",
    )
    p.add_argument("--n-freq", type=int, default=config.N_FREQ_EVAL)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--support-stride", type=int, default=config.SUPPORT_STRIDE)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON path (default results/arch_train/spatial_query_<hold>.json).",
    )
    args = p.parse_args()
    train_split = TRAIN_FROM_HOLD[args.hold_out]
    device = _device()
    kernel_model = kernel_stats = None
    serial = True
    stoch_layout = "xi_cov"
    if args.ckpt is not None and args.ckpt.is_file():
        kernel_model, kblob, kernel_stats, _ = _load_model(args.ckpt, device)
        serial = bool(kblob.get("serial_tf1d", True))
        stoch_layout = str(kblob.get("stoch_layout", "xi_cov"))
        kernel_stats = stats_from_checkpoint(kblob)
    ship_model = ship_stats = None
    if args.ship_ckpt.is_file():
        ship_model, sblob, _, _ = _load_model(args.ship_ckpt, device)
        ship_stats = stats_from_checkpoint(sblob)
        if kernel_model is None:
            serial = bool(sblob.get("serial_tf1d", True))
            stoch_layout = str(sblob.get("stoch_layout", "xi_cov"))

    tests = mix_test_parts()
    domains = {}
    for dname, (cache, idx) in tests.items():
        if not (cache / "r_nom_signed.npy").is_file():
            print(f"[spatial-query] skip {dname}: missing {cache}", flush=True)
            continue
        print(f"[spatial-query] {dname} hold={args.hold_out} n={len(idx)}", flush=True)
        domains[dname] = score_domain(
            dname,
            cache,
            idx,
            hold_split=args.hold_out,
            train_split=train_split,
            kernel_model=kernel_model,
            kernel_stats=kernel_stats,
            ship_model=ship_model,
            ship_stats=ship_stats,
            serial=serial,
            stoch_layout=stoch_layout,
            support_stride=args.support_stride if kernel_model is not None else 0,
            n_freq=args.n_freq,
            batch_size=args.batch_size,
            device=device,
        )
    iid = domains.get("iid") or {}
    kill = bool(iid) and not bool(iid.get("beats_interpolate_p", False))
    report = {
        "hold_out": args.hold_out,
        "train_split": train_split,
        "ckpt": None if args.ckpt is None else str(args.ckpt),
        "ship_ckpt": str(args.ship_ckpt),
        "kill_keep_ship": kill,
        "domains": domains,
    }
    out = args.out or (OUT_DIR / f"spatial_query_{args.hold_out}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=2))
    md = out.with_suffix(".md")
    write_markdown(report, md)
    print(json.dumps(_jsonable({"out": str(out), "kill_keep_ship": kill}), indent=2))
    if kill:
        print(
            "[spatial-query] KILL: hold-out Pearson is not better than interpolate-p; "
            "keep checkpoints/M7680_gino_rebal_ft.pt",
            flush=True,
        )


if __name__ == "__main__":
    main()
