#!/usr/bin/env python3
"""Permutation importance of ξ vs RF scalars on leftover GINO / Mscale.

The stochastic branch is ``ξ (2 K_XI) + CoV`` (17-d). Old leftover
checkpoints still use ``ξ + [rH, aHV, CoV, ξ_damp]`` (20-d); layout is
inferred from ``stoch_mlp`` in-features.

    python scoring/score_stoch_permutation.py --batch-size 16
    python scoring/score_stoch_permutation.py --arm mscale_T4B4=checkpoints/iid2000_gino_fno_mscaleT4B4.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from mix_ladder import mix_test_parts  # noqa: E402
from scoring.score_mscale_distributions import (  # noqa: E402
    default_ckpt_map,
    gof_af_per_sample,
    load_domain_arrays,
    pearson_freq_per_sample,
    summarize_values,
)

CONTROL = "iid2000_gino"
SHIPPED = "shipped_M7680"
PERM_SEED = 0


def stoch_slices(k_xi: int = config.K_XI, layout: str = "xi_cov") -> dict[str, slice]:
    """Named views into the stochastic vector."""
    k2 = 2 * int(k_xi)
    if layout == "legacy20":
        return {
            "xi": slice(0, k2),
            "rH": slice(k2, k2 + 1),
            "aHV": slice(k2 + 1, k2 + 2),
            "CoV": slice(k2 + 2, k2 + 3),
            "xi_damp": slice(k2 + 3, k2 + 4),
            "scalars": slice(k2, k2 + 4),
            "all": slice(0, k2 + 4),
        }
    if layout == "cov_only":
        return {
            "CoV": slice(0, 1),
            "all": slice(0, 1),
        }
    return {
        "xi": slice(0, k2),
        "CoV": slice(k2, k2 + 1),
        "all": slice(0, k2 + 1),
    }


def stoch_channel_names(k_xi: int = config.K_XI, layout: str = "xi_cov") -> list[str]:
    names = [f"xi{i}_{p}" for i in range(int(k_xi)) for p in ("re", "im")]
    if layout == "legacy20":
        names.extend(["rH", "aHV", "CoV", "xi_damp"])
    elif layout == "cov_only":
        return ["CoV"]
    else:
        names.append("CoV")
    return names


def apply_channel_op(
    stoch: np.ndarray,
    sl: slice,
    *,
    op: str,
    perm: np.ndarray | None = None,
) -> np.ndarray:
    """Copy ``stoch`` and shuffle or zero ``sl`` along the sample axis."""
    out = np.array(stoch, copy=True, dtype=np.float32)
    if op == "identity":
        return out
    if op == "zero":
        out[:, sl] = 0.0
        return out
    if op == "shuffle":
        if perm is None:
            raise ValueError("shuffle requires a permutation of the sample axis")
        perm = np.asarray(perm, dtype=int)
        if perm.shape != (out.shape[0],):
            raise ValueError(f"perm shape {perm.shape} != ({out.shape[0]},)")
        out[:, sl] = out[perm][:, sl]
        return out
    raise ValueError(f"unknown op {op!r}")


def permutation_ops(layout: str = "xi_cov") -> list[tuple[str, str, str]]:
    """(op_name, kind, slice_name). Baseline is identity on the full vector."""
    if layout == "legacy20":
        return [
            ("baseline", "identity", "all"),
            ("shuffle_xi", "shuffle", "xi"),
            ("shuffle_rH", "shuffle", "rH"),
            ("shuffle_aHV", "shuffle", "aHV"),
            ("shuffle_CoV", "shuffle", "CoV"),
            ("shuffle_xi_damp", "shuffle", "xi_damp"),
            ("shuffle_scalars", "shuffle", "scalars"),
            ("shuffle_all", "shuffle", "all"),
            ("zero_scalars", "zero", "scalars"),
            ("zero_all", "zero", "all"),
        ]
    if layout == "cov_only":
        return [
            ("baseline", "identity", "all"),
            ("shuffle_CoV", "shuffle", "CoV"),
            ("shuffle_all", "shuffle", "all"),
            ("zero_all", "zero", "all"),
        ]
    return [
        ("baseline", "identity", "all"),
        ("shuffle_xi", "shuffle", "xi"),
        ("shuffle_CoV", "shuffle", "CoV"),
        ("shuffle_all", "shuffle", "all"),
        ("zero_all", "zero", "all"),
    ]


def rel_l2_per_sample(tf_true: np.ndarray, tf_pred: np.ndarray) -> np.ndarray:
    y = np.asarray(tf_true, dtype=np.float64).reshape(len(tf_true), -1)
    p = np.asarray(tf_pred, dtype=np.float64).reshape(len(tf_pred), -1)
    num = np.linalg.norm(p - y, axis=1)
    den = np.maximum(np.linalg.norm(y, axis=1), 1e-12)
    return num / den


def delta_pack(
    base: np.ndarray, treat: np.ndarray, *, higher_better: bool
) -> dict[str, Any]:
    a = np.asarray(base, dtype=np.float64)
    b = np.asarray(treat, dtype=np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size == 0:
        return {"n": 0}
    diff = b - a
    wins = (b > a) if higher_better else (b < a)
    return {
        "n": int(a.size),
        "mean_delta": float(np.mean(diff)),
        "median_delta": float(np.median(diff)),
        "win_rate": float(np.mean(wins)),
    }


def _metrics_from_hats(
    freq: np.ndarray, tf_ops: np.ndarray, hats: np.ndarray
) -> dict[str, Any]:
    pearson = pearson_freq_per_sample(tf_ops, hats)
    gof = gof_af_per_sample(freq, tf_ops, hats)
    rel = rel_l2_per_sample(tf_ops, hats)
    return {
        "pearson_freq": summarize_values(pearson),
        "gof_af": summarize_values(gof),
        "rel_l2_TF": summarize_values(rel),
        "_pearson": pearson,
        "_gof": gof,
        "_rel": rel,
    }


def _public_metrics(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "pearson_freq": rec["pearson_freq"],
        "gof_af": rec["gof_af"],
        "rel_l2_TF": rec["rel_l2_TF"],
    }


def stoch_weight_profile(model) -> dict[str, Any] | None:
    from model import gno_core

    core = gno_core(model)
    mlp = getattr(core, "stoch_mlp", None)
    if mlp is None or not hasattr(mlp, "__getitem__"):
        return None
    linear = mlp[0]
    w = linear.weight.detach().float().cpu().abs()
    col = w.mean(0).numpy()
    layout = (
        "legacy20"
        if int(col.size) == 2 * config.K_XI + 4
        else ("cov_only" if int(col.size) == 1 else "xi_cov")
    )
    names = stoch_channel_names(layout=layout)
    if col.size != len(names):
        names = [f"c{i}" for i in range(int(col.size))]
    k2 = min(2 * config.K_XI, int(col.size))
    return {
        "per_channel_mean_abs": {n: float(v) for n, v in zip(names, col)},
        "xi_mean_abs": float(col[:k2].mean()) if k2 else None,
        "scalars_mean_abs": float(col[k2:].mean()) if col.size > k2 else None,
    }


def _stack_stoch(ds) -> torch.Tensor:
    return torch.stack([item["stoch"] for item in ds._cache], dim=0).clone()


def _set_stoch(ds, stoch: torch.Tensor) -> None:
    for i in range(len(ds._cache)):
        ds._cache[i]["stoch"] = stoch[i].clone()


def _predict(model, blob, stats, ds, device, batch_size: int) -> np.ndarray:
    from torch.utils.data import DataLoader

    from train import _forward
    from unified_metrics import tf_from_residual

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    t_mean = stats["target_mean"].to(device)
    t_std = stats["target_std"].to(device)
    mode = blob.get("branch_mode", "single")
    log_residual = bool(blob.get("log_residual", False))
    hats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pred_n = _forward(
                model,
                batch["fields"].to(device),
                batch["stoch"].to(device),
                batch["trunk_y"].to(device),
                mode,
                geom_flags=batch.get("geom_flags"),
            )
            pred = (pred_n * t_std + t_mean).cpu().numpy()
            hats.append(
                tf_from_residual(batch["tf1d"].numpy(), pred, log_residual=log_residual)
            )
    n_rec, n_freq = ds.n_rec, len(ds.f_idx)
    return np.concatenate(hats, axis=0).reshape(len(ds), n_rec, n_freq)


def score_ckpt_domain(
    *,
    model,
    blob: dict[str, Any],
    stats: dict[str, Any],
    trunk_set: str,
    cache_dir: Path,
    test_idx: np.ndarray,
    arrays: dict[str, np.ndarray],
    batch_size: int,
    device,
) -> dict[str, Any]:
    from data import ResidualDeepONetDataset, dataset_kwargs_from_blob
    from model import apply_query_freq
    from train import apply_norms

    ds = ResidualDeepONetDataset(
        cache_dir,
        test_idx,
        target="R_nom",
        trunk_set=trunk_set,
        n_freq=config.N_FREQ_EVAL,
        serial_tf1d=bool(blob.get("serial_tf1d", True)),
        **dataset_kwargs_from_blob(blob),
    )
    apply_norms(ds, stats)
    apply_query_freq(model, getattr(ds, "freq_s", None))
    base_stoch = _stack_stoch(ds)
    layout = str(blob.get("stoch_layout", "xi_cov"))
    slices = stoch_slices(layout=layout)
    rng = np.random.default_rng(PERM_SEED)
    n = int(base_stoch.shape[0])
    perm = rng.permutation(n)
    tf_ops = arrays["tf_opensees"]
    freq = arrays["freq"]

    ops_out: dict[str, Any] = {}
    baseline_raw: dict[str, np.ndarray] | None = None
    for op_name, kind, slice_name in permutation_ops(layout=layout):
        mutated = apply_channel_op(
            base_stoch.cpu().numpy(),
            slices[slice_name],
            op=kind,
            perm=perm,
        )
        _set_stoch(ds, torch.from_numpy(mutated))
        hats = _predict(model, blob, stats, ds, device, batch_size)
        rec = _metrics_from_hats(freq, tf_ops, hats)
        public = _public_metrics(rec)
        if kind == "identity":
            baseline_raw = {
                "pearson": rec["_pearson"],
                "gof": rec["_gof"],
                "rel": rec["_rel"],
            }
        elif baseline_raw is not None:
            public["delta_vs_baseline"] = {
                "pearson_freq": delta_pack(
                    baseline_raw["pearson"], rec["_pearson"], higher_better=True
                ),
                "gof_af": delta_pack(
                    baseline_raw["gof"], rec["_gof"], higher_better=False
                ),
                "rel_l2_TF": delta_pack(
                    baseline_raw["rel"], rec["_rel"], higher_better=False
                ),
            }
        ops_out[op_name] = public
        p, g = rec["pearson_freq"], rec["gof_af"]
        dp = public.get("delta_vs_baseline", {}).get("pearson_freq", {})
        dg = public.get("delta_vs_baseline", {}).get("gof_af", {})
        print(
            f"    {op_name:<22} P={p['mean']:.4f} "
            f"ΔP={dp.get('mean_delta', 0.0):+.4f}  "
            f"A={g['p50']:.4f} ΔA={dg.get('median_delta', 0.0):+.4f}",
            flush=True,
        )
    return ops_out


def run(*, ckpts: dict[str, Path], batch_size: int) -> dict[str, Any]:
    from train import _device

    device = _device()
    print(f"device={device}", flush=True)
    tests = mix_test_parts()
    arrays_by = {d: load_domain_arrays(cache, idx) for d, (cache, idx) in tests.items()}
    present = {n: p for n, p in ckpts.items() if p.is_file()}
    missing = [n for n, p in ckpts.items() if not p.is_file()]
    from scoring.eval_ood import _load_residual_model

    arms_out: dict[str, Any] = {}
    for name, ckpt in present.items():
        print(f"=== {name}  {ckpt.name} ===", flush=True)
        model, blob, stats, trunk_set = _load_residual_model(ckpt, device)
        weights = stoch_weight_profile(model)
        domains_out: dict[str, Any] = {}
        for domain, (cache, idx) in tests.items():
            print(f"  {domain} n={len(idx)}", flush=True)
            domains_out[domain] = score_ckpt_domain(
                model=model,
                blob=blob,
                stats=stats,
                trunk_set=trunk_set,
                cache_dir=cache,
                test_idx=idx,
                arrays=arrays_by[domain],
                batch_size=batch_size,
                device=device,
            )
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        arms_out[name] = {
            "ckpt": str(ckpt),
            "stoch_weight": weights,
            "domains": domains_out,
        }
    return {
        "k_xi": int(config.K_XI),
        "stoch_dim": int(2 * config.K_XI + 4),
        "perm_seed": PERM_SEED,
        "ops": [op for op, _, _ in permutation_ops()],
        "n_by_domain": {d: int(len(tests[d][1])) for d in tests},
        "missing_ckpts": missing,
        "device": str(device),
        "note": (
            "Shuffle permutes z-scored channels across nested-test samples; "
            "zero replaces them with the train-split mean. Fields and trunk stay paired."
        ),
        "arms": arms_out,
    }


def _print_brief(blob: dict[str, Any]) -> None:
    print(
        f"\n{'arm':<16} {'dom':<16} {'op':<22} {'P mean':>8} {'ΔP':>8} {'A med':>8} {'ΔA':>8}"
    )
    for name, arm in blob.get("arms", {}).items():
        for domain, ops in arm.get("domains", {}).items():
            for op_name, rec in ops.items():
                p = rec["pearson_freq"]
                g = rec["gof_af"]
                d = rec.get("delta_vs_baseline", {})
                dp = d.get("pearson_freq", {}).get("mean_delta", 0.0)
                dg = d.get("gof_af", {}).get("median_delta", 0.0)
                print(
                    f"{name:<16} {domain:<16} {op_name:<22} "
                    f"{p['mean']:8.4f} {dp:+8.4f} {g['p50']:8.4f} {dg:+8.4f}"
                )


def _strip_values(blob: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(blob))
    for arm in out.get("arms", {}).values():
        for ops in arm.get("domains", {}).values():
            for rec in ops.values():
                for key in ("pearson_freq", "gof_af", "rel_l2_TF"):
                    if key in rec and isinstance(rec[key], dict):
                        rec[key].pop("values", None)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--out",
        type=Path,
        default=config.RESULTS_DIR / "arch_train" / "stoch_permutation.json",
    )
    p.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override/add an arm (repeatable). Default: IID2000 Mscale screen + shipped.",
    )
    args = p.parse_args()
    ckpts = default_ckpt_map()
    for spec in args.arm:
        name, path = spec.split("=", 1)
        ckpts[name] = Path(path)
    blob = run(ckpts=ckpts, batch_size=args.batch_size)
    _print_brief(blob)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2))
    compact = args.out.with_name(args.out.stem + "_compact.json")
    compact.write_text(json.dumps(_strip_values(blob), indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {compact}")


if __name__ == "__main__":
    main()
