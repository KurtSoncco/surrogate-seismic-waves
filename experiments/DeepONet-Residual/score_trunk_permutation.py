#!/usr/bin/env python3
"""Eval-only trunk-channel permutation on a frozen leftover checkpoint.

Shuffles one trunk channel at a time across the sample axis (``f*``, ``sin``,
``cos``, ``x/λ``, ``log TF1D`` when serial). Default target is the M1400 T4B4
ξ+CoV control. Scores central-recorder Pearson / Anderson on IID and dipping.

    python score_trunk_permutation.py
    python score_trunk_permutation.py --ckpt checkpoints/m1400_gino_fno_mscaleT4B4_xicov.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from mix_ladder import mix_test_parts  # noqa: E402
from score_mscale_distributions import (  # noqa: E402
    CENTRAL_REC,
    gof_af_per_sample,
    load_domain_arrays,
    pearson_freq_per_sample,
    summarize_values,
)
from score_stoch_permutation import (  # noqa: E402
    _predict,
    _public_metrics,
    delta_pack,
    rel_l2_per_sample,
)

PERM_SEED = 0
CONTROL_CKPT = config.CHECKPOINT_DIR / "m1400_gino_fno_mscaleT4B4_xicov.pt"
DOMAINS = ("iid", "ood_dipping")
BASE_TRUNK_NAMES = ("f_star", "sin_f", "cos_f", "x_over_lambda")


def trunk_channel_names(*, serial: bool, trunk_dim: int) -> list[str]:
    names = list(BASE_TRUNK_NAMES)
    if serial:
        names.append("log_tf1d")
    while len(names) < int(trunk_dim):
        names.append(f"trunk_{len(names)}")
    return names[: int(trunk_dim)]


def shuffle_trunk_channel(
    trunk: np.ndarray, channel: int, perm: np.ndarray
) -> np.ndarray:
    """Copy ``trunk`` (N, Q, D) and permute ``channel`` along the sample axis."""
    out = np.array(trunk, copy=True, dtype=np.float32)
    perm = np.asarray(perm, dtype=int)
    if perm.shape != (out.shape[0],):
        raise ValueError(f"perm shape {perm.shape} != ({out.shape[0]},)")
    if not (0 <= int(channel) < out.shape[-1]):
        raise ValueError(f"channel {channel} out of range for dim {out.shape[-1]}")
    out[:, :, int(channel)] = out[perm][:, :, int(channel)]
    return out


def _stack_trunk(ds) -> torch.Tensor:
    return torch.stack([item["trunk_y"] for item in ds._cache], dim=0).clone()


def _set_trunk(ds, trunk: torch.Tensor) -> None:
    for i in range(len(ds._cache)):
        ds._cache[i]["trunk_y"] = trunk[i].clone()


def _metrics_from_hats(
    freq: np.ndarray, tf_ops: np.ndarray, hats: np.ndarray
) -> dict[str, Any]:
    pearson = pearson_freq_per_sample(tf_ops, hats, recorder=CENTRAL_REC)
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

    serial = bool(blob.get("serial_tf1d", True))
    ds = ResidualDeepONetDataset(
        cache_dir,
        test_idx,
        target="R_nom",
        trunk_set=trunk_set,
        n_freq=config.N_FREQ_EVAL,
        serial_tf1d=serial,
        **dataset_kwargs_from_blob(blob),
    )
    apply_norms(ds, stats)
    apply_query_freq(model, getattr(ds, "freq_s", None))
    base_trunk = _stack_trunk(ds)
    trunk_np = base_trunk.cpu().numpy()
    names = trunk_channel_names(serial=serial, trunk_dim=int(trunk_np.shape[-1]))
    rng = np.random.default_rng(PERM_SEED)
    perm = rng.permutation(int(trunk_np.shape[0]))
    tf_ops = arrays["tf_opensees"]
    freq = arrays["freq"]

    ops_out: dict[str, Any] = {}
    baseline_raw: dict[str, np.ndarray] | None = None
    ops: list[tuple[str, int | None]] = [("baseline", None)]
    ops.extend((f"shuffle_{name}", i) for i, name in enumerate(names))
    for op_name, channel in ops:
        if channel is None:
            mutated = trunk_np
        else:
            mutated = shuffle_trunk_channel(trunk_np, channel, perm)
        _set_trunk(ds, torch.from_numpy(np.ascontiguousarray(mutated)))
        hats = _predict(model, blob, stats, ds, device, batch_size)
        rec = _metrics_from_hats(freq, tf_ops, hats)
        public = _public_metrics(rec)
        if channel is None:
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


def run(*, ckpt: Path, batch_size: int) -> dict[str, Any]:
    from eval_ood import _load_residual_model
    from train import _device

    device = _device()
    print(f"device={device}  ckpt={ckpt}", flush=True)
    tests = mix_test_parts()
    model, blob, stats, trunk_set = _load_residual_model(ckpt, device)
    serial = bool(blob.get("serial_tf1d", True))
    domains_out: dict[str, Any] = {}
    for domain in DOMAINS:
        cache, idx = tests[domain]
        arrays = load_domain_arrays(cache, idx)
        print(f"=== {domain} n={len(idx)} ===", flush=True)
        domains_out[domain] = score_ckpt_domain(
            model=model,
            blob=blob,
            stats=stats,
            trunk_set=trunk_set,
            cache_dir=cache,
            test_idx=idx,
            arrays=arrays,
            batch_size=batch_size,
            device=device,
        )
    return {
        "ckpt": str(ckpt),
        "serial_tf1d": serial,
        "n_mscale_trunk": int(blob.get("n_mscale_trunk", 1)),
        "n_mscale_branch": int(blob.get("n_mscale_branch", 1)),
        "pearson_recorder": int(CENTRAL_REC),
        "domains": domains_out,
    }


def _print_brief(blob: dict[str, Any]) -> None:
    print(f"\n{'dom':<16} {'op':<22} {'P mean':>8} {'ΔP':>8} {'A med':>8} {'ΔA':>8}")
    for domain, ops in blob.get("domains", {}).items():
        for op_name, rec in ops.items():
            p = rec["pearson_freq"]
            g = rec["gof_af"]
            d = rec.get("delta_vs_baseline", {})
            dp = d.get("pearson_freq", {}).get("mean_delta", 0.0)
            dg = d.get("gof_af", {}).get("median_delta", 0.0)
            print(
                f"{domain:<16} {op_name:<22} "
                f"{p['mean']:8.4f} {dp:+8.4f} {g['p50']:8.4f} {dg:+8.4f}"
            )


def _strip_values(blob: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(blob))
    for ops in out.get("domains", {}).values():
        for rec in ops.values():
            for key in ("pearson_freq", "gof_af", "rel_l2_TF"):
                if key in rec and isinstance(rec[key], dict):
                    rec[key].pop("values", None)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ckpt", type=Path, default=CONTROL_CKPT)
    p.add_argument(
        "--out",
        type=Path,
        default=config.RESULTS_DIR / "arch_train" / "m1400_t4b4_trunkperm.json",
    )
    args = p.parse_args()
    if not args.ckpt.is_file():
        raise SystemExit(f"missing checkpoint {args.ckpt}")
    blob = run(ckpt=args.ckpt, batch_size=args.batch_size)
    _print_brief(blob)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2))
    compact = args.out.with_name(args.out.stem + "_compact.json")
    compact.write_text(json.dumps(_strip_values(blob), indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {compact}")


if __name__ == "__main__":
    main()
