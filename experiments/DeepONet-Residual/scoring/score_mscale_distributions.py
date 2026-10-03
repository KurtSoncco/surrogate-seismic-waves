#!/usr/bin/env python3
"""Shared nested-test scoring helpers (Pearson / Anderson GOF / TF arrays).

Permutation scripts and leftover tests import this module. Default checkpoint
map is the shipped GINO rebal FT only; extra arms can be passed with ``--arm``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from mix_ladder import mix_test_parts  # noqa: E402
from response_variability.metrics import (  # noqa: E402
    anderson_frequency_domain,
    pearson as pearson_1d,
)

CENTRAL_REC = config.N_LATERAL // 2
DOMAINS = ("iid", "ood_dipping", "ood_three_layer")
DEFAULT_SCORE_DOMAINS = ("iid", "ood_dipping")
DEFAULT_ARMS = (("shipped_M7680", "M7680_gino_rebal_ft.pt"),)
SHIPPED = "shipped_M7680"


def summarize_values(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    qs = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "values": [float(v) for v in x],
    }


def pearson_freq_per_sample(
    tf_true: np.ndarray,
    tf_pred: np.ndarray,
    *,
    recorder: int | None = None,
) -> np.ndarray:
    """Pearson of |TF|(f). Default: mean over recorders. Pass ``recorder`` for one station."""
    y = np.asarray(tf_true, dtype=np.float64)
    p = np.asarray(tf_pred, dtype=np.float64)
    if p.ndim == 2:
        p = np.broadcast_to(p[:, None, :], y.shape)
    n = y.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    recs = (recorder,) if recorder is not None else range(int(y.shape[1]))
    for i in range(n):
        cors: list[float] = []
        for r in recs:
            c = pearson_1d(p[i, r], y[i, r])
            if np.isfinite(c):
                cors.append(c)
        if cors:
            out[i] = float(np.mean(cors))
    return out


def gof_af_per_sample(
    freq: np.ndarray,
    tf_true: np.ndarray,
    tf_pred: np.ndarray,
    *,
    central: int = CENTRAL_REC,
    f0_2d: np.ndarray | None = None,
) -> np.ndarray:
    """Anderson ln|TF| L1 weighted at the 2D-TF fundamental (lower better)."""
    y = np.asarray(tf_true, dtype=np.float64)
    p = np.asarray(tf_pred, dtype=np.float64)
    n = y.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    f0 = (
        np.asarray(f0_2d, dtype=np.float64).ravel()
        if f0_2d is not None
        else np.full(n, np.nan)
    )
    for i in range(n):
        ref = y[i, central] if y.ndim == 3 else y[i]
        if p.ndim == 3:
            cand = p[i, central]
        elif p.ndim == 2:
            cand = p[i]
        else:
            continue
        center = float(f0[i]) if i < f0.size else float("nan")
        out[i] = anderson_frequency_domain(
            freq,
            ref,
            cand,
            f_weight_center=center if np.isfinite(center) else None,
            f_weight_width=1.5,
        )
    return out


def load_domain_arrays(cache_dir: Path, test_idx: np.ndarray) -> dict[str, np.ndarray]:
    test_idx = np.asarray(test_idx, dtype=int)
    cache_dir = Path(cache_dir)
    tf2d_path = cache_dir / "tf2d.npy"
    if tf2d_path.is_file():
        tf_ops = np.asarray(
            np.load(tf2d_path, mmap_mode="r")[test_idx], dtype=np.float64
        )
    elif config.TF_PER_SAMPLE_PATH.is_file():
        tf_all = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
        sidx = np.load(cache_dir / "sample_indices.npy")[test_idx]
        tf_ops = np.asarray(tf_all[sidx], dtype=np.float64)
    else:
        raise FileNotFoundError(
            f"{cache_dir} has no tf2d.npy and {config.TF_PER_SAMPLE_PATH} is missing"
        )
    freq_path = cache_dir / "freq.npy"
    freq = np.load(freq_path if freq_path.is_file() else config.TF_FREQ_PATH)
    return {
        "tf_opensees": tf_ops,
        "tf_haskell_nominal": np.asarray(
            np.load(cache_dir / "tf1d_nom.npy", mmap_mode="r")[test_idx],
            dtype=np.float64,
        ),
        "freq": np.asarray(freq, dtype=float),
        "local_idx": test_idx,
    }


def default_ckpt_map() -> dict[str, Path]:
    return {name: config.CHECKPOINT_DIR / fname for name, fname in DEFAULT_ARMS}


def run(
    *, ckpts: dict[str, Path], batch_size: int, include_haskell: bool
) -> dict[str, Any]:
    from response_variability.evals.eval_iid import predict_gino

    tests = mix_test_parts()
    arrays_by = {d: load_domain_arrays(cache, idx) for d, (cache, idx) in tests.items()}
    present = {n: p for n, p in ckpts.items() if p.is_file()}
    missing = [n for n, p in ckpts.items() if not p.is_file()]
    domains_out: dict[str, dict[str, Any]] = {d: {} for d in DOMAINS}
    for domain, (cache, idx) in tests.items():
        arrays = arrays_by[domain]
        tf_ops, freq = arrays["tf_opensees"], arrays["freq"]
        print(f"=== {domain} n={len(idx)} ===", flush=True)
        if include_haskell:
            rec = {
                "pearson_freq": summarize_values(
                    pearson_freq_per_sample(tf_ops, arrays["tf_haskell_nominal"])
                ),
                "gof_af": summarize_values(
                    gof_af_per_sample(freq, tf_ops, arrays["tf_haskell_nominal"])
                ),
                "n": int(tf_ops.shape[0]),
                "ckpt": None,
            }
            domains_out[domain]["haskell_nom"] = rec
        for name, ckpt in present.items():
            print(f"  {name}  {ckpt.name}", flush=True)
            hats = predict_gino(
                cache_dir=cache, test_idx=idx, ckpt_path=ckpt, batch_size=batch_size
            )
            domains_out[domain][name] = {
                "pearson_freq": summarize_values(pearson_freq_per_sample(tf_ops, hats)),
                "gof_af": summarize_values(gof_af_per_sample(freq, tf_ops, hats)),
                "n": int(tf_ops.shape[0]),
                "ckpt": str(ckpt),
            }
    return {
        "n_by_domain": {d: int(len(tests[d][1])) for d in tests},
        "missing_ckpts": missing,
        "domains": domains_out,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--out",
        type=Path,
        default=config.RESULTS_DIR / "arch_train" / "nested_distributions.json",
    )
    p.add_argument("--no-haskell", action="store_true")
    p.add_argument("--arm", action="append", default=[], metavar="NAME=PATH")
    args = p.parse_args()
    ckpts = default_ckpt_map()
    for spec in args.arm:
        name, path = spec.split("=", 1)
        ckpts[name] = Path(path)
    blob = run(
        ckpts=ckpts, batch_size=args.batch_size, include_haskell=not args.no_haskell
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
