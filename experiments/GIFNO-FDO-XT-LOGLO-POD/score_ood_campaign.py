#!/usr/bin/env python3
"""
Zero-shot score of Box OOD campaigns (ood_dipping / ood_three_layer).

Loads the surrogate once, caches ground-truth TFs, skips per-case plots by
default, and writes a stratified CSV + JSON summary.

Example:
  export GIFNO_MODEL_DIR=~/surrogate-seismic-waves/checkpoints/tier2_pod64_n2000
  export GIFNO_POD_NUM_MODES=64 GIFNO_LATENT_CHANNELS=128 GIFNO_NUM_FNO_LAYERS=5
  export GIFNO_DEEPONET_LATENT_DIM=128 SEISKIT_ROOT=~/seiskit

  cd experiments/GIFNO-FDO-XT-LOGLO-POD
  uv run python score_ood_campaign.py --campaign both
  uv run python score_ood_campaign.py --campaign dipping --limit 32  # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config

config.setup_import_paths()

from capability_check import (  # noqa: E402
    build_input_from_h5,
    compare_tfs,
    load_model,
    predict_tf,
)

DEFAULT_BOX = Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data")
DEFAULT_OUT = (
    Path.home() / "surrogate-seismic-waves" / "checkpoints" / "ood_campaign_scores"
)
_EPS = 1e-12


def _box_root() -> Path:
    env = os.environ.get("GIFNO_OOD_ROOT") or os.environ.get("GIFNO_DATA_ROOT")
    if env:
        p = Path(env)
        if (p / "ood_dipping").is_dir() or p.name.startswith("ood_"):
            return p if p.name.startswith("ood_") else p
        return p
    for cand in (
        DEFAULT_BOX,
        Path("/mnt/box_lab/Projects/Neural Operator/data"),
    ):
        if cand.is_dir():
            return cand
    return DEFAULT_BOX


def campaign_dirs(root: Path, campaign: str) -> list[tuple[str, Path]]:
    mapping = {
        "dipping": root / "ood_dipping",
        "three_layer": root / "ood_three_layer",
    }
    if campaign == "both":
        return [(k, v) for k, v in mapping.items() if v.is_dir()]
    if campaign not in mapping:
        raise ValueError(f"Unknown campaign {campaign!r}")
    return [(campaign, mapping[campaign])]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _gt_worker(args: tuple[str, str, str, bool]) -> tuple[str, str, float]:
    """Compute and cache GT TF in a worker process. Returns (slug, status, seconds)."""
    h5_str, cache_str, case_dir_str, force = args
    h5_path = Path(h5_str)
    cache_dir = Path(case_dir_str)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tf_path = cache_dir / "tf_true.npy"
    freq_path = cache_dir / "freq.npy"
    t0 = time.time()
    if not force and tf_path.is_file() and freq_path.is_file():
        return (cache_str, "cached", time.time() - t0)
    # Late import so workers pick up SEISKIT_ROOT
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as _cfg  # noqa: WPS433

    _cfg.setup_import_paths()
    from capability_check import compute_ground_truth_tf  # noqa: WPS433

    tf, freq = compute_ground_truth_tf(h5_path)
    np.save(tf_path, tf)
    np.save(freq_path, freq)
    return (cache_str, "computed", time.time() - t0)


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + _EPS))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def metrics_from_tfs(
    tf_pred: np.ndarray, tf_true: np.ndarray, freq: np.ndarray
) -> dict[str, float]:
    """Lightweight metrics (no plots)."""
    n_rec = tf_true.shape[0]
    rel = np.array(
        [_rel_l2(tf_pred[r], tf_true[r]) for r in range(n_rec)], dtype=np.float64
    )
    pear = np.array(
        [_pearson(tf_pred[r], tf_true[r]) for r in range(n_rec)], dtype=np.float64
    )
    log_rel = np.array(
        [
            _rel_l2(
                np.log(np.maximum(tf_pred[r], _EPS)),
                np.log(np.maximum(tf_true[r], _EPS)),
            )
            for r in range(n_rec)
        ],
        dtype=np.float64,
    )
    central = n_rec // 2
    out: dict[str, float] = {
        "rel_l2_mean": float(np.mean(rel)),
        "rel_l2_median": float(np.median(rel)),
        "rel_l2_central": float(rel[central]),
        "pearson_mean": float(np.nanmean(pear)),
        "pearson_median": float(np.nanmedian(pear)),
        "pearson_central": float(pear[central]),
        "logspec_rel_l2_mean": float(np.mean(log_rel)),
    }
    for name, band in (
        ("low", config.FREQ_BAND_LOW),
        ("mid", config.FREQ_BAND_MID),
        ("high", config.FREQ_BAND_HIGH),
    ):
        lo, hi = band
        mask = (freq >= lo) & (freq <= hi)
        if not np.any(mask):
            continue
        out[f"rel_l2_band_{name}_mean"] = float(
            np.mean([_rel_l2(tf_pred[r, mask], tf_true[r, mask]) for r in range(n_rec)])
        )
    return out


def stratify_summary(rows: list[dict[str, Any]], key: str, n_bins: int = 4) -> dict:
    """Quantile-bin continuous keys; group categorical as-is."""
    vals = []
    for r in rows:
        if key not in r or r[key] in ("", None):
            continue
        try:
            vals.append(float(r[key]))
        except (TypeError, ValueError):
            continue
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    # Heuristic: few unique → treat as categorical groups
    uniq = np.unique(np.round(arr, 6))
    if len(uniq) <= 12:
        edges = None
        labels = [str(u) for u in uniq]
        bin_ids = [int(np.argmin(np.abs(uniq - v))) for v in arr]
    else:
        qs = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(arr, qs))
        if len(edges) < 2:
            return {}
        bin_ids = np.clip(np.digitize(arr, edges[1:-1], right=True), 0, len(edges) - 2)
        labels = [
            f"[{edges[i]:.3g},{edges[i + 1]:.3g})" for i in range(len(edges) - 1)
        ]
    by_bin: dict[str, list[dict]] = {lab: [] for lab in labels}
    # Remap rows that had the key
    j = 0
    for r in rows:
        if key not in r or r[key] in ("", None):
            continue
        try:
            float(r[key])
        except (TypeError, ValueError):
            continue
        by_bin[labels[bin_ids[j]]].append(r)
        j += 1

    summary = {}
    for lab, group in by_bin.items():
        if not group:
            continue
        summary[lab] = {
            "n": len(group),
            "rel_l2_mean": float(np.mean([float(g["rel_l2_mean"]) for g in group])),
            "pearson_mean": float(np.mean([float(g["pearson_mean"]) for g in group])),
        }
    return summary


def plot_stratification(
    campaign: str, rows: list[dict[str, Any]], out_dir: Path
) -> None:
    keys = []
    if campaign == "dipping":
        keys = ["dip_angle_deg", "H_discretized", "CoV", "rH"]
    else:
        keys = ["H1_discretized", "CoV1", "rH1", "Vs_contrast"]
    present = [k for k in keys if any(k in r for r in rows)]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 3.6))
    if len(present) == 1:
        axes = [axes]
    for ax, key in zip(axes, present):
        xs, ys = [], []
        for r in rows:
            if key not in r:
                continue
            try:
                xs.append(float(r[key]))
                ys.append(float(r["rel_l2_mean"]))
            except (TypeError, ValueError):
                continue
        ax.scatter(xs, ys, s=8, alpha=0.35, edgecolors="none")
        ax.set_xlabel(key)
        ax.set_ylabel("rel_l2_mean")
        ax.set_title(f"{campaign}: {key}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{campaign}_rel_l2_vs_params.png", dpi=140)
    plt.close(fig)


def score_campaign(
    name: str,
    camp_dir: Path,
    *,
    out_root: Path,
    model: torch.nn.Module,
    device: torch.device,
    limit: int | None,
    workers: int,
    force_gt: bool,
    write_plots: bool,
) -> list[dict[str, Any]]:
    manifest = load_manifest(camp_dir / "manifest.csv")
    h5_dir = camp_dir / "h5"
    if limit is not None:
        manifest = manifest[:limit]

    case_root = out_root / name
    case_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for row in manifest:
        idx = int(row["index"])
        h5_path = h5_dir / f"run_{idx}.h5"
        if not h5_path.is_file():
            print(f"[warn] missing {h5_path}")
            continue
        case_dir = case_root / f"run_{idx}"
        jobs.append((str(h5_path), f"{name}/run_{idx}", str(case_dir), force_gt))

    print(f"[{name}] Precomputing GT TFs for {len(jobs)} cases (workers={workers}) ...")
    t_gt0 = time.time()
    if workers <= 1:
        for job in jobs:
            slug, status, dt = _gt_worker(job)
            if status == "computed":
                print(f"  GT {slug} {dt:.1f}s")
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_gt_worker, j) for j in jobs]
            for fut in as_completed(futs):
                slug, status, dt = fut.result()
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(
                        f"  GT progress {done}/{len(jobs)} "
                        f"(last {slug} {status} {dt:.1f}s)"
                    )
    print(f"[{name}] GT phase done in {time.time() - t_gt0:.1f}s")

    rows_out: list[dict[str, Any]] = []
    print(f"[{name}] Inference on {len(jobs)} cases ...")
    t_inf0 = time.time()
    for i, (h5_str, slug, case_dir_str, _) in enumerate(jobs):
        h5_path = Path(h5_str)
        case_dir = Path(case_dir_str)
        tf_true = np.load(case_dir / "tf_true.npy")
        freq = np.load(case_dir / "freq.npy")
        x = build_input_from_h5(h5_path)
        tf_pred = predict_tf(model, x, device)
        np.save(case_dir / "tf_pred.npy", tf_pred)
        m = metrics_from_tfs(tf_pred, tf_true, freq)
        if write_plots and i < 5:
            compare_tfs(
                tf_pred, tf_true, freq, case_name=slug, out_dir=case_dir
            )
        # Align by index from path
        idx = int(Path(case_dir_str).name.split("_")[1])
        man_row = next((r for r in manifest if int(r["index"]) == idx), {})
        out = {
            "campaign": name,
            "index": idx,
            "h5_path": str(h5_path),
            **{k: man_row.get(k, "") for k in man_row},
            **m,
        }
        rows_out.append(out)
        if (i + 1) % 50 == 0 or (i + 1) == len(jobs):
            print(
                f"  infer {i + 1}/{len(jobs)}  "
                f"rel_l2={m['rel_l2_mean']:.3f} pearson={m['pearson_mean']:.3f}"
            )
    print(f"[{name}] Inference done in {time.time() - t_inf0:.1f}s")

    # CSV
    csv_path = case_root / "per_case_metrics.csv"
    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    strat_keys = (
        ["dip_angle_deg", "H_discretized", "CoV", "rH", "Vs1"]
        if name == "dipping"
        else ["H1_discretized", "H2_discretized", "CoV1", "rH1", "Vs_contrast", "Vs_mid"]
    )
    stratified = {k: stratify_summary(rows_out, k) for k in strat_keys}
    summary = {
        "campaign": name,
        "n": len(rows_out),
        "rel_l2_mean": float(np.mean([r["rel_l2_mean"] for r in rows_out])),
        "rel_l2_median": float(np.median([r["rel_l2_mean"] for r in rows_out])),
        "pearson_mean": float(np.mean([r["pearson_mean"] for r in rows_out])),
        "pearson_median": float(np.median([r["pearson_mean"] for r in rows_out])),
        "logspec_rel_l2_mean": float(
            np.mean([r["logspec_rel_l2_mean"] for r in rows_out])
        ),
        "stratified": stratified,
    }
    (case_root / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_stratification(name, rows_out, case_root)
    print(
        f"[{name}] SUMMARY n={summary['n']}  "
        f"rel_l2={summary['rel_l2_mean']:.4f}  pearson={summary['pearson_mean']:.4f}"
    )
    return rows_out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score OOD dipping/three_layer campaigns")
    p.add_argument(
        "--campaign",
        choices=("dipping", "three_layer", "both"),
        default="both",
    )
    p.add_argument("--box-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force-gt", action="store_true")
    p.add_argument("--plots", action="store_true", help="Write plots for first 5 cases")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.box_root or _box_root()
    checkpoint = Path(args.checkpoint or config.MODEL_SAVE_PATH)
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    device = torch.device(args.device if args.device else str(config.DEVICE))
    print(f"[ood] box_root={root}")
    print(f"[ood] checkpoint={checkpoint} device={device}")
    model = load_model(checkpoint, device)

    all_rows: list[dict[str, Any]] = []
    for name, camp_dir in campaign_dirs(root, args.campaign):
        if not camp_dir.is_dir():
            print(f"ERROR: missing {camp_dir}", file=sys.stderr)
            return 1
        all_rows.extend(
            score_campaign(
                name,
                camp_dir,
                out_root=args.out_dir,
                model=model,
                device=device,
                limit=args.limit,
                workers=args.workers,
                force_gt=args.force_gt,
                write_plots=args.plots,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "all_cases.json").write_text(json.dumps(all_rows, indent=2))
    by_camp: dict[str, list] = {}
    for r in all_rows:
        by_camp.setdefault(r["campaign"], []).append(r)
    overall = {
        c: {
            "n": len(rows),
            "rel_l2_mean": float(np.mean([x["rel_l2_mean"] for x in rows])),
            "pearson_mean": float(np.mean([x["pearson_mean"] for x in rows])),
        }
        for c, rows in by_camp.items()
    }
    (args.out_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2))
    print(f"[ood] Wrote {args.out_dir / 'overall_summary.json'}")
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
