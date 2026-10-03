#!/usr/bin/env python3
"""
Homogeneous / low-CoV sanity + optional full-depth OOD re-eval.

1. Pick lowest-CoV in-dist GIFNO cases (or CoV below a threshold), compare
   surrogate TF vs 1D Thomson–Haskell on the central column. Homogeneous
   columns should approach 1D; large gaps imply GRF-texture memorization.
2. If --depth-checkpoint is given, re-score a sample of OOD H5s with that
   full-depth (or other) checkpoint and compare to the depth-1 baseline
   metrics already written by score_ood_campaign.py.

Example:
  export GIFNO_MODEL_DIR=~/surrogate-seismic-waves/checkpoints/tier2_pod64_n2000
  export GIFNO_POD_NUM_MODES=64 GIFNO_LATENT_CHANNELS=128 ...
  export GIFNO_DATA_ROOT="/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data"

  uv run python sanity_homog_depth.py --n-low-cov 40
  uv run python sanity_homog_depth.py --ood-compare --ood-limit 64
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config

config.setup_import_paths()

_hs_path = (
    Path(__file__).resolve().parents[1] / "DeepONet-Residual" / "haskell_baseline.py"
)
_hs_spec = importlib.util.spec_from_file_location("haskell_baseline", _hs_path)
if _hs_spec is None or _hs_spec.loader is None:
    raise ImportError(f"Cannot load haskell_baseline from {_hs_path}")
_hs_mod = importlib.util.module_from_spec(_hs_spec)
_hs_spec.loader.exec_module(_hs_mod)
haskell_af_within = _hs_mod.haskell_af_within

from capability_check import (  # noqa: E402
    _rel_l2,
    build_input_from_h5,
    load_model,
    predict_tf,
)
from data_loader import _resolve_h5_path, load_manifest  # noqa: E402
from score_ood_campaign import _pearson  # noqa: E402

_EPS = 1e-12
DEFAULT_OUT = (
    Path.home() / "surrogate-seismic-waves" / "checkpoints" / "sanity_homog_depth"
)


def _cov_from_row(row: dict) -> float | None:
    for key in ("CoV", "cov", "CoV1"):
        if key in row and row[key] not in ("", None):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass
    return None


def low_cov_haskell_sanity(
    *,
    model: torch.nn.Module,
    device: torch.device,
    n_cases: int,
    cov_max: float | None,
    out_dir: Path,
) -> dict[str, Any]:
    manifest = load_manifest(config.MANIFEST_PATH)
    tf_array = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
    freq = np.load(config.TF_FREQ_PATH)
    recorder_x = np.load(config.TF_RESULTS_DIR / "recorder_x_idx.npy")
    central_r = int(np.argmin(np.abs(recorder_x - config.NX // 2)))
    # map cropped index: recorder_x is on cropped strip
    central_x = int(recorder_x[central_r])

    scored = []
    for i, row in enumerate(manifest):
        if i >= tf_array.shape[0]:
            break
        cov = _cov_from_row(row)
        if cov is None:
            continue
        scored.append((cov, i, row))
    scored.sort(key=lambda t: t[0])
    if cov_max is not None:
        scored = [t for t in scored if t[0] <= cov_max]
    selected = scored[:n_cases]
    print(
        f"[homog] selected {len(selected)} lowest-CoV cases "
        f"(range {selected[0][0]:.3f}–{selected[-1][0]:.3f})"
    )

    rows_out = []
    for cov, idx, row in selected:
        h5_path = _resolve_h5_path(row["h5_path"])
        import h5py

        with h5py.File(h5_path, "r") as f:
            vs = f["Vs_realization_2D"][:]
            zeta = f["Damping_zeta"][:]
            dz = float(f["grid"].attrs.get("dz", 1.0))
            vs2 = float(f["params"].attrs.get("Vs2", vs[-1].mean()))
            sl = slice(config.X_SLICE_START, config.X_SLICE_END)
            vs_c = vs[:, sl]
            zeta_c = zeta[:, sl]
            col_vs = vs_c[:, central_x]
            col_zeta = zeta_c[:, central_x]

        # 1D Haskell on realized central column
        af = haskell_af_within(
            freq,
            col_vs,
            col_zeta,
            dz=dz,
            vs_rock=vs2,
        )
        tf_true = tf_array[idx]  # (R, F)
        x = build_input_from_h5(h5_path)
        tf_pred = predict_tf(model, x, device)

        # Surrogate vs OpenSees (central)
        s_vs_o = _rel_l2(tf_pred[central_r], tf_true[central_r])
        # Surrogate vs Haskell
        s_vs_h = _rel_l2(tf_pred[central_r], af)
        # OpenSees vs Haskell (how 2D the case still is)
        o_vs_h = _rel_l2(tf_true[central_r], af)
        rows_out.append(
            {
                "index": idx,
                "CoV": cov,
                "rel_l2_surr_vs_opensees": s_vs_o,
                "rel_l2_surr_vs_haskell": s_vs_h,
                "rel_l2_opensees_vs_haskell": o_vs_h,
                "pearson_surr_vs_opensees": _pearson(
                    tf_pred[central_r], tf_true[central_r]
                ),
                "pearson_surr_vs_haskell": _pearson(tf_pred[central_r], af),
                "pearson_opensees_vs_haskell": _pearson(tf_true[central_r], af),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "low_cov_haskell.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    summary = {
        "n": len(rows_out),
        "cov_min": float(rows_out[0]["CoV"]),
        "cov_max": float(rows_out[-1]["CoV"]),
        "mean_rel_l2_surr_vs_opensees": float(
            np.mean([r["rel_l2_surr_vs_opensees"] for r in rows_out])
        ),
        "mean_rel_l2_surr_vs_haskell": float(
            np.mean([r["rel_l2_surr_vs_haskell"] for r in rows_out])
        ),
        "mean_rel_l2_opensees_vs_haskell": float(
            np.mean([r["rel_l2_opensees_vs_haskell"] for r in rows_out])
        ),
        "interpretation": (
            "If OpenSees≈Haskell (low o_vs_h) but surrogate stays far from both, "
            "the model failed the homogeneous limit. If OpenSees still differs "
            "from Haskell, residual 2D coupling remains even at low CoV."
        ),
    }
    (out_dir / "low_cov_haskell_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    covs = [r["CoV"] for r in rows_out]
    ax.plot(
        covs,
        [r["rel_l2_opensees_vs_haskell"] for r in rows_out],
        "o-",
        label="OpenSees vs Haskell",
    )
    ax.plot(
        covs,
        [r["rel_l2_surr_vs_haskell"] for r in rows_out],
        "s--",
        label="Surrogate vs Haskell",
    )
    ax.plot(
        covs,
        [r["rel_l2_surr_vs_opensees"] for r in rows_out],
        "^:",
        label="Surrogate vs OpenSees",
    )
    ax.set_xlabel("CoV")
    ax.set_ylabel("Central-recorder rel L2")
    ax.set_title("Low-CoV homogeneous sanity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "low_cov_haskell.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


def ood_depth_compare(
    *,
    baseline_ckpt: Path,
    depth_ckpt: Path | None,
    ood_scores_root: Path,
    box_root: Path,
    device: torch.device,
    limit: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Re-score a subset of OOD cases with baseline and optional depth ckpt."""
    from score_ood_campaign import metrics_from_tfs  # noqa: PLC0415

    results: dict[str, Any] = {"baseline": str(baseline_ckpt), "depth": None}
    model_base = load_model(baseline_ckpt, device)
    model_depth = (
        load_model(depth_ckpt, device) if depth_ckpt and depth_ckpt.is_file() else None
    )
    if model_depth is None:
        print(
            "[depth] No full-depth checkpoint found — writing baseline-only "
            "subset metrics and a note. Train loglo_pod_depth_full to enable."
        )
        results["note"] = "depth checkpoint missing"
    else:
        results["depth"] = str(depth_ckpt)

    per_camp = {}
    for camp in ("dipping", "three_layer"):
        camp_scores = ood_scores_root / camp
        csv_path = camp_scores / "per_case_metrics.csv"
        h5_dir = box_root / f"ood_{camp}" / "h5"
        if not csv_path.is_file():
            # fall back: first `limit` runs
            indices = list(range(limit))
            baseline_from_csv = {}
        else:
            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            # stratified sample: sort by rel_l2 and take evenly spaced
            rows_sorted = sorted(rows, key=lambda r: float(r["rel_l2_mean"]))
            step = max(1, len(rows_sorted) // limit)
            picked = rows_sorted[::step][:limit]
            indices = [int(r["index"]) for r in picked]
            baseline_from_csv = {
                int(r["index"]): float(r["rel_l2_mean"]) for r in picked
            }

        rows_out = []
        for idx in indices:
            h5_path = h5_dir / f"run_{idx}.h5"
            case_dir = camp_scores / f"run_{idx}"
            if not h5_path.is_file():
                continue
            # Prefer cached GT from score_ood_campaign
            if (case_dir / "tf_true.npy").is_file():
                tf_true = np.load(case_dir / "tf_true.npy")
                freq = np.load(case_dir / "freq.npy")
            else:
                from capability_check import compute_ground_truth_tf

                tf_true, freq = compute_ground_truth_tf(h5_path)
                case_dir.mkdir(parents=True, exist_ok=True)
                np.save(case_dir / "tf_true.npy", tf_true)
                np.save(case_dir / "freq.npy", freq)

            x = build_input_from_h5(h5_path)
            pred_b = predict_tf(model_base, x, device)
            mb = metrics_from_tfs(pred_b, tf_true, freq)
            row = {
                "index": idx,
                "baseline_rel_l2": mb["rel_l2_mean"],
                "baseline_pearson": mb["pearson_mean"],
            }
            if idx in baseline_from_csv:
                row["csv_rel_l2"] = baseline_from_csv[idx]
            if model_depth is not None:
                pred_d = predict_tf(model_depth, x, device)
                md = metrics_from_tfs(pred_d, tf_true, freq)
                row["depth_rel_l2"] = md["rel_l2_mean"]
                row["depth_pearson"] = md["pearson_mean"]
                row["delta_rel_l2_depth_minus_base"] = (
                    md["rel_l2_mean"] - mb["rel_l2_mean"]
                )
            rows_out.append(row)

        per_camp[camp] = {
            "n": len(rows_out),
            "baseline_rel_l2_mean": float(
                np.mean([r["baseline_rel_l2"] for r in rows_out])
            )
            if rows_out
            else None,
            "depth_rel_l2_mean": float(np.mean([r["depth_rel_l2"] for r in rows_out]))
            if rows_out and model_depth is not None
            else None,
            "cases": rows_out,
        }
        print(
            f"[depth] {camp}: n={per_camp[camp]['n']} "
            f"base_rel_l2={per_camp[camp]['baseline_rel_l2_mean']} "
            f"depth_rel_l2={per_camp[camp]['depth_rel_l2_mean']}"
        )

    results["campaigns"] = per_camp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ood_depth_compare.json").write_text(json.dumps(results, indent=2))
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Homogeneous + depth OOD sanity")
    p.add_argument("--n-low-cov", type=int, default=40)
    p.add_argument("--cov-max", type=float, default=None)
    p.add_argument("--skip-homog", action="store_true")
    p.add_argument("--ood-compare", action="store_true")
    p.add_argument("--ood-limit", type=int, default=64)
    p.add_argument(
        "--ood-scores-root",
        type=Path,
        default=Path.home()
        / "surrogate-seismic-waves"
        / "checkpoints"
        / "ood_campaign_scores",
    )
    p.add_argument(
        "--box-root",
        type=Path,
        default=Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data"),
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--depth-checkpoint",
        type=Path,
        default=None,
        help="Optional full-depth best_model.pt (LOGLO_DEPTH_STRIDE=1)",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ckpt = Path(args.checkpoint or config.MODEL_SAVE_PATH)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1
    device = torch.device(args.device if args.device else str(config.DEVICE))
    model = load_model(ckpt, device)

    if not args.skip_homog:
        low_cov_haskell_sanity(
            model=model,
            device=device,
            n_cases=args.n_low_cov,
            cov_max=args.cov_max,
            out_dir=args.out_dir,
        )

    if args.ood_compare:
        # Search common locations for depth_full if not passed
        depth_ckpt = args.depth_checkpoint
        if depth_ckpt is None:
            candidates = [
                Path.home()
                / "surrogate-seismic-waves"
                / "checkpoints"
                / "loglo_pod_depth_full"
                / "best_model.pt",
                Path(os.environ.get("GIFNO_TF_DIR", ""))
                / "models"
                / "fdo_xt_loglo_pod"
                / "sweep"
                / "loglo_pod_depth_full"
                / "best_model.pt",
            ]
            for c in candidates:
                if c.is_file():
                    depth_ckpt = c
                    break
        ood_depth_compare(
            baseline_ckpt=ckpt,
            depth_ckpt=depth_ckpt,
            ood_scores_root=args.ood_scores_root,
            box_root=args.box_root,
            device=device,
            limit=args.ood_limit,
            out_dir=args.out_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
