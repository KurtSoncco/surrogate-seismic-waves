#!/usr/bin/env python3
"""
Peak-frequency robustness vs theoretical f0 = Vs1 / (4 H).

For each realization (Sobol point × RF seed):
  1. Compute f0 from Vs1 and H_discretized
  2. Run surrogate on Vs/zeta/coords from H5 (no accelerations)
  3. In [0.6, 1.4] × f0, find dominant |TF| peak (scipy.signal.find_peaks)
  4. Record |f_surrogate − f0| and f0 / f_surrogate

With ``--compare-opensees``, also compute OpenSees |TF| from H5 accelerations
(seiskit TTF) and plot surrogate vs OpenSees vs f0.

Example:
  export GIFNO_DATA_ROOT="/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data"
  export GIFNO_H5_DIR="${GIFNO_DATA_ROOT}/h5"
  export GIFNO_TF_DIR="${GIFNO_DATA_ROOT}/transfer_function"
  export GIFNO_MODEL_DIR=~/surrogate-seismic-waves/checkpoints/xt_lat128_d128

  cd experiments/GIFNO-FDO-XT-LOGLO-POD
  uv run python seed_robustness/peak_f0_robustness_xt.py --split test
  uv run python seed_robustness/peak_f0_robustness_xt.py --split all   # full corpus
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks
from tqdm import tqdm

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

import config  # noqa: E402

config.setup_import_paths()

from capability_check import (  # noqa: E402
    build_input_from_h5,
    compute_ground_truth_tf,
    load_model,
)

DEFAULT_SEISKIT_DATA = Path.home() / "seiskit" / "neural-operator" / "data"
DEFAULT_OUT_ROOT = (
    Path.home() / "surrogate-seismic-waves" / "checkpoints" / "peak_f0_robustness_xt"
)
DEFAULT_SEEDS_PER_SAMPLE = 30
DEFAULT_WINDOW_LO = 0.6
DEFAULT_WINDOW_HI = 1.4
_EPS = 1e-12


@dataclass(frozen=True)
class RunRecord:
    run_index: int
    sample_id: int
    replicate_id: int
    rf_seed: int
    vs1: float
    h_discretized: float
    cov: float
    manifest_sample_idx: int


def _load_sobol_module(seiskit_data_dir: Path):
    sobol_path = seiskit_data_dir / "sobol.py"
    spec = importlib.util.spec_from_file_location("sobol", sobol_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {sobol_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sobol"] = mod
    spec.loader.exec_module(mod)
    return mod


def sobol_physics_by_run_index(
    seiskit_data_dir: Path,
    *,
    seeds_per_sample: int = DEFAULT_SEEDS_PER_SAMPLE,
) -> dict[int, tuple[float, float]]:
    """run_index -> (Vs1, H_discretized)."""
    sobol = _load_sobol_module(seiskit_data_dir)
    manifest = sobol.build_manifest(
        sample_count=256,
        seeds_per_sample=seeds_per_sample,
    )
    return {e.index: (e.Vs1, e.H_discretized) for e in manifest}


def load_gifno_manifest_rows(
    manifest_path: Path,
    *,
    limit: int | None = None,
    sample_id: int | None = None,
    seeds_per_sample: int = DEFAULT_SEEDS_PER_SAMPLE,
    physics: dict[int, tuple[float, float]],
    dataset_indices: set[int] | None = None,
) -> list[RunRecord]:
    rows: list[RunRecord] = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for dataset_idx, row in enumerate(csv.DictReader(f)):
            if limit is not None and dataset_idx >= limit:
                break
            if dataset_indices is not None and dataset_idx not in dataset_indices:
                continue
            run_index = int(row["run_index"])
            sid = run_index // seeds_per_sample
            if sample_id is not None and sid != sample_id:
                continue
            vs1, h = physics.get(run_index, (float("nan"), float("nan")))
            rows.append(
                RunRecord(
                    run_index=run_index,
                    sample_id=sid,
                    replicate_id=run_index % seeds_per_sample,
                    rf_seed=int(row["rf_seed"]),
                    vs1=vs1,
                    h_discretized=h,
                    cov=float(row["CoV"]),
                    manifest_sample_idx=dataset_idx,
                )
            )
    return rows


def split_dataset_indices(
    n: int,
    *,
    train_split: float,
    val_split: float,
    seed: int,
) -> dict[str, list[int]]:
    """Same 70/15/15 split as GIFNO data_loader.get_data_loaders."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def theoretical_f0(vs1: float, h: float) -> float:
    return float(vs1 / (4.0 * max(h, _EPS)))


def find_peak_frequency(
    freq: np.ndarray,
    tf_curve: np.ndarray,
    f0: float,
    *,
    window_lo: float = DEFAULT_WINDOW_LO,
    window_hi: float = DEFAULT_WINDOW_HI,
) -> tuple[float, float]:
    """
    Dominant |TF| peak in [window_lo, window_hi] × f0.

    Returns (f_peak_hz, peak_amplitude). NaNs if window empty or f0 invalid.
    """
    if not np.isfinite(f0) or f0 <= 0:
        return float("nan"), float("nan")

    f_lo = window_lo * f0
    f_hi = window_hi * f0
    mask = (freq >= f_lo) & (freq <= f_hi)
    if not np.any(mask):
        return float("nan"), float("nan")

    f_win = freq[mask]
    y_win = np.abs(tf_curve[mask]).astype(np.float64)
    if y_win.size == 0 or np.max(y_win) <= _EPS:
        return float("nan"), float("nan")

    peaks, props = find_peaks(y_win, prominence=0.0)
    if peaks.size == 0:
        idx = int(np.argmax(y_win))
        return float(f_win[idx]), float(y_win[idx])

    best_local = int(peaks[np.argmax(y_win[peaks])])
    return float(f_win[best_local]), float(y_win[best_local])


def central_recorder_index() -> int:
    rec = config.recorder_x_indices()
    return int(np.argmin(np.abs(rec - config.NX // 2)))


def h5_path(h5_dir: Path, run_index: int) -> Path:
    return h5_dir / f"run_{run_index}.h5"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _dist_summary(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def plot_histograms(
    out_dir: Path,
    abs_err: np.ndarray,
    f0_over_fsur: np.ndarray,
    *,
    title_suffix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    abs_err = abs_err[np.isfinite(abs_err)]
    f0_over_fsur = f0_over_fsur[np.isfinite(f0_over_fsur)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if abs_err.size:
        axes[0].hist(abs_err, bins=40, color="steelblue", edgecolor="white", alpha=0.9)
        axes[0].axvline(
            np.median(abs_err),
            color="crimson",
            ls="--",
            lw=1.5,
            label=f"median={np.median(abs_err):.3f} Hz",
        )
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("|f_surrogate − f0| (Hz)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Peak freq error{title_suffix}")
    axes[0].grid(True, alpha=0.3)

    if f0_over_fsur.size:
        axes[1].hist(
            f0_over_fsur, bins=40, color="darkorange", edgecolor="white", alpha=0.9
        )
        axes[1].axvline(1.0, color="black", ls="-", lw=1, alpha=0.5)
        axes[1].axvline(
            np.median(f0_over_fsur),
            color="crimson",
            ls="--",
            lw=1.5,
            label=f"median={np.median(f0_over_fsur):.3f}",
        )
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("f0 / f_surrogate")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"f0 ratio{title_suffix}")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "peak_f0_histograms.png", dpi=150)
    plt.close(fig)


def plot_opensees_comparison(
    out_dir: Path,
    *,
    abs_err_sur: np.ndarray,
    abs_err_oo: np.ndarray,
    ratio_sur: np.ndarray,
    ratio_oo: np.ndarray,
    abs_sur_oo: np.ndarray,
    f_sur: np.ndarray,
    f_oo: np.ndarray,
    title_suffix: str = "",
) -> None:
    """Surrogate vs OpenSees peak-frequency comparison plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    s = abs_err_sur[np.isfinite(abs_err_sur)]
    o = abs_err_oo[np.isfinite(abs_err_oo)]
    if s.size:
        ax.hist(
            s,
            bins=40,
            alpha=0.55,
            label=f"surrogate (med={np.median(s):.3f} Hz)",
            color="steelblue",
        )
    if o.size:
        ax.hist(
            o,
            bins=40,
            alpha=0.55,
            label=f"OpenSees (med={np.median(o):.3f} Hz)",
            color="seagreen",
        )
    ax.set_xlabel("|f_peak − f0| (Hz)")
    ax.set_ylabel("count")
    ax.set_title(f"Peak error vs f0{title_suffix}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    rs = ratio_sur[np.isfinite(ratio_sur)]
    ro = ratio_oo[np.isfinite(ratio_oo)]
    if rs.size:
        ax.hist(
            rs,
            bins=40,
            alpha=0.55,
            label=f"surrogate (med={np.median(rs):.3f})",
            color="darkorange",
        )
    if ro.size:
        ax.hist(
            ro,
            bins=40,
            alpha=0.55,
            label=f"OpenSees (med={np.median(ro):.3f})",
            color="mediumseagreen",
        )
    ax.axvline(1.0, color="black", ls="-", lw=1, alpha=0.4)
    ax.set_xlabel("f0 / f_peak")
    ax.set_ylabel("count")
    ax.set_title(f"f0 ratio{title_suffix}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    d = abs_sur_oo[np.isfinite(abs_sur_oo)]
    if d.size:
        ax.hist(d, bins=40, color="purple", alpha=0.85, edgecolor="white")
        ax.axvline(
            np.median(d),
            color="crimson",
            ls="--",
            lw=1.5,
            label=f"median={np.median(d):.3f} Hz",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("|f_surrogate − f_opensees| (Hz)")
    ax.set_ylabel("count")
    ax.set_title("Surrogate vs OpenSees peak mismatch")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    mask = np.isfinite(f_sur) & np.isfinite(f_oo)
    if np.any(mask):
        ax.scatter(
            f_oo[mask], f_sur[mask], s=8, alpha=0.35, c="0.25", edgecolors="none"
        )
        lo = float(min(f_oo[mask].min(), f_sur[mask].min()))
        hi = float(max(f_oo[mask].max(), f_sur[mask].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y=x")
        ax.legend(fontsize=8)
    ax.set_xlabel("f_opensees (Hz)")
    ax.set_ylabel("f_surrogate (Hz)")
    ax.set_title("Peak frequency scatter")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "peak_f0_opensees_comparison.png", dpi=150)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Surrogate peak f vs theoretical f0=Vs1/(4H)"
    )
    p.add_argument("--h5-dir", type=Path, default=None)
    p.add_argument("--gifno-manifest", type=Path, default=None)
    p.add_argument("--freq-path", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--seiskit-data-dir", type=Path, default=DEFAULT_SEISKIT_DATA)
    p.add_argument("--seeds-per-sample", type=int, default=DEFAULT_SEEDS_PER_SAMPLE)
    p.add_argument(
        "--sample-id", type=int, default=None, help="Optional single Sobol point"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap manifest rows before split (matches training --limit)",
    )
    p.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
        help="Dataset split (default: test hold-out, seed=42, 70/15/15)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Split RNG seed (default: config.SEED)"
    )
    p.add_argument("--train-split", type=float, default=None)
    p.add_argument("--val-split", type=float, default=None)
    p.add_argument("--window-lo", type=float, default=DEFAULT_WINDOW_LO)
    p.add_argument("--window-hi", type=float, default=DEFAULT_WINDOW_HI)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--compare-opensees",
        action="store_true",
        help="Compute OpenSees |TF| from H5 accelerations and plot vs surrogate",
    )
    return p.parse_args()


@torch.no_grad()
def run_batch(
    model: torch.nn.Module,
    h5_paths: list[Path],
    device: torch.device,
) -> list[np.ndarray]:
    if not h5_paths:
        return []
    grids = [build_input_from_h5(p) for p in h5_paths]
    batch = torch.from_numpy(np.stack(grids)).to(device)
    out = model(batch).cpu().numpy()
    rec = config.recorder_x_indices()
    return [out[i, rec, :].astype(np.float32) for i in range(out.shape[0])]


def main() -> int:
    args = _parse_args()
    h5_dir = args.h5_dir or Path(
        __import__("os").environ.get("GIFNO_H5_DIR", config.H5_DIR)
    )
    manifest_path = args.gifno_manifest or config.MANIFEST_PATH
    freq_path = args.freq_path or config.TF_FREQ_PATH
    checkpoint = Path(args.checkpoint or config.MODEL_SAVE_PATH)

    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1
    if not freq_path.is_file():
        print(f"ERROR: freq not found: {freq_path}", file=sys.stderr)
        return 1

    freq = np.load(freq_path)
    physics = sobol_physics_by_run_index(
        args.seiskit_data_dir, seeds_per_sample=args.seeds_per_sample
    )

    with manifest_path.open(newline="", encoding="utf-8") as f:
        n_manifest = sum(
            1
            for i, _ in enumerate(csv.DictReader(f))
            if args.limit is None or i < args.limit
        )

    dataset_indices: set[int] | None = None
    split_name = args.split
    if split_name != "all":
        splits = split_dataset_indices(
            n_manifest,
            train_split=args.train_split
            if args.train_split is not None
            else config.TRAIN_SPLIT,
            val_split=args.val_split
            if args.val_split is not None
            else config.VAL_SPLIT,
            seed=args.seed if args.seed is not None else config.SEED,
        )
        dataset_indices = set(splits[split_name])
        print(
            f"[peak_f0] split={split_name}  n={len(dataset_indices)} / {n_manifest}  "
            f"(seed={args.seed or config.SEED})"
        )

    out_dir = args.out_dir
    if split_name != "all":
        out_dir = args.out_dir / split_name

    records = load_gifno_manifest_rows(
        manifest_path,
        limit=args.limit,
        sample_id=args.sample_id,
        seeds_per_sample=args.seeds_per_sample,
        physics=physics,
        dataset_indices=dataset_indices,
    )
    if not records:
        print("ERROR: no manifest rows to process", file=sys.stderr)
        return 1

    device = torch.device(args.device if args.device else str(config.DEVICE))
    model = load_model(checkpoint, device)
    c_rec = central_recorder_index()

    result_rows: list[dict] = []
    pending_paths: list[Path] = []
    pending_meta: list[RunRecord] = []

    def flush_batch() -> None:
        nonlocal pending_paths, pending_meta
        if not pending_paths:
            return
        tf_preds = run_batch(model, pending_paths, device)
        for path, rec, tf_pred in zip(pending_paths, pending_meta, tf_preds):
            f0 = theoretical_f0(rec.vs1, rec.h_discretized)
            freq_use = freq
            f_sur, amp_sur = find_peak_frequency(
                freq_use,
                tf_pred[c_rec],
                f0,
                window_lo=args.window_lo,
                window_hi=args.window_hi,
            )

            f_oo = float("nan")
            amp_oo = float("nan")
            if args.compare_opensees:
                tf_true, freq_gt = compute_ground_truth_tf(path)
                freq_use = freq_gt
                f_oo, amp_oo = find_peak_frequency(
                    freq_gt,
                    tf_true[c_rec],
                    f0,
                    window_lo=args.window_lo,
                    window_hi=args.window_hi,
                )

            abs_err_sur = abs(f_sur - f0) if np.isfinite(f_sur) else float("nan")
            ratio_sur = (
                f0 / f_sur if np.isfinite(f_sur) and f_sur > _EPS else float("nan")
            )
            abs_err_oo = abs(f_oo - f0) if np.isfinite(f_oo) else float("nan")
            ratio_oo = f0 / f_oo if np.isfinite(f_oo) and f_oo > _EPS else float("nan")
            abs_sur_oo = (
                abs(f_sur - f_oo)
                if np.isfinite(f_sur) and np.isfinite(f_oo)
                else float("nan")
            )

            row = {
                "run_index": rec.run_index,
                "sample_id": rec.sample_id,
                "replicate_id": rec.replicate_id,
                "rf_seed": rec.rf_seed,
                "Vs1": rec.vs1,
                "H_discretized": rec.h_discretized,
                "CoV": rec.cov,
                "f0_hz": f0,
                "f_surrogate_hz": f_sur,
                "peak_amplitude_surrogate": amp_sur,
                "abs_f_err_hz": abs_err_sur,
                "f0_over_f_surrogate": ratio_sur,
                "window_lo_hz": args.window_lo * f0,
                "window_hi_hz": args.window_hi * f0,
            }
            if args.compare_opensees:
                row.update(
                    {
                        "f_opensees_hz": f_oo,
                        "peak_amplitude_opensees": amp_oo,
                        "abs_f_opensees_err_hz": abs_err_oo,
                        "f0_over_f_opensees": ratio_oo,
                        "abs_f_surrogate_opensees_hz": abs_sur_oo,
                    }
                )
            result_rows.append(row)
        pending_paths = []
        pending_meta = []

    for rec in tqdm(records, desc="peak_f0"):
        path = h5_path(h5_dir, rec.run_index)
        if not path.is_file():
            continue
        pending_paths.append(path)
        pending_meta.append(rec)
        if len(pending_paths) >= args.batch_size:
            flush_batch()
    flush_batch()

    if not result_rows:
        print(f"ERROR: no H5 files processed under {h5_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_opensees" if args.compare_opensees else ""
    csv_name = f"per_run_peak_f0{tag}.csv"
    write_csv(out_dir / csv_name, result_rows)

    abs_err = np.array([r["abs_f_err_hz"] for r in result_rows], dtype=np.float64)
    ratio = np.array([r["f0_over_f_surrogate"] for r in result_rows], dtype=np.float64)
    suffix = f" [{split_name}, n={len(result_rows)}]"
    plot_histograms(out_dir, abs_err, ratio, title_suffix=suffix)

    summary = {
        "split": split_name,
        "compare_opensees": args.compare_opensees,
        "n_processed": len(result_rows),
        "n_requested": len(records),
        "n_manifest": n_manifest,
        "split_seed": args.seed if args.seed is not None else config.SEED,
        "train_split": args.train_split
        if args.train_split is not None
        else config.TRAIN_SPLIT,
        "val_split": args.val_split if args.val_split is not None else config.VAL_SPLIT,
        "window": [args.window_lo, args.window_hi],
        "f0_formula": "Vs1 / (4 * H_discretized)",
        "abs_f_err_hz_surrogate": _dist_summary(abs_err),
        "f0_over_f_surrogate": _dist_summary(ratio),
        "checkpoint": str(checkpoint),
        "h5_dir": str(h5_dir),
    }

    if args.compare_opensees:
        abs_err_oo = np.array(
            [r["abs_f_opensees_err_hz"] for r in result_rows], dtype=np.float64
        )
        ratio_oo = np.array(
            [r["f0_over_f_opensees"] for r in result_rows], dtype=np.float64
        )
        abs_sur_oo = np.array(
            [r["abs_f_surrogate_opensees_hz"] for r in result_rows], dtype=np.float64
        )
        f_sur = np.array([r["f_surrogate_hz"] for r in result_rows], dtype=np.float64)
        f_oo = np.array([r["f_opensees_hz"] for r in result_rows], dtype=np.float64)
        plot_opensees_comparison(
            out_dir,
            abs_err_sur=abs_err,
            abs_err_oo=abs_err_oo,
            ratio_sur=ratio,
            ratio_oo=ratio_oo,
            abs_sur_oo=abs_sur_oo,
            f_sur=f_sur,
            f_oo=f_oo,
            title_suffix=suffix,
        )
        summary["abs_f_err_hz_opensees"] = _dist_summary(abs_err_oo)
        summary["f0_over_f_opensees"] = _dist_summary(ratio_oo)
        summary["abs_f_surrogate_opensees_hz"] = _dist_summary(abs_sur_oo)

    summary_path = out_dir / f"summary{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[peak_f0] processed {len(result_rows)} runs -> {out_dir}")
    print(
        f"  |f_sur−f0| median={summary['abs_f_err_hz_surrogate']['median']:.4f} Hz  "
        f"f0/f_sur median={summary['f0_over_f_surrogate']['median']:.4f}"
    )
    if args.compare_opensees:
        print(
            f"  |f_oo−f0| median={summary['abs_f_err_hz_opensees']['median']:.4f} Hz  "
            f"|f_sur−f_oo| median={summary['abs_f_surrogate_opensees_hz']['median']:.4f} Hz"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
