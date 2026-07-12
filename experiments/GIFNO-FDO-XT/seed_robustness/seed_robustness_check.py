#!/usr/bin/env python3
"""
Seed-number cross-validation at a fixed Sobol sample_id.

Treats ``rf_seed`` as a hidden parameter: the surrogate sees only Vs/zeta grids
from each realization, not the seed value. Ground truth TFs come from the
precomputed ``tf_per_sample.npy`` cache (no H5 accelerations required).

Example:
  # Baseline XT checkpoint
  export GIFNO_MODEL_DIR=~/surrogate-seismic-waves/checkpoints/xt_lat128_d128
  export GIFNO_LATENT_CHANNELS=128 GIFNO_DEEPONET_LATENT_DIM=128

  # Seed-conditional recipe checkpoint (scale-split + dual-path + softplus)
  # export GIFNO_SEED_CONDITIONAL_RECIPE=1
  # export GIFNO_MODEL_DIR=~/surrogate-seismic-waves/checkpoints/xt_seed_conditional

  cd experiments/GIFNO-FDO-XT
  uv run python seed_robustness/seed_robustness_check.py --sample-id 0 --max-seeds 30
  uv run python seed_robustness/analyze_convergence.py \\
    --sample-dir ~/surrogate-seismic-waves/checkpoints/seed_robustness_xt/sample_0000
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

import config  # noqa: E402

config.setup_import_paths()

from capability_check import (  # noqa: E402
    build_input_from_h5,
    compute_ground_truth_tf,
    load_model,
    predict_tf,
)
from metrics import pearson_1d  # noqa: E402

DEFAULT_SEEDS_PER_SAMPLE_TRAINING = 30
DEFAULT_SEISKIT_DATA = Path.home() / "seiskit" / "neural-operator" / "data"
DEFAULT_OUT_ROOT = (
    Path.home() / "surrogate-seismic-waves" / "checkpoints" / "seed_robustness_xt"
)
DEFAULT_SEEDS_PER_SAMPLE = 50
DEFAULT_SEED_COUNTS = (10, 20, 30)
DEFAULT_N_SUBSETS = 5
DEFAULT_SUBSET_RNG = 42
_EPS = 1e-12


@dataclass(frozen=True)
class ReplicateRow:
    """One RF realization at a fixed Sobol physical point."""

    run_index: int
    replicate_id: int
    rf_seed: int
    Vs1: float
    H_discretized: float
    CoV: float
    rH: float
    aHV: float
    Vs2: float
    manifest_sample_idx: int


class CachedTruthTF:
    """Ground truth from tf_per_sample.npy + GIFNO manifest (no accel I/O)."""

    def __init__(
        self,
        tf_path: Path,
        freq_path: Path,
        manifest_path: Path,
    ) -> None:
        self.tf = np.load(tf_path, mmap_mode="r")
        self.freq = np.load(freq_path)
        self._by_run_index: dict[int, tuple[int, dict]] = {}
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                run_index = int(row["run_index"])
                sample_idx = int(row["sample_idx"])
                self._by_run_index[run_index] = (sample_idx, row)

    def truth_tf(self, run_index: int) -> np.ndarray:
        if run_index not in self._by_run_index:
            raise KeyError(f"run_index {run_index} not in manifest")
        sample_idx, _ = self._by_run_index[run_index]
        return np.asarray(self.tf[sample_idx], dtype=np.float32)

    @property
    def frequency(self) -> np.ndarray:
        return self.freq


def gifno_manifest_rows_for_sample(
    manifest_path: Path,
    sample_id: int,
    *,
    seeds_per_sample: int = DEFAULT_SEEDS_PER_SAMPLE_TRAINING,
) -> list[ReplicateRow]:
    """Rows for one Sobol sample_id using flattened run_index layout."""
    start = sample_id * seeds_per_sample
    end = start + seeds_per_sample
    rows: list[ReplicateRow] = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            run_index = int(row["run_index"])
            if run_index < start or run_index >= end:
                continue
            replicate_id = run_index - start
            rows.append(
                ReplicateRow(
                    run_index=run_index,
                    replicate_id=replicate_id,
                    rf_seed=int(row["rf_seed"]),
                    Vs1=float("nan"),
                    H_discretized=float(row["H_discretized"]),
                    CoV=float(row["CoV"]),
                    rH=float("nan"),
                    aHV=float("nan"),
                    Vs2=float("nan"),
                    manifest_sample_idx=int(row["sample_idx"]),
                )
            )
    return sorted(rows, key=lambda r: r.replicate_id)


def enrich_physics_from_sobol(
    rows: list[ReplicateRow],
    seiskit_data_dir: Path,
    sample_id: int,
    *,
    seeds_per_sample: int,
) -> list[ReplicateRow]:
    """Fill Vs1/rH/aHV/Vs2 from canonical sobol manifest when available."""
    sobol = _load_sobol_module(seiskit_data_dir)
    manifest = sobol.build_manifest(
        sample_count=256,
        seeds_per_sample=seeds_per_sample,
    )
    by_rep = {e.replicate_id: e for e in manifest if e.sample_id == sample_id}
    out: list[ReplicateRow] = []
    for row in rows:
        e = by_rep.get(row.replicate_id)
        if e is None:
            out.append(row)
            continue
        out.append(
            ReplicateRow(
                run_index=row.run_index,
                replicate_id=row.replicate_id,
                rf_seed=row.rf_seed,
                Vs1=e.Vs1,
                H_discretized=e.H_discretized,
                CoV=e.CoV,
                rH=e.rH,
                aHV=e.aHV,
                Vs2=e.Vs2,
                manifest_sample_idx=row.manifest_sample_idx,
            )
        )
    return out


def _load_sobol_module(seiskit_data_dir: Path):
    sobol_path = seiskit_data_dir / "sobol.py"
    spec = importlib.util.spec_from_file_location("sobol", sobol_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {sobol_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sobol"] = mod
    spec.loader.exec_module(mod)
    return mod


def manifest_entries_for_sample(
    seiskit_data_dir: Path,
    sample_id: int,
    *,
    seeds_per_sample: int,
    manifest_path: Path | None = None,
) -> list:
    sobol = _load_sobol_module(seiskit_data_dir)
    if manifest_path and manifest_path.is_file():
        manifest = sobol.load_manifest_csv(manifest_path)
        return sorted(
            [e for e in manifest if e.sample_id == sample_id],
            key=lambda e: e.replicate_id,
        )
    manifest = sobol.build_manifest(
        sample_count=256,
        seeds_per_sample=seeds_per_sample,
    )
    return sorted(
        [e for e in manifest if e.sample_id == sample_id],
        key=lambda e: e.replicate_id,
    )


def h5_path_for_index(h5_dir: Path, index: int) -> Path:
    return h5_dir / f"run_{index}.h5"


def resolve_h5_dirs(primary: Path, extra: Path | None) -> list[Path]:
    dirs = [primary]
    if extra is not None and extra.is_dir() and extra.resolve() != primary.resolve():
        dirs.append(extra)
    return dirs


def find_h5(index: int, h5_dirs: list[Path]) -> Path | None:
    for d in h5_dirs:
        p = h5_path_for_index(d, index)
        if p.is_file():
            return p
    return None


def sigma_ln_per_freq(stack: np.ndarray) -> np.ndarray:
    """sigma_ln across axis 0 for each frequency (positive values only)."""
    x = np.asarray(stack, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D stack, got {x.shape}")
    n_seeds, n_freq = x.shape
    out = np.zeros(n_freq, dtype=np.float64)
    for j in range(n_freq):
        col = x[:, j]
        col = col[col > 0]
        if col.size < 2:
            out[j] = 0.0
        else:
            out[j] = float(np.std(np.log(col), ddof=1))
    return out


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    num = np.linalg.norm(pred - true)
    den = np.linalg.norm(true) + _EPS
    return float(num / den)


def central_recorder_index() -> int:
    rec = config.recorder_x_indices()
    return int(np.argmin(np.abs(rec - config.NX // 2)))


def run_per_seed(
    rows: list[ReplicateRow],
    h5_dirs: list[Path],
    checkpoint: Path,
    device: torch.device,
    truth: CachedTruthTF | None,
    *,
    max_seeds: int,
    truth_source: Literal["cache", "accel"] = "cache",
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, int]:
    """Forward surrogate on Vs/zeta grids; truth from cache or H5 accel."""
    model = load_model(checkpoint, device)
    out_rows: list[dict[str, Any]] = []
    truth_stack: list[np.ndarray] = []
    pred_stack: list[np.ndarray] = []
    freq_ref: np.ndarray = (
        truth.frequency if truth is not None else np.array([], dtype=np.float32)
    )
    missing = 0
    c = central_recorder_index()

    for row in rows:
        if row.replicate_id >= max_seeds:
            break
        h5_path = find_h5(row.run_index, h5_dirs)
        if h5_path is None:
            missing += 1
            continue

        if truth_source == "cache":
            if truth is None:
                raise ValueError("truth cache required for truth_source=cache")
            tf_true = truth.truth_tf(row.run_index)
        else:
            tf_true, freq_ref = compute_ground_truth_tf(h5_path)

        x = build_input_from_h5(h5_path)
        tf_pred = predict_tf(model, x, device)
        truth_stack.append(tf_true[c])
        pred_stack.append(tf_pred[c])
        out_rows.append(
            {
                "run_index": row.run_index,
                "replicate_id": row.replicate_id,
                "rf_seed": row.rf_seed,
                "h5_path": str(h5_path),
                "rel_l2_central": _rel_l2(tf_pred[c], tf_true[c]),
                "pearson_central": float(pearson_1d(tf_pred[c], tf_true[c])),
                "rel_l2_mean": float(
                    np.mean(
                        [
                            _rel_l2(tf_pred[r], tf_true[r])
                            for r in range(tf_true.shape[0])
                        ]
                    )
                ),
            }
        )

    if not out_rows:
        raise FileNotFoundError(
            f"No H5 files found under {h5_dirs} (missing={missing})"
        )
    if freq_ref.size == 0:
        raise RuntimeError("frequency axis not set")

    return out_rows, freq_ref, np.stack(truth_stack), np.stack(pred_stack), missing


def seed_cv_metrics(
    truth_stack: np.ndarray,
    pred_stack: np.ndarray,
    freq: np.ndarray,
    *,
    seed_counts: tuple[int, ...],
    n_subsets: int,
    rng_seed: int,
) -> list[dict[str, Any]]:
    """Subsample ensembles of size N and compare sigma_ln to full reference."""
    n_avail = truth_stack.shape[0]
    ref_truth_sigma = sigma_ln_per_freq(truth_stack)
    ref_pred_sigma = sigma_ln_per_freq(pred_stack)
    ref_truth_median = np.median(truth_stack, axis=0)
    ref_pred_median = np.median(pred_stack, axis=0)

    rng = np.random.default_rng(rng_seed)
    rows: list[dict[str, Any]] = []

    for n in seed_counts:
        if n > n_avail:
            continue
        for subset_id in range(n_subsets):
            idx = np.sort(rng.choice(n_avail, size=n, replace=False))
            t_sub = truth_stack[idx]
            p_sub = pred_stack[idx]
            t_sigma = sigma_ln_per_freq(t_sub)
            p_sigma = sigma_ln_per_freq(p_sub)

            # RMSE of sigma_ln curves vs full reference
            sigma_rmse_truth = float(np.sqrt(np.mean((t_sigma - ref_truth_sigma) ** 2)))
            sigma_rmse_pred = float(np.sqrt(np.mean((p_sigma - ref_pred_sigma) ** 2)))
            median_rmse_truth = float(
                np.sqrt(np.mean((np.median(t_sub, axis=0) - ref_truth_median) ** 2))
                / (np.linalg.norm(ref_truth_median) + _EPS)
            )
            median_rmse_pred = float(
                np.sqrt(np.mean((np.median(p_sub, axis=0) - ref_pred_median) ** 2))
                / (np.linalg.norm(ref_pred_median) + _EPS)
            )

            rows.append(
                {
                    "n_seeds": n,
                    "subset_id": subset_id,
                    "n_available": n_avail,
                    "sigma_ln_rmse_truth": sigma_rmse_truth,
                    "sigma_ln_rmse_pred": sigma_rmse_pred,
                    "median_af_rmse_truth": median_rmse_truth,
                    "median_af_rmse_pred": median_rmse_pred,
                    "sigma_ln_mean_truth": float(np.mean(t_sigma)),
                    "sigma_ln_mean_pred": float(np.mean(p_sigma)),
                    "sigma_ln_mean_ref_truth": float(np.mean(ref_truth_sigma)),
                    "sigma_ln_mean_ref_pred": float(np.mean(ref_pred_sigma)),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed-number CV at fixed Sobol sample_id")
    p.add_argument("--sample-id", type=int, default=0)
    p.add_argument("--seeds-per-sample", type=int, default=DEFAULT_SEEDS_PER_SAMPLE)
    p.add_argument("--max-seeds", type=int, default=50)
    p.add_argument("--h5-dir", type=Path, default=None, help="Primary SOBOL_H5_DIR")
    p.add_argument(
        "--extra-h5-dir",
        type=Path,
        default=None,
        help="Supplemental H5 dir for replicate_id >= 30",
    )
    p.add_argument("--manifest-path", type=Path, default=None)
    p.add_argument("--seiskit-data-dir", type=Path, default=DEFAULT_SEISKIT_DATA)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument(
        "--seed-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEED_COUNTS),
    )
    p.add_argument("--n-subsets", type=int, default=DEFAULT_N_SUBSETS)
    p.add_argument("--subset-rng", type=int, default=DEFAULT_SUBSET_RNG)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--truth-source",
        choices=("cache", "accel"),
        default="cache",
        help="cache: tf_per_sample.npy (default); accel: recompute from H5 recorders",
    )
    p.add_argument("--gifno-manifest", type=Path, default=None)
    p.add_argument("--tf-path", type=Path, default=None)
    p.add_argument("--freq-path", type=Path, default=None)
    p.add_argument(
        "--training-seeds-per-sample",
        type=int,
        default=DEFAULT_SEEDS_PER_SAMPLE_TRAINING,
        help="Seeds per Sobol point in the training corpus (default 30).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    h5_dir = args.h5_dir or Path(
        __import__("os").environ.get("SOBOL_H5_DIR", config.H5_DIR)
    )
    extra_h5 = args.extra_h5_dir
    if extra_h5 is None:
        extra_env = __import__("os").environ.get("SEED_ROBUSTNESS_EXTRA_H5_DIR")
        if extra_env:
            extra_h5 = Path(extra_env)

    checkpoint = Path(args.checkpoint or config.MODEL_SAVE_PATH)
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    manifest_path = args.gifno_manifest or config.MANIFEST_PATH
    tf_path = args.tf_path or config.TF_PER_SAMPLE_PATH
    freq_path = args.freq_path or config.TF_FREQ_PATH
    truth: CachedTruthTF | None = None
    if args.truth_source == "cache":
        if not (manifest_path.is_file() and tf_path.is_file() and freq_path.is_file()):
            print(
                f"ERROR: cache truth requires manifest, tf, freq under GIFNO_TF_DIR "
                f"(manifest={manifest_path})",
                file=sys.stderr,
            )
            return 1
        truth = CachedTruthTF(tf_path, freq_path, manifest_path)

    seeds_per = args.training_seeds_per_sample
    replicate_rows = gifno_manifest_rows_for_sample(
        manifest_path, args.sample_id, seeds_per_sample=seeds_per
    )
    replicate_rows = enrich_physics_from_sobol(
        replicate_rows,
        args.seiskit_data_dir,
        args.sample_id,
        seeds_per_sample=seeds_per,
    )
    replicate_rows = [r for r in replicate_rows if r.replicate_id < args.max_seeds]
    if not replicate_rows:
        print(
            f"ERROR: no manifest rows for sample_id={args.sample_id}", file=sys.stderr
        )
        return 1

    device = torch.device(args.device if args.device else str(config.DEVICE))
    h5_dirs = resolve_h5_dirs(h5_dir, extra_h5)

    print(
        f"[seed_robustness] sample_id={args.sample_id}  max_seeds={args.max_seeds}  "
        f"truth={args.truth_source}"
    )
    print(f"[seed_robustness] H5 dirs: {h5_dirs}")

    if args.truth_source == "cache" and truth is None:
        return 1

    per_seed_rows, freq, truth_arr, pred_arr, missing = run_per_seed(
        replicate_rows,
        h5_dirs,
        checkpoint,
        device,
        truth,
        max_seeds=args.max_seeds,
        truth_source=args.truth_source,
    )
    print(
        f"[seed_robustness] processed {len(per_seed_rows)} replicates (missing H5: {missing})"
    )

    cv_rows = seed_cv_metrics(
        truth_arr,
        pred_arr,
        freq,
        seed_counts=tuple(args.seed_counts),
        n_subsets=args.n_subsets,
        rng_seed=args.subset_rng,
    )

    sample_dir = args.out_dir / f"sample_{args.sample_id:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(sample_dir / "freq.npy", freq)
    np.save(sample_dir / "truth_af_central_stack.npy", truth_arr)
    np.save(sample_dir / "pred_af_central_stack.npy", pred_arr)

    phys = replicate_rows[0]
    meta = {
        "sample_id": args.sample_id,
        "n_replicates_processed": len(per_seed_rows),
        "n_replicates_missing": missing,
        "max_seeds": args.max_seeds,
        "seeds_per_sample": seeds_per,
        "truth_source": args.truth_source,
        "Vs1": phys.Vs1,
        "H_discretized": phys.H_discretized,
        "CoV": phys.CoV,
        "rH": phys.rH,
        "aHV": phys.aHV,
        "Vs2": phys.Vs2,
        "h5_dirs": [str(d) for d in h5_dirs],
        "tf_path": str(tf_path),
        "manifest_path": str(manifest_path),
        "checkpoint": str(checkpoint),
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    write_csv(sample_dir / "per_seed_metrics.csv", per_seed_rows)
    write_csv(sample_dir / "seed_cv_metrics.csv", cv_rows)

    summary_path = args.out_dir / "summary.json"
    summaries = []
    if summary_path.is_file():
        summaries = json.loads(summary_path.read_text())
    summaries = [s for s in summaries if s.get("sample_id") != args.sample_id]
    summaries.append(
        {
            **meta,
            "rel_l2_central_mean": float(
                np.mean([r["rel_l2_central"] for r in per_seed_rows])
            ),
            "rel_l2_central_std": float(
                np.std([r["rel_l2_central"] for r in per_seed_rows])
            ),
            "pearson_central_mean": float(
                np.mean([r["pearson_central"] for r in per_seed_rows])
            ),
            "sigma_ln_mean_truth": float(np.mean(sigma_ln_per_freq(truth_arr))),
            "sigma_ln_mean_pred": float(np.mean(sigma_ln_per_freq(pred_arr))),
        }
    )
    summary_path.write_text(json.dumps(summaries, indent=2))

    print(f"[seed_robustness] Wrote -> {sample_dir}")
    print(
        f"  rel_l2_central mean={summaries[-1]['rel_l2_central_mean']:.4f}  "
        f"sigma_ln truth={summaries[-1]['sigma_ln_mean_truth']:.4f}  "
        f"pred={summaries[-1]['sigma_ln_mean_pred']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
